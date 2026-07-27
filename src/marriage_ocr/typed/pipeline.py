from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import cv2

from marriage_ocr.config import load_runtime_config
from marriage_ocr.logging_config import get_logger
from marriage_ocr.models import ExtractedRecord
from marriage_ocr.typed.csv_writer import TypedCsvStore
from marriage_ocr.typed.extractor import FIELD_OUTPUT_NAMES, extract_raw_fields
from marriage_ocr.typed.loader import discover_typed_pdfs, render_typed_pdf
from marriage_ocr.typed.models import (
    FieldDiagnostic,
    PageOcrResult,
    ProcessingStatus,
    RawField,
    RenderedPage,
    RetryCrop,
    TypedBatchResult,
    TypedDocumentResult,
    ValidationSummary,
)
from marriage_ocr.typed.normalizer import build_extracted_record
from marriage_ocr.typed.retry import create_retry_crops, extract_retry_raw_fields, prefer_retry_value
from marriage_ocr.typed.validator import RETRY_PRIORITY, status_for_result, validate_record
from marriage_ocr.typed.vision import TypedVisionClient


ProgressCallback = Callable[[str], None]


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, ProcessingStatus):
        return value.value
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, set):
        return sorted(value)
    return str(value)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _ocr_micro_batch(pages: Sequence[RenderedPage], client: TypedVisionClient) -> tuple[PageOcrResult, ...]:
    return client.annotate_pages(pages)


def _write_region_overlay(
    page: RenderedPage,
    fields: Mapping[str, RawField],
    output_path: Path,
) -> None:
    image = cv2.imread(str(page.image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read rendered page for overlay: {page.image_path}")
    for field_key, raw_field in fields.items():
        if raw_field.page_number != page.page_number:
            continue
        region = raw_field.region
        x1 = int(region.x1 * page.width)
        y1 = int(region.y1 * page.height)
        x2 = int(region.x2 * page.width)
        y2 = int(region.y2 * page.height)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), 2)
        cv2.putText(
            image,
            field_key,
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise OSError(f"Failed to write region overlay: {output_path}")


def _record_failure(source_file: str, error_message: str, *, debug_dir: Path | None = None) -> TypedDocumentResult:
    record = ExtractedRecord(source_file=source_file)
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        _write_json(debug_dir / "validation.json", {"error_message": error_message, "status": "FAILED"})
    return TypedDocumentResult(
        record=record,
        source_file=source_file,
        processing_status=ProcessingStatus.FAILED,
        error_message=error_message,
    )


def _serialise_summary(summary: ValidationSummary) -> dict[str, object]:
    return {
        "diagnostics": {key: asdict(value) for key, value in summary.diagnostics.items()},
        "retry_fields": list(summary.retry_fields),
        "failed_fields": list(summary.failed_fields),
        "meaningful_field_count": summary.meaningful_field_count,
    }


def _is_success_status(status: ProcessingStatus) -> bool:
    return status in {ProcessingStatus.SUCCESS, ProcessingStatus.SUCCESS_WITH_RETRY}


def _process_retry_fields(
    *,
    source_file: str,
    source_stem: str,
    debug_dir: Path,
    pages: tuple[RenderedPage, RenderedPage],
    raw_fields: dict[str, RawField],
    summary: ValidationSummary,
    client: TypedVisionClient,
    word_confidence_threshold: float,
    min_age: int,
    max_age: int,
    max_retry_fields: int,
) -> tuple[dict[str, RawField], int, dict[str, object]]:
    retry_fields = tuple(summary.retry_fields[:max_retry_fields])
    if not retry_fields:
        return raw_fields, 0, {}

    retry_dir = debug_dir / "retries"
    retry_crops = create_retry_crops(
        pages=pages,
        field_keys=retry_fields,
        retry_dir=debug_dir,
        padding_ratio=0.05,
    )
    retry_results = extract_retry_raw_fields(retry_crops, client)
    retry_payload: dict[str, object] = {}

    updated_fields = dict(raw_fields)
    retry_count = 0
    for crop in retry_crops:
        original = raw_fields[crop.field_key]
        retry_field = retry_results.get(crop.field_key)
        if retry_field is None:
            continue
        retry_payload[crop.field_key] = {
            "original": {
                "raw_text": original.raw_text,
                "confidence": original.confidence,
            },
            "retried": {
                "raw_text": retry_field.raw_text,
                "confidence": retry_field.confidence,
            },
        }
        _write_json(
            debug_dir / "retries" / f"{crop.field_key}.json",
            {
                "field_key": crop.field_key,
                "original": {
                    "raw_text": original.raw_text,
                    "confidence": original.confidence,
                    "region": asdict(original.region),
                },
                "retried": {
                    "raw_text": retry_field.raw_text,
                    "confidence": retry_field.confidence,
                    "region": asdict(retry_field.region),
                },
            },
        )
        original_record = build_extracted_record(updated_fields)
        original_summary = validate_record(
            original_record,
            updated_fields,
            word_confidence_threshold=word_confidence_threshold,
            min_age=min_age,
            max_age=max_age,
            max_retry_fields=max_retry_fields,
        )
        retry_candidate_fields = dict(updated_fields)
        retry_candidate_fields[crop.field_key] = retry_field
        retry_record = build_extracted_record(retry_candidate_fields)
        retry_summary = validate_record(
            retry_record,
            retry_candidate_fields,
            word_confidence_threshold=word_confidence_threshold,
            min_age=min_age,
            max_age=max_age,
            max_retry_fields=max_retry_fields,
        )
        current_valid = original_summary.diagnostics[crop.field_key].valid
        retry_valid = retry_summary.diagnostics[crop.field_key].valid
        chosen = prefer_retry_value(
            original,
            retry_field,
            original_valid=current_valid,
            retry_valid=retry_valid,
        )
        if chosen is retry_field:
            updated_fields[crop.field_key] = retry_field
            retry_count += 1
    if retry_payload:
        _write_json(retry_dir / f"{source_stem}_retry_summary.json", retry_payload)
    return updated_fields, retry_count, retry_payload


def _process_single_pdf(
    pdf: Path,
    *,
    debug_path: Path,
    client: TypedVisionClient,
    typed_cfg: Mapping[str, object],
    retry_cfg: Mapping[str, object],
    validation_cfg: Mapping[str, object],
    pages: tuple[RenderedPage, RenderedPage] | None = None,
    ocr_results_by_page: Mapping[tuple[str, int], PageOcrResult] | None = None,
) -> TypedDocumentResult:
    source_file = pdf.name
    source_stem = pdf.stem
    document_debug_dir = debug_path / source_stem
    document_debug_dir.mkdir(parents=True, exist_ok=True)
    try:
        if pages is None:
            pages = render_typed_pdf(pdf, document_debug_dir, dpi=int(typed_cfg.get("pdf_dpi", 300)))
    except Exception as error:
        return _record_failure(source_file, str(error), debug_dir=document_debug_dir)

    page_results: list[PageOcrResult] = []
    if ocr_results_by_page is not None:
        page_results = [
            ocr_results_by_page.get((source_file, page.page_number))
            for page in pages
        ]
        if any(result is None for result in page_results):
            return _record_failure(source_file, "Missing OCR results", debug_dir=document_debug_dir)
    else:
        try:
            page_results = list(_ocr_micro_batch(pages, client))
        except Exception as error:
            return _record_failure(source_file, str(error), debug_dir=document_debug_dir)

    assert all(result is not None for result in page_results)
    page_results = [result for result in page_results if result is not None]
    if any(result.error_message for result in page_results):
        return _record_failure(
            source_file,
            "; ".join(result.error_message for result in page_results if result.error_message),
            debug_dir=document_debug_dir,
        )

    _write_json(document_debug_dir / "full_page_vision.json", [asdict(result) for result in page_results])
    raw_fields = extract_raw_fields(page_results, boundary_tolerance=float(typed_cfg.get("region_boundary_tolerance", 0.01)))
    _write_json(document_debug_dir / "extracted_raw.json", {key: asdict(field) for key, field in raw_fields.items()})

    for page in pages:
        _write_region_overlay(page, raw_fields, document_debug_dir / f"page_{page.page_number}_regions.png")

    record = build_extracted_record(raw_fields)
    summary = validate_record(
        record,
        raw_fields,
        word_confidence_threshold=float(typed_cfg.get("word_confidence_threshold", 0.75)),
        min_age=int(validation_cfg.get("min_age", 16)),
        max_age=int(validation_cfg.get("max_age", 120)),
        max_retry_fields=int(retry_cfg.get("max_fields_per_pdf", 6)),
    )
    _write_json(document_debug_dir / "validation.json", _serialise_summary(summary))

    if summary.retry_fields:
        try:
            retry_updated_fields, retry_count, retry_payload = _process_retry_fields(
                source_file=source_file,
                source_stem=source_stem,
                debug_dir=document_debug_dir,
                pages=pages,
                raw_fields=raw_fields,
                summary=summary,
                client=client,
                word_confidence_threshold=float(typed_cfg.get("word_confidence_threshold", 0.75)),
                min_age=int(validation_cfg.get("min_age", 16)),
                max_age=int(validation_cfg.get("max_age", 120)),
                max_retry_fields=int(retry_cfg.get("max_fields_per_pdf", 6)),
            )
        except Exception as error:
            return _record_failure(source_file, str(error), debug_dir=document_debug_dir)
        raw_fields = retry_updated_fields
        record = build_extracted_record(raw_fields)
        summary = validate_record(
            record,
            raw_fields,
            word_confidence_threshold=float(typed_cfg.get("word_confidence_threshold", 0.75)),
            min_age=int(validation_cfg.get("min_age", 16)),
            max_age=int(validation_cfg.get("max_age", 120)),
            max_retry_fields=int(retry_cfg.get("max_fields_per_pdf", 6)),
        )
        _write_json(document_debug_dir / "extracted_raw.json", {key: asdict(field) for key, field in raw_fields.items()})
        _write_json(document_debug_dir / "validation.json", _serialise_summary(summary))
    else:
        retry_count = 0

    status = status_for_result(summary, retry_count=retry_count)
    record.source_file = source_file
    record.source_page = 1
    record.source_record = source_stem
    record.crop_folder = str(document_debug_dir)
    _write_json(document_debug_dir / "extracted_normalised.json", asdict(record))
    return TypedDocumentResult(
        record=record,
        source_file=source_file,
        processing_status=status,
        failed_fields=summary.failed_fields,
        retry_count=retry_count,
    )


def _process_micro_batch(
    *,
    pdfs: Sequence[Path],
    debug_path: Path,
    client: TypedVisionClient,
    typed_cfg: Mapping[str, object],
    retry_cfg: Mapping[str, object],
    validation_cfg: Mapping[str, object],
) -> tuple[TypedDocumentResult, ...]:
    render_workers = max(1, int(typed_cfg.get("render_workers", 4)))
    rendered: dict[str, tuple[RenderedPage, RenderedPage] | None] = {}
    errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=render_workers) as executor:
        futures = {
            pdf.name: executor.submit(
                render_typed_pdf,
                pdf,
                debug_path / pdf.stem,
                dpi=int(typed_cfg.get("pdf_dpi", 300)),
            )
            for pdf in pdfs
        }
        for source_file, future in futures.items():
            try:
                rendered[source_file] = future.result()
            except Exception as error:
                rendered[source_file] = None
                errors[source_file] = str(error)

    ordered_pages: list[RenderedPage] = []
    for pdf in pdfs:
        pages = rendered.get(pdf.name)
        if pages is None:
            continue
        ordered_pages.extend(sorted(pages, key=lambda page: page.page_number))

    ocr_results_by_page: dict[tuple[str, int], PageOcrResult] | None = None
    try:
        if ordered_pages:
            ocr_results = _ocr_micro_batch(ordered_pages, client)
            ocr_results_by_page = {(result.source_file, result.page_number): result for result in ocr_results}
    except Exception:
        ocr_results_by_page = {}
        for pdf in pdfs:
            pages = rendered.get(pdf.name)
            if pages is None:
                continue
            try:
                results = _ocr_micro_batch(pages, client)
            except Exception as error:
                errors[pdf.name] = str(error)
                continue
            for result in results:
                ocr_results_by_page[(result.source_file, result.page_number)] = result

    results: list[TypedDocumentResult] = []
    for pdf in pdfs:
        if pdf.name in errors and rendered.get(pdf.name) is None:
            results.append(_record_failure(pdf.name, errors[pdf.name], debug_dir=debug_path / pdf.stem))
            continue
        try:
            results.append(
                _process_single_pdf(
                    pdf,
                    debug_path=debug_path,
                    client=client,
                    typed_cfg=typed_cfg,
                    retry_cfg=retry_cfg,
                    validation_cfg=validation_cfg,
                    pages=pages,
                    ocr_results_by_page=ocr_results_by_page,
                )
            )
        except Exception as error:
            results.append(_record_failure(pdf.name, str(error), debug_dir=debug_path / pdf.stem))
    return tuple(results)


def process_typed_input(
    *,
    input_path: Path,
    output_path: Path,
    debug_path: Path,
    config_path: Path,
    reset_output: bool = False,
    skip_existing: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> TypedBatchResult:
    loaded = load_runtime_config(config_path)
    typed_cfg = dict(loaded.data.get("typed", {}))
    retry_cfg = dict(typed_cfg.get("retry", {}))
    validation_cfg = dict(typed_cfg.get("validation", {}))
    vision_cfg = dict(loaded.data.get("ocr", {}).get("google_vision", {}))
    pdfs = discover_typed_pdfs(input_path)
    store = TypedCsvStore.load(
        output_path,
        reset_output=reset_output,
        skip_existing=skip_existing,
    )
    pending = [pdf for pdf in pdfs if not store.should_skip(pdf.name)]
    skipped = tuple(pdf.name for pdf in pdfs if store.should_skip(pdf.name))
    client = TypedVisionClient(
        language_hints=tuple(vision_cfg.get("language_hints", ("ms", "en"))),
        api_attempts=int(retry_cfg.get("api_attempts", 3)),
        initial_delay_seconds=float(retry_cfg.get("initial_delay_seconds", 1)),
        backoff_multiplier=float(retry_cfg.get("backoff_multiplier", 2)),
        request_batch_size=int(retry_cfg.get("request_batch_size", 16)),
    )
    completed: list[TypedDocumentResult] = []
    batch_size = int(typed_cfg.get("pdf_batch_size", 4))
    for start in range(0, len(pending), batch_size):
        micro_batch = pending[start : start + batch_size]
        batch_results = _process_micro_batch(
            pdfs=micro_batch,
            debug_path=debug_path,
            client=client,
            typed_cfg=typed_cfg,
            retry_cfg=retry_cfg,
            validation_cfg=validation_cfg,
        )
        for result in batch_results:
            store.upsert(result)
            completed.append(result)
        store.flush()
        if progress_callback is not None:
            progress_callback(
                f"Processed {min(start + batch_size, len(pending))}/{len(pending)} typed PDF(s)"
            )

    ordered = tuple(sorted(completed, key=lambda result: result.source_file.casefold()))
    status_counts = Counter(result.processing_status.value for result in ordered)
    return TypedBatchResult(
        records=ordered,
        discovered_pdfs=len(pdfs),
        written_rows=len(ordered),
        skipped_files=skipped,
        status_counts=dict(status_counts),
    )
