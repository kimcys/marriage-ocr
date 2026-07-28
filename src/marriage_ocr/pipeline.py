from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping

from marriage_ocr.config import load_runtime_config
from marriage_ocr.logging_config import get_logger
from marriage_ocr.models import ExtractedRecord
from marriage_ocr.refinement import field_refinement as refinement_engine
from marriage_ocr.refinement.audit import save_refinement_audit_sidecar, write_refinement_audit
from marriage_ocr.refinement.field_refinement import refine_field
from marriage_ocr.refinement.models import (
    FieldCandidate,
    FieldRefinementAuditRow,
    FieldRefinementDecision,
    FieldRefinementSettings,
)
from marriage_ocr.validation import validate_record
from llm import GeminiRecordExtractor, merge_parser_and_gemini


@dataclass(frozen=True)
class ProcessProgress:
    page_index: int
    page_total: int
    source_file: str
    source_page: int
    detected_records: int
    parsed_records: int
    message: str


@dataclass(frozen=True)
class ProcessResult:
    records: list[ExtractedRecord]
    total_pages: int
    total_detected_records: int
    total_parsed_records: int
    status_counts: dict[str, int]
    output_path: Path | None
    debug_path: Path
    refinement_ocr_calls: int = 0
    refinement_audit_rows: list[FieldRefinementAuditRow] = field(default_factory=list)


ProgressCallback = Callable[[ProcessProgress], None]


REFINEMENT_FIELD_CROPS: dict[str, str] = {
    "nama_suami": "suami_isteri",
    "ic_lama_suami": "suami_isteri",
    "ic_baru_suami": "suami_isteri",
    "nama_isteri": "suami_isteri",
    "ic_lama_isteri": "suami_isteri",
    "ic_baru_isteri": "suami_isteri",
    "nama_pendaftar": "pendaftar",
    "nama_wali": "wali",
    "saksi_1": "saksi",
    "saksi_2": "saksi",
    "tarikh_nikah": "tarikh_nikah",
    "tarikh_keluar": "tarikh_keluar",
}


def process_input(
    *,
    input_path: Path,
    output_path: Path | None,
    debug_path: Path,
    config_path: Path,
    retain_debug_artifacts: bool | None = None,
    reset_output: bool = False,
    layout_only: bool = False,
    skip_existing: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> ProcessResult:
    """Run the OCR pipeline for an input path and return the parsed records."""

    from marriage_ocr.cropper import save_record_crops
    from marriage_ocr.document_loader import load_document_pages, write_image
    from marriage_ocr.exporter import export_records_to_xlsx
    from marriage_ocr.layout import create_record_overlay, create_table_overlay, detect_layout
    from marriage_ocr.ocr import build_ocr_engine, run_ocr_on_page_layout, run_ocr_on_record_crops
    from marriage_ocr.parser import parse_record_ocr_output, save_parsed_record
    from marriage_ocr.postprocess import correct_bil_sequence
    from marriage_ocr.preprocess import PreprocessSettings, preprocess_image
    from marriage_ocr.validation import estimate_layout_confidence

    loaded = load_runtime_config(config_path)
    cfg = loaded.data
    logger = get_logger("marriage_ocr.process")
    debug_cfg = cfg.get("debug", {})
    if retain_debug_artifacts is None:
        retain_debug_artifacts = bool(debug_cfg.get("retain_artifacts", False))

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_debug_workspace: tempfile.TemporaryDirectory[str] | None = None
    debug_root = debug_path
    if retain_debug_artifacts:
        debug_root.mkdir(parents=True, exist_ok=True)
    else:
        temp_debug_workspace = tempfile.TemporaryDirectory(prefix="marriage-ocr-debug-")
        debug_root = Path(temp_debug_workspace.name)

    input_cfg = cfg.get("input", {})
    preprocessing_cfg = cfg.get("preprocessing", {})
    ocr_cfg = cfg.get("ocr", {})
    export_cfg = cfg.get("export", {})
    layout_cfg = cfg.get("layout", {})
    validation_cfg = cfg.get("validation", {})
    llm_cfg = dict(cfg.get("llm", {}))
    refinement_settings = FieldRefinementSettings.from_config(cfg)
    validation_input = {
        **validation_cfg,
        "min_average_confidence": ocr_cfg.get("min_average_confidence", 0.50),
    }
    ocr_save_raw_json = bool(ocr_cfg.get("save_raw_json", True)) and retain_debug_artifacts
    llm_cfg["save_raw_json"] = bool(llm_cfg.get("save_raw_json", True)) and retain_debug_artifacts
    gemini_processor = None if layout_only else _build_gemini_record_processor(
        llm_cfg,
        validation_config=validation_input,
    )
    gemini_state: dict[str, bool] = {"disabled": False}

    pages = load_document_pages(
        input_path,
        input_cfg.get("allowed_extensions", []),
        pdf_dpi=int(input_cfg.get("pdf_dpi", 200)),
    )

    settings = PreprocessSettings(
        processing_width=int(preprocessing_cfg.get("processing_width", 2200)),
        expected_landscape=bool(preprocessing_cfg.get("expected_landscape", True)),
        denoise_kernel_size=int(preprocessing_cfg.get("denoise_kernel_size", 5)),
        threshold_method=str(preprocessing_cfg.get("threshold_method", "adaptive")),
        adaptive_block_size=int(preprocessing_cfg.get("adaptive_block_size", 31)),
        adaptive_c=int(preprocessing_cfg.get("adaptive_c", 15)),
        deskew_enabled=bool(preprocessing_cfg.get("deskew_enabled", True)),
        deskew_max_angle=float(preprocessing_cfg.get("deskew_max_angle", 7.5)),
        min_rotation_for_deskew=float(preprocessing_cfg.get("min_rotation_for_deskew", 0.15)),
        hough_threshold=int(preprocessing_cfg.get("hough_threshold", 150)),
        hough_min_line_length_ratio=float(preprocessing_cfg.get("hough_min_line_length_ratio", 0.35)),
        hough_max_line_gap=int(preprocessing_cfg.get("hough_max_line_gap", 20)),
    )

    logger.info("Discovered %s page(s)", len(pages))
    _emit_progress(
        progress_callback,
        page_index=0,
        page_total=len(pages),
        source_file="",
        source_page=0,
        detected_records=0,
        parsed_records=0,
        message=f"Discovered {len(pages)} page(s)",
    )

    total_records = 0
    total_ocr_cells = 0
    total_refinement_ocr_calls = 0
    total_parsed_records = 0
    status_counts: dict[str, int] = {}
    refinement_audit_rows: list[FieldRefinementAuditRow] = []
    validated_records: list[ExtractedRecord] = []
    ocr_engine = None if layout_only else build_ocr_engine(ocr_cfg)

    for index, page in enumerate(pages, start=1):
        page_debug_dir = debug_root / page.debug_name
        write_image(page_debug_dir / "original.jpg", page.image)

        processed = preprocess_image(page.image, settings)
        write_image(page_debug_dir / "preprocessed.jpg", processed.binary)
        layout = detect_layout(processed.color, processed.binary, layout_cfg)
        write_image(page_debug_dir / "table_lines_overlay.jpg", create_table_overlay(processed.color, layout))
        write_image(page_debug_dir / "record_boxes_overlay.jpg", create_record_overlay(processed.color, layout))
        saved_records = save_record_crops(page_debug_dir, layout, processed.color, write_image)
        total_records += len(layout.records)
        page_ocr_cells = sum(len(record.cell_paths) for record in saved_records)

        page_parsed_records = 0
        if ocr_engine is not None:
            ocr_mode = str(
                ocr_cfg.get("mode", "full_page" if ocr_engine.name == "google_vision" else "cell_crops")
            ).strip().lower()
            if ocr_mode in {"full_page", "page", "page_layout"}:
                page_ocr_path = page_debug_dir / "google_vision_full_page.jpg"
                write_image(page_ocr_path, processed.color)
                record_ocr_outputs = run_ocr_on_page_layout(
                    page_ocr_path,
                    layout,
                    saved_records,
                    ocr_engine,
                    save_raw_json=ocr_save_raw_json,
                )
                total_ocr_cells += 1
                page_ocr_cells = 1
            else:
                record_ocr_outputs = run_ocr_on_record_crops(
                    saved_records,
                    ocr_engine,
                    save_raw_json=ocr_save_raw_json,
                )
                total_ocr_cells += page_ocr_cells

            for record_output, record_layout, saved_record in zip(
                record_ocr_outputs,
                layout.records,
                saved_records,
                strict=True,
            ):
                parsed_record = parse_record_ocr_output(
                    record_output,
                    include_crop_folder=retain_debug_artifacts,
                )
                refined_record = parsed_record
                record_refinement_rows: list[FieldRefinementAuditRow] = []
                if ocr_engine is not None and refinement_settings.enabled:
                    refined_record, record_refinement_rows, refinement_calls = refine_record_fields(
                        parsed_record=parsed_record,
                        record_output=record_output,
                        record_crops=saved_record,
                        ocr_engine=ocr_engine,
                        settings=refinement_settings,
                        source_file=str(page.relative_source),
                        source_page=page.source_page,
                    )
                    total_refinement_ocr_calls += refinement_calls
                    refinement_audit_rows.extend(record_refinement_rows)
                if retain_debug_artifacts:
                    save_parsed_record(refined_record, record_output.record_dir / "parsed_record.json")
                    if record_refinement_rows:
                        save_refinement_audit_sidecar(record_output.record_dir, record_refinement_rows)
                layout_confidence = estimate_layout_confidence(
                    marker_present=record_layout.marker_box is not None,
                    cell_count=len(record_layout.cells),
                    record_height=record_layout.box.height,
                    min_record_height=int(layout_cfg.get("min_record_height_px", 80)),
                    max_record_height=int(layout_cfg.get("max_record_height_px", 280)),
                )
                validated_record = _validate_record_with_optional_gemini(
                    parsed_record=refined_record,
                    record_output=record_output,
                    layout_confidence=layout_confidence,
                    gemini_processor=gemini_processor,
                    gemini_state=gemini_state,
                    validation_config=validation_input,
                    logger=logger,
                    source_file=str(page.relative_source),
                    source_page=page.source_page,
                )
                validated_record = replace(
                    validated_record,
                    source_file=str(page.relative_source),
                    source_page=page.source_page,
                    source_record=validated_record.source_record or f"record_{record_output.record_index:03d}",
                )
                if retain_debug_artifacts:
                    save_parsed_record(validated_record, record_output.record_dir / "validated_record.json")
                status_counts[validated_record.status_review] = status_counts.get(validated_record.status_review, 0) + 1
                validated_records.append(validated_record)
                page_parsed_records += 1
                total_parsed_records += 1

        message = (
            f"[green]{index}/{len(pages)}[/green] "
            f"{page.relative_source} page {page.source_page}: "
            f"detected {len(layout.records)} record(s)"
            + (
                f"; OCR saved for {page_ocr_cells} cell crop(s); parsed and validated {page_parsed_records} record(s)"
                if ocr_engine is not None
                else "; layout-only mode skipped OCR"
            )
            + (
                f"; retained debug artifacts at {page_debug_dir}"
                if retain_debug_artifacts
                else "; debug artifacts were not retained"
            )
        )
        _emit_progress(
            progress_callback,
            page_index=index,
            page_total=len(pages),
            source_file=str(page.relative_source),
            source_page=page.source_page,
            detected_records=len(layout.records),
            parsed_records=page_parsed_records,
            message=message,
        )
        logger.info(
            "Processed page %s/%s source=%s page=%s records=%s ocr_cells=%s debug_dir=%s",
            index,
            len(pages),
            page.relative_source,
            page.source_page,
            len(layout.records),
            page_ocr_cells if ocr_engine is not None else 0,
            page_debug_dir,
        )

    postprocess_cfg = dict(cfg.get("postprocess", {}))
    bil_sequence_cfg = dict(postprocess_cfg.get("bil_sequence", {}))
    if validated_records and bil_sequence_cfg.get("enabled", True):
        validated_records = correct_bil_sequence(
            validated_records,
            enabled=bool(bil_sequence_cfg.get("enabled", True)),
            start_number=bil_sequence_cfg.get("start_number"),
            year=bil_sequence_cfg.get("year"),
        )

    status_summary = ", ".join(f"{name}={count}" for name, count in sorted(status_counts.items())) or "no validated records"
    export_summary = None
    if ocr_engine is not None and output_path is not None:
        export_summary = export_records_to_xlsx(
            validated_records,
            output_path,
            export_cfg,
            reset_output=reset_output,
            skip_existing=skip_existing,
        )
    if retain_debug_artifacts and refinement_audit_rows:
        write_refinement_audit(refinement_audit_rows, debug_root / "refinement_audit.csv")

    if ocr_engine is not None:
        completion_message = "[bold green]Marriage OCR process complete[/bold green] "
        completion_message += (
            f"generated preprocessing, layout, OCR results for {total_ocr_cells} cell crop(s), "
            f"and parsed/validated {total_parsed_records} record(s) across {total_records} record(s) on {len(pages)} page(s)"
        )
        if total_refinement_ocr_calls:
            completion_message += f"; refinement OCR calls {total_refinement_ocr_calls}"
        completion_message += f" [{status_summary}]"
        if export_summary is not None:
            completion_message += (
                f"; XLSX wrote {export_summary.written_count} row(s) and skipped "
                f"{export_summary.skipped_duplicates} duplicate(s) to {export_summary.output_path}"
            )
        completion_message += (
            f"; retained debug artifacts at {debug_root}"
            if retain_debug_artifacts
            else "; debug artifacts were not retained"
        )
    else:
        completion_message = (
            f"[bold green]Marriage OCR process complete[/bold green] generated preprocessing, layout, "
            f"and {total_records} record crop(s) across {len(pages)} page(s)"
        )
    _emit_progress(
        progress_callback,
        page_index=len(pages),
        page_total=len(pages),
        source_file="",
        source_page=0,
        detected_records=total_records,
        parsed_records=total_parsed_records,
        message=completion_message,
    )
    logger.info(
        "Process completed records=%s parsed_records=%s ocr_cells=%s statuses=%s export=%s",
        total_records,
        total_parsed_records,
        total_ocr_cells,
        status_summary,
        export_summary.output_path if export_summary is not None else None,
    )

    if loaded.env_file is not None:
        logger.debug("Effective environment sourced from %s", loaded.env_file)

    result = ProcessResult(
        records=validated_records,
        total_pages=len(pages),
        total_detected_records=total_records,
        total_parsed_records=total_parsed_records,
        status_counts=status_counts,
        output_path=export_summary.output_path if export_summary is not None else None,
        debug_path=debug_path,
        refinement_ocr_calls=total_refinement_ocr_calls,
        refinement_audit_rows=refinement_audit_rows,
    )
    if temp_debug_workspace is not None:
        temp_debug_workspace.cleanup()
    return result


def _emit_progress(
    progress_callback: ProgressCallback | None,
    *,
    page_index: int,
    page_total: int,
    source_file: str,
    source_page: int,
    detected_records: int,
    parsed_records: int,
    message: str,
) -> None:
    if progress_callback is None:
        return

    progress_callback(
        ProcessProgress(
            page_index=page_index,
            page_total=page_total,
            source_file=source_file,
            source_page=source_page,
            detected_records=detected_records,
            parsed_records=parsed_records,
            message=message,
        )
    )


class _CountingOcrEngineProxy:
    def __init__(self, engine: Any) -> None:
        self._engine = engine
        self.name = getattr(engine, "name", "refinement")
        self.read_calls = 0

    def read_image(self, image_path: str | Path):
        self.read_calls += 1
        return self._engine.read_image(image_path)


def refine_record_fields(
    *,
    parsed_record: ExtractedRecord,
    record_output: Any,
    record_crops: Any,
    ocr_engine: Any,
    settings: FieldRefinementSettings,
    source_file: str | None = None,
    source_page: int | None = None,
) -> tuple[ExtractedRecord, list[FieldRefinementAuditRow], int]:
    if not settings.enabled:
        return parsed_record, [], 0

    refined_record = parsed_record
    audit_rows: list[FieldRefinementAuditRow] = []
    total_retry_ocr_calls = 0

    for field_name, crop_name in REFINEMENT_FIELD_CROPS.items():
        crop_path = getattr(record_crops, "cell_paths", {}).get(crop_name)
        if crop_path is None:
            continue

        current_value = getattr(refined_record, field_name, None)
        original_value = _refinement_original_value(refined_record, field_name, current_value)
        counting_engine = _CountingOcrEngineProxy(ocr_engine)
        try:
            decision = refine_field(
                field_name,
                original_value,
                parsed_value=current_value,
                crop_path=crop_path,
                ocr_engine=counting_engine,
                settings=settings,
            )
        except Exception as error:
            decision = _build_failed_refinement_decision(
                field_name=field_name,
                original_value=original_value,
                selected_value=current_value,
                reason=f"refinement_error:{error.__class__.__name__}",
            )
        retry_calls = counting_engine.read_calls
        total_retry_ocr_calls += retry_calls

        if decision.selected_value is not None and decision.selected_value != current_value:
            refined_record = replace(refined_record, **{field_name: decision.selected_value})

        audit_rows.append(
            _build_refinement_audit_row(
                decision=decision,
                source_file=source_file,
                source_page=source_page,
                record_index=getattr(record_output, "record_index", 0),
                crop_path=crop_path,
                retry_count=retry_calls,
            )
        )

    return refined_record, audit_rows, total_retry_ocr_calls


def _refinement_original_value(
    record: ExtractedRecord,
    field_name: str,
    parsed_value: str | None,
) -> str | None:
    raw_field_map = {
        "tarikh_nikah": "tarikh_nikah_raw",
        "tarikh_keluar": "tarikh_keluar_raw",
    }
    raw_field_name = raw_field_map.get(field_name)
    if raw_field_name is None:
        return parsed_value
    return getattr(record, raw_field_name, None) or parsed_value


def _build_failed_refinement_decision(
    *,
    field_name: str,
    original_value: str | None,
    selected_value: str | None,
    reason: str,
) -> FieldRefinementDecision:
    candidates: tuple[FieldCandidate, ...] = ()
    if isinstance(selected_value, str) and selected_value.strip():
        candidates = (
            FieldCandidate(
                value=selected_value.strip(),
                source="original_ocr",
                validity_score=0.0,
                ocr_confidence=None,
                plausibility_score=0.0,
                similarity_score=1.0,
                substitutions=0,
                metadata={"field_name": field_name},
            ),
        )
    return FieldRefinementDecision(
        field_name=field_name,
        original_value=original_value,
        selected_value=selected_value,
        candidates=candidates,
        selected_candidate=candidates[0] if candidates else None,
        requires_review=True,
        reason=reason,
    )


def _build_refinement_audit_row(
    *,
    decision: Any,
    source_file: str | None,
    source_page: int | None,
    record_index: int,
    crop_path: Path,
    retry_count: int,
) -> FieldRefinementAuditRow:
    selected_candidate = decision.selected_candidate
    original_candidate = refinement_engine._find_original_candidate(list(decision.candidates))
    selected_score = refinement_engine._candidate_score(selected_candidate, reference_value=decision.original_value)
    original_score = refinement_engine._candidate_score(original_candidate, reference_value=decision.original_value)
    return FieldRefinementAuditRow(
        source_file=source_file or "",
        page_number=source_page or 0,
        record_index=record_index,
        field_name=decision.field_name,
        original_value=decision.original_value,
        selected_value=decision.selected_value,
        original_score=original_score,
        selected_score=selected_score,
        correction_type=(
            str(selected_candidate.metadata.get("correction_type", selected_candidate.source))
            if selected_candidate is not None
            else "original_ocr"
        ),
        candidate_source=(selected_candidate.source if selected_candidate is not None else "original_ocr"),
        reason=decision.reason,
        requires_review=decision.requires_review,
        crop_path=str(crop_path),
        retry_count=retry_count,
    )


def _build_gemini_record_processor(
    llm_config: Mapping[str, Any],
    *,
    validation_config: Mapping[str, Any],
) -> Callable[[ExtractedRecord, Any], ExtractedRecord] | None:
    if not bool(llm_config.get("enabled", False)):
        return None

    provider = str(llm_config.get("provider", "")).strip().lower()
    if provider != "gemini":
        raise ValueError(
            "Unsupported llm.provider value: "
            f"{provider or '(missing)'}. Only 'gemini' is implemented."
        )

    extractor = GeminiRecordExtractor(llm_config)
    prefer_gemini_threshold = float(llm_config.get("prefer_gemini_threshold", 0.70))
    review_below_field_confidence = float(llm_config.get("review_below_field_confidence", 0.80))

    def _process(
        parser_record: ExtractedRecord,
        record_output: Any,
        *,
        layout_confidence: float,
    ) -> ExtractedRecord:
        gemini_result = extractor.extract_record(
            record_crop_path=record_output.record_dir / "full_record.jpg",
            ocr_cells=record_output.cell_results,
        )
        return merge_parser_and_gemini(
            parser_record=parser_record,
            gemini_result=gemini_result,
            cell_results=record_output.cell_results,
            validation_config=validation_config,
            layout_confidence=layout_confidence,
            prefer_gemini_threshold=prefer_gemini_threshold,
            review_below_field_confidence=review_below_field_confidence,
        )

    return _process


def _validate_record_with_optional_gemini(
    *,
    parsed_record: ExtractedRecord,
    record_output: Any,
    layout_confidence: float,
    gemini_processor: Callable[[ExtractedRecord, Any], ExtractedRecord] | None,
    gemini_state: dict[str, bool] | None,
    validation_config: Mapping[str, Any],
    logger: Any,
    source_file: str,
    source_page: int,
) -> ExtractedRecord:
    if gemini_processor is None or (gemini_state is not None and gemini_state.get("disabled", False)):
        return validate_record(
            parsed_record,
            record_output.cell_results,
            validation_config,
            layout_confidence=layout_confidence,
        )

    try:
        return gemini_processor(
            parsed_record,
            record_output,
            layout_confidence=layout_confidence,
        )
    except Exception as error:
        logger.warning(
            "Gemini semantic extraction failed for %s page %s; falling back to parser-only validation: %s",
            source_file,
            source_page,
            error,
        )
        validated_record = validate_record(
            parsed_record,
            record_output.cell_results,
            validation_config,
            layout_confidence=layout_confidence,
        )
        fallback_reason = f"Gemini unavailable: {error.__class__.__name__}"
        review_reason = list(validated_record.review_reason or [])
        if fallback_reason not in review_reason:
            review_reason.append(fallback_reason)
        status_review = validated_record.status_review
        if status_review == "OK":
            status_review = "REVIEW"
        if gemini_state is not None and _should_disable_gemini_for_run(error):
            gemini_state["disabled"] = True
            logger.warning(
                "Gemini disabled for the remainder of this run after %s on %s page %s",
                error.__class__.__name__,
                source_file,
                source_page,
            )
        return replace(
            validated_record,
            review_reason=review_reason,
            status_review=status_review,
        )


def _should_disable_gemini_for_run(error: Exception) -> bool:
    error_text = " ".join(
        str(part)
        for part in (
            error.__class__.__name__,
            *getattr(error, "args", ()),
            str(error),
        )
    ).lower()
    return any(
        token in error_text
        for token in (
            "reported as leaked",
            "permission_denied",
            "permission denied",
            "unauthenticated",
            "resource_exhausted",
            "quota exceeded",
            "too many requests",
            "429",
            "403",
            "401",
        )
    )
