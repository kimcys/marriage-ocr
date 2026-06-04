from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping

from marriage_ocr.config import load_runtime_config
from marriage_ocr.logging_config import get_logger
from marriage_ocr.models import ExtractedRecord
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


ProgressCallback = Callable[[ProcessProgress], None]


def process_input(
    *,
    input_path: Path,
    output_path: Path | None,
    debug_path: Path,
    config_path: Path,
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

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    debug_path.mkdir(parents=True, exist_ok=True)

    input_cfg = cfg.get("input", {})
    preprocessing_cfg = cfg.get("preprocessing", {})
    ocr_cfg = cfg.get("ocr", {})
    export_cfg = cfg.get("export", {})
    layout_cfg = cfg.get("layout", {})
    validation_cfg = cfg.get("validation", {})
    llm_cfg = cfg.get("llm", {})
    validation_input = {
        **validation_cfg,
        "min_average_confidence": ocr_cfg.get("min_average_confidence", 0.50),
    }
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
    total_parsed_records = 0
    status_counts: dict[str, int] = {}
    validated_records: list[ExtractedRecord] = []
    ocr_engine = None if layout_only else build_ocr_engine(ocr_cfg)

    for index, page in enumerate(pages, start=1):
        page_debug_dir = debug_path / page.debug_name
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
                    save_raw_json=bool(ocr_cfg.get("save_raw_json", True)),
                )
                total_ocr_cells += 1
                page_ocr_cells = 1
            else:
                record_ocr_outputs = run_ocr_on_record_crops(
                    saved_records,
                    ocr_engine,
                    save_raw_json=bool(ocr_cfg.get("save_raw_json", True)),
                )
                total_ocr_cells += page_ocr_cells

            for record_output, record_layout in zip(record_ocr_outputs, layout.records, strict=True):
                parsed_record = parse_record_ocr_output(record_output)
                save_parsed_record(parsed_record, record_output.record_dir / "parsed_record.json")
                layout_confidence = estimate_layout_confidence(
                    marker_present=record_layout.marker_box is not None,
                    cell_count=len(record_layout.cells),
                    record_height=record_layout.box.height,
                    min_record_height=int(layout_cfg.get("min_record_height_px", 80)),
                    max_record_height=int(layout_cfg.get("max_record_height_px", 280)),
                )
                validated_record = _validate_record_with_optional_gemini(
                    parsed_record=parsed_record,
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
            + f"; saved debug to {page_debug_dir}"
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

    completion_message = "[bold green]Marriage OCR process complete[/bold green] " + (
        f"generated preprocessing, layout, crops, OCR JSON for {total_ocr_cells} cell crop(s), "
        f"and parsed/validated {total_parsed_records} record(s) across {total_records} record(s) on {len(pages)} page(s)"
        f" [{status_summary}]"
        + (
            f"; XLSX wrote {export_summary.written_count} row(s) and skipped "
            f"{export_summary.skipped_duplicates} duplicate(s) to {export_summary.output_path}"
            if export_summary is not None
            else ""
        )
        if ocr_engine is not None
        else f"generated preprocessing, layout, and {total_records} record crop(s) across {len(pages)} page(s)"
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

    return ProcessResult(
        records=validated_records,
        total_pages=len(pages),
        total_detected_records=total_records,
        total_parsed_records=total_parsed_records,
        status_counts=status_counts,
        output_path=export_summary.output_path if export_summary is not None else None,
        debug_path=debug_path,
    )


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
