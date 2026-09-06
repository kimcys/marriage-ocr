"""Full-Gemini-per-page pipeline for handwritten documents -- one Gemini
call reads a whole page image and returns every record on it, with no
Google Vision OCR step, no deterministic parser, and no per-cell cropping.

Selected via `pipeline.engine: gemini_page` in a handwritten config (see
config/handwritten.yaml and friends); process_input() in pipeline.py checks
that flag and delegates here before doing anything Vision-specific. Returns
the same ProcessResult shape as the Vision-based path so batch_runner.py,
the `process` CLI command, and everything downstream (DB insert, XLSX
export, the Streamlit review UI) work unchanged.

Why this exists: Vision OCR is ~76% of the current handwritten pipeline's
cost (cell_crops mode issues ~9-10 Vision calls per record, more once
field-refinement retries are counted), and real testing against actual
scans this session found full-Gemini-per-page's record segmentation held up
as well as, or better than, the current layout detector's (which has a
confirmed, separate bug -- see layout.py's _optimize_record_starts
docstring). See gemini_page_extractor.py's module docstring for the fuller
rationale and gemini_extractor.py's GeminiPageExtractor for the actual call.

What's genuinely different here, not just cheaper: there is no second
extraction path to disagree with Gemini, so review-flagging can't work the
way it does for the Vision-based pipeline (parser vs. Gemini agreement).
validate_gemini_only_record() in validation.py is the replacement -- see its
docstring for why it leans on field-value sanity checks rather than
Gemini's own reported confidence.
"""
from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path
import tempfile
from typing import Any

from marriage_ocr.config import load_runtime_config
from marriage_ocr.error_reporting import write_error_report
from marriage_ocr.logging_config import get_logger
from marriage_ocr.models import ExtractedRecord
from marriage_ocr.pipeline import ProcessResult, _emit_progress
from marriage_ocr.validation import validate_gemini_only_record
from llm import GeminiPageExtractor


def process_input_gemini_page(
    *,
    input_path: Path,
    output_path: Path | None,
    debug_path: Path,
    config_path: Path,
    retain_debug_artifacts: bool | None = None,
    reset_output: bool = False,
    skip_existing: bool = False,
    progress_callback=None,
) -> ProcessResult:
    """Same external contract as pipeline.process_input -- see its
    docstring for the parameter meanings. Called from there when
    `pipeline.engine: gemini_page` is set; not meant to be imported
    directly by callers that also need to handle the Vision-based path.
    """
    from marriage_ocr.document_loader import load_document_pages, write_image
    from marriage_ocr.exporter import export_records_to_csv, export_records_to_xlsx
    from marriage_ocr.parser import save_parsed_record
    from marriage_ocr.postprocess import correct_bil_sequence
    from marriage_ocr.preprocess import PreprocessSettings, preprocess_image

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
    export_cfg = cfg.get("export", {})
    validation_cfg = cfg.get("validation", {})
    llm_cfg = dict(cfg.get("llm", {}))
    record_type = str(cfg.get("record_type", "nikah")).strip().lower()
    layout_variant = str(cfg.get("layout_variant", "legacy")).strip().lower()
    llm_cfg.setdefault("record_type", record_type)
    llm_cfg.setdefault("layout_variant", layout_variant)
    llm_cfg["save_raw_json"] = bool(llm_cfg.get("save_raw_json", True)) and retain_debug_artifacts

    extractor = GeminiPageExtractor(llm_cfg)

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
    total_gemini_calls = 0
    status_counts: dict[str, int] = {}
    validated_records: list[ExtractedRecord] = []
    failed_pages: list[str] = []

    for index, page in enumerate(pages, start=1):
        try:
            page_debug_dir = debug_root / page.debug_name
            write_image(page_debug_dir / "original.jpg", page.image)

            processed = preprocess_image(page.image, settings)
            page_image_path = page_debug_dir / "preprocessed_color.jpg"
            write_image(page_image_path, processed.color)

            gemini_results = extractor.extract_page(page_image_path)
            total_gemini_calls += 1

            page_records_dir = page_debug_dir / "records"
            for record_index, gemini_result in enumerate(gemini_results, start=1):
                record = replace(
                    gemini_result.record,
                    source_file=str(page.relative_source),
                    source_page=page.source_page,
                    source_record=f"record_{record_index:03d}",
                )
                validated_record = validate_gemini_only_record(
                    record,
                    field_confidence=gemini_result.field_confidence,
                    uncertain_fields=gemini_result.uncertain_fields,
                    validation_config=validation_cfg,
                    record_type=record_type,
                )

                if retain_debug_artifacts:
                    record_dir = page_records_dir / f"record_{record_index:03d}"
                    record_dir.mkdir(parents=True, exist_ok=True)
                    # No per-record crop exists in this pipeline (there's no
                    # cropping step at all) -- copy the whole page in so a
                    # reviewer has something to check the record against.
                    shutil.copyfile(page_image_path, record_dir / "full_record.jpg")
                    save_parsed_record(validated_record, record_dir / "validated_record.json")

                status_counts[validated_record.status_review] = (
                    status_counts.get(validated_record.status_review, 0) + 1
                )
                validated_records.append(validated_record)
                total_records += 1

            message = (
                f"[green]{index}/{len(pages)}[/green] {page.relative_source} page {page.source_page}: "
                f"Gemini returned {len(gemini_results)} record(s)"
            )
            _emit_progress(
                progress_callback,
                page_index=index,
                page_total=len(pages),
                source_file=str(page.relative_source),
                source_page=page.source_page,
                detected_records=len(gemini_results),
                parsed_records=len(gemini_results),
                message=message,
            )
            logger.info(
                "Processed page %s/%s source=%s page=%s records=%s",
                index,
                len(pages),
                page.relative_source,
                page.source_page,
                len(gemini_results),
            )
        except Exception as error:
            failed_pages.append(str(page.relative_source))
            logger.exception(
                "Page %s/%s (%s page %s) failed and will be skipped",
                index,
                len(pages),
                page.relative_source,
                page.source_page,
            )
            write_error_report(
                error,
                command_name="process_page_gemini_page",
                extra_context={
                    "source_file": str(page.relative_source),
                    "source_page": page.source_page,
                    "page_index": index,
                    "page_total": len(pages),
                },
            )
            continue

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
    if output_path is not None:
        if output_path.suffix.lower() == ".csv":
            export_summary = export_records_to_csv(
                validated_records, output_path, export_cfg,
                reset_output=reset_output, skip_existing=skip_existing,
            )
        else:
            export_summary = export_records_to_xlsx(
                validated_records, output_path, export_cfg,
                reset_output=reset_output, skip_existing=skip_existing,
            )

    completion_message = (
        "[bold green]Marriage OCR process complete[/bold green] "
        f"generated {total_records} record(s) across {len(pages)} page(s) "
        f"via {total_gemini_calls} Gemini page call(s) [{status_summary}]"
    )
    if export_summary is not None:
        export_label = "CSV" if export_summary.output_path.suffix.lower() == ".csv" else "XLSX"
        completion_message += (
            f"; {export_label} wrote {export_summary.written_count} row(s) and skipped "
            f"{export_summary.skipped_duplicates} duplicate(s) to {export_summary.output_path}"
        )
    completion_message += (
        f"; retained debug artifacts at {debug_root}" if retain_debug_artifacts else "; debug artifacts were not retained"
    )
    if failed_pages:
        completion_message += (
            f" [bold red]-- {len(failed_pages)} page(s) FAILED and were skipped, "
            f"records for other pages were still saved; see error reports[/bold red]"
        )

    _emit_progress(
        progress_callback,
        page_index=len(pages),
        page_total=len(pages),
        source_file="",
        source_page=0,
        detected_records=total_records,
        parsed_records=total_records,
        message=completion_message,
    )
    logger.info(
        "Process completed records=%s gemini_calls=%s statuses=%s export=%s failed_pages=%s",
        total_records,
        total_gemini_calls,
        status_summary,
        export_summary.output_path if export_summary is not None else None,
        len(failed_pages),
    )
    if failed_pages:
        logger.warning("Pages that failed and were skipped: %s", failed_pages)

    return ProcessResult(
        records=validated_records,
        total_pages=len(pages),
        total_detected_records=total_records,
        total_parsed_records=total_records,
        status_counts=status_counts,
        output_path=export_summary.output_path if export_summary is not None else None,
        debug_path=debug_path,
        refinement_ocr_calls=0,
        refinement_audit_rows=[],
        failed_pages=failed_pages,
        gemini_calls=total_gemini_calls,
        gemini_calls_skipped=0,
    )
