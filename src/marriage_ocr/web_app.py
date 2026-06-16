from __future__ import annotations

import csv
from io import BytesIO
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence
from zipfile import ZIP_DEFLATED, ZipFile

if __package__ in {None, ""}:
    src_root = Path(__file__).resolve().parents[1]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

import pandas as pd
import streamlit as st

from marriage_ocr.batch_exporter import (
    export_csv_rows_to_parts,
    export_records_to_csv,
    export_records_to_csv_parts,
    export_records_to_xlsx_parts,
)
from marriage_ocr.config import load_runtime_config
from marriage_ocr.exporter import XLSX_COLUMNS, record_from_export_dict
from marriage_ocr.job_store import DEFAULT_JOBS_ROOT, OcrJob, create_job, list_jobs, load_job, save_job
from marriage_ocr.models import ExtractedRecord
from marriage_ocr.pipeline import ProcessProgress, ProcessResult, process_input


DEFAULT_CONFIG_PATH = Path("config/production.yaml")
DEFAULT_BATCH_SIZE = 5000
DEFAULT_ROWS_PER_XLSX_PART = 5000
MAX_PREVIEW_ROWS = 200
UPLOAD_TYPES = ["jpg", "jpeg", "png", "tif", "tiff", "pdf"]

ORIGINAL_CSV_NAME = "original_records.csv"
LEGACY_CSV_NAME = "records.csv"
CORRECTED_CSV_NAME = "corrected_records.csv"
CORRECTED_JSON_NAME = "corrected_records.json"
RECORDS_PART_PREFIX = "records_part"

EDITOR_COLUMNS = [
    "Bil",
    "Nama Suami",
    "IC Lama Suami",
    "IC Baru Suami",
    "ID Suami Raw",
    "Umur Suami",
    "Nama Isteri",
    "IC Lama Isteri",
    "IC Baru Isteri",
    "ID Isteri Raw",
    "Umur Isteri",
    "Mas Kahwin",
    "Mas Kahwin Raw",
    "Nama Pendaftar",
    "Alamat Pendaftar",
    "Nama Wali",
    "Hubungan Wali",
    "Saksi 1",
    "Saksi 2",
    "Tarikh Nikah",
    "Tarikh Nikah Raw",
    "Tarikh Keluar",
    "Tarikh Keluar Raw",
    "Remarks",
    "Confidence",
    "Status Review",
    "Review Reason",
    "Source File",
    "Source Page",
    "Source Record",
]


st.set_page_config(page_title="Marriage OCR Runner", layout="wide")


def main() -> None:
    st.title("Marriage OCR Runner")
    st.caption("Upload image/PDF files, review one batch at a time, and export corrected outputs.")

    settings = render_advanced_settings()
    upload_tab, review_tab, export_tab = st.tabs(["Upload & Run", "Review & Edit", "Export"])

    with upload_tab:
        render_upload_run(settings)
    with review_tab:
        render_review_edit(settings)
    with export_tab:
        render_export(settings)


def render_advanced_settings() -> dict[str, Any]:
    return {
        "config_path": DEFAULT_CONFIG_PATH,
        "batch_size": DEFAULT_BATCH_SIZE,
        "rows_per_xlsx_part": DEFAULT_ROWS_PER_XLSX_PART,
        "layout_only": False,
    }


def render_upload_run(settings: dict[str, Any]) -> None:
    st.subheader("Upload & Run")

    uploaded_files = list(
        st.file_uploader(
            "Upload image/PDF files",
            type=UPLOAD_TYPES,
            accept_multiple_files=True,
        )
        or []
    )

    if not uploaded_files:
        st.info("Upload one or more files to start.")
    else:
        file_batches = _split_uploaded_files(uploaded_files, int(settings["batch_size"]))
        st.write(f"Uploaded files: {len(uploaded_files)}")
        st.caption(", ".join(uploaded_file.name for uploaded_file in uploaded_files))
        st.caption(
            f"This run will create {len(file_batches)} batch(es) of up to {int(settings['batch_size'])} file(s) each."
        )

    run_requested = st.button("Run OCR", type="primary", disabled=not uploaded_files)

    if run_requested:
        config_path = Path(settings["config_path"])
        if not config_path.exists():
            st.error(f"Config path does not exist: {config_path}")
            return

        export_config = load_runtime_config(config_path).data.get("export", {})
        file_batches = _split_uploaded_files(uploaded_files, int(settings["batch_size"]))

        overall_progress = st.progress(0.0)
        overall_text = st.empty()
        batch_progress = st.progress(0.0)
        batch_text = st.empty()
        batch_meta = st.empty()

        processed_batches: list[dict[str, Any]] = []

        for batch_index, batch_files in enumerate(file_batches, start=1):
            overall_text.write(f"Running batch {batch_index}/{len(file_batches)}")
            batch_progress.progress(0.0)

            job = create_job(create_debug_dir=False)
            _save_uploaded_files(batch_files, Path(job.input_dir))

            job.status = "RUNNING"
            job.message = f"Batch {batch_index}/{len(file_batches)} started"
            save_job(job)

            def on_progress(progress: ProcessProgress) -> None:
                percent = 0.0 if progress.page_total <= 0 else min(1.0, progress.page_index / progress.page_total)
                batch_progress.progress(percent)
                batch_text.write(f"Batch {batch_index}/{len(file_batches)}: {progress.message}")
                batch_meta.caption(
                    "Page "
                    f"{progress.page_index}/{max(progress.page_total, 1)} | "
                    f"Detected {progress.detected_records} | Parsed {progress.parsed_records}"
                )
                job.message = f"Batch {batch_index}/{len(file_batches)}: {progress.message}"
                save_job(job)

            try:
                result = process_input(
                    input_path=Path(job.input_dir),
                    output_path=None,
                    debug_path=Path(job.debug_dir),
                    config_path=config_path,
                    reset_output=True,
                    layout_only=bool(settings["layout_only"]),
                    skip_existing=False,
                    progress_callback=on_progress,
                )
            except Exception as error:
                job.status = "FAILED"
                job.message = str(error)
                save_job(job)
                st.error(f"Batch {batch_index} failed: {error}")
                return

            original_csv_path = export_records_to_csv(result.records, Path(job.output_dir) / ORIGINAL_CSV_NAME)
            csv_part_paths, xlsx_paths = _prepare_original_export_parts(
                result.records,
                Path(job.output_dir),
                export_config,
                rows_per_file=int(settings["rows_per_xlsx_part"]),
            )

            job.status = "DONE"
            job.message = "OCR completed"
            job.total_pages = result.total_pages
            job.total_detected_records = result.total_detected_records
            job.total_parsed_records = result.total_parsed_records
            save_job(job)

            processed_batches.append(
                {
                    "job": load_job(job.job_id, root=DEFAULT_JOBS_ROOT),
                    "result": result,
                    "source_csv_path": original_csv_path,
                    "csv_part_paths": csv_part_paths,
                    "xlsx_paths": xlsx_paths,
                }
            )
            overall_progress.progress(batch_index / len(file_batches))

        overall_progress.progress(1.0)
        batch_progress.progress(1.0)
        st.session_state["last_job_ids"] = [item["job"].job_id for item in processed_batches]

        st.success(f"OCR completed for {len(processed_batches)} batch(es).")
        st.dataframe(pd.DataFrame(_jobs_table_rows([item["job"] for item in processed_batches])), use_container_width=True)

        selected_batch_id = st.selectbox(
            "Preview batch",
            options=[item["job"].job_id for item in processed_batches],
            format_func=lambda job_id: _job_label(job_id, [item["job"] for item in processed_batches]),
            key="upload_batch_preview",
        )
        selected_batch = next(item for item in processed_batches if item["job"].job_id == selected_batch_id)
        render_result_metrics(selected_batch["result"])
        render_result_downloads(
            source_csv_path=selected_batch["source_csv_path"],
            csv_parts=selected_batch["csv_part_paths"],
            xlsx_parts=selected_batch["xlsx_paths"],
            download_key_prefix=f"upload_{selected_batch_id}",
            source_label="Original batch records",
        )
        render_preview_rows(_build_preview_rows(selected_batch["result"].records))
        return

    last_job_ids = st.session_state.get("last_job_ids", [])
    if last_job_ids:
        recent_jobs = []
        for job_id in last_job_ids:
            try:
                recent_jobs.append(load_job(job_id, root=DEFAULT_JOBS_ROOT))
            except FileNotFoundError:
                continue
        if recent_jobs:
            st.divider()
            st.caption("Recent run")
            st.dataframe(pd.DataFrame(_jobs_table_rows(recent_jobs)), use_container_width=True)


def render_review_edit(settings: dict[str, Any]) -> None:
    st.subheader("Review & Edit")
    jobs = _jobs_with_records()
    if not jobs:
        st.info("No completed batches with exported records yet.")
        return

    selected_job_id = st.selectbox(
        "Select batch",
        options=[job.job_id for job in jobs],
        format_func=lambda job_id: _job_label(job_id, jobs),
        key="review_job_select",
    )
    selected_job = next(job for job in jobs if job.job_id == selected_job_id)
    output_dir = Path(selected_job.output_dir)

    rows, source_label, _ = _load_rows_for_review(output_dir)
    if not rows:
        st.warning("No records found for the selected batch.")
        return

    render_job_metrics(selected_job)
    st.caption(f"Editing source: {source_label}")

    start_index, end_index = _row_window_selector(len(rows), key_prefix=f"review_{selected_job.job_id}")
    st.caption(f"Editing rows {start_index + 1}-{end_index} of {len(rows)}.")

    editor_df = pd.DataFrame(_slice_editor_rows(rows, start_index, end_index), columns=EDITOR_COLUMNS)
    editor_df.index = range(start_index + 1, end_index + 1)
    edited_df = st.data_editor(
        editor_df,
        use_container_width=True,
        hide_index=False,
        num_rows="fixed",
        key=f"editor_{selected_job.job_id}_{start_index}",
    )

    action_columns = st.columns(2)
    if action_columns[0].button("Save Corrections", key=f"save_{selected_job.job_id}_{start_index}"):
        edited_rows = _normalize_dataframe_rows(edited_df)
        merged_rows = _merge_edited_rows(rows, edited_rows, start_index)
        corrected_csv_path, corrected_json_path = _write_corrected_rows(output_dir, merged_rows)
        st.success(f"Saved corrections to {corrected_csv_path.name} and {corrected_json_path.name}.")
        st.rerun()

    if action_columns[1].button("Reset Corrections", key=f"reset_{selected_job.job_id}"):
        _reset_corrections(output_dir)
        st.success("Reset corrections for the selected batch.")
        st.rerun()

    corrected_csv_path = output_dir / CORRECTED_CSV_NAME
    corrected_json_path = output_dir / CORRECTED_JSON_NAME
    if corrected_csv_path.exists() or corrected_json_path.exists():
        download_columns = st.columns(2)
        if corrected_csv_path.exists():
            with corrected_csv_path.open("rb") as handle:
                download_columns[0].download_button(
                    "Download corrected_records.csv",
                    data=handle,
                    file_name=corrected_csv_path.name,
                    mime="text/csv",
                    key=f"review_corrected_csv_{selected_job.job_id}",
                )
        if corrected_json_path.exists():
            with corrected_json_path.open("rb") as handle:
                download_columns[1].download_button(
                    "Download corrected_records.json",
                    data=handle,
                    file_name=corrected_json_path.name,
                    mime="application/json",
                    key=f"review_corrected_json_{selected_job.job_id}",
                )


def render_export(settings: dict[str, Any]) -> None:
    st.subheader("Export")
    jobs = _jobs_with_records()
    if not jobs:
        st.info("No completed batches with exported records yet.")
        return

    selected_job_id = st.selectbox(
        "Select batch",
        options=[job.job_id for job in jobs],
        format_func=lambda job_id: _job_label(job_id, jobs),
        key="export_job_select",
    )
    selected_job = next(job for job in jobs if job.job_id == selected_job_id)
    output_dir = Path(selected_job.output_dir)

    rows, source_label, source_csv_path = _load_preferred_export_rows(output_dir)
    if not rows or source_csv_path is None:
        st.warning("No exportable records found for the selected batch.")
        return

    render_job_metrics(selected_job)
    st.caption(f"Export source: {source_label}")

    export_config = _load_export_config(Path(settings["config_path"]))
    csv_part_paths, preferred_xlsx_paths = _prepare_preferred_export_parts(
        rows,
        output_dir,
        export_config,
        rows_per_file=int(settings["rows_per_xlsx_part"]),
    )

    render_result_downloads(
        source_csv_path=source_csv_path,
        csv_parts=csv_part_paths,
        xlsx_parts=preferred_xlsx_paths,
        download_key_prefix=f"export_{selected_job.job_id}",
        source_label=source_label,
    )
    render_preview_rows(_build_preview_rows_from_export_rows(rows))


def render_job_metrics(job: OcrJob) -> None:
    st.caption(f"Batch ID: {job.job_id}")
    status_col, pages_col, detected_col, parsed_col = st.columns(4)
    status_col.metric("Status", job.status)
    pages_col.metric("Pages", job.total_pages)
    detected_col.metric("Detected Records", job.total_detected_records)
    parsed_col.metric("Parsed Records", job.total_parsed_records)
    st.caption(f"Created: {job.created_at} | Updated: {job.updated_at}")
    if job.message:
        st.write(f"Message: {job.message}")


def render_result_metrics(result: ProcessResult) -> None:
    st.subheader("Result Metrics")
    pages_col, detected_col, parsed_col = st.columns(3)
    pages_col.metric("Pages", result.total_pages)
    detected_col.metric("Detected Records", result.total_detected_records)
    parsed_col.metric("Parsed Records", result.total_parsed_records)

    if result.status_counts:
        status_metrics = st.columns(len(result.status_counts))
        for index, (status, count) in enumerate(sorted(result.status_counts.items())):
            status_metrics[index].metric(status, count)


def render_result_downloads(
    source_csv_path: Path | None,
    csv_parts: list[Path],
    xlsx_parts: list[Path],
    *,
    download_key_prefix: str,
    source_label: str,
) -> None:
    st.subheader("Downloads")
    st.caption(source_label)

    csv_payload = _build_download_payload(
        csv_parts,
        fallback_path=source_csv_path,
        single_file_name="records.csv",
        archive_name="records_csv.zip",
        mime_type="text/csv",
    )
    if csv_payload is not None:
        csv_file_name, csv_data, csv_mime = csv_payload
        st.download_button(
            "Download records CSV",
            data=csv_data,
            file_name=csv_file_name,
            mime=csv_mime,
            key=f"{download_key_prefix}_csv",
        )

    xlsx_payload = _build_download_payload(
        xlsx_parts,
        fallback_path=None,
        single_file_name="records.xlsx",
        archive_name="records_xlsx.zip",
        mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    if xlsx_payload is None:
        st.caption("No XLSX parts were generated.")
        return

    xlsx_file_name, xlsx_data, xlsx_mime = xlsx_payload
    st.download_button(
        "Download records XLSX",
        data=xlsx_data,
        file_name=xlsx_file_name,
        mime=xlsx_mime,
        key=f"{download_key_prefix}_xlsx",
    )


def render_preview_rows(rows: list[dict[str, Any]]) -> None:
    st.subheader("Preview")
    st.caption(f"Showing at most the first {MAX_PREVIEW_ROWS} records.")
    if not rows:
        st.warning("No records were parsed.")
        return
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def _split_uploaded_files(uploaded_files: Sequence[Any], batch_size: int) -> list[list[Any]]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    return [list(uploaded_files[index : index + batch_size]) for index in range(0, len(uploaded_files), batch_size)]


def _jobs_with_records() -> list[OcrJob]:
    return [job for job in list_jobs() if _original_csv_path(Path(job.output_dir)) is not None or _corrected_csv_path(Path(job.output_dir)).exists()]


def _load_rows_for_review(output_dir: Path) -> tuple[list[dict[str, Any]], str, Path | None]:
    corrected_csv = _corrected_csv_path(output_dir)
    if corrected_csv.exists():
        return _load_export_rows(corrected_csv), "Corrected batch records", corrected_csv

    original_csv = _original_csv_path(output_dir)
    if original_csv is None:
        return [], "No records available", None
    return _load_export_rows(original_csv), "Original batch records", original_csv


def _load_preferred_export_rows(output_dir: Path) -> tuple[list[dict[str, Any]], str, Path | None]:
    return _load_rows_for_review(output_dir)


def _load_export_rows(csv_path: Path) -> list[dict[str, Any]]:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, Any]] = []
        for row in reader:
            rows.append({column: row.get(column, "") for column in XLSX_COLUMNS})
        return rows


def _write_export_rows_csv(rows: Sequence[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=XLSX_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in XLSX_COLUMNS})
    return output_path


def _write_corrected_rows(output_dir: Path, rows: Sequence[dict[str, Any]]) -> tuple[Path, Path]:
    corrected_csv = _write_export_rows_csv(rows, _corrected_csv_path(output_dir))
    corrected_json = output_dir / CORRECTED_JSON_NAME
    normalized_rows = [_normalize_row(row) for row in rows]
    corrected_json.write_text(json.dumps(normalized_rows, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    _delete_export_parts(output_dir)

    return corrected_csv, corrected_json


def _reset_corrections(output_dir: Path) -> None:
    for path in [output_dir / CORRECTED_CSV_NAME, output_dir / CORRECTED_JSON_NAME]:
        if path.exists():
            path.unlink()
    _delete_export_parts(output_dir)


def _prepare_original_export_parts(
    records: Sequence[ExtractedRecord],
    output_dir: Path,
    export_config: dict[str, Any],
    *,
    rows_per_file: int,
) -> tuple[list[Path], list[Path]]:
    _delete_export_parts(output_dir)
    csv_parts = export_records_to_csv_parts(
        records,
        output_dir,
        rows_per_file=rows_per_file,
        filename_prefix=RECORDS_PART_PREFIX,
    )
    xlsx_parts = export_records_to_xlsx_parts(
        records,
        output_dir,
        export_config,
        rows_per_file=rows_per_file,
        filename_prefix=RECORDS_PART_PREFIX,
    )
    return csv_parts, xlsx_parts


def _prepare_preferred_export_parts(
    rows: Sequence[dict[str, Any]],
    output_dir: Path,
    export_config: dict[str, Any],
    *,
    rows_per_file: int,
) -> tuple[list[Path], list[Path]]:
    if not rows:
        return [], []

    _delete_export_parts(output_dir)
    csv_parts = export_csv_rows_to_parts(
        rows,
        output_dir,
        rows_per_file=rows_per_file,
        filename_prefix=RECORDS_PART_PREFIX,
    )
    records = [record_from_export_dict(row) for row in rows]
    xlsx_parts = export_records_to_xlsx_parts(
        records,
        output_dir,
        export_config,
        rows_per_file=rows_per_file,
        filename_prefix=RECORDS_PART_PREFIX,
    )
    return csv_parts, xlsx_parts


def _load_export_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {}
    return dict(load_runtime_config(config_path).data.get("export", {}))


def _slice_editor_rows(rows: Sequence[dict[str, Any]], start_index: int, end_index: int) -> list[dict[str, Any]]:
    sliced = rows[start_index:end_index]
    return [{column: row.get(column) for column in EDITOR_COLUMNS} for row in sliced]


def _merge_edited_rows(
    all_rows: Sequence[dict[str, Any]],
    edited_rows: Sequence[dict[str, Any]],
    start_index: int,
) -> list[dict[str, Any]]:
    merged_rows = [dict(row) for row in all_rows]
    for offset, edited_row in enumerate(edited_rows):
        target_index = start_index + offset
        if target_index >= len(merged_rows):
            break
        for column in EDITOR_COLUMNS:
            merged_rows[target_index][column] = edited_row.get(column)
    return merged_rows


def _normalize_dataframe_rows(frame: Any) -> list[dict[str, Any]]:
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)
    cleaned = frame.where(pd.notna(frame), None)
    return [_normalize_row(row) for row in cleaned.to_dict(orient="records")]


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in row.items():
        normalized[key] = _normalize_value(value)
    return normalized


def _normalize_value(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def _row_window_selector(total_rows: int, *, key_prefix: str) -> tuple[int, int]:
    if total_rows <= MAX_PREVIEW_ROWS:
        return 0, total_rows

    max_start = max(1, total_rows)
    start_row = int(
        st.number_input(
            "Row window start",
            min_value=1,
            max_value=max_start,
            value=1,
            step=MAX_PREVIEW_ROWS,
            key=f"{key_prefix}_row_window",
        )
    )
    start_index = start_row - 1
    end_index = min(start_index + MAX_PREVIEW_ROWS, total_rows)
    return start_index, end_index


def _build_preview_rows(records: Iterable[ExtractedRecord], *, limit: int = MAX_PREVIEW_ROWS) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        rows.append(
            {
                "Bil": record.bil,
                "Nama Suami": record.nama_suami,
                "Nama Isteri": record.nama_isteri,
                "Tarikh Nikah": record.tarikh_nikah,
                "Confidence": record.confidence,
                "Status Review": record.status_review,
                "Source File": record.source_file,
                "Source Page": record.source_page,
                "Source Record": record.source_record,
            }
        )
        if len(rows) >= limit:
            break
    return rows


def _build_preview_rows_from_export_rows(rows: Sequence[dict[str, Any]], *, limit: int = MAX_PREVIEW_ROWS) -> list[dict[str, Any]]:
    return [
        {
            "Bil": row.get("Bil"),
            "Nama Suami": row.get("Nama Suami"),
            "Nama Isteri": row.get("Nama Isteri"),
            "Tarikh Nikah": row.get("Tarikh Nikah"),
            "Confidence": row.get("Confidence"),
            "Status Review": row.get("Status Review"),
            "Source File": row.get("Source File"),
            "Source Page": row.get("Source Page"),
            "Source Record": row.get("Source Record"),
        }
        for row in rows[:limit]
    ]


def _job_label(job_id: str, jobs: list[OcrJob]) -> str:
    job = next(job for job in jobs if job.job_id == job_id)
    return f"{job.job_id} | {job.status} | parsed {job.total_parsed_records}"


def _jobs_table_rows(jobs: list[OcrJob]) -> list[dict[str, Any]]:
    return [
        {
            "job_id": job.job_id,
            "status": job.status,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
            "message": job.message,
            "total_pages": job.total_pages,
            "total_detected_records": job.total_detected_records,
            "total_parsed_records": job.total_parsed_records,
        }
        for job in jobs
    ]


def _save_uploaded_files(uploaded_files: Iterable[Any], input_dir: Path) -> list[Path]:
    input_dir.mkdir(parents=True, exist_ok=True)
    used_names: dict[str, int] = {}
    saved_paths: list[Path] = []

    for uploaded_file in uploaded_files:
        original_name = Path(str(uploaded_file.name)).name
        stem = Path(original_name).stem
        suffix = Path(original_name).suffix
        index = used_names.get(original_name, 0)
        used_names[original_name] = index + 1
        target_name = original_name if index == 0 else f"{stem}_{index}{suffix}"

        target_path = input_dir / target_name
        target_path.write_bytes(bytes(uploaded_file.getbuffer()))
        saved_paths.append(target_path)

    return saved_paths


def _original_csv_path(output_dir: Path) -> Path | None:
    for path in [output_dir / ORIGINAL_CSV_NAME, output_dir / LEGACY_CSV_NAME]:
        if path.exists():
            return path
    return None


def _corrected_csv_path(output_dir: Path) -> Path:
    return output_dir / CORRECTED_CSV_NAME


def _delete_export_parts(output_dir: Path) -> None:
    for pattern in [f"{RECORDS_PART_PREFIX}_*.csv", f"{RECORDS_PART_PREFIX}_*.xlsx"]:
        for path in output_dir.glob(pattern):
            path.unlink()


def _build_download_payload(
    paths: Sequence[Path],
    *,
    fallback_path: Path | None,
    single_file_name: str,
    archive_name: str,
    mime_type: str,
) -> tuple[str, bytes, str] | None:
    existing_paths = [path for path in paths if path.exists()]
    if not existing_paths and fallback_path is not None and fallback_path.exists():
        existing_paths = [fallback_path]

    if not existing_paths:
        return None

    if len(existing_paths) == 1:
        return single_file_name, existing_paths[0].read_bytes(), mime_type

    archive_buffer = BytesIO()
    with ZipFile(archive_buffer, "w", compression=ZIP_DEFLATED) as archive:
        for path in existing_paths:
            archive.writestr(path.name, path.read_bytes())

    return archive_name, archive_buffer.getvalue(), "application/zip"


if __name__ == "__main__":
    main()
