# Streamlit-Only OCR Website Execution Plan

## Goal

Add a Streamlit-only web workflow to the existing `marriage-ocr` codebase so users can:

```text
Upload image/PDF/folder batch -> Run OCR -> Preview records -> Download CSV/XLSX -> Continue review/correction
```

The implementation must reuse the existing OCR pipeline, parser, validator, debug crops, review UI, and exporter. Do not rebuild the OCR logic inside Streamlit.

This plan is designed for two stages:

1. **Phase 1: Internal Streamlit upload runner** for small to medium batches.
2. **Phase 2: Streamlit batch manager** for large-scale runs toward 1.8 million records.

---

## Existing Codebase Observations

Current repository already contains the important building blocks:

```text
src/marriage_ocr/cli.py
  Existing Typer commands:
  - process
  - review
  - export-training

src/marriage_ocr/review_app.py
  Existing Streamlit review/correction UI.

src/marriage_ocr/exporter.py
  Existing XLSX exporter.

src/marriage_ocr/document_loader.py
  Existing image/PDF loading support.

src/marriage_ocr/ocr/
  Existing OCR engine runtime.

src/marriage_ocr/parser/
  Existing parser.

src/marriage_ocr/validation.py
  Existing validation and review status logic.
```

The new Streamlit website should call/refactor the same processing flow used by:

```bash
python -m marriage_ocr.cli process \
  --input input \
  --output output/daftar_perkahwinan.xlsx \
  --debug debug \
  --config config/production.yaml
```

---

## Recommended Final Streamlit Flow

```text
Streamlit App: src/marriage_ocr/web_app.py

Page 1: Upload & Run
  - Upload JPG/JPEG/PNG/TIF/TIFF/PDF files
  - Select config file
  - Select batch size
  - Select output format
  - Run OCR
  - Show progress

Page 2: Job Results
  - Show detected records count
  - Show OK / REVIEW / FAILED counts
  - Show preview table
  - Download CSV
  - Download XLSX parts
  - Open review queue instructions

Page 3: Batch History
  - Show previous jobs from data/jobs/
  - Resume failed/incomplete batch
  - Download previous exports

Existing Review App
  - Keep `review_app.py` as the detailed human correction UI.
```

---

## Target Folder Structure

Add these files and folders:

```text
src/marriage_ocr/
  pipeline.py              # New reusable process function extracted from cli.py
  web_app.py               # New Streamlit upload/run website
  batch_exporter.py        # New CSV/XLSX split export helpers
  job_store.py             # New simple local job metadata store

scripts/
  run_web.sh               # Optional convenience launcher

data/
  jobs/
    <job_id>/
      input/
      output/
      debug/
      metadata.json
      records.csv
      records_part_001.xlsx
      records_part_002.xlsx
      errors.jsonl
```

Do not remove:

```text
src/marriage_ocr/review_app.py
src/marriage_ocr/cli.py
src/marriage_ocr/exporter.py
```

---

## Why Add `pipeline.py`

Currently the processing logic lives mostly inside `cli.py`. Streamlit can call the CLI with `subprocess`, but that is weaker because:

- hard to show progress,
- hard to catch partial results,
- hard to run batches,
- hard to unit test,
- hard to reuse without shell commands.

Move the body of `cli.process()` into a reusable function.

The CLI will become a thin wrapper around that function.

---

## Phase 1 Implementation Checklist

### Step 1: Create `src/marriage_ocr/pipeline.py`

Create a reusable function:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from marriage_ocr.models import ExtractedRecord


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
    """Run the existing OCR pipeline from Python code.

    This function should contain the processing logic currently inside
    `cli.process()`. The CLI and Streamlit app should both call this.
    """
    raise NotImplementedError
```

Then move the implementation from `cli.process()` into this function.

Keep all current behavior:

- config loading,
- document loading,
- preprocessing,
- layout detection,
- crop saving,
- OCR,
- parsing,
- validation,
- bil sequence correction,
- XLSX export,
- debug artifacts,
- error reporting.

### Step 2: Update `src/marriage_ocr/cli.py`

Replace the large body of `process()` with a call to `process_input()`.

Expected shape:

```python
from marriage_ocr.pipeline import process_input

result = process_input(
    input_path=input,
    output_path=output,
    debug_path=debug,
    config_path=config,
    reset_output=reset_output,
    layout_only=layout_only,
    skip_existing=skip_existing,
    progress_callback=None,
)
```

Keep the existing Typer options unchanged so existing commands still work.

### Step 3: Add CSV Export Support

Create `src/marriage_ocr/batch_exporter.py`.

Purpose:

- export all records to CSV,
- export XLSX in parts,
- avoid Excel row-limit issues.

Excel limit reminder:

```text
1 XLSX sheet max rows = 1,048,576 rows including header
Safe per file target = 500,000 records
Recommended operational batch = 5,000 records
```

Use smaller files first:

```text
records_part_001.xlsx = 5,000 to 50,000 records during testing
records_part_002.xlsx = next batch
```

Initial implementation:

```python
from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from marriage_ocr.exporter import XLSX_COLUMNS, export_records_to_xlsx
from marriage_ocr.models import ExtractedRecord


def export_records_to_csv(records: Iterable[ExtractedRecord], output_path: Path) -> Path:
    record_list = list(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=XLSX_COLUMNS)
        writer.writeheader()
        for record in record_list:
            writer.writerow(_record_to_export_dict(record))

    return output_path


def export_records_to_xlsx_parts(
    records: Iterable[ExtractedRecord],
    output_dir: Path,
    export_config: dict,
    *,
    rows_per_file: int = 5000,
) -> list[Path]:
    record_list = list(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for part_index, start in enumerate(range(0, len(record_list), rows_per_file), start=1):
        chunk = record_list[start:start + rows_per_file]
        part_path = output_dir / f"records_part_{part_index:03d}.xlsx"
        export_records_to_xlsx(
            chunk,
            part_path,
            export_config,
            reset_output=True,
            skip_existing=False,
        )
        paths.append(part_path)

    return paths


def _record_to_export_dict(record: ExtractedRecord) -> dict[str, object]:
    # Keep this mapping aligned with exporter._record_to_row().
    return {
        "Bil": record.bil,
        "Nama Suami": record.nama_suami,
        "IC Lama Suami": record.ic_lama_suami,
        "IC Baru Suami": record.ic_baru_suami,
        "ID Suami Raw": record.id_suami_raw,
        "Umur Suami": record.umur_suami,
        "Nama Isteri": record.nama_isteri,
        "IC Lama Isteri": record.ic_lama_isteri,
        "IC Baru Isteri": record.ic_baru_isteri,
        "ID Isteri Raw": record.id_isteri_raw,
        "Umur Isteri": record.umur_isteri,
        "Mas Kahwin": record.mas_kahwin,
        "Mas Kahwin Raw": record.mas_kahwin_raw,
        "Nama Pendaftar": record.nama_pendaftar,
        "Alamat Pendaftar": record.alamat_pendaftar,
        "Nama Wali": record.nama_wali,
        "Hubungan Wali": record.hubungan_wali,
        "Saksi 1": record.saksi_1,
        "Saksi 2": record.saksi_2,
        "Tarikh Nikah": record.tarikh_nikah,
        "Tarikh Nikah Raw": record.tarikh_nikah_raw,
        "Tarikh Keluar": record.tarikh_keluar,
        "Tarikh Keluar Raw": record.tarikh_keluar_raw,
        "Remarks": record.remarks,
        "Confidence": record.confidence,
        "Status Review": record.status_review,
        "Review Reason": "; ".join(record.review_reason),
        "Source File": record.source_file,
        "Source Page": record.source_page,
        "Source Record": record.source_record,
        "Crop Folder": record.crop_folder,
        "Raw Bil": record.raw_bil,
        "Raw Suami Isteri": record.raw_suami_isteri,
        "Raw Pendaftar": record.raw_pendaftar,
        "Raw Wali": record.raw_wali,
        "Raw Hubungan Wali": record.raw_hubungan_wali,
        "Raw Saksi": record.raw_saksi,
        "Raw Tarikh Nikah": record.raw_tarikh_nikah,
        "Raw Tarikh Keluar": record.raw_tarikh_keluar,
        "Raw Remarks": record.raw_remarks,
        "Raw OCR JSON": record.raw_ocr_json,
        "Created At": record.created_at,
        "Updated At": record.updated_at,
    }
```

Later improvement: expose `_record_to_row()` from `exporter.py` instead of duplicating mapping.

### Step 4: Add Local Job Store

Create `src/marriage_ocr/job_store.py`.

Purpose:

- create unique job IDs,
- save uploaded files,
- save job metadata,
- track status,
- allow batch history in Streamlit.

Initial implementation:

```python
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4


@dataclass
class OcrJob:
    job_id: str
    status: str
    created_at: str
    updated_at: str
    input_dir: str
    output_dir: str
    debug_dir: str
    message: str = ""
    total_pages: int = 0
    total_detected_records: int = 0
    total_parsed_records: int = 0


def create_job(root: Path = Path("data/jobs")) -> OcrJob:
    now = datetime.now().isoformat(timespec="seconds")
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:8]
    job_dir = root / job_id
    input_dir = job_dir / "input"
    output_dir = job_dir / "output"
    debug_dir = job_dir / "debug"

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    job = OcrJob(
        job_id=job_id,
        status="CREATED",
        created_at=now,
        updated_at=now,
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        debug_dir=str(debug_dir),
    )
    save_job(job, root=root)
    return job


def save_job(job: OcrJob, root: Path = Path("data/jobs")) -> None:
    job.updated_at = datetime.now().isoformat(timespec="seconds")
    metadata_path = root / job.job_id / "metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(asdict(job), indent=2), encoding="utf-8")


def load_job(job_id: str, root: Path = Path("data/jobs")) -> OcrJob:
    metadata_path = root / job_id / "metadata.json"
    return OcrJob(**json.loads(metadata_path.read_text(encoding="utf-8")))


def list_jobs(root: Path = Path("data/jobs")) -> list[OcrJob]:
    if not root.exists():
        return []
    jobs = []
    for metadata_path in sorted(root.glob("*/metadata.json"), reverse=True):
        jobs.append(OcrJob(**json.loads(metadata_path.read_text(encoding="utf-8"))))
    return jobs
```

### Step 5: Create `src/marriage_ocr/web_app.py`

This is the new Streamlit website.

Initial UI:

```python
from __future__ import annotations

from pathlib import Path
import shutil

import pandas as pd
import streamlit as st

from marriage_ocr.batch_exporter import export_records_to_csv, export_records_to_xlsx_parts
from marriage_ocr.config import load_runtime_config
from marriage_ocr.exporter import XLSX_COLUMNS
from marriage_ocr.job_store import create_job, list_jobs, save_job
from marriage_ocr.pipeline import ProcessProgress, process_input


st.set_page_config(page_title="Marriage OCR Runner", layout="wide")


def main() -> None:
    st.title("Marriage OCR Runner")
    st.caption("Upload image/PDF files, run OCR, preview records, and download CSV/XLSX.")

    page = st.sidebar.radio("Page", ["Upload & Run", "Batch History"])

    if page == "Upload & Run":
        render_upload_run()
    else:
        render_batch_history()


def render_upload_run() -> None:
    config_path = Path(st.sidebar.text_input("Config path", "config/production.yaml"))
    rows_per_xlsx = st.sidebar.number_input("Rows per XLSX part", min_value=100, max_value=500000, value=5000, step=100)
    layout_only = st.sidebar.checkbox("Layout only", value=False)

    uploaded_files = st.file_uploader(
        "Upload image/PDF files",
        type=["jpg", "jpeg", "png", "tif", "tiff", "pdf"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Upload one or more files to start.")
        return

    st.write(f"Uploaded files: {len(uploaded_files)}")

    if st.button("Run OCR", type="primary"):
        job = create_job()
        input_dir = Path(job.input_dir)
        output_dir = Path(job.output_dir)
        debug_dir = Path(job.debug_dir)

        for uploaded_file in uploaded_files:
            target_path = input_dir / uploaded_file.name
            target_path.write_bytes(uploaded_file.getbuffer())

        job.status = "RUNNING"
        job.message = "OCR started"
        save_job(job)

        progress_bar = st.progress(0)
        progress_text = st.empty()

        def on_progress(progress: ProcessProgress) -> None:
            percent = progress.page_index / max(progress.page_total, 1)
            progress_bar.progress(percent)
            progress_text.write(progress.message)

        try:
            result = process_input(
                input_path=input_dir,
                output_path=output_dir / "records.xlsx",
                debug_path=debug_dir,
                config_path=config_path,
                reset_output=True,
                layout_only=layout_only,
                skip_existing=False,
                progress_callback=on_progress,
            )

            csv_path = export_records_to_csv(result.records, output_dir / "records.csv")
            xlsx_parts = export_records_to_xlsx_parts(
                result.records,
                output_dir,
                load_runtime_config(config_path).data.get("export", {}),
                rows_per_file=int(rows_per_xlsx),
            )

            job.status = "DONE"
            job.message = "OCR completed"
            job.total_pages = result.total_pages
            job.total_detected_records = result.total_detected_records
            job.total_parsed_records = result.total_parsed_records
            save_job(job)

            st.success("OCR completed")
            render_result_downloads(csv_path, xlsx_parts)
            render_preview(result.records)

        except Exception as error:
            job.status = "FAILED"
            job.message = str(error)
            save_job(job)
            st.error(f"OCR failed: {error}")


def render_result_downloads(csv_path: Path, xlsx_parts: list[Path]) -> None:
    st.subheader("Downloads")

    with csv_path.open("rb") as file:
        st.download_button(
            "Download CSV",
            data=file,
            file_name=csv_path.name,
            mime="text/csv",
        )

    for path in xlsx_parts:
        with path.open("rb") as file:
            st.download_button(
                f"Download {path.name}",
                data=file,
                file_name=path.name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )


def render_preview(records) -> None:
    st.subheader("Preview")
    if not records:
        st.warning("No records were parsed.")
        return

    rows = []
    for record in records[:200]:
        rows.append({
            "Bil": record.bil,
            "Nama Suami": record.nama_suami,
            "Nama Isteri": record.nama_isteri,
            "Tarikh Nikah": record.tarikh_nikah,
            "Confidence": record.confidence,
            "Status Review": record.status_review,
            "Source File": record.source_file,
            "Source Page": record.source_page,
            "Source Record": record.source_record,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def render_batch_history() -> None:
    st.subheader("Batch History")
    jobs = list_jobs()
    if not jobs:
        st.info("No jobs yet.")
        return

    rows = [job.__dict__ for job in jobs]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


if __name__ == "__main__":
    main()
```

### Step 6: Add CLI Command to Launch Web App

In `src/marriage_ocr/cli.py`, add:

```python
@app.command("web")
def web(
    port: int = typer.Option(8502, "--port", min=1, max=65535),
) -> None:
    app_path = Path(__file__).with_name("web_app.py")
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.headless",
        "true",
        "--server.address",
        "127.0.0.1",
        "--server.port",
        str(port),
    ]
    subprocess.run(command, check=True, env=os.environ.copy())
```

Run with:

```bash
python -m marriage_ocr.cli web --port 8502
```

Existing review app remains:

```bash
python -m marriage_ocr.cli review --debug debug --config config/production.yaml --port 8501
```

---

## Phase 2: Batch Processing for 1.8 Million Records

For 1.8 million records, do not process everything as one Streamlit run and do not create one huge XLSX.

Use job folders and chunked processing.

Recommended chunk size:

```text
Testing: 100 records per batch
Pilot: 1,000 records per batch
Production: 5,000 records per batch
Maximum suggested Streamlit batch: 10,000 records
```

### Batch Folder Design

```text
data/jobs/<job_id>/
  input/
    uploaded files
  batches/
    batch_000001/
      input/
      debug/
      output/
        records.csv
        records_part_001.xlsx
      metadata.json
    batch_000002/
      input/
      debug/
      output/
        records.csv
        records_part_001.xlsx
      metadata.json
  output/
    all_records_manifest.json
    combined_records.csv
    records_part_001.xlsx
    records_part_002.xlsx
  metadata.json
```

### Why Batch by Files/Pages Instead of Records

The code only knows the record count after layout detection. Therefore batching should be based on:

- number of uploaded files,
- PDF pages,
- or estimated pages per batch.

After processing, the export can split by actual record count.

### Add Batch Split Helper Later

Create function in `job_store.py` or new `batch_runner.py`:

```python
def split_input_files(input_dir: Path, batch_dir: Path, files_per_batch: int) -> list[Path]:
    ...
```

For PDFs, start simple:

- process one PDF per batch,
- or split PDFs outside the app first.

Later enhancement:

- add PDF page split using PyMuPDF.

---

## Streamlit UI Design Details

### Sidebar Controls

```text
Config path: config/production.yaml
Rows per XLSX part: 5000
Layout only: false
Skip existing: false
Output mode:
  - CSV + XLSX parts
  - CSV only
Debug output: enabled
```

### Main Upload Page

```text
[Upload files]
[Run OCR]

Progress:
  Page 3 / 100
  Source: file_001.pdf page 3
  Detected records: 12
  Parsed records: 12
```

### Result Summary

Show metrics:

```text
Total pages
Detected records
Parsed records
OK records
Review records
Failed records
```

Show table preview with first 200 records only.

Do not display 1.8 million rows in Streamlit.

### Downloads

```text
Download CSV
Download records_part_001.xlsx
Download records_part_002.xlsx
Download debug ZIP
```

For debug ZIP, create later only when needed because debug folders can be very large.

---

## XLSX Split Strategy

For normal batches:

```text
rows_per_xlsx = 5,000
```

For large export:

```text
rows_per_xlsx = 500,000
```

Never exceed:

```text
1,000,000 records per XLSX sheet
```

Safer production values:

```text
CSV primary archive: unlimited split by batch
XLSX review files: 5,000 to 50,000 rows
XLSX final delivery: 500,000 rows per file
```

For 1.8 million records:

```text
records_part_001.xlsx  500,000 rows
records_part_002.xlsx  500,000 rows
records_part_003.xlsx  500,000 rows
records_part_004.xlsx  300,000 rows
```

---

## Performance Rules for 1.8 Million Records

### Do

- Process in batches.
- Save each batch result immediately.
- Keep CSV as the primary output.
- Use XLSX only as delivery/review output.
- Save debug output per job/batch.
- Keep error logs in JSONL.
- Resume from completed batches.
- Store reviewed corrections separately.

### Do Not

- Do not keep all 1.8 million records in Streamlit memory.
- Do not show all records in `st.dataframe`.
- Do not write one XLSX with all rows.
- Do not run OCR as one giant request.
- Do not delete debug folders until QA is done.

---

## Testing Plan

### Unit Tests to Add

```text
tests/test_batch_exporter.py
  - exports CSV with expected columns
  - splits XLSX into expected number of files
  - handles empty records

tests/test_job_store.py
  - creates job folder
  - writes metadata.json
  - loads job
  - lists jobs newest first

tests/test_pipeline.py
  - process_input works with mock OCR
  - progress_callback is called
```

### Manual Test Sequence

Run from project root:

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e .[dev]
cp .env.example .env
```

Use mock OCR first if available in config:

```yaml
ocr:
  engine: mock
```

Start Streamlit runner:

```bash
.venv/bin/python -m streamlit run src/marriage_ocr/web_app.py --server.port 8502
```

Or after adding CLI command:

```bash
.venv/bin/python -m marriage_ocr.cli web --port 8502
```

Test uploads:

```text
1 image
3 images
1 PDF
folder-equivalent using multi-file upload
```

Expected outputs:

```text
data/jobs/<job_id>/input/
data/jobs/<job_id>/debug/
data/jobs/<job_id>/output/records.csv
data/jobs/<job_id>/output/records.xlsx
data/jobs/<job_id>/output/records_part_001.xlsx
data/jobs/<job_id>/metadata.json
```

Then open review UI against the job debug folder:

```bash
.venv/bin/python -m marriage_ocr.cli review \
  --debug data/jobs/<job_id>/debug \
  --config config/production.yaml \
  --port 8501
```

---

## Production Runbook

### Small Batch Run

```text
1. Start web app.
2. Upload 1 to 100 files.
3. Run OCR.
4. Download CSV/XLSX.
5. Open review app for corrections.
6. Export corrected XLSX.
```

### Large Batch Run

```text
1. Prepare input folders by year/month/volume.
2. Process 5,000 records target per batch.
3. Keep output CSV per batch.
4. Review only REVIEW/FAILED records first.
5. Export corrected data.
6. Merge CSV files after QA.
7. Generate final XLSX parts only at the end.
```

### Suggested Production Directory

```text
storage/
  input_raw/
    year_1960/
    year_1961/
  jobs/
    job_...
  exports/
    csv/
    xlsx_parts/
  review/
  archive/
```

---

## Configuration Additions

Add to `config/default.yaml` and `config/production.yaml`:

```yaml
web:
  jobs_dir: data/jobs
  default_rows_per_xlsx: 5000
  max_preview_rows: 200
  allow_debug_zip: false
  default_config_path: config/production.yaml

batch:
  target_records_per_batch: 5000
  files_per_batch: 100
  xlsx_rows_per_file: 5000
  csv_primary: true
```

---

## Risks and Mitigations

### Risk: Google Vision cost and quota

Mitigation:

- pilot with 1,000 records first,
- estimate cost per page,
- cache raw OCR JSON,
- do not re-OCR reviewed pages unnecessarily,
- use layout-only mode to test crops before OCR.

### Risk: Streamlit process interruption

Mitigation:

- save batch outputs immediately,
- save job metadata after each batch,
- support resume later,
- keep debug folders per batch.

### Risk: Low handwriting accuracy

Mitigation:

- route low confidence to review,
- export verified training data,
- improve parser rules,
- eventually train/fine-tune local OCR if needed.

### Risk: Huge debug storage

Mitigation:

- compress old debug folders,
- keep only REVIEW/FAILED crops long-term,
- archive raw OCR JSON separately,
- define retention policy.

---

## Milestone Plan

### Milestone 1: Basic Streamlit Runner

Deliverables:

- `web_app.py`
- upload files
- run existing OCR pipeline
- preview first 200 rows
- download CSV and XLSX

Acceptance criteria:

```text
Can upload current sample images and get downloadable output from browser.
```

### Milestone 2: Pipeline Refactor

Deliverables:

- `pipeline.py`
- CLI `process` uses `pipeline.process_input()`
- Streamlit uses same function
- progress callback works

Acceptance criteria:

```text
CLI result and Streamlit result match for the same input.
```

### Milestone 3: Batch Export

Deliverables:

- `batch_exporter.py`
- CSV export
- XLSX split export
- tests

Acceptance criteria:

```text
Can split 12,000 test records into 3 XLSX files at 5,000 rows each.
```

### Milestone 4: Job History

Deliverables:

- `job_store.py`
- Batch History page
- job metadata
- previous download support

Acceptance criteria:

```text
A completed job remains visible after app restart.
```

### Milestone 5: Large Batch Mode

Deliverables:

- file/page batching
- resume incomplete batches
- per-batch CSV
- per-batch metadata

Acceptance criteria:

```text
Can process a large folder in batches without losing completed outputs if one batch fails.
```

---

## Immediate Next Coding Order

Use this exact order:

```text
1. Create src/marriage_ocr/pipeline.py.
2. Move logic from cli.process into process_input().
3. Make cli.process call process_input().
4. Add src/marriage_ocr/batch_exporter.py.
5. Add src/marriage_ocr/job_store.py.
6. Add src/marriage_ocr/web_app.py.
7. Add cli command: python -m marriage_ocr.cli web --port 8502.
8. Add tests for job_store and batch_exporter.
9. Run pytest.
10. Test Streamlit with sample uploaded images.
```

---

## Developer Notes

### Keep OCR and UI separated

Bad pattern:

```python
# Do not put OCR parsing directly inside Streamlit button code.
```

Good pattern:

```python
# Streamlit calls process_input().
result = process_input(...)
```

### Keep CSV as source of truth for large data

For 1.8 million records:

```text
CSV/Database = primary
XLSX = generated delivery artifact
Streamlit preview = limited to 200 rows
```

### Keep Review App Separate for Now

Do not merge `review_app.py` and `web_app.py` in the first implementation.

Reason:

- upload/run workflow and human verification workflow are different,
- existing review app already works,
- merging too early will make the code harder to stabilize.

Later, both can be combined into one multipage Streamlit app.

---

## Definition of Done

The Streamlit-only OCR website is done when:

```text
1. User can upload image/PDF files in browser.
2. OCR runs using existing pipeline.
3. App shows progress and summary metrics.
4. App previews parsed records.
5. App provides CSV download.
6. App provides XLSX split downloads.
7. Debug folder is generated per job.
8. Existing review UI can review the generated debug folder.
9. Job history survives app restart.
10. CLI process still works exactly as before.
```
