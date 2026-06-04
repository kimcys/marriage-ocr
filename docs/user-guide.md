# User Guide

## 1. Install

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e .[dev]
cp .env.example .env
```

The CLI loads `.env` automatically when it exists. Runtime overrides can also be passed with environment variables such as `MARRIAGE_OCR_LOG_LEVEL=DEBUG`.

The default OCR engine is `google_vision`. Before running real OCR, export `GOOGLE_APPLICATION_CREDENTIALS` to a valid Google Cloud service-account JSON file.
If you enable the Gemini semantic extractor, export a single `GEMINI_API_KEY` as well.

Example:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/absolute/path/to/service-account.json"
```

The repository still keeps PaddleOCR as an optional alternative engine, but it is no longer the default path.

## 2. Process Input Files

```bash
.venv/bin/python -m marriage_ocr.cli process \
  --input input \
  --output output/daftar_perkahwinan.xlsx \
  --debug debug \
  --config config/production.yaml \
  --reset-output
```

What you get:

- `output/daftar_perkahwinan.xlsx`
- page-level debug overlays under `debug/`
- per-record crops, raw OCR, parsed JSON, validated JSON
- log file under `logs/`

## 3. Batch Into Postgres With Gemini

This is the path that runs Google Vision OCR, the existing parser, and the Gemini merge step before writing records into Postgres.

```bash
.venv/bin/python -m marriage_ocr.batch_runner \
  --input-dir input \
  --batch-name run_001 \
  --output-dir runs/batch_output \
  --config-path config/production.yaml
```

Then export Excel/CSV from Postgres:

```bash
.venv/bin/python -m marriage_ocr.export_from_postgres \
  --output-dir exports/final_xlsx \
  --csv-path exports/final_records.csv
```

## 4. Review Records

```bash
.venv/bin/python -m marriage_ocr.cli review \
  --debug debug \
  --config config/production.yaml \
  --port 8501
```

Open `http://127.0.0.1:8501`.

In the review UI you can:

- inspect the full record crop
- inspect each cell crop
- edit normalized fields
- edit cell-level OCR labels for training
- mark the record as verified
- export corrected XLSX
- export training data

Corrections are saved into `corrected_record.json` inside each record directory.

## 5. Export Training Data

```bash
.venv/bin/python -m marriage_ocr.cli export-training \
  --debug debug \
  --output-dir data/ground_truth \
  --config config/production.yaml \
  --verified-only \
  --reset-output
```

This produces:

- `data/ground_truth/labels.tsv`
- `data/ground_truth/train.tsv`
- `data/ground_truth/val.tsv`
- `data/ground_truth/manifest.jsonl`
- `data/ground_truth/stats.json`
- `data/ground_truth/training_crops/<cell_name>/*.jpg`

The label format is:

```text
image_path<TAB>label_text
```

## 6. Logs And Error Reports

Every CLI command writes a timestamped log file under `logs/`.

If a command fails unexpectedly, the CLI also writes a JSON error report under `logs/error_reports/` with:

- exception type and traceback
- command arguments
- config path
- current working directory
- relevant `MARRIAGE_OCR_*` environment variables

## 7. Config Files

- `config/default.yaml`: local development defaults
- `config/production.yaml`: packaged runtime defaults
- `.env`: environment overrides

Supported override styles:

```text
MARRIAGE_OCR_LOG_LEVEL=DEBUG
MARRIAGE_OCR_OCR_ENGINE=google_vision
MARRIAGE_OCR__TRAINING_EXPORT__VALIDATION_RATIO=0.10
```

## 8. Docker

Build:

```bash
docker build -t marriage-ocr:latest .
```

Run process:

```bash
docker run --rm \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/creds/google-vision.json \
  -v "$PWD/input:/app/input" \
  -v "$PWD/output:/app/output" \
  -v "$PWD/debug:/app/debug" \
  -v "$PWD/data:/app/data" \
  -v "$PWD/logs:/app/logs" \
  -v "$HOME/Documents/your-service-account.json:/app/creds/google-vision.json:ro" \
  marriage-ocr:latest \
  python -m marriage_ocr.cli process \
    --input input \
    --output output/daftar_perkahwinan.xlsx \
    --debug debug \
    --config config/production.yaml
```

Run review UI:

```bash
docker run --rm -p 8501:8501 \
  -v "$PWD/debug:/app/debug" \
  -v "$PWD/data:/app/data" \
  -v "$PWD/logs:/app/logs" \
  marriage-ocr:latest \
  python -m marriage_ocr.cli review \
    --debug debug \
    --config config/production.yaml \
    --port 8501
```

## 9. Known Limits

- The default OCR path now depends on Google Vision credentials and outbound network access.
- Training export is structurally correct, but its usefulness depends on human-corrected cell labels.
- Streamlit review requires a machine that can bind a local port.
