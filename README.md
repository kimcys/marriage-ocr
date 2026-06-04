# Marriage OCR

OCR-assisted pipeline for extracting handwritten `Daftar Perkahwinan Orang Islam` records into XLSX, review artifacts, and OCR training data.

## What It Does

- Processes images and PDFs into per-record debug crops.
- Runs OCR and rule-based parsing.
- Validates extracted records and exports XLSX.
- Provides a Streamlit review UI for human correction.
- Exports corrected cell crops and label files for OCR fine-tuning.

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e .[dev]
cp .env.example .env
```

For the packaged default flow, that is enough. The default OCR engine is now Google Vision, so before running real OCR you must export `GOOGLE_APPLICATION_CREDENTIALS` to a valid Google Cloud service-account JSON file. PaddleOCR remains available only as an optional alternative engine.

If you want the Gemini semantic extractor, export a single `GEMINI_API_KEY`.

Run the pipeline with the packaged production settings:

```bash
.venv/bin/python -m marriage_ocr.cli process \
  --input input \
  --output output/daftar_perkahwinan.xlsx \
  --debug debug \
  --config config/production.yaml \
  --reset-output
```

Run the DB batch flow with Gemini enabled in `config/production.yaml`:

```bash
.venv/bin/python -m marriage_ocr.batch_runner \
  --input-dir input \
  --batch-name run_001 \
  --output-dir runs/batch_output \
  --config-path config/production.yaml
```

That writes the merged workbook to `runs/batch_output/exports/run_001_merged.xlsx` and stores the row data in Postgres.

Export Excel/CSV from Postgres after the batch run:

```bash
.venv/bin/python -m marriage_ocr.export_from_postgres \
  --output-dir exports/final_xlsx \
  --csv-path exports/final_records.csv
```

Start the human review UI:

```bash
.venv/bin/python -m marriage_ocr.cli review \
  --debug debug \
  --config config/production.yaml \
  --port 8501
```

Export OCR training data from reviewed records:

```bash
.venv/bin/python -m marriage_ocr.cli export-training \
  --debug debug \
  --output-dir data/ground_truth \
  --config config/production.yaml \
  --verified-only \
  --reset-output
```

## Runtime Files

- Config: `config/default.yaml`, `config/production.yaml`
- Env template: `.env.example`
- Logs: `logs/*.log`
- Error reports: `logs/error_reports/*.json`
- Review export: `data/reviewed_exports/*.xlsx`
- Training data: `data/ground_truth/`

## Docker

Build:

```bash
docker build -t marriage-ocr:latest .
```

Run processing:

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

## More Docs

- [User Guide](docs/user-guide.md)
- [Handover Document](docs/handover.md)
