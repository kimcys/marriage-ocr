# Marriage OCR

OCR-assisted pipeline for extracting handwritten `Daftar Perkahwinan Orang Islam` records into XLSX, review artifacts, and OCR training data.

## What It Does

- Processes images and PDFs into per-record debug crops.
- Runs OCR and rule-based parsing.
- Runs a conservative handwritten field-refinement pass for names, ICs, and dates before final validation.
- Validates extracted records and exports XLSX.
- Provides a Streamlit review UI for human correction.
- Exports corrected cell crops and label files for OCR fine-tuning.

## Handwritten Refinement

The handwritten pipeline keeps the existing OCR and export flow intact, then applies a lightweight refinement pass only to suspicious handwritten names, IC values, and dates before final validation. It uses the same OCR engine that the page already ran with and records every evaluated field in refinement audit metadata.

Supported fields:

- Names: `nama_suami`, `nama_isteri`, `nama_pendaftar`, `nama_wali`, `saksi_1`, `saksi_2`
- IC values: `ic_lama_*`, `ic_baru_*`, `id_*`
- Dates: `tarikh_nikah`, `tarikh_keluar`

The feature is enabled by default through `ocr.field_refinement.enabled: true`. Disable it with:

```yaml
ocr:
  field_refinement:
    enabled: false
```

Available settings:

- `enabled`
- `max_variants_per_field`
- `minimum_candidate_score`
- `minimum_score_improvement`
- `save_retry_images`
- `retry_names`
- `retry_ic_numbers`
- `retry_dates`

Operational notes:

- The refinement pass may trigger extra OCR calls for suspicious fields, so enabling it can increase OCR cost and runtime.
- Uncertain results remain conservative: low-confidence retry candidates fall back to the original parsed value and are marked for review instead of being forced into the export.
- General name-dictionary autocorrection is intentionally avoided. The pipeline only applies narrow substitutions and retry OCR because broad dictionary replacement creates false positives for Malay and Arabic-derived personal names.
- When `debug.retain_artifacts: true` is enabled, the pipeline writes aggregate audit data to `debug/refinement_audit.csv` and per-record sidecars to `debug/<page>/records/<record>/refinement_audit.json`.
- The audit CSV columns are `source_file`, `page_number`, `record_index`, `field_name`, `original_value`, `selected_value`, `original_score`, `selected_score`, `correction_type`, `candidate_source`, `reason`, `requires_review`, `crop_path`, `retry_count`.
- To collect the first 25 reviewed records for a quick baseline benchmark, process with retained debug artifacts enabled, review records in the Streamlit UI, mark them verified, then run `build_refinement_baseline(debug_path, limit=25)` against that debug root.
- Known limitation: the baseline helper measures exact matches only for audited name, IC, and date fields that already have review bundles and refinement sidecars; it does not score free-form remarks or non-refined fields.

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e .[dev]
cp .env.example .env
```

For the packaged default flow, that is enough. The default OCR engine is now Google Vision, so before running real OCR you must export `GOOGLE_APPLICATION_CREDENTIALS` to a valid Google Cloud service-account JSON file. PaddleOCR remains available only as an optional alternative engine.

If you want the Gemini semantic extractor, export a single `GEMINI_API_KEY`.
By default, the runtime keeps only the CSV/XLSX exports and does not retain page crops or JSON debug artifacts. Enable `debug.retain_artifacts: true` in the config only when you need review or training outputs.

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

That writes the merged workbook to `runs/batch_output/exports/run_001_merged.xlsx` and stores the row data in Postgres. Debug crops and JSON files are not retained unless you explicitly enable them in config.

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

Process typed Borang 4B PDFs into a separate CSV:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/service-account.json
python -m marriage_ocr.cli process-typed \
  --input input/typed \
  --output output/typed_records.csv \
  --debug debug/typed \
  --config config/typed_borang4b.yaml \
  --reset-output
```

Typed input and output stay separate from the handwritten pipeline:

```text
input/
├── handwritten/
└── typed/

output/
├── handwritten_records.csv
└── typed_records.csv
```

The typed workflow writes one CSV row per PDF, supports only two-page Borang 4B forms, and keeps `Tarikh Keluar` blank. Use `--skip-existing` to skip already-successful typed rows while reprocessing review or failed rows.

## Runtime Files

- Config: `config/default.yaml`, `config/production.yaml`
- Env template: `.env.example`
- Logs: `logs/*.log`
- Error reports: `logs/error_reports/*.json`
- Review export: `data/reviewed_exports/*.xlsx`
- Training data: `data/ground_truth/`
- Typed output: `output/typed_records.csv`
- Typed debug artifacts: `debug/typed/`

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
