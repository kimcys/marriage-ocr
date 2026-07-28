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
By default, debug artifacts are not retained. The processing commands leave only the tabular export unless `debug.retain_artifacts: true` is enabled in the config.

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
- no retained page crops or JSON artifacts by default
- log file under `logs/`

For handwritten pages, use the dedicated handwritten config instead:

```bash
.venv/bin/python -m marriage_ocr.cli process \
  --input input/handwritten \
  --output output/handwritten_records.csv \
  --debug debug/handwritten \
  --config config/handwritten.yaml \
  --reset-output
```

That profile uses cell-crop OCR, higher DPI, stronger preprocessing, the handwritten refinement defaults, Gemini for a second-pass cleanup, and the handwritten aggressive prompt mode. It is tuned aggressively for handwritten fields across the whole row. If you do not have a Gemini key available, switch `llm.enabled` off in a copy of the config.

## 3. Batch Into Postgres With Gemini

This is the path that runs Google Vision OCR, the existing parser, and the Gemini merge step before writing records into Postgres.

```bash
.venv/bin/python -m marriage_ocr.batch_runner \
  --input-dir input \
  --batch-name run_001 \
  --output-dir runs/batch_output \
  --config-path config/production.yaml
```

That writes the merged workbook to `runs/batch_output/exports/run_001_merged.xlsx` and stores the row data in Postgres.
It does not keep the page crops or JSON debug artifacts unless you enable debug retention in the config.

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
This workflow requires retained debug artifacts, so set `debug.retain_artifacts: true` before processing if you plan to review records later.

In the review UI you can:

- inspect the full record crop
- inspect each cell crop
- edit normalized fields
- edit cell-level OCR labels for training
- mark the record as verified
- export corrected XLSX
- export training data

Corrections are saved into `corrected_record.json` inside each record directory.

The review bundle also loads `refinement_audit.json` when it exists, so the UI can show the original value, selected value, retry source, scores, retry count, and review flag for each refined field.

## 5. Handwritten Refinement

The handwritten `process` command keeps the existing parsing and export behavior, then runs a conservative refinement stage before final validation. It only targets suspicious handwritten fields that are already supported by the parser:

- names: `nama_suami`, `nama_isteri`, `nama_pendaftar`, `nama_wali`, `saksi_1`, `saksi_2`
- IC values: `ic_lama_*`, `ic_baru_*`, `id_*`
- dates: `tarikh_nikah`, `tarikh_keluar`

The refinement stage uses the configured OCR engine for retry reads on deterministic crop variants. It never applies broad dictionary replacement to personal names. That omission is deliberate: aggressive dictionary autocorrection introduces too many false positives for real person names, so the shipped behavior stays narrow and reviewable.

Default config lives under `ocr.field_refinement`:

```yaml
ocr:
  field_refinement:
    enabled: true
    max_variants_per_field: 3
    minimum_candidate_score: 0.75
```

The runtime also supports these optional settings:

- `minimum_score_improvement`
- `save_retry_images`
- `retry_names`
- `retry_ic_numbers`
- `retry_dates`

Disable the refinement pass completely with:

```yaml
ocr:
  field_refinement:
    enabled: false
```

When disabled, the pipeline skips the retry stage entirely and keeps the original handwritten flow unchanged.

How uncertain fields are handled:

- suspicious fields are retried only when the parsed value looks unsafe
- stronger retry candidates are accepted only when they clear the configured score thresholds
- weak retry candidates fall back to the original parsed value
- unresolved fields stay flagged for human review instead of being silently rewritten

Cost and artifact notes:

- refinement can increase OCR cost because each suspicious field may trigger additional OCR reads
- aggregate audit output is written to `debug/refinement_audit.csv` only when `debug.retain_artifacts: true`
- per-record audit metadata is written to `debug/<page>/records/<record>/refinement_audit.json`

The audit CSV schema is:

- `source_file`
- `page_number`
- `record_index`
- `field_name`
- `original_value`
- `selected_value`
- `original_score`
- `selected_score`
- `correction_type`
- `candidate_source`
- `reason`
- `requires_review`
- `crop_path`
- `retry_count`

## 6. Baseline Benchmark Helper

Use the lightweight benchmark helper after you have a reviewed sample:

1. Run `process` with `debug.retain_artifacts: true`.
2. Open the review UI and verify corrected records.
3. Collect at least the first 25 verified records in normal review order.
4. Run `build_refinement_baseline(debug_path, limit=25)`.

The helper reads the existing review bundles and refinement audit sidecars. It reports:

- `record_count`
- `name_exact_match_count`
- `ic_exact_match_count`
- `date_exact_match_count`

Known limitations:

- it only measures exact matches against the reviewed active record
- it only covers audited name, IC, and date fields
- it is a quick regression baseline, not a full precision/recall evaluation

## 7. Export Training Data

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

It also requires retained debug artifacts from the original OCR run.

The label format is:

```text
image_path<TAB>label_text
```

## 8. Logs And Error Reports

Every CLI command writes a timestamped log file under `logs/`.

If a command fails unexpectedly, the CLI also writes a JSON error report under `logs/error_reports/` with:

- exception type and traceback
- command arguments
- config path
- current working directory
- relevant `MARRIAGE_OCR_*` environment variables

## 9. Config Files

- `config/default.yaml`: local development defaults
- `config/production.yaml`: packaged runtime defaults
- `.env`: environment overrides

Supported override styles:

```text
MARRIAGE_OCR_LOG_LEVEL=DEBUG
MARRIAGE_OCR_OCR_ENGINE=google_vision
MARRIAGE_OCR__TRAINING_EXPORT__VALIDATION_RATIO=0.10
```

## 10. Docker

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

## 11. Known Limits

- The default OCR path now depends on Google Vision credentials and outbound network access.
- Handwritten refinement can increase OCR calls when suspicious fields trigger retries.
- Training export is structurally correct, but its usefulness depends on human-corrected cell labels.
- Streamlit review requires a machine that can bind a local port.
