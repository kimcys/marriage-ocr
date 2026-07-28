# Handover Document

## Scope

This repository packages the current MVP for:

- image and PDF ingestion
- preprocessing
- layout and record detection
- OCR integration
- rule-based parsing
- validation and confidence scoring
- conservative handwritten refinement with retry audit artifacts
- XLSX export
- Streamlit review UI
- training data export

## Operational Entry Points

- Process input: `python -m marriage_ocr.cli process`
- Review UI: `python -m marriage_ocr.cli review`
- Export training data: `python -m marriage_ocr.cli export-training`

The packaged runtime defaults are in `config/production.yaml`.

## Runtime Artifacts

- `output/`: exported XLSX
- `debug/`: overlays, record crops, JSON artifacts, and `refinement_audit.csv` when retained
- `data/reviewed_exports/`: corrected review exports
- `data/ground_truth/`: OCR training dataset
- `logs/`: timestamped command logs
- `logs/error_reports/`: JSON crash reports

Per-record handwritten refinement metadata is stored as `refinement_audit.json` inside each retained record directory.

## Current OCR State

The repository ships with:

- `google_vision` enabled by default
- `mock` and `paddle` retained as fallback/alternative engines

For real OCR, export `GOOGLE_APPLICATION_CREDENTIALS` to a valid Google Cloud service-account JSON file. PaddleOCR remains available only as an optional alternative.

Handwritten refinement is enabled by default under `ocr.field_refinement.enabled`. It only targets suspicious names, ICs, and dates and reuses the configured OCR engine for retry reads on deterministic crop variants.

## Configuration Model

Configuration layers:

1. YAML file passed with `--config`
2. `.env` file in repo root, or path from `MARRIAGE_OCR_ENV_FILE`
3. live environment variables

Supported override patterns:

- alias style: `MARRIAGE_OCR_LOG_LEVEL=DEBUG`
- nested style: `MARRIAGE_OCR__OCR__ENGINE=google_vision`

Relevant handwritten refinement settings:

- `ocr.field_refinement.enabled`
- `ocr.field_refinement.max_variants_per_field`
- `ocr.field_refinement.minimum_candidate_score`
- `ocr.field_refinement.minimum_score_improvement`
- `ocr.field_refinement.save_retry_images`
- `ocr.field_refinement.retry_names`
- `ocr.field_refinement.retry_ic_numbers`
- `ocr.field_refinement.retry_dates`

Setting `ocr.field_refinement.enabled: false` skips the retry stage entirely and preserves the original handwritten pipeline behavior.

The refinement logic intentionally avoids general name-dictionary autocorrection. Broad dictionary replacement caused too many false positives for real personal names, so the production path is limited to conservative substitutions plus retry OCR.

## Failure Handling

All CLI commands now do three things on failure:

1. write a rotating log file
2. write a JSON error report
3. exit with code `1`

Error report payload includes command context, traceback, config path, and relevant environment variables.

## Deployment Notes

- Local install uses `pip install -e .[dev]`
- Container build uses the included `Dockerfile`
- Containerized Google Vision runs must pass `GOOGLE_APPLICATION_CREDENTIALS` and mount the service-account JSON file
- Streamlit review needs a bound port such as `8501`
- Mounted volumes are required for `input`, `output`, `debug`, `data`, and `logs` in Docker usage

Operational refinement notes:

- additional OCR calls can increase runtime and API cost when suspicious fields trigger retries
- uncertain retry candidates are not forced into output; they fall back to the original parsed value and remain reviewable
- `debug/refinement_audit.csv` is only written when debug artifacts are retained
- the audit CSV columns are `source_file`, `page_number`, `record_index`, `field_name`, `original_value`, `selected_value`, `original_score`, `selected_score`, `correction_type`, `candidate_source`, `reason`, `requires_review`, `crop_path`, `retry_count`
- the lightweight benchmark helper reads the existing review store plus refinement audit rows and can summarize the first 25 verified records with `build_refinement_baseline(debug_path, limit=25)`

## Known Gaps

- No authentication or multi-user review workflow
- No background job queue
- No API surface beyond CLI and Streamlit
- No container image validation in CI
- Real OCR runtime dependencies are still optional
- The benchmark helper is an exact-match baseline for audited name, IC, and date fields only, not a full evaluation framework.

## Recommended Next Steps

1. Add CI for tests, linting, and container build verification.
2. Add explicit secrets/runtime guidance for Google Vision in container deployments.
3. Turn the refinement benchmark helper into a tracked evaluation workflow for the first 25 verified review bundles.
4. Add retention policy for logs and generated debug artifacts.
5. Decide whether the legacy PaddleOCR adapter should remain supported long term.
