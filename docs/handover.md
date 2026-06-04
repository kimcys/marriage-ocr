# Handover Document

## Scope

This repository packages the current MVP for:

- image and PDF ingestion
- preprocessing
- layout and record detection
- OCR integration
- rule-based parsing
- validation and confidence scoring
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
- `debug/`: overlays, record crops, JSON artifacts
- `data/reviewed_exports/`: corrected review exports
- `data/ground_truth/`: OCR training dataset
- `logs/`: timestamped command logs
- `logs/error_reports/`: JSON crash reports

## Current OCR State

The repository ships with:

- `google_vision` enabled by default
- `mock` and `paddle` retained as fallback/alternative engines

For real OCR, export `GOOGLE_APPLICATION_CREDENTIALS` to a valid Google Cloud service-account JSON file. PaddleOCR remains available only as an optional alternative.

## Configuration Model

Configuration layers:

1. YAML file passed with `--config`
2. `.env` file in repo root, or path from `MARRIAGE_OCR_ENV_FILE`
3. live environment variables

Supported override patterns:

- alias style: `MARRIAGE_OCR_LOG_LEVEL=DEBUG`
- nested style: `MARRIAGE_OCR__OCR__ENGINE=google_vision`

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

## Known Gaps

- No authentication or multi-user review workflow
- No background job queue
- No API surface beyond CLI and Streamlit
- No container image validation in CI
- Real OCR runtime dependencies are still optional

## Recommended Next Steps

1. Add CI for tests, linting, and container build verification.
2. Add explicit secrets/runtime guidance for Google Vision in container deployments.
3. Add audit metadata for per-field review changes.
4. Add retention policy for logs and generated debug artifacts.
5. Decide whether the legacy PaddleOCR adapter should remain supported long term.
