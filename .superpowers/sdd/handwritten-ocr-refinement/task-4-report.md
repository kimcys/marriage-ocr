# Task 4 Report - Wire Refinement Into The Handwritten Process Pipeline

## Status

Implemented the handwritten pipeline wiring so parsed records are refined before final validation, retry OCR calls are counted and surfaced to callers, and the CLI prints a small refinement summary without changing the existing export path.

## Files Changed

- `src/marriage_ocr/pipeline.py`
- `src/marriage_ocr/cli.py`
- `tests/test_pipeline.py`
- `tests/test_pipeline_gemini.py`
- `tests/__init__.py`
- `tests/typed/__init__.py`

## What Was Implemented

- Inserted field refinement between parsing and final validation in the handwritten `process_input(...)` flow.
- Reused the existing Task 3 `refine_field(...)` engine instead of duplicating retry OCR logic.
- Preserved the full-page OCR path by refining against the saved crop map rather than assuming cell-crop OCR only.
- Added `refinement_ocr_calls` and backward-compatible `refinement_audit_rows` to `ProcessResult`.
- Counted refinement retry OCR calls with a proxy engine scoped only to the refinement stage.
- Kept export behavior backward compatible by leaving the final validated/exported record path intact.
- Made refinement failures field-local so one bad retry does not stop other fields or records.
- Surfaced the retry OCR count in the CLI with a small post-run summary line.
- Added handwritten pipeline tests covering ordering, disablement, retry counting, safe fallback, failure isolation, export compatibility, CLI output, and refined-record Gemini flow.
- Added package markers under `tests/` to avoid pytest collection collisions between handwritten and typed pipeline test modules.

## Tests Run

1. `.venv/bin/python -m pytest tests/test_pipeline.py tests/test_pipeline_gemini.py -q`
2. `.venv/bin/python -m pytest tests/test_pipeline.py tests/test_pipeline_gemini.py tests/refinement/test_field_refinement.py tests/test_ocr_runtime.py -q`
3. `.venv/bin/python -m pytest -q`

## Test Results

- handwritten pipeline and Gemini tests: pass
- surrounding refinement/OCR regression tests: pass
- full suite: pass (`165 passed`)

## Concerns

- Refinement audit rows are collected in memory only for now. Audit CSV output and review-surface integration remain intentionally out of scope for this task.
