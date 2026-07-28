# Task 5 Implementation Report

Date: 2026-07-28

## Summary

Implemented refinement audit persistence and review surfacing without changing the existing XLSX export schema. The pipeline now writes deterministic refinement audit artifacts when debug artifacts are retained, the review store loads per-record refinement sidecars, and the review UI surfaces the metadata in an additional expander instead of redesigning the workflow.

## Changes

### `src/marriage_ocr/refinement/audit.py`

- Added `REFINEMENT_AUDIT_COLUMNS` with the exact column order required by the brief.
- Added `write_refinement_audit()` to write `refinement_audit.csv` deterministically with UTF-8 CSV output.
- Added `save_refinement_audit_sidecar()` to persist per-record review metadata to `refinement_audit.json`.
- Added `load_refinement_audit_sidecar()` to restore sidecar rows as `FieldRefinementAuditRow` instances.
- Serialized CSV booleans explicitly as lowercase `true` / `false` for stable output.

### `src/marriage_ocr/refinement/__init__.py`

- Re-exported the new audit helpers for consistent refinement package access.

### `src/marriage_ocr/pipeline.py`

- Persisted per-record `refinement_audit.json` sidecars alongside retained debug record artifacts.
- Persisted aggregate `debug/refinement_audit.csv` when debug artifacts are retained and audit rows exist.
- Kept write behavior explicit and deterministic by using the already-collected `ProcessResult.refinement_audit_rows` order.
- Fixed the disabled-refinement branch so sidecar writing does not reference uninitialized per-record audit rows.

### `src/marriage_ocr/review_store.py`

- Extended `ReviewBundle` with `refinement_audit_rows`.
- Loaded `refinement_audit.json` sidecars when building review bundles.

### `src/marriage_ocr/review_app.py`

- Added a `Refinement Audit` expander to show original value, selected value, scores, source, retry count, review flag, and reason.
- Preserved the existing review flow and editing form.

## Compatibility

- No changes were made to `src/marriage_ocr/exporter.py`.
- No changes were made to `src/marriage_ocr/training_export.py`.
- Existing XLSX export schema remains unchanged and compatible.

## Test-Driven Workflow

1. Confirmed the starting red state with:
   - `./.venv/bin/python -m pytest tests/refinement/test_audit.py tests/test_review_store.py -v`
   - Failure: `ModuleNotFoundError: No module named 'marriage_ocr.refinement.audit'`
2. Implemented the missing audit module and review-store integration.
3. Ran the focused suite and got green.
4. Added a focused pipeline test to lock the automatic retained-debug artifact behavior.
5. Ran the broader related suite.
6. Found and fixed one regression in the disabled-refinement path.
7. Re-ran the broader related suite to green.

## Verification

Focused:

- `./.venv/bin/python -m pytest tests/refinement/test_audit.py tests/test_review_store.py -v`

Broader related suite:

- `./.venv/bin/python -m pytest tests/refinement/test_audit.py tests/test_review_store.py tests/test_exporter.py tests/test_training_export.py tests/test_pipeline.py -v`

Result:

- 23 tests passed in the broader related suite.

## Notes

- Automatic `refinement_audit.csv` writing only happens when debug artifacts are retained, matching the retained-debug workflow and avoiding unexpected writes in non-debug runs.
- The aggregate CSV is written at `debug/refinement_audit.csv`.
- Per-record review sidecars are written at each record directory as `refinement_audit.json`.

## Post-Review Fix

Addressed reviewer feedback in commit `3108cee`:

- Preserved `correction_type` separately from `candidate_source` by storing and reading `correction_type` in candidate metadata.
- Disabled refinement no longer writes empty per-record audit sidecars.
- Added targeted tests covering both regressions.

Final verification after the fix:

- `./.venv/bin/python -m pytest -q` -> `170 passed, 5 warnings`
