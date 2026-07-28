# Handwritten OCR Mode-Aware Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve handwritten OCR accuracy, validation, confidence scoring, and review routing by introducing an explicit document-mode boundary while keeping typed processing stable.

**Architecture:** Keep the current typed and handwritten entry points, but add a shared mode-aware extraction layer that branches early on `DocumentType`. Handwritten records get layered OCR candidate generation, field-level normalization and validation, confidence scoring, and review routing; typed records keep their current simpler path with only the new mode abstraction and any safe shared validation helpers.

**Tech Stack:** Python 3.14, Typer CLI, Google Vision OCR, Gemini semantic extraction, optional PaddleOCR adapter, Streamlit review UI, PostgreSQL, pytest, openpyxl, Pillow/OpenCV.

## Global Constraints

- Do not train or integrate a custom handwriting model.
- Do not break existing typed OCR behavior.
- Do not require live OCR credentials in the standard test suite.
- Preserve raw OCR values and auditability; never overwrite original OCR text destructively.
- Keep paid OCR integrations behind the existing provider interfaces.
- Use incremental changes with tests for each independently reviewable step.
- Keep production and development configuration separate.

## File Map

- `src/marriage_ocr/document_type.py`: document-mode enum and config resolution helpers.
- `src/marriage_ocr/pipeline.py`: handwritten versus typed branching, mode-aware strategy selection, and shared orchestration.
- `src/marriage_ocr/models.py`: shared record and field-level extraction models.
- `src/marriage_ocr/field_extraction.py`: handwritten field extraction, candidate, validation, and review data models.
- `src/marriage_ocr/ocr/__init__.py`: OCR engine wiring, OCR cache hooks, and handwritten retry-image handling.
- `src/marriage_ocr/refinement/preprocess.py`: deterministic retry-image generation variants.
- `src/marriage_ocr/refinement/field_refinement.py`: handwritten retry orchestration and candidate scoring.
- `src/marriage_ocr/refinement/text_corrections.py`: name, IC, date, and address candidate generation helpers.
- `src/llm/gemini_extractor.py`: handwritten aggressive literal-transcription prompt mode.
- `src/llm/record_merge.py`: handwritten merge and confidence selection rules.
- `src/marriage_ocr/validation.py`: field validation, cross-field validation, and review routing.
- `src/marriage_ocr/review_app.py`: field-level candidate display and reviewer actions.
- `src/marriage_ocr/review_store.py`: persistence of reviewed values and candidate metadata.
- `src/marriage_ocr/db_postgres.py`: additive persistence changes for provenance and review data.
- `src/marriage_ocr/refinement/audit.py`: audit CSV and JSON serialization.
- `src/marriage_ocr/refinement/benchmark.py`: baseline evaluation helper for verified handwritten records.
- `config/*.yaml`: typed versus handwritten mode settings.
- `tests/*`: unit, integration, and regression coverage.
- `README.md`, `docs/user-guide.md`, `docs/handover.md`: operator documentation.

---

### Task 1: Add explicit document mode and config routing

**Files:**
- Create: `src/marriage_ocr/document_type.py`
- Modify: `src/marriage_ocr/pipeline.py`
- Modify: `src/marriage_ocr/cli.py`
- Modify: `src/marriage_ocr/config.py`
- Modify: `config/default.yaml`
- Modify: `config/production.yaml`
- Modify: `config/handwritten.yaml`
- Modify: `config/typed_borang4b.yaml`
- Test: `tests/test_runtime_packaging.py`
- Test: `tests/test_pipeline.py`
- Test: `tests/test_pipeline_gemini.py`

**Interfaces:**
- Consumes: existing command names, existing config structures, `process_input(...)`
- Produces: `DocumentType.TYPED`, `DocumentType.HANDWRITTEN`, and a mode resolution helper used by the pipeline

- [ ] **Step 1: Write the failing test**

Add a focused test that asserts document mode resolution is deterministic and command-aware:

```python
def test_resolve_document_type_prefers_typed_command():
    assert resolve_document_type({}, command_name="process-typed") is DocumentType.TYPED
    assert resolve_document_type({}, command_name="process") is DocumentType.HANDWRITTEN
```

Also add a pipeline test that confirms handwritten `process` routes through the handwritten mode while `process-typed` remains typed-specific.

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
./.venv/bin/python -m pytest tests/test_runtime_packaging.py tests/test_pipeline.py tests/test_pipeline_gemini.py -q
```

Expected: the new assertions fail because `DocumentType` and explicit mode routing do not exist yet.

- [ ] **Step 3: Write minimal implementation**

Add `src/marriage_ocr/document_type.py` with:

```python
from enum import Enum

class DocumentType(str, Enum):
    TYPED = "typed"
    HANDWRITTEN = "handwritten"
```

Add a small resolver:

```python
def resolve_document_type(config: Mapping[str, Any], *, command_name: str | None = None) -> DocumentType:
    ...
```

Pass the resolved mode through `process_input(...)` and keep typed behavior unchanged except for the new mode selection.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
./.venv/bin/python -m pytest tests/test_runtime_packaging.py tests/test_pipeline.py tests/test_pipeline_gemini.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/marriage_ocr/document_type.py src/marriage_ocr/pipeline.py src/marriage_ocr/cli.py src/marriage_ocr/config.py config/default.yaml config/production.yaml config/handwritten.yaml config/typed_borang4b.yaml tests/test_runtime_packaging.py tests/test_pipeline.py tests/test_pipeline_gemini.py
git commit -m "feat: add explicit handwritten and typed document modes"
```

---

### Task 2: Introduce shared field extraction and handwritten literal/normalization split

**Files:**
- Create: `src/marriage_ocr/field_extraction.py`
- Modify: `src/marriage_ocr/models.py`
- Modify: `src/llm/gemini_extractor.py`
- Modify: `src/llm/record_merge.py`
- Modify: `src/marriage_ocr/parser/__init__.py`
- Test: `tests/test_gemini_extractor.py`
- Test: `tests/test_misc_parsers.py`
- Test: `tests/test_pipeline_gemini.py`

**Interfaces:**
- Consumes: `ExtractedRecord`, OCR cell hints, Gemini output payloads, existing parser outputs
- Produces: a typed `FieldExtraction` model and handwritten-safe literal/normalized values without destroying raw OCR text

- [ ] **Step 1: Write the failing test**

Add a unit test that asserts a handwritten field extraction keeps raw text, normalized text, candidate values, and provider metadata separate:

```python
def test_field_extraction_preserves_raw_and_normalized_values():
    extraction = build_field_extraction(
        field_name="nama_suami",
        raw_value="AHMAD B1N ALI",
        normalized_value="AHMAD BIN ALI",
        candidates=["AHMAD B1N ALI", "AHMAD BIN ALI"],
        provider="google_vision",
        image_variant="retry_grayscale",
    )
    assert extraction.raw_value == "AHMAD B1N ALI"
    assert extraction.normalized_value == "AHMAD BIN ALI"
    assert extraction.candidates == ["AHMAD B1N ALI", "AHMAD BIN ALI"]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
./.venv/bin/python -m pytest tests/test_gemini_extractor.py tests/test_misc_parsers.py tests/test_pipeline_gemini.py -q
```

Expected: fail because the shared field extraction layer does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Create `FieldExtraction` with:

```python
@dataclass(frozen=True)
class FieldExtraction:
    field_name: str
    raw_value: str | None
    normalized_value: str | None
    candidates: list[str]
    confidence: float
    validation_status: str
    validation_reasons: list[str]
    provider: str | None
    image_variant: str | None
    human_corrected: bool
    verified_value: str | None
```

Add handwritten-specific literal transcription handling in `GeminiRecordExtractor` so the prompt can preserve raw text and explicitly separate raw versus normalized outputs.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
./.venv/bin/python -m pytest tests/test_gemini_extractor.py tests/test_misc_parsers.py tests/test_pipeline_gemini.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/marriage_ocr/field_extraction.py src/marriage_ocr/models.py src/llm/gemini_extractor.py src/llm/record_merge.py src/marriage_ocr/parser/__init__.py tests/test_gemini_extractor.py tests/test_misc_parsers.py tests/test_pipeline_gemini.py
git commit -m "feat: preserve handwritten field extraction provenance"
```

---

### Task 3: Add handwritten image variants, OCR cache hooks, and retry candidate generation

**Files:**
- Modify: `src/marriage_ocr/refinement/preprocess.py`
- Modify: `src/marriage_ocr/refinement/field_refinement.py`
- Modify: `src/marriage_ocr/ocr/__init__.py`
- Modify: `src/marriage_ocr/ocr_cache.py`
- Modify: `config/handwritten.yaml`
- Test: `tests/refinement/test_preprocess.py`
- Test: `tests/test_ocr_cache.py`
- Test: `tests/test_pipeline.py`
- Test: `tests/test_ocr_runtime.py`

**Interfaces:**
- Consumes: crop paths, OCR provider config, image preprocessing settings, retry settings
- Produces: deterministic retry variants, cache keys, and OCR retry candidates for suspicious handwritten fields only

- [ ] **Step 1: Write the failing test**

Add a test for deterministic variant generation and OCR cache keys:

```python
def test_build_retry_variants_emits_color_grayscale_thresholded():
    variants = build_retry_variants(crop_path, padding_ratio=0.05)
    assert [variant.name for variant in variants] == ["original_color", "grayscale_contrast", "thresholded_upscaled"]

def test_ocr_cache_key_changes_with_model_config():
    assert build_ocr_cache_key(path, provider="gemini", config={"model": "a"}) != build_ocr_cache_key(path, provider="gemini", config={"model": "b"})
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
./.venv/bin/python -m pytest tests/refinement/test_preprocess.py tests/test_ocr_cache.py tests/test_ocr_runtime.py -q
```

Expected: fail because the handwritten variant/caching contract is not implemented yet.

- [ ] **Step 3: Write minimal implementation**

Implement deterministic image variants in `refinement/preprocess.py`:

- padded original color crop
- grayscale contrast-enhanced crop
- conservatively thresholded upscaled crop

Add an OCR cache key helper in `ocr_cache.py` keyed by image hash plus provider/config.

Wire `field_refinement.py` to request retry OCR only for suspicious handwritten fields and to keep retry images only when debug retention is enabled.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
./.venv/bin/python -m pytest tests/refinement/test_preprocess.py tests/test_ocr_cache.py tests/test_ocr_runtime.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/marriage_ocr/refinement/preprocess.py src/marriage_ocr/refinement/field_refinement.py src/marriage_ocr/ocr/__init__.py src/marriage_ocr/ocr_cache.py config/handwritten.yaml tests/refinement/test_preprocess.py tests/test_ocr_cache.py tests/test_ocr_runtime.py
git commit -m "feat: add handwritten retry variants and OCR caching"
```

---

### Task 4: Improve handwritten validation, confidence, and review routing

**Files:**
- Modify: `src/marriage_ocr/validation.py`
- Modify: `src/marriage_ocr/refinement/models.py`
- Modify: `src/marriage_ocr/refinement/text_corrections.py`
- Modify: `src/marriage_ocr/review_app.py`
- Modify: `src/marriage_ocr/review_store.py`
- Test: `tests/test_validation.py`
- Test: `tests/test_review_store.py`
- Test: `tests/test_web_app.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: field extractions, OCR confidence, retry candidate metadata, review bundles
- Produces: field validation results, field confidence scores, and `AUTO_ACCEPT` / `FIELD_REVIEW` / `FULL_RECORD_REVIEW` routing

- [ ] **Step 1: Write the failing test**

Add tests that assert:

```python
def test_handwritten_route_becomes_field_review_when_only_name_is_uncertain():
    ...

def test_field_confidence_penalizes_multiple_competing_candidates():
    ...
```

Also add a review-store/UI test that verifies candidate metadata and validation reasons remain visible after saving a correction.

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
./.venv/bin/python -m pytest tests/test_validation.py tests/test_review_store.py tests/test_web_app.py tests/test_pipeline.py -q
```

Expected: fail because the richer handwritten validation and routing are not implemented yet.

- [ ] **Step 3: Write minimal implementation**

Add handwritten validation helpers for:

- identity-number classification
- date candidate validation and ambiguity handling
- conservative name and address normalization
- field-level confidence scoring
- review-route selection

Extend the review store/UI to display:

- raw value
- normalized value
- candidate list
- confidence
- validation reasons
- review route

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
./.venv/bin/python -m pytest tests/test_validation.py tests/test_review_store.py tests/test_web_app.py tests/test_pipeline.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/marriage_ocr/validation.py src/marriage_ocr/refinement/models.py src/marriage_ocr/refinement/text_corrections.py src/marriage_ocr/review_app.py src/marriage_ocr/review_store.py tests/test_validation.py tests/test_review_store.py tests/test_web_app.py tests/test_pipeline.py
git commit -m "feat: improve handwritten validation and review routing"
```

---

### Task 5: Persist provenance, baseline evaluation, and documentation

**Files:**
- Modify: `src/marriage_ocr/db_postgres.py`
- Modify: `src/marriage_ocr/refinement/audit.py`
- Modify: `src/marriage_ocr/refinement/benchmark.py`
- Modify: `README.md`
- Modify: `docs/user-guide.md`
- Modify: `docs/handover.md`
- Test: `tests/test_db_postgres.py`
- Test: `tests/refinement/test_benchmark.py`
- Test: `tests/test_runtime_packaging.py`

**Interfaces:**
- Consumes: field-level provenance, reviewed records, debug sidecars, and persisted records
- Produces: additive database columns or JSONB provenance, a baseline evaluation command path, and operator documentation

- [ ] **Step 1: Write the failing test**

Add a DB compatibility test that asserts existing records still load after provenance columns are introduced, plus a benchmark test that confirms the first 25 verified review bundles are counted correctly.

```python
def test_records_table_accepts_field_extraction_metadata():
    ...

def test_build_refinement_baseline_counts_verified_bundles_only():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
./.venv/bin/python -m pytest tests/test_db_postgres.py tests/refinement/test_benchmark.py tests/test_runtime_packaging.py -q
```

Expected: fail because the schema and evaluation/reporting surfaces are not wired for the new provenance data yet.

- [ ] **Step 3: Write minimal implementation**

Add the smallest additive persistence path needed for handwritten provenance, such as:

- `document_type`
- `review_route`
- `field_extractions` JSONB

Keep existing rows readable and existing exports unchanged.

Update the benchmark helper so it can summarize the first 25 verified handwritten review bundles without inventing accuracy results.

Document:

- how handwritten mode differs from typed mode
- which thresholds are configurable
- how to run the handwritten pipeline
- how to inspect review and benchmark artifacts

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
./.venv/bin/python -m pytest tests/test_db_postgres.py tests/refinement/test_benchmark.py tests/test_runtime_packaging.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/marriage_ocr/db_postgres.py src/marriage_ocr/refinement/audit.py src/marriage_ocr/refinement/benchmark.py README.md docs/user-guide.md docs/handover.md tests/test_db_postgres.py tests/refinement/test_benchmark.py tests/test_runtime_packaging.py
git commit -m "feat: persist handwritten provenance and document the workflow"
```

---

## Self-Review Checklist

- The plan covers the document-mode boundary, handwritten field extraction, OCR retries, validation/confidence, review routing, persistence, benchmark support, and docs.
- The plan keeps typed behavior stable and isolates the handwritten refactor.
- The plan avoids a custom handwriting model and avoids requiring live OCR credentials in tests.
- The plan does not use placeholders like TBD or implement later.
- Each task has a concrete failing test, a focused implementation target, a verification command, and a commit step.

