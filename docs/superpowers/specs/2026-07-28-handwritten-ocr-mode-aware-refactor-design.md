# Handwritten OCR Mode-Aware Refactor Design

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve handwritten OCR accuracy, validation, confidence scoring, and review routing by introducing an explicit document-mode boundary while keeping typed processing stable.

**Architecture:** Keep the current typed and handwritten entry points, but add a shared mode-aware extraction layer that branches early on `DocumentType`. Handwritten records get layered OCR candidate generation, field-level normalization/validation, confidence scoring, and review routing; typed records keep their current simpler path with only the new mode abstraction and any shared validation helpers that are safe to reuse.

**Tech Stack:** Python 3.14, Typer CLI, Google Vision OCR, Gemini semantic extraction, optional PaddleOCR adapter, Streamlit review UI, PostgreSQL, pytest, openpyxl, Pillow/OpenCV.

## Global Constraints

- Do not train or integrate a custom handwriting model.
- Do not break existing typed OCR behavior.
- Do not require live OCR credentials in the standard test suite.
- Preserve raw OCR values and auditability; never overwrite original OCR text destructively.
- Keep paid OCR integrations behind the existing provider interfaces.
- Use incremental changes with tests for each independently reviewable step.
- Keep production and development configuration separate.

---

## 1. Current State

### Current OCR Flow

The repository already has two operating paths:

- `marriage_ocr.cli process` for handwritten records
- `marriage_ocr.cli process-typed` for typed Borang 4B PDFs

The handwritten path currently runs:

1. load document pages
2. preprocess pages and detect layout
3. run Google Vision full-page OCR or crop OCR fallback
4. parse the OCR text into `ExtractedRecord`
5. run handwritten field refinement on names, ICs, and dates
6. optionally merge Gemini semantic extraction into the record
7. validate the record
8. export CSV/XLSX and review artifacts

The typed path already has its own OCR, retry, validation, and CSV export flow.

### Typed Versus Handwritten Handling

The typed pipeline is already more structured and more accurate in practice. It uses:

- typed PDF rendering
- typed OCR client
- typed retry crops
- typed validator
- typed CSV store

The handwritten path is more heuristic. It currently relies on:

- page or crop OCR from Google Vision
- parser normalization
- field refinement retries
- Gemini merge and validation
- record-level confidence and review reasons

### Gemini Integration Status

Gemini is already implemented and active in the handwritten path. It is used as a semantic extractor after Google Vision OCR, not as the primary OCR engine. The handwritten profile also already has a special prompt mode that is more aggressive for handwritten rows.

### Existing Image Preprocessing

The handwritten pipeline already supports:

- page preprocessing
- adaptive thresholding
- deskewing
- table detection
- crop saving
- retry crop generation for suspicious fields

Typed processing has its own PDF rendering and retry-crop flow.

### Existing Validation Rules

Handwritten validation currently checks:

- husband and wife names
- old/new IC presence and broad structure
- age range
- marriage date validity
- mas kahwin presence
- layout confidence
- OCR confidence

The validator is mostly record-level and binary. It does not yet model field candidates, field-level validation reasons, or a typed handwritten acceptance strategy.

### Existing Confidence Calculation

Handwritten confidence is a heuristic record-level score that deducts points for missing or invalid fields and low OCR/layout confidence. Gemini can influence the final confidence, but there is no shared field-confidence model across OCR providers, normalization, and validation.

### Existing Database Fields

The current PostgreSQL schema stores compact record data, raw JSON blobs, status, confidence, review metadata, and timestamps. It does not store:

- candidate histories
- field-level confidence
- validation reasons per field
- OCR provider/model per field
- image variant per field
- correction provenance

### Existing Human-Review Workflow

The Streamlit review UI already shows:

- the full record crop
- cell crops
- raw OCR payload
- refinement audit rows
- editable record fields
- corrected record export
- training-data export

It is still mostly a record-edit UI. It does not yet present a first-class field-candidate workflow for handwritten review.

### Current Test Coverage

The repository already has meaningful coverage for:

- OCR runtime adapters
- parser logic
- validation
- handwritten refinement
- Gemini extraction and merge behavior
- typed pipeline behavior
- review store/UI
- exports
- database access

The suite is currently green on this branch.

### Main Technical Weaknesses

1. No explicit document-mode abstraction at the core pipeline boundary.
2. Handwritten extraction still uses record-level heuristics instead of typed field extraction artifacts.
3. Validation is too binary for ambiguous handwritten OCR.
4. Confidence is not field-aware enough to support safe auto-accept versus review routing.
5. Review UI is not candidate-driven.
6. The database schema does not preserve enough field-level provenance for a reliable learning loop.

### Files Likely To Require Modification

- `src/marriage_ocr/pipeline.py`
- `src/marriage_ocr/models.py`
- `src/marriage_ocr/ocr/__init__.py`
- `src/marriage_ocr/parser/__init__.py`
- `src/marriage_ocr/validation.py`
- `src/marriage_ocr/refinement/models.py`
- `src/marriage_ocr/refinement/field_refinement.py`
- `src/marriage_ocr/refinement/text_corrections.py`
- `src/llm/gemini_extractor.py`
- `src/llm/record_merge.py`
- `src/marriage_ocr/review_app.py`
- `src/marriage_ocr/review_store.py`
- `src/marriage_ocr/db_postgres.py`
- `src/marriage_ocr/config.py`
- `config/*.yaml`
- `tests/*`
- `README.md`
- `docs/user-guide.md`
- `docs/handover.md`

---

## 2. Proposed Architecture

### Document Mode Boundary

Introduce an explicit document mode:

- `DocumentType.TYPED`
- `DocumentType.HANDWRITTEN`

The mode should be determined early from the command path or config and passed through the pipeline. This should not require duplicating the entire pipeline; it should select different strategies at a few defined seams:

- preprocessing
- OCR candidate generation
- normalization rules
- confidence weighting
- review routing thresholds
- export defaults

### Shared Extraction Model

Add a shared field-level extraction model that preserves:

- raw OCR transcription
- normalized value
- candidates
- source provider
- image variant
- confidence
- validation status
- validation reasons
- review flag
- verified value

This should live in the main model layer or a focused extraction module and be reused by handwritten validation, review, and export. Typed processing can continue using its current simpler result shape until the mode-aware refactor needs the same field-level model.

### Handwritten Strategy

For handwritten records, the pipeline should:

1. run primary OCR
2. parse the literal transcription
3. normalize safely without destroying the raw text
4. validate each field
5. generate retry candidates only for suspicious fields
6. run secondary OCR only for those fields
7. compare candidates and apply cross-field validation
8. classify review outcome
9. export record plus audit metadata

The handwritten strategy should be conservative about auto-accepting names, ICs, and dates unless the candidate and validation evidence are strong.

### Typed Strategy

Typed processing should keep its current optimized flow. The only planned shared addition is the mode abstraction and any safe shared validation utilities. Typed should not inherit the more expensive handwritten retry behavior by default.

### Confidence and Review Routing

Confidence should be computed per field and then rolled up to a record score. Review routing should support:

- `AUTO_ACCEPT`
- `FIELD_REVIEW`
- `FULL_RECORD_REVIEW`

These outcomes should be derived from:

- field validity
- OCR confidence
- agreement between OCR passes
- candidate counts
- cross-field consistency
- image quality signals

### Review and Learning Loop

Review artifacts should preserve:

- original OCR
- selected value
- rejected candidates
- review reason
- provider/model
- crop reference
- correction history

The review UI should continue to work as a Streamlit app, but its field editing surface should become aware of field candidates and review reasons.

---

## 3. Phase 1 Scope

This phase is intentionally limited to design and the smallest implementation boundary needed to make the handwritten path more reliable without destabilizing typed records.

### In Scope

- define `DocumentType`
- route typed and handwritten through a mode-aware strategy boundary
- improve handwritten field extraction and confidence handling
- keep Gemini and Google Vision in the handwritten path
- preserve raw and normalized values
- improve handwritten validation and review routing
- update review artifacts to carry field-level provenance
- add tests for the new mode boundary and handwritten-specific behavior

### Out of Scope For Phase 1

- a full rewrite of the typed pipeline
- a custom handwriting model
- a new review application framework
- a major PostgreSQL redesign beyond the minimum required to persist new metadata
- a new OCR provider

---

## 4. Validation Design

### Identity Classification

Replace generic identity validation with a classifier that can distinguish:

- `NEW_IC`
- `OLD_IC`
- `MILITARY_ID`
- `POLICE_ID`
- `PASSPORT`
- `UNKNOWN`

Do not invent unsupported Malaysian patterns. Use configuration plus verified examples from the existing codebase and tests. Candidate validation should:

- preserve raw text
- remove separators only for analysis
- generate safe OCR substitutions only in valid numeric positions
- accept only a single corrected candidate when ambiguity exists

### Date Validation

Dates should:

- preserve raw OCR text
- generate candidate interpretations from collapsed or separated text
- validate real calendar dates
- resolve two-digit years using register context where available
- avoid silently choosing between multiple plausible dates

Cross-field checks should flag, not automatically rewrite:

- marriage date before issue date
- implausible age/date combinations
- impossible calendar dates

### Name Processing

Names should:

- preserve raw text
- normalize whitespace and punctuation safely
- keep particles like `BIN`, `BINTI`, `BT`, `MOHD`, `ABDUL`
- avoid spellchecker-style autocorrection
- use verified-value ranking only as a signal, not as automatic replacement

### Address Handling

Address normalization should be conservative:

- normalize common abbreviations
- preserve historical spelling
- use local verified values only for ranking
- never rewrite an address merely because it is not a modern standard spelling

### Field Confidence

Field confidence should combine:

- primary OCR confidence
- retry OCR agreement
- format validity
- cross-field consistency
- candidate count
- image quality
- correction penalty

It should be treated as a review score, not a calibrated probability unless calibration is explicitly measured.

### Review Routing

Routes should be:

- `AUTO_ACCEPT` when critical fields are valid and confidence is high
- `FIELD_REVIEW` when only specific fields are uncertain
- `FULL_RECORD_REVIEW` when segmentation, crop quality, or cross-field consistency is too poor

---

## 5. Database Proposal

Phase 1 should avoid a disruptive schema rewrite. The minimum coherent addition is to preserve field-level provenance for handwritten review and future evaluation.

Likely additions:

- raw OCR transcription per refined field
- normalized field value
- field confidence
- validation reasons
- candidate source/model metadata
- review route

Backward compatibility must be preserved:

- existing records remain readable
- existing exports continue to work
- old review artifacts remain valid

If the current schema is insufficient for the new audit trail, add the smallest migration that preserves all existing columns and records.

Rollback requirement:

- migration must be additive or have a documented downgrade path
- do not drop or rename existing columns in this phase

---

## 6. Testing Strategy

### Unit Tests

Add tests for:

- mode selection
- handwritten versus typed strategy routing
- field preservation of raw and normalized values
- identity classification and candidate handling
- date candidate generation and calendar validation
- name-safe normalization and ranking
- field confidence calculation
- review routing
- Gemini structured-output validation

### Integration Tests

Add tests for:

- handwritten pipeline with primary OCR success
- handwritten retry OCR on suspicious fields
- fallback when retry OCR fails
- typed pipeline unchanged behavior
- review artifact preservation

### Regression Tests

Cover known handwritten issues:

- malformed names
- old IC and modern IC confusion
- collapsed dates
- address noise
- guardian relationship noise
- witness names
- dowry extraction

All OCR APIs must remain mocked in tests.

---

## 7. Risks And Trade-offs

### Risks

- more aggressive handwritten acceptance could increase false auto-accepts
- field-level model changes may require review UI updates
- schema additions may need a migration if provenance is persisted
- Gemini may still produce plausible but incorrect text if the prompt is too permissive

### Trade-offs

- keeping typed behavior stable means not all structural improvements will be shared immediately
- a conservative handwritten fallback can keep review workload higher, but that is preferable to silently accepting wrong data
- a small additive schema change is safer than a wide database rewrite

### Security And Privacy

The repo already handles sensitive personal data. The refactor must continue to:

- keep secrets in env/config, not source code
- avoid logging raw IC values in production logs
- limit debug artifact retention
- clean up temporary images

---

## 8. Implementation Outline

This spec intentionally stops before detailed task planning.

Next steps after review:

1. turn this design into a file-by-file implementation plan
2. split the handwritten extraction refactor into testable tasks
3. implement the mode-aware boundary first
4. add handwritten field extraction and validation improvements
5. update the review UI and persistence only where the new data needs to be surfaced

