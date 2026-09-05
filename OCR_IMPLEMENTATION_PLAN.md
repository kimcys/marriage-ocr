# OCR Implementation Plan — Nikah / Cerai / Rujuk

Synthesized from our full planning session, checked directly against your
`marriage-ocr` repo (not just the README). Target output schema below matches
`OCR_Data_Column.xlsx` exactly.

---

## 1. What your code already does well (confirmed by reading it)

No need to rebuild these — they're solid:

- **PDF rasterization is already correct.** `typed/loader.py` and
  `document_loader.py` both rasterize every page via PyMuPDF before OCR runs.
  There is no embedded-text-layer bug.
- **Row/column detection for handwritten ledgers is real CV, not fixed
  pixels.** `layout.py` detects ruled table lines and the red BIL markers to
  find record boundaries dynamically. Fixed ratios are a fallback only.
- **Typed-form position drift is already corrected.** `typed/template.py`
  anchor-matches known phrases (e.g. "MAKLUMAT PASANGAN") and shifts the
  expected field regions by the observed offset.
- **Gemini is already grounded, schema-forced, and has a null escape hatch.**
  `src/llm/gemini_extractor.py` sends the image as primary evidence with
  Vision OCR as secondary hint, forces `response_schema`, and instructs the
  model never to guess an unclear value. It already returns per-field
  `field_confidence` and `uncertain_fields`.
- **Input ingestion already recurses through nested folders.** Point
  `--input` at your unsorted OneDrive dump as-is; no pre-sorting needed for a
  single run.

## 2. The real gaps, ranked by what to do first

### Gap 1 — Model tier (5-minute change, do this first)
`config/handwritten.yaml` sets `llm.model: gemini-2.5-flash`. Change to
`gemini-3.1-pro` and re-run against your existing reviewed samples to see if
accuracy moves. Zero other code changes required for this test.

### Gap 2 — Everything assumes Nikah only
Confirmed in four separate places:
- The Gemini prompt text literally says *"extracting ONE handwritten row
  from a Malay Islamic marriage register"* — hardcoded.
- `layout.py`'s fallback column names (`wali`, `hubungan_wali`, `saksi`...)
  are Nikah-specific.
- `typed/template.py` only has `BORANG_4B_REGIONS` — nothing for Borang 3A
  or any Cerai/Rujuk typed certificate.
- The Postgres `records` table has typed columns for `nama_suami`,
  `tarikh_nikah`, `mas_kahwin`, `wali` — nothing for Cerai or Rujuk fields.

Section 3 below gives you the record-type-specific prompts, schemas, and DB
columns needed to close this gap for **handwritten** ledgers. Section 4
covers what's needed for **typed** certificates, with an honest caveat about
what I can and can't give you without real image measurements.

### Gap 3 — No Batch API wiring
`gemini_extractor.py` calls `generate_content` synchronously per record.
Worth adding once Gaps 1–2 are validated on a small sample — no point
batching a prompt you're still tuning.

---

## 3. Handwritten ledgers — ready to use

Three record types, each needs its own prompt + schema, following the exact
pattern already used in `gemini_extractor.py` (image primary, OCR hints
secondary, null instead of guessing, common OCR-correction substitutions
reused as-is since they're script-level, not content-level fixes).

See `record_schemas.py` for the actual code — one prompt + one
`response_schema` per type, drop-in compatible with your existing
`GeminiRecordExtractor` class (just needs a `record_type` parameter added to
pick which prompt/schema pair to use — see the `RECORD_TYPES` dict at the
bottom of that file).

**Only Nikah is validated against real samples from this session.** Cerai and
Rujuk prompts are built from the ledger samples you shared (Daftar
Perceraian, and the pre-printed "Divorce Register"/Daftar Rujok book) — worth
a small pilot batch against your own ground truth before trusting them at
volume, same as we discussed for any new template.

**Jawi note on the Rujuk ledger:** the pre-printed column headers on that
register are in Jawi script, but every sample of actual *filled-in* data we
looked at was ordinary Rumi/Latin script. The prompt below assumes that
holds across your corpus — worth spot-checking a wider sample before
assuming no Jawi OCR is needed.

## 4. Typed certificates — plan, with an honest gap

Typed docs don't go through Gemini at all currently — Vision + fixed
regions, corrected by the anchor-transform system in `typed/template.py`.
Adding Borang 3A (or any typed Cerai/Rujuk certificate) means writing a new
`BORANG_3A_REGIONS` / `BORANG_3A_ANCHORS` dict in the same shape.

**I'm not fabricating those coordinates.** They're page-position ratios that
need to be measured against your actual rendered page images — the same way
the existing Borang 4B ones clearly were. Inventing plausible-looking numbers
here would silently produce wrong positions, which is exactly the failure
mode this whole session has been about avoiding. The method to build them:
render a sample Borang 3A page at your configured DPI, overlay a coordinate
grid, and read off the same anchor phrases and field regions
`estimate_transform()` already expects — "SURAT PERAKUAN NIKAH", the numbered
fields 1–8, the Taliq declaration page. Happy to do this with you directly if
you upload a few more Borang 3A samples at full resolution so the actual
pixel measurement is verifiable rather than guessed.

## 5. Database schema — matches your Excel exactly

See `migration_add_cerai_rujuk.sql`. Adds a `record_type` column and the
Cerai/Rujuk-specific columns to your existing `records` table, rather than
building separate tables per type — matches how your Excel template already
represents this (one wide table, nullable columns, populated per category).

## 6. Suggested order of execution

1. Run Gap 1 (model swap) against your existing reviewed ground truth.
2. Apply the SQL migration.
3. Wire the Cerai/Rujuk prompts+schemas from `record_schemas.py` into
   `gemini_extractor.py` behind a `record_type` parameter.
4. Pilot on a small stratified sample across both new types before full
   volume — same reasoning as the original bake-off.
5. Batch API wiring once 1–4 are validated.
6. Typed Borang 3A / Cerai / Rujuk certificates — once real samples are
   available for coordinate measurement.
