"""Prompt + response_schema pairs for Cerai / Rujuk handwritten extraction.

Drop-in for gemini_extractor.py: GeminiRecordExtractor selects a prompt +
schema pair from RECORD_TYPES by `config["record_type"]` (default "nikah",
which keeps using its own hardcoded, already-validated prompt/schema in
gemini_extractor.py -- the "nikah" entry here exists for completeness/future
unification but is not currently consumed).

Field naming and layout assumptions below are grounded in real sample
ledger photos (input/cerai/*, input/rujuk/*), not guessed. Those samples
show TWO physically different layouts per record type across the years:

- "legacy" (seen through ~2009): ~10 dedicated ruled columns.
- "modern" (seen from ~2024): only 5 columns; everything the legacy layout
  put in dedicated columns (talak count, original nikah date, a
  cross-reference bil) is folded into one free-text "Catatan" column using
  inline labels, e.g. "TALAQ : 1", "T.NIKAH : 03.03.2013",
  "BIL.DAFTAR : 0655/2013".

Both eras share ONE Gemini JSON schema per record type (every field
nullable) -- only the *prompt* text and the *layout config*
(fallback_column_ratios) differ by layout_variant, selected via the
`layout_variant` config key ("legacy" default, or "modern").

Open questions flagged rather than silently resolved (confirm against a
pilot before trusting at volume):
  - Cerai legacy has a right-page "Jumlah" column whose exact relationship
    to "Keadaan Talak" (a similar-looking value on some rows) is unclear
    from the image alone -- both are captured as separate fields
    (keadaan_talak, jumlah_talak).
  - The stacked number under BIL (no_rujukan) vs. the modern "No. Siri"
    (no_siri) may turn out to be the same concept under two names --
    kept separate until confirmed.
"""
from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Shared instructions, reused verbatim across record types -- these are
# script-level corrections (common misreadings of Jawi-influenced Rumi
# script), not content-specific, so they apply regardless of record type.
# ---------------------------------------------------------------------------
_SHARED_RULES = """
Use the image as the primary source. Use the Google Vision OCR cell hints
only as secondary evidence. If the OCR hint conflicts with the image,
prefer the image.

Return only JSON matching the schema. Do not include markdown.

Rules:
- Do not invent missing values. Use null if unreadable.
- Never guess a plausible-looking date, IC number, or name when the
  handwriting is unclear or partially illegible -- put whatever you can
  actually make out in the matching *_raw field and set the normalized
  field to null instead. A wrong value is worse than a missing one.
- Common OCR corrections: 4J/ITJ/14J -> HJ.; BIR/8IN -> BIN;
  BINT!/BINT1 -> BINTI; BAP9/B4PA -> BAPA.
- IC/passport values: this corpus spans both the old short-form IC
  (e.g. "2108418") and the modern 12-digit format (YYMMDD-PB-####).
  Transcribe exactly what is visible; do not convert between formats or
  pad digits. Put uncertain readings in the matching *_raw field.
- Dates: put the original visible text in the matching *_raw field;
  normalize to YYYY-MM-DD only when unambiguous. Where both a Hijri and a
  Masihi (Gregorian) date are visible for the same event, capture both --
  do not derive one from the other.
- The "Tandatangan" (signature) column sometimes also has printed/written
  phone numbers or names alongside the signature scrawl -- if you can read
  any such text, put it in hal_hal_lain instead of discarding it. Do not
  attempt to identify or transcribe the signature scrawl itself.
- Return field_confidence as an array of objects with `field` and
  `confidence`. Put fields below 0.70 confidence into uncertain_fields.
""".strip()


def _confidence_block() -> dict[str, Any]:
    return {
        "field_confidence": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "field": {"type": "STRING", "nullable": False},
                    "confidence": {"type": "NUMBER", "nullable": False},
                },
                "required": ["field", "confidence"],
            },
            "nullable": True,
        },
        "uncertain_fields": {"type": "ARRAY", "items": {"type": "STRING"}, "nullable": True},
        "notes": {"type": "ARRAY", "items": {"type": "STRING"}, "nullable": True},
    }


def _schema_from_properties(properties: dict[str, Any]) -> dict[str, Any]:
    full_properties = {**properties, **_confidence_block()}
    return {
        "type": "OBJECT",
        "properties": full_properties,
        "required": list(full_properties.keys()),
    }


_LEGACY_LAYOUT_NOTE = """
This page uses the OLDER ledger layout: dedicated ruled columns for each
piece of information (place, registrar, talak/reconciliation detail, etc.
each have their own column). Read each field from its own column. The
"Hal-Hal Lain" / remarks column is genuinely free text -- do not force
structure onto it.
""".strip()

_MODERN_LAYOUT_NOTE = """
This page uses the NEWER, more compact ledger layout: only Bil/reference
info, names+IC, one date column, signature+date-out, and one free-text
"Catatan" column -- there are no separate columns for place, registrar, or
talak/reconciliation detail on this layout. Instead, "Catatan" contains
short inline-labelled notes such as "TALAQ : 1", "T.NIKAH : 03.03.2013",
"T.CERAI : 27.11.2022", "BIL.DAFTAR : 0655/2013". Parse each labelled value
you can find into its matching field below, but ALSO copy the entire
Catatan cell verbatim into catatan_raw so nothing is lost if a label isn't
recognized. Do not fabricate a value for a field just because the modern
layout usually has one -- if a label is genuinely absent from Catatan,
leave that field null.
""".strip()


def _layout_note(layout_variant: str) -> str:
    return _MODERN_LAYOUT_NOTE if layout_variant.strip().lower() == "modern" else _LEGACY_LAYOUT_NOTE


def _build_prompt(document_description: str, field_notes: str, cell_hints_json: str, layout_variant: str) -> str:
    return f"""
You are extracting ONE handwritten row from {document_description}.

{_layout_note(layout_variant)}

{_SHARED_RULES}

{field_notes}

Google Vision OCR cell hints:
{cell_hints_json}
""".strip()


# ---------------------------------------------------------------------------
# NIKAH -- reference only; gemini_extractor.py currently uses its own
# already-validated hardcoded prompt/schema for record_type == "nikah"
# instead of this entry. Kept here so all three types are documented in one
# place and so future unification has a starting point.
# ---------------------------------------------------------------------------
NIKAH_FIELD_NOTES = """
Field-specific notes:
- nama_wali: the guardian's name, from the "Nama Wali" / "MAKLUMAT WALI"
  section.
- hubungan_wali: the guardian's relationship to the bride, e.g. BAPA,
  BAPA KANDUNG, ABANG, WALI HAKIM, DATUK.
- saksi_1 / saksi_2: the two marriage witnesses as TWO SEPARATE fields.
- mas_kahwin: usually contains RM and a numeric amount when visible.
- tarikh_nikah: the Masihi (Gregorian) marriage date.
""".strip()

NIKAH_PROPERTIES: dict[str, Any] = {
    "bil": {"type": "STRING", "nullable": True},
    "nama_suami": {"type": "STRING", "nullable": True},
    "ic_lama_suami": {"type": "STRING", "nullable": True},
    "ic_baru_suami": {"type": "STRING", "nullable": True},
    "id_suami_raw": {"type": "STRING", "nullable": True},
    "umur_suami": {"type": "INTEGER", "nullable": True},
    "nama_isteri": {"type": "STRING", "nullable": True},
    "ic_lama_isteri": {"type": "STRING", "nullable": True},
    "ic_baru_isteri": {"type": "STRING", "nullable": True},
    "id_isteri_raw": {"type": "STRING", "nullable": True},
    "umur_isteri": {"type": "INTEGER", "nullable": True},
    "mas_kahwin": {"type": "STRING", "nullable": True},
    "mas_kahwin_raw": {"type": "STRING", "nullable": True},
    "nama_pendaftar": {"type": "STRING", "nullable": True},
    "alamat_pendaftar": {"type": "STRING", "nullable": True},
    "nama_wali": {"type": "STRING", "nullable": True},
    "hubungan_wali": {"type": "STRING", "nullable": True},
    "saksi_1": {"type": "STRING", "nullable": True},
    "saksi_2": {"type": "STRING", "nullable": True},
    "tarikh_nikah": {"type": "STRING", "nullable": True},
    "tarikh_nikah_raw": {"type": "STRING", "nullable": True},
    "tarikh_keluar": {"type": "STRING", "nullable": True},
    "tarikh_keluar_raw": {"type": "STRING", "nullable": True},
    "remarks": {"type": "STRING", "nullable": True},
}


# ---------------------------------------------------------------------------
# CERAI -- fields grounded in input/cerai/*.jpg (1997, 2009, and 2024
# samples). See module docstring for the legacy/modern layout split.
# ---------------------------------------------------------------------------
CERAI_FIELD_NOTES = """
Field-specific notes:
- bil: the record's own sequential number, e.g. "1/97" or "1585".
- no_rujukan: a second number stacked directly under/beside bil on the
  legacy layout (e.g. "007847") -- a distinct reference number, not part
  of bil itself.
- no_siri: "No. Siri" serial number, stacked under bil on the modern
  layout (e.g. "150689"). May also appear as a later red-ink annotation
  added to an older legacy-layout page when a certified copy was issued --
  capture it if present even on a legacy page.
- tarikh_daftar: the registration date, stacked under bil (distinct from
  tarikh_cerai, the actual divorce date, which has its own column).
- nama_suami / ic_suami: husband's name and single IC/passport number (this
  corpus wants one IC field per person here, unlike Nikah's split
  ic_lama/ic_baru). ic_suami_raw holds anything illegible.
- nama_isteri / ic_isteri: same, for the wife.
- tempat_cerai: place/authority of the divorce, legacy layout only, e.g.
  "TANPA KEBENARAN MAHKAMAH", "KEBENARAN MAHKAMAH".
- nama_pendaftar: registrar's name, legacy layout only.
- keadaan_talak: the type of talak, e.g. "RAJIE", "TALAK TIGA (BAIN
  KUBRA)". On the modern layout this is usually just a count like
  "TALAQ : 1" inside Catatan -- if so, put that value here.
- jumlah_talak: a second, separate "Jumlah" column seen on the legacy
  layout's right-hand page, near Tarikh Cerai. Its exact relationship to
  keadaan_talak is not fully confirmed -- capture it verbatim as its own
  value, do not merge it into keadaan_talak.
- tarikh_nikah: the ORIGINAL marriage date being dissolved, only present
  via the modern layout's Catatan ("T.NIKAH : ...").
- tarikh_cerai: the divorce date, always in its own dedicated column on
  both layouts.
- bil_daftar_rujukan: a cross-reference bil to another registration,
  modern layout only, from Catatan's "BIL.DAFTAR : ..." label.
- catatan_raw: modern layout only -- the ENTIRE free-text Catatan cell,
  verbatim, in addition to whatever you parsed out of it above.
- hal_hal_lain: legacy layout's dedicated remarks column, e.g. "Diambil
  oleh isteri", or any modern-layout Tandatangan-column phone numbers/names
  per the shared rules above.
""".strip()

CERAI_PROPERTIES: dict[str, Any] = {
    "bil": {"type": "STRING", "nullable": True},
    "no_rujukan": {"type": "STRING", "nullable": True},
    "no_siri": {"type": "STRING", "nullable": True},
    "tarikh_daftar": {"type": "STRING", "nullable": True},
    "nama_suami": {"type": "STRING", "nullable": True},
    "ic_suami": {"type": "STRING", "nullable": True},
    "ic_suami_raw": {"type": "STRING", "nullable": True},
    "nama_isteri": {"type": "STRING", "nullable": True},
    "ic_isteri": {"type": "STRING", "nullable": True},
    "ic_isteri_raw": {"type": "STRING", "nullable": True},
    "tempat_cerai": {"type": "STRING", "nullable": True},
    "nama_pendaftar": {"type": "STRING", "nullable": True},
    "keadaan_talak": {"type": "STRING", "nullable": True},
    "jumlah_talak": {"type": "STRING", "nullable": True},
    "tarikh_nikah": {"type": "STRING", "nullable": True},
    "tarikh_cerai": {"type": "STRING", "nullable": True},
    "tarikh_cerai_raw": {"type": "STRING", "nullable": True},
    "tarikh_keluar": {"type": "STRING", "nullable": True},
    "tarikh_keluar_raw": {"type": "STRING", "nullable": True},
    "bil_daftar_rujukan": {"type": "STRING", "nullable": True},
    "catatan_raw": {"type": "STRING", "nullable": True},
    "hal_hal_lain": {"type": "STRING", "nullable": True},
}


# ---------------------------------------------------------------------------
# RUJUK -- fields grounded in input/rujuk/*.jpg (1990 and 2024 samples).
# ---------------------------------------------------------------------------
RUJUK_FIELD_NOTES = """
Field-specific notes:
- bil: the record's own sequential number, e.g. "1/90" or "00399".
- no_rujukan: a reference code stacked under bil on the legacy layout
  (e.g. "H 631075").
- no_siri: serial number stacked under bil, legacy (e.g. "NO. 006401") or
  modern ("No. Siri", e.g. "022868") layout alike.
- tarikh_daftar: the registration date, stacked under bil.
- nama_suami / umur_suami / ic_suami: husband's name, age in years (legacy
  layout shows this explicitly, e.g. "28 THN"), and single IC/passport
  number. ic_suami_raw holds anything illegible.
- nama_isteri / umur_isteri / ic_isteri: same, for the wife.
- tempat_rujuk: place the reconciliation was registered, legacy layout
  only, e.g. "PAID SHAH ALAM".
- nama_pendaftar: registrar/office name, legacy layout only.
- bil_cerai: a cross-reference to the ORIGINAL DIVORCE record's own bil
  number (e.g. "14/90"), NOT a description of the talak -- this is the
  legacy layout's dedicated "Bil Cerai" column, or the modern layout's
  Catatan "BIL.DAFTAR : ..." label.
- rujuk_kali: which reconciliation attempt this is, e.g. "PERTAMA" (legacy
  layout's "Rujuk Kali" column).
- tarikh_cerai: the original divorce date, only present via the modern
  layout's Catatan ("T.CERAI : ...").
- tarikh_nikah: the original marriage date, only present via the modern
  layout's Catatan ("T.NIKAH : ...").
- tarikh_rujuk: the reconciliation date, always in its own dedicated
  column on both layouts.
- catatan_raw: modern layout only -- the ENTIRE free-text Catatan cell,
  verbatim, in addition to whatever you parsed out of it above.
- hal_hal_lain: legacy layout's dedicated remarks column, e.g. "diambil
  oleh suami", or any modern-layout Tandatangan-column phone numbers/names
  per the shared rules above.
- IMPORTANT: this ledger's pre-printed column headers may be in Jawi
  script, but the actual handwritten entries (names, IC numbers, dates)
  are expected to be ordinary Rumi/Latin script based on samples reviewed.
  If you encounter Jawi script in a filled-in data field itself (not just
  the printed header), set that field to null and add a note explaining
  what you saw -- do not attempt to transliterate.
""".strip()

RUJUK_PROPERTIES: dict[str, Any] = {
    "bil": {"type": "STRING", "nullable": True},
    "no_rujukan": {"type": "STRING", "nullable": True},
    "no_siri": {"type": "STRING", "nullable": True},
    "tarikh_daftar": {"type": "STRING", "nullable": True},
    "nama_suami": {"type": "STRING", "nullable": True},
    "umur_suami": {"type": "INTEGER", "nullable": True},
    "ic_suami": {"type": "STRING", "nullable": True},
    "ic_suami_raw": {"type": "STRING", "nullable": True},
    "nama_isteri": {"type": "STRING", "nullable": True},
    "umur_isteri": {"type": "INTEGER", "nullable": True},
    "ic_isteri": {"type": "STRING", "nullable": True},
    "ic_isteri_raw": {"type": "STRING", "nullable": True},
    "tempat_rujuk": {"type": "STRING", "nullable": True},
    "nama_pendaftar": {"type": "STRING", "nullable": True},
    "bil_cerai": {"type": "STRING", "nullable": True},
    "rujuk_kali": {"type": "STRING", "nullable": True},
    "tarikh_cerai": {"type": "STRING", "nullable": True},
    "tarikh_nikah": {"type": "STRING", "nullable": True},
    "tarikh_rujuk": {"type": "STRING", "nullable": True},
    "tarikh_rujuk_raw": {"type": "STRING", "nullable": True},
    "tarikh_keluar": {"type": "STRING", "nullable": True},
    "tarikh_keluar_raw": {"type": "STRING", "nullable": True},
    "catatan_raw": {"type": "STRING", "nullable": True},
    "hal_hal_lain": {"type": "STRING", "nullable": True},
}


NIKAH_SCHEMA = _schema_from_properties(NIKAH_PROPERTIES)
CERAI_SCHEMA = _schema_from_properties(CERAI_PROPERTIES)
RUJUK_SCHEMA = _schema_from_properties(RUJUK_PROPERTIES)


def nikah_prompt(cell_hints_json: str, layout_variant: str = "legacy") -> str:
    return _build_prompt(
        "a Malay Islamic marriage register, Daftar Perkahwinan Orang Islam",
        NIKAH_FIELD_NOTES,
        cell_hints_json,
        layout_variant,
    )


def cerai_prompt(cell_hints_json: str, layout_variant: str = "legacy") -> str:
    return _build_prompt(
        "a Malay Islamic divorce register, Daftar Perceraian Orang Islam",
        CERAI_FIELD_NOTES,
        cell_hints_json,
        layout_variant,
    )


def rujuk_prompt(cell_hints_json: str, layout_variant: str = "legacy") -> str:
    return _build_prompt(
        "a Malay Islamic reconciliation register, Daftar Rujuk Orang Islam",
        RUJUK_FIELD_NOTES,
        cell_hints_json,
        layout_variant,
    )


# ---------------------------------------------------------------------------
# GeminiRecordExtractor picks a pair by config["record_type"].
# ---------------------------------------------------------------------------
RECORD_TYPES: dict[str, dict[str, Any]] = {
    "nikah": {"prompt_builder": nikah_prompt, "schema": NIKAH_SCHEMA},
    "cerai": {"prompt_builder": cerai_prompt, "schema": CERAI_SCHEMA},
    "rujuk": {"prompt_builder": rujuk_prompt, "schema": RUJUK_SCHEMA},
}
