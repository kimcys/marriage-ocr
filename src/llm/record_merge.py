from __future__ import annotations

from dataclasses import replace
from statistics import mean
from typing import Any, Mapping

from marriage_ocr.models import ExtractedRecord, OcrResult
from marriage_ocr.validation import validate_record

from . import record_schemas
from .gemini_extractor import GeminiRecordResult

_META_SCHEMA_KEYS = {"field_confidence", "uncertain_fields", "notes"}


def _fields_from_schema(properties: Mapping[str, Any]) -> list[str]:
    return [name for name in properties if name not in _META_SCHEMA_KEYS]


NIKAH_CRITICAL_FIELDS = {
    "bil",
    "nama_suami",
    "nama_isteri",
    "ic_lama_suami",
    "ic_baru_suami",
    "ic_lama_isteri",
    "ic_baru_isteri",
    "umur_suami",
    "umur_isteri",
    "tarikh_nikah",
}

NIKAH_MERGE_FIELDS = [
    "bil",
    "nama_suami", "ic_lama_suami", "ic_baru_suami", "id_suami_raw", "umur_suami",
    "nama_isteri", "ic_lama_isteri", "ic_baru_isteri", "id_isteri_raw", "umur_isteri",
    "mas_kahwin", "mas_kahwin_raw",
    "nama_pendaftar", "alamat_pendaftar",
    "nama_wali", "hubungan_wali",
    "saksi_1", "saksi_2",
    "tarikh_nikah", "tarikh_nikah_raw",
    "tarikh_keluar", "tarikh_keluar_raw",
    "remarks",
]

NIKAH_CELL_MAP = {
    "bil": "bil",
    "nama_suami": "suami_isteri",
    "ic_lama_suami": "suami_isteri",
    "ic_baru_suami": "suami_isteri",
    "id_suami_raw": "suami_isteri",
    "umur_suami": "suami_isteri",
    "nama_isteri": "suami_isteri",
    "ic_lama_isteri": "suami_isteri",
    "ic_baru_isteri": "suami_isteri",
    "id_isteri_raw": "suami_isteri",
    "umur_isteri": "suami_isteri",
    "mas_kahwin": "suami_isteri",
    "mas_kahwin_raw": "suami_isteri",
    "nama_pendaftar": "pendaftar",
    "alamat_pendaftar": "pendaftar",
    "nama_wali": "wali",
    "hubungan_wali": "hubungan_wali",
    "saksi_1": "saksi",
    "saksi_2": "saksi",
    "tarikh_nikah": "tarikh_nikah",
    "tarikh_nikah_raw": "tarikh_nikah",
    "tarikh_keluar": "tarikh_keluar",
    "tarikh_keluar_raw": "tarikh_keluar",
    "remarks": "remarks",
}

# Backward-compatible aliases (pre-record_type module surface).
CRITICAL_FIELDS = NIKAH_CRITICAL_FIELDS
MERGE_FIELDS = NIKAH_MERGE_FIELDS

# Cerai/Rujuk field lists are derived straight from the Gemini schema
# properties in record_schemas.py so the two stay in sync automatically
# instead of drifting -- every schema field is a merge candidate.
CERAI_MERGE_FIELDS = _fields_from_schema(record_schemas.CERAI_PROPERTIES)
RUJUK_MERGE_FIELDS = _fields_from_schema(record_schemas.RUJUK_PROPERTIES)

CERAI_CRITICAL_FIELDS = {"bil", "nama_suami", "nama_isteri", "ic_suami", "ic_isteri", "tarikh_cerai"}
RUJUK_CRITICAL_FIELDS = {"bil", "nama_suami", "nama_isteri", "ic_suami", "ic_isteri", "tarikh_rujuk"}

# Cell-name mapping for the legacy 10-column layout; fields that only ever
# come from the modern layout's free-text "Catatan" column (or that have no
# known cell mapping yet) point at "catatan" -- _ocr_field_confidence already
# falls back to 0.0 when a mapped cell name isn't present in cell_results,
# so this is safe to reuse across both layout_variant configs unchanged.
CERAI_CELL_MAP = {
    "bil": "bil",
    "no_rujukan": "bil",
    "no_siri": "bil",
    "tarikh_daftar": "bil",
    "nama_suami": "suami_isteri",
    "ic_suami": "suami_isteri",
    "ic_suami_raw": "suami_isteri",
    "nama_isteri": "suami_isteri",
    "ic_isteri": "suami_isteri",
    "ic_isteri_raw": "suami_isteri",
    "tempat_cerai": "tempat_cerai",
    "nama_pendaftar": "pendaftar",
    "keadaan_talak": "keadaan_talak",
    "jumlah_talak": "jumlah",
    "tarikh_nikah": "catatan",
    "tarikh_cerai": "tarikh_cerai",
    "tarikh_cerai_raw": "tarikh_cerai",
    "tarikh_keluar": "tarikh_keluar",
    "tarikh_keluar_raw": "tarikh_keluar",
    "bil_daftar_rujukan": "catatan",
    "catatan_raw": "catatan",
    "hal_hal_lain": "hal_hal_lain",
}

RUJUK_CELL_MAP = {
    "bil": "bil",
    "no_rujukan": "bil",
    "no_siri": "bil",
    "tarikh_daftar": "bil",
    "nama_suami": "suami_isteri",
    "umur_suami": "suami_isteri",
    "ic_suami": "suami_isteri",
    "ic_suami_raw": "suami_isteri",
    "nama_isteri": "suami_isteri",
    "umur_isteri": "suami_isteri",
    "ic_isteri": "suami_isteri",
    "ic_isteri_raw": "suami_isteri",
    "tempat_rujuk": "tempat_rujuk",
    "nama_pendaftar": "pendaftar",
    "bil_cerai": "bil_cerai",
    "rujuk_kali": "rujuk_kali",
    "tarikh_cerai": "catatan",
    "tarikh_nikah": "catatan",
    "tarikh_rujuk": "tarikh_rujuk",
    "tarikh_rujuk_raw": "tarikh_rujuk",
    "tarikh_keluar": "tarikh_keluar",
    "tarikh_keluar_raw": "tarikh_keluar",
    "catatan_raw": "catatan",
    "hal_hal_lain": "hal_hal_lain",
}

MERGE_FIELDS_BY_TYPE: dict[str, list[str]] = {
    "nikah": NIKAH_MERGE_FIELDS,
    "cerai": CERAI_MERGE_FIELDS,
    "rujuk": RUJUK_MERGE_FIELDS,
}
CRITICAL_FIELDS_BY_TYPE: dict[str, set[str]] = {
    "nikah": NIKAH_CRITICAL_FIELDS,
    "cerai": CERAI_CRITICAL_FIELDS,
    "rujuk": RUJUK_CRITICAL_FIELDS,
}
CELL_MAP_BY_TYPE: dict[str, dict[str, str]] = {
    "nikah": NIKAH_CELL_MAP,
    "cerai": CERAI_CELL_MAP,
    "rujuk": RUJUK_CELL_MAP,
}


def merge_parser_and_gemini(
    *,
    parser_record: ExtractedRecord,
    gemini_result: GeminiRecordResult,
    cell_results: Mapping[str, OcrResult],
    validation_config: Mapping[str, Any],
    layout_confidence: float = 1.0,
    prefer_gemini_threshold: float = 0.70,
    review_below_field_confidence: float = 0.80,
    record_type: str = "nikah",
) -> ExtractedRecord:
    """Merge deterministic parser output with Gemini semantic extraction.

    Selection rule:
    - Keep parser value when Gemini value is missing.
    - Use Gemini value when parser value is missing.
    - If both exist and match after normalization, keep parser spelling and mark agreement.
    - If they disagree, prefer Gemini only when its field confidence is high enough.
    """

    record_type = record_type.strip().lower()
    merge_fields = MERGE_FIELDS_BY_TYPE.get(record_type, NIKAH_MERGE_FIELDS)
    critical_fields = CRITICAL_FIELDS_BY_TYPE.get(record_type, NIKAH_CRITICAL_FIELDS)
    cell_map = CELL_MAP_BY_TYPE.get(record_type, NIKAH_CELL_MAP)

    gemini_record = gemini_result.record
    chosen: dict[str, Any] = {}
    reasons = list(parser_record.review_reason or [])
    field_confidences: list[float] = []

    for field in merge_fields:
        parser_value = getattr(parser_record, field, None)
        gemini_value = getattr(gemini_record, field, None)
        gemini_conf = float(gemini_result.field_confidence.get(field, gemini_record.confidence or 0.0))

        if _blank(gemini_value):
            chosen[field] = parser_value
            if not _blank(parser_value):
                field_confidences.append(_ocr_field_confidence(field, cell_results, cell_map))
            continue

        if _blank(parser_value):
            chosen[field] = gemini_value
            field_confidences.append(gemini_conf)
            if field in critical_fields:
                # There's no parser value to cross-check this against, so this is
                # Gemini's word alone for a critical field (e.g. tarikh_nikah).
                # LLMs are prone to reporting high confidence for a fabricated-but-
                # plausible value, so don't rely on gemini_conf to catch this --
                # force review unconditionally rather than trust the model's own
                # confidence about its own unverified guess.
                reasons.append(f"{field}: filled by Gemini with no parser corroboration; review required")
            else:
                reasons.append(f"{field}: filled by Gemini")
            continue

        if _norm(parser_value) == _norm(gemini_value):
            chosen[field] = parser_value
            field_confidences.append(max(gemini_conf, _ocr_field_confidence(field, cell_results, cell_map)))
            continue

        if gemini_conf >= prefer_gemini_threshold:
            chosen[field] = gemini_value
            field_confidences.append(gemini_conf * 0.95)
            reasons.append(f"{field}: parser/Gemini disagreement; chose Gemini")
        else:
            chosen[field] = parser_value
            field_confidences.append(min(gemini_conf, _ocr_field_confidence(field, cell_results, cell_map)))
            reasons.append(f"{field}: parser/Gemini disagreement; review required")

    chosen["record_type"] = record_type.upper()
    merged = replace(parser_record, **chosen)

    low_conf_fields = [
        field for field, confidence in gemini_result.field_confidence.items()
        if confidence < review_below_field_confidence and field in critical_fields
    ]
    for field in low_conf_fields:
        reasons.append(f"low Gemini confidence: {field}")

    if gemini_result.uncertain_fields:
        reasons.append("Gemini uncertain fields: " + ", ".join(gemini_result.uncertain_fields))

    merged.review_reason = _dedupe(reasons)
    merged.confidence = round(mean(field_confidences), 4) if field_confidences else parser_record.confidence

    validated = validate_record(
        merged,
        cell_results,
        validation_config,
        layout_confidence=layout_confidence,
        record_type=record_type,
    )

    # Never allow OK when a critical parser/Gemini disagreement remains, or when a
    # critical field's only source is an uncorroborated Gemini guess.
    if any(
        "disagreement" in reason or "low Gemini confidence" in reason or "no parser corroboration" in reason
        for reason in merged.review_reason
    ):
        validated.status_review = "REVIEW"
        validated.review_reason = _dedupe(list(validated.review_reason or []) + merged.review_reason)

    return validated


def _ocr_field_confidence(field: str, cell_results: Mapping[str, OcrResult], cell_map: Mapping[str, str]) -> float:
    cell_name = cell_map.get(field)
    if not cell_name or cell_name not in cell_results:
        return 0.0
    return float(cell_results[cell_name].average_confidence or 0.0)


def _blank(value: Any) -> bool:
    return value is None or str(value).strip() == ""


def _norm(value: Any) -> str:
    return " ".join(str(value).upper().replace(".", "").replace(",", "").split())


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        out.append(value)
        seen.add(value)
    return out
