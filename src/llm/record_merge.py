from __future__ import annotations

from dataclasses import replace
from statistics import mean
from typing import Any, Mapping

from marriage_ocr.models import ExtractedRecord, OcrResult
from marriage_ocr.validation import validate_record

from .gemini_extractor import GeminiRecordResult

CRITICAL_FIELDS = {
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

MERGE_FIELDS = [
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


def merge_parser_and_gemini(
    *,
    parser_record: ExtractedRecord,
    gemini_result: GeminiRecordResult,
    cell_results: Mapping[str, OcrResult],
    validation_config: Mapping[str, Any],
    layout_confidence: float = 1.0,
    prefer_gemini_threshold: float = 0.70,
    review_below_field_confidence: float = 0.80,
) -> ExtractedRecord:
    """Merge deterministic parser output with Gemini semantic extraction.

    Selection rule:
    - Keep parser value when Gemini value is missing.
    - Use Gemini value when parser value is missing.
    - If both exist and match after normalization, keep parser spelling and mark agreement.
    - If they disagree, prefer Gemini only when its field confidence is high enough.
    """

    gemini_record = gemini_result.record
    chosen: dict[str, Any] = {}
    reasons = list(parser_record.review_reason or [])
    field_confidences: list[float] = []

    for field in MERGE_FIELDS:
        parser_value = getattr(parser_record, field, None)
        gemini_value = getattr(gemini_record, field, None)
        gemini_conf = float(gemini_result.field_confidence.get(field, gemini_record.confidence or 0.0))

        if _blank(gemini_value):
            chosen[field] = parser_value
            if not _blank(parser_value):
                field_confidences.append(_ocr_field_confidence(field, cell_results))
            continue

        if _blank(parser_value):
            chosen[field] = gemini_value
            field_confidences.append(gemini_conf)
            reasons.append(f"{field}: filled by Gemini")
            continue

        if _norm(parser_value) == _norm(gemini_value):
            chosen[field] = parser_value
            field_confidences.append(max(gemini_conf, _ocr_field_confidence(field, cell_results)))
            continue

        if gemini_conf >= prefer_gemini_threshold:
            chosen[field] = gemini_value
            field_confidences.append(gemini_conf * 0.95)
            reasons.append(f"{field}: parser/Gemini disagreement; chose Gemini")
        else:
            chosen[field] = parser_value
            field_confidences.append(min(gemini_conf, _ocr_field_confidence(field, cell_results)))
            reasons.append(f"{field}: parser/Gemini disagreement; review required")

    merged = replace(parser_record, **chosen)

    low_conf_fields = [
        field for field, confidence in gemini_result.field_confidence.items()
        if confidence < review_below_field_confidence and field in CRITICAL_FIELDS
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
    )

    # Never allow OK when a critical parser/Gemini disagreement remains.
    if any("disagreement" in reason or "low Gemini confidence" in reason for reason in merged.review_reason):
        validated.status_review = "REVIEW"
        validated.review_reason = _dedupe(list(validated.review_reason or []) + merged.review_reason)

    return validated


def _ocr_field_confidence(field: str, cell_results: Mapping[str, OcrResult]) -> float:
    mapping = {
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
    cell_name = mapping.get(field)
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
