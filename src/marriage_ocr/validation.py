from __future__ import annotations

from dataclasses import replace
from statistics import mean
from typing import Any, Mapping

from marriage_ocr.models import ExtractedRecord, OcrResult
from marriage_ocr.refinement.text_corrections import (
    is_suspicious_name,
    is_valid_date,
    is_valid_malaysian_ic,
)


def validate_record(
    record: ExtractedRecord,
    cell_results: Mapping[str, OcrResult],
    validation_config: Mapping[str, Any],
    *,
    layout_confidence: float = 1.0,
    layout_ok: bool = True,
) -> ExtractedRecord:
    validated = replace(record)
    reasons: list[str] = []
    critical = False
    confidence = 1.0

    if not layout_ok:
        validated.confidence = 0.0
        validated.status_review = "FAILED_LAYOUT"
        validated.review_reason = ["layout detection failed"]
        return validated

    nonempty_results = [result for result in cell_results.values() if result.text.strip()]
    if not nonempty_results:
        validated.confidence = 0.0
        validated.status_review = "FAILED_OCR"
        validated.review_reason = ["OCR returned empty text"]
        return validated

    if not validated.nama_suami:
        confidence -= 0.20
        reasons.append("missing husband name")
        critical = True
    elif _is_name_suspicious(validated.nama_suami):
        confidence -= 0.15
        reasons.append("suspicious symbols in husband name")

    if not validated.nama_isteri:
        confidence -= 0.20
        reasons.append("missing wife name")
        critical = True
    elif _is_name_suspicious(validated.nama_isteri):
        confidence -= 0.15
        reasons.append("suspicious symbols in wife name")

    husband_ic_valid = _has_valid_ic(validated.ic_lama_suami, validated.ic_baru_suami)
    wife_ic_valid = _has_valid_ic(validated.ic_lama_isteri, validated.ic_baru_isteri)
    if not husband_ic_valid:
        confidence -= 0.15
        reasons.append("missing or invalid husband IC")
    if not wife_ic_valid:
        confidence -= 0.15
        reasons.append("missing or invalid wife IC")
    if not husband_ic_valid and not wife_ic_valid:
        critical = True
        reasons.append("missing both IC values")

    if not _age_valid(validated.umur_suami, validation_config):
        confidence -= 0.10
        reasons.append("invalid husband age")
        critical = True

    if not _age_valid(validated.umur_isteri, validation_config):
        confidence -= 0.10
        reasons.append("invalid wife age")
        critical = True

    if bool(validation_config.get("require_mas_kahwin", True)) and not validated.mas_kahwin:
        confidence -= 0.10
        reasons.append("missing mas kahwin")

    if validated.mas_kahwin and validated.mas_kahwin_raw and "RM" not in validated.mas_kahwin_raw.upper():
        reasons.append("mas kahwin missing RM prefix")

    if bool(validation_config.get("require_tarikh_nikah", True)) and not is_valid_date(validated.tarikh_nikah):
        confidence -= 0.10
        reasons.append("invalid nikah date")
        critical = True

    if validated.tarikh_keluar_raw and not is_valid_date(validated.tarikh_keluar):
        confidence -= 0.10
        reasons.append("invalid keluar date")

    if not validated.nama_pendaftar:
        reasons.append("missing pendaftar name")
    if not validated.alamat_pendaftar:
        reasons.append("missing pendaftar address")
    if not validated.nama_wali:
        reasons.append("missing wali name")
    if not validated.hubungan_wali:
        reasons.append("missing wali relationship")
    if not validated.saksi_1:
        reasons.append("missing saksi 1")
    if not validated.saksi_2:
        reasons.append("missing saksi 2")

    min_average_confidence = float(validation_config.get("min_average_confidence", 0.50))
    average_ocr_confidence = mean([result.average_confidence for result in nonempty_results])
    if average_ocr_confidence < min_average_confidence:
        confidence -= 0.10
        reasons.append("low OCR confidence")

    if layout_confidence < 0.75:
        confidence -= 0.20
        reasons.append("low layout confidence")

    validated.confidence = max(0.0, round(confidence, 4))
    validated.review_reason = _dedupe_preserve_order(reasons)

    ok_threshold = float(validation_config.get("ok_confidence_threshold", 0.85))
    validated.status_review = (
        "OK"
        if validated.confidence >= ok_threshold and not critical and not validated.review_reason
        else "REVIEW"
    )
    return validated


def estimate_layout_confidence(
    *,
    marker_present: bool,
    cell_count: int,
    record_height: int,
    min_record_height: int,
    max_record_height: int,
) -> float:
    confidence = 1.0
    if not marker_present:
        confidence -= 0.35
    if cell_count < 8:
        confidence -= 0.20
    if record_height < int(min_record_height * 0.75):
        confidence -= 0.20
    if record_height > int(max_record_height * 1.5):
        confidence -= 0.20
    return max(0.0, min(1.0, confidence))


def _has_valid_ic(old_ic: str | None, new_ic: str | None) -> bool:
    return is_valid_malaysian_ic(old_ic) or is_valid_malaysian_ic(new_ic)


def _age_valid(age: int | None, validation_config: Mapping[str, Any]) -> bool:
    if age is None:
        return False
    minimum = int(validation_config.get("min_age", 15))
    maximum = int(validation_config.get("max_age", 100))
    return minimum <= age <= maximum


def _is_name_suspicious(name: str) -> bool:
    return is_suspicious_name(name)


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        deduped.append(value)
        seen.add(value)
    return deduped
