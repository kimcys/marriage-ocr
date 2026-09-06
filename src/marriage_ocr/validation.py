from __future__ import annotations

from dataclasses import replace
from statistics import mean
from typing import Any, Mapping

from marriage_ocr.models import ExtractedRecord, OcrResult
from marriage_ocr.refinement.text_corrections import (
    is_suspicious_ic,
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
    record_type: str = "nikah",
) -> ExtractedRecord:
    validated = replace(record, record_type=str(record_type or "nikah").strip().upper())

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

    scorer = _SCORERS.get(str(record_type or "nikah").strip().lower(), _score_nikah)
    reasons, critical, confidence = scorer(validated, validation_config)

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


def validate_gemini_only_record(
    record: ExtractedRecord,
    *,
    field_confidence: Mapping[str, float],
    uncertain_fields: Any = (),
    validation_config: Mapping[str, Any],
    record_type: str = "nikah",
) -> ExtractedRecord:
    """validate_record's counterpart for a pipeline with no Vision OCR at
    all (see gemini_page_pipeline.py) -- there's no cell_results to check
    for "OCR returned nothing" or to average into a confidence score.

    Reuses the SAME per-field sanity scorers validate_record uses
    (_score_nikah/_score_cerai/_score_rujuk): IC format, age range, date
    validity, required fields present. Those checks are about the
    extracted VALUES, not about how they were extracted, so they're just as
    meaningful here as when Vision was in the loop -- unlike Gemini's own
    field_confidence, which real testing found too flat across records
    (0.966-0.979 regardless of actual accuracy) to rank which ones need a
    human. uncertain_fields is added as a direct, blunter penalty precisely
    because that flat confidence can't be trusted to carry the signal on
    its own.
    """
    validated = replace(record, record_type=str(record_type or "nikah").strip().upper())

    scorer = _SCORERS.get(str(record_type or "nikah").strip().lower(), _score_nikah)
    reasons, critical, confidence = scorer(validated, validation_config)

    if field_confidence:
        min_average_confidence = float(validation_config.get("min_average_confidence", 0.50))
        average_field_confidence = mean(field_confidence.values())
        if average_field_confidence < min_average_confidence:
            confidence -= 0.10
            reasons.append("low Gemini field confidence")
    else:
        confidence -= 0.10
        reasons.append("no Gemini field confidence reported")

    if uncertain_fields:
        confidence -= 0.05 * len(list(uncertain_fields))
        reasons.append("Gemini uncertain fields: " + ", ".join(str(field) for field in uncertain_fields))

    validated.confidence = max(0.0, round(confidence, 4))
    validated.review_reason = _dedupe_preserve_order(reasons)

    ok_threshold = float(validation_config.get("ok_confidence_threshold", 0.85))
    validated.status_review = (
        "OK"
        if validated.confidence >= ok_threshold and not critical and not validated.review_reason
        else "REVIEW"
    )
    return validated


def parser_confidence_clears_threshold(record: ExtractedRecord, *, min_confidence: float) -> bool:
    """True when a parser-only validated record is trustworthy enough to
    skip a Gemini call entirely -- status_review must already be "OK" (no
    critical failures, no review reasons) AND confidence must clear a bar
    stricter than validate_record's own ok_confidence_threshold, since this
    decision forgoes Gemini's second opinion altogether rather than just
    flagging for human review. Shared by pipeline.py's synchronous path and
    llm/gemini_batch_extractor.py's offline batch path so the two apply the
    exact same bar.
    """
    return record.status_review == "OK" and (record.confidence or 0.0) >= min_confidence


def _score_nikah(validated: ExtractedRecord, validation_config: Mapping[str, Any]) -> tuple[list[str], bool, float]:
    reasons: list[str] = []
    critical = False
    confidence = 1.0

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

    if is_suspicious_ic(validated.ic_lama_suami) or is_suspicious_ic(validated.ic_baru_suami):
        confidence -= 0.15
        reasons.append("suspicious husband IC (implausible digits)")
    if is_suspicious_ic(validated.ic_lama_isteri) or is_suspicious_ic(validated.ic_baru_isteri):
        confidence -= 0.15
        reasons.append("suspicious wife IC (implausible digits)")

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

    return reasons, critical, confidence


def _score_spouse_pair_ic(
    validated: ExtractedRecord,
    reasons: list[str],
    confidence: float,
) -> tuple[float, bool]:
    """Shared IC scoring for Cerai/Rujuk, which use one ic_suami/ic_isteri
    field per person instead of Nikah's split ic_lama_*/ic_baru_*."""
    critical = False
    husband_ic_valid = is_valid_malaysian_ic(validated.ic_suami)
    wife_ic_valid = is_valid_malaysian_ic(validated.ic_isteri)
    if not husband_ic_valid:
        confidence -= 0.15
        reasons.append("missing or invalid husband IC")
    if not wife_ic_valid:
        confidence -= 0.15
        reasons.append("missing or invalid wife IC")
    if not husband_ic_valid and not wife_ic_valid:
        critical = True
        reasons.append("missing both IC values")

    if is_suspicious_ic(validated.ic_suami):
        confidence -= 0.15
        reasons.append("suspicious husband IC (implausible digits)")
    if is_suspicious_ic(validated.ic_isteri):
        confidence -= 0.15
        reasons.append("suspicious wife IC (implausible digits)")

    return confidence, critical


def _score_names(validated: ExtractedRecord, reasons: list[str], confidence: float) -> tuple[float, bool]:
    critical = False
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

    return confidence, critical


def _score_cerai(validated: ExtractedRecord, validation_config: Mapping[str, Any]) -> tuple[list[str], bool, float]:
    reasons: list[str] = []
    confidence = 1.0
    critical = False

    confidence, name_critical = _score_names(validated, reasons, confidence)
    critical = critical or name_critical

    confidence, ic_critical = _score_spouse_pair_ic(validated, reasons, confidence)
    critical = critical or ic_critical

    if bool(validation_config.get("require_tarikh_cerai", True)) and not is_valid_date(validated.tarikh_cerai):
        confidence -= 0.10
        reasons.append("invalid or missing cerai date")
        critical = True

    if validated.tarikh_keluar_raw and not is_valid_date(validated.tarikh_keluar):
        confidence -= 0.10
        reasons.append("invalid keluar date")

    return reasons, critical, confidence


def _score_rujuk(validated: ExtractedRecord, validation_config: Mapping[str, Any]) -> tuple[list[str], bool, float]:
    reasons: list[str] = []
    confidence = 1.0
    critical = False

    confidence, name_critical = _score_names(validated, reasons, confidence)
    critical = critical or name_critical

    confidence, ic_critical = _score_spouse_pair_ic(validated, reasons, confidence)
    critical = critical or ic_critical

    # Ages are only present on the legacy layout -- soft check only, not
    # critical, since the modern layout legitimately omits them.
    if validated.umur_suami is not None and not _age_valid(validated.umur_suami, validation_config):
        confidence -= 0.10
        reasons.append("invalid husband age")
    if validated.umur_isteri is not None and not _age_valid(validated.umur_isteri, validation_config):
        confidence -= 0.10
        reasons.append("invalid wife age")

    if bool(validation_config.get("require_tarikh_rujuk", True)) and not is_valid_date(validated.tarikh_rujuk):
        confidence -= 0.10
        reasons.append("invalid or missing rujuk date")
        critical = True

    if validated.tarikh_keluar_raw and not is_valid_date(validated.tarikh_keluar):
        confidence -= 0.10
        reasons.append("invalid keluar date")

    return reasons, critical, confidence


_SCORERS = {
    "nikah": _score_nikah,
    "cerai": _score_cerai,
    "rujuk": _score_rujuk,
}


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
