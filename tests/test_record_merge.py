from __future__ import annotations

from marriage_ocr.models import ExtractedRecord, OcrResult
from llm.gemini_extractor import GeminiRecordResult
from llm.record_merge import merge_parser_and_gemini


def _cell_results() -> dict[str, OcrResult]:
    return {"suami_isteri": OcrResult(text="some text", average_confidence=0.9)}


def test_gemini_filled_critical_field_forces_review_despite_high_confidence() -> None:
    # Regression: when the deterministic parser found nothing for a critical
    # field (e.g. the nikah date cell was unreadable), Gemini's value is the
    # *only* source -- there is nothing to cross-check it against. LLMs are
    # prone to reporting high self-confidence for a fabricated-but-plausible
    # value, so this must force review regardless of Gemini's own confidence
    # score, rather than trusting it.
    parser_record = ExtractedRecord(
        nama_suami="ALI BIN ABU",
        nama_isteri="SITI BINTI ALI",
        tarikh_nikah=None,  # parser could not read the date at all
    )
    gemini_record = ExtractedRecord(
        nama_suami="ALI BIN ABU",
        nama_isteri="SITI BINTI ALI",
        tarikh_nikah="2005-01-01",  # a confident-looking but unverifiable guess
    )
    gemini_result = GeminiRecordResult(
        record=gemini_record,
        field_confidence={"tarikh_nikah": 0.95},
    )

    merged = merge_parser_and_gemini(
        parser_record=parser_record,
        gemini_result=gemini_result,
        cell_results=_cell_results(),
        validation_config={"require_tarikh_nikah": False},
    )

    assert merged.tarikh_nikah == "2005-01-01"
    assert merged.status_review == "REVIEW"
    assert any("no parser corroboration" in reason for reason in merged.review_reason)


def test_gemini_filled_non_critical_field_does_not_force_review() -> None:
    parser_record = ExtractedRecord(
        nama_suami="ALI BIN ABU",
        nama_isteri="SITI BINTI ALI",
        remarks=None,
    )
    gemini_record = ExtractedRecord(
        nama_suami="ALI BIN ABU",
        nama_isteri="SITI BINTI ALI",
        remarks="Diambil oleh suami",
    )
    gemini_result = GeminiRecordResult(
        record=gemini_record,
        field_confidence={"remarks": 0.95},
    )

    merged = merge_parser_and_gemini(
        parser_record=parser_record,
        gemini_result=gemini_result,
        cell_results=_cell_results(),
        validation_config={},
    )

    assert merged.remarks == "Diambil oleh suami"
    assert not any("no parser corroboration" in reason for reason in merged.review_reason)
