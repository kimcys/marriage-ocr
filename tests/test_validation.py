from marriage_ocr.models import OcrResult
from marriage_ocr.parser import parse_record_ocr
from marriage_ocr.validation import (
    is_suspicious_ic,
    is_suspicious_name,
    is_valid_date,
    is_valid_malaysian_ic,
    validate_record,
)


VALIDATION_CONFIG = {
    "ok_confidence_threshold": 0.85,
    "min_age": 15,
    "max_age": 100,
    "require_mas_kahwin": True,
    "require_tarikh_nikah": True,
    "min_average_confidence": 0.50,
}


def test_validation_marks_good_record_ok() -> None:
    cell_results = {
        "bil": OcrResult(text="12", average_confidence=0.98),
        "suami_isteri": OcrResult(
            text="\n".join(
                [
                    "MOHAMAD BIN YASMIN",
                    "A 1192345 25 TAHUN",
                    "SITI BINTI ALI",
                    "900101101234 23 THN",
                    "RM 8O.OO",
                ]
            ),
            average_confidence=0.96,
        ),
        "pendaftar": OcrResult(text="MOHD SALLEH\nKAMPUNG BARU", average_confidence=0.95),
        "wali": OcrResult(text="ABDUL RAHMAN", average_confidence=0.94),
        "hubungan_wali": OcrResult(text="BAPA", average_confidence=0.94),
        "saksi": OcrResult(text="1) AHMAD BIN ALI\n2) OSMAN BIN DIN", average_confidence=0.95),
        "tarikh_nikah": OcrResult(text="27.8.94", average_confidence=0.95),
        "tarikh_keluar": OcrResult(text="2.6.95", average_confidence=0.95),
        "remarks": OcrResult(text="TIADA", average_confidence=0.90),
    }

    parsed = parse_record_ocr(cell_results, source_record="record_012")
    validated = validate_record(parsed, cell_results, VALIDATION_CONFIG, layout_confidence=1.0)

    assert validated.status_review == "OK"
    assert validated.review_reason == []
    assert validated.confidence >= 0.85


def test_validation_marks_bad_ocr_review_with_reasons() -> None:
    cell_results = {
        "bil": OcrResult(text="MOCK_OCR[RECORD_001:BIL]", average_confidence=0.20),
        "suami_isteri": OcrResult(text="MOCK_OCR[RECORD_001:SUAMI_ISTERI]", average_confidence=0.20),
        "pendaftar": OcrResult(text="MOCK_OCR[RECORD_001:PENDAFTAR]", average_confidence=0.20),
        "wali": OcrResult(text="MOCK_OCR[RECORD_001:WALI]", average_confidence=0.20),
        "hubungan_wali": OcrResult(text="MOCK_OCR[RECORD_001:HUBUNGAN_WALI]", average_confidence=0.20),
        "saksi": OcrResult(text="MOCK_OCR[RECORD_001:SAKSI]", average_confidence=0.20),
        "tarikh_nikah": OcrResult(text="MOCK_OCR[RECORD_001:TARIKH_NIKAH]", average_confidence=0.20),
        "tarikh_keluar": OcrResult(text="MOCK_OCR[RECORD_001:TARIKH_KELUAR]", average_confidence=0.20),
        "remarks": OcrResult(text="MOCK_OCR[RECORD_001:REMARKS]", average_confidence=0.20),
    }

    parsed = parse_record_ocr(cell_results, source_record="record_001")
    validated = validate_record(parsed, cell_results, VALIDATION_CONFIG, layout_confidence=0.6)

    assert validated.status_review == "REVIEW"
    assert validated.confidence < 0.85
    assert "missing husband name" in validated.review_reason
    assert "missing wife name" in validated.review_reason
    assert "low OCR confidence" in validated.review_reason
    assert "low layout confidence" in validated.review_reason


def test_validation_marks_empty_ocr_failed() -> None:
    cell_results = {
        "bil": OcrResult(text="", average_confidence=0.0),
        "suami_isteri": OcrResult(text="", average_confidence=0.0),
    }

    parsed = parse_record_ocr(cell_results, source_record="record_000")
    validated = validate_record(parsed, cell_results, VALIDATION_CONFIG, layout_confidence=1.0)

    assert validated.status_review == "FAILED_OCR"
    assert validated.review_reason == ["OCR returned empty text"]
    assert validated.confidence == 0.0


def test_validation_helpers_cover_ic_date_and_name_rules() -> None:
    assert is_valid_malaysian_ic("900101-10-1234") is True
    assert is_valid_malaysian_ic("991332-10-1234") is False
    assert is_valid_malaysian_ic("A1192345") is True
    assert is_valid_malaysian_ic("R/F119395") is True
    assert is_valid_date("2024-01-01") is True
    assert is_valid_date("27-08-1994") is True
    assert is_valid_date("31-02-1994") is False
    assert is_suspicious_name("SITI B1NTI ALI") is True
    assert is_suspicious_name("SITI BINTI ALI") is False
    assert is_suspicious_ic("1111111") is True
    assert is_suspicious_ic("0000000") is True
    assert is_suspicious_ic("A1192345") is False
    assert is_suspicious_ic("1234567") is False
    assert is_suspicious_ic(None) is False


def test_validation_flags_suspicious_all_identical_digit_ic() -> None:
    # Regression: real OCR read a handwritten "A.1111111" as "4.1111111", and
    # line-splitting discarded the leading "4." as a garbage token, leaving a
    # bare "1111111" -- 7 identical digits. That passes is_valid_malaysian_ic's
    # plain-digit format check with no red flag, so it needs its own check.
    cell_results = {
        "bil": OcrResult(text="1", average_confidence=0.9),
        "suami_isteri": OcrResult(text="ZABA BIN MOHD ZAIN\nA.1471242-24 TAHUN.", average_confidence=0.9),
    }
    parsed = parse_record_ocr(cell_results, source_record="record_000")
    parsed.ic_lama_isteri = "1111111"

    validated = validate_record(parsed, cell_results, VALIDATION_CONFIG, layout_confidence=1.0)

    assert "suspicious wife IC (implausible digits)" in validated.review_reason
    assert validated.status_review != "OK"
