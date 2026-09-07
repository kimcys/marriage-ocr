from marriage_ocr.models import ExtractedRecord, OcrResult
from marriage_ocr.parser import parse_record_ocr
from marriage_ocr.validation import (
    is_suspicious_ic,
    is_suspicious_name,
    is_valid_date,
    is_valid_malaysian_ic,
    parser_confidence_clears_threshold,
    validate_gemini_only_record,
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
    assert validated.missing_fields == []
    assert validated.confidence >= 0.85


def test_parser_confidence_clears_threshold_requires_ok_and_high_confidence() -> None:
    ok_high = ExtractedRecord(status_review="OK", confidence=0.95)
    ok_borderline = ExtractedRecord(status_review="OK", confidence=0.90)
    ok_low = ExtractedRecord(status_review="OK", confidence=0.86)
    review_high_confidence = ExtractedRecord(status_review="REVIEW", confidence=0.99)

    assert parser_confidence_clears_threshold(ok_high, min_confidence=0.90) is True
    assert parser_confidence_clears_threshold(ok_borderline, min_confidence=0.90) is True
    assert parser_confidence_clears_threshold(ok_low, min_confidence=0.90) is False
    assert parser_confidence_clears_threshold(review_high_confidence, min_confidence=0.90) is False


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
    # missing_fields is the structured (column-name) counterpart of
    # review_reason's free text, used by marriage-be to render an editable
    # input for exactly the fields that are actually absent.
    assert "Nama Suami" in validated.missing_fields
    assert "Nama Isteri" in validated.missing_fields
    # "low OCR confidence"/"low layout confidence" aren't tied to one field.
    assert len(validated.missing_fields) < len(validated.review_reason)


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
    # A present-but-suspicious IC isn't "missing" -- it already has a value
    # to correct via the normal edit path, not a blank input.
    assert "IC Lama Isteri" not in validated.missing_fields
    assert "IC Baru Isteri" not in validated.missing_fields


CERAI_CELL_RESULTS = {
    "bil": OcrResult(text="1/97", average_confidence=0.95),
    "suami_isteri": OcrResult(text="ABDULLAH B. ABD HAMID\nZUBAIDAH BTE YAHYA", average_confidence=0.9),
}

RUJUK_CELL_RESULTS = {
    "bil": OcrResult(text="1/90", average_confidence=0.95),
    "suami_isteri": OcrResult(text="RAZALI BIN HARUN\nSARIMAH BT. MOHD AMIN", average_confidence=0.9),
}


def test_validate_record_cerai_ok_with_valid_ic_and_date() -> None:
    record = ExtractedRecord(
        record_type="CERAI",
        bil="1/97",
        nama_suami="ABDULLAH B. ABD HAMID",
        ic_suami="A0394566",
        nama_isteri="ZUBAIDAH BTE YAHYA",
        ic_isteri="A0287541",
        tarikh_cerai="1997-01-21",
    )

    validated = validate_record(
        record,
        CERAI_CELL_RESULTS,
        VALIDATION_CONFIG,
        layout_confidence=1.0,
        record_type="cerai",
    )

    assert validated.status_review == "OK"
    assert validated.record_type == "CERAI"
    # Nikah-only fields (mas_kahwin, wali, saksi) must not be scored for Cerai.
    assert "missing mas kahwin" not in validated.review_reason
    assert "missing wali name" not in validated.review_reason


def test_validate_record_cerai_requires_tarikh_cerai() -> None:
    record = ExtractedRecord(
        record_type="CERAI",
        bil="1/97",
        nama_suami="ABDULLAH B. ABD HAMID",
        ic_suami="A0394566",
        nama_isteri="ZUBAIDAH BTE YAHYA",
        ic_isteri="A0287541",
        tarikh_cerai=None,
    )

    validated = validate_record(
        record,
        CERAI_CELL_RESULTS,
        VALIDATION_CONFIG,
        layout_confidence=1.0,
        record_type="cerai",
    )

    assert validated.status_review == "REVIEW"
    assert "invalid or missing cerai date" in validated.review_reason
    assert "Tarikh Cerai" in validated.missing_fields


def test_validate_record_rujuk_ok_without_ages_present() -> None:
    # Ages are only on the legacy layout -- the modern layout genuinely
    # omits them and must not be penalized for that.
    record = ExtractedRecord(
        record_type="RUJUK",
        bil="1/90",
        nama_suami="RAZALI BIN HARUN",
        ic_suami="6200510",
        nama_isteri="SARIMAH BT. MOHD AMIN",
        ic_isteri="6042214",
        tarikh_rujuk="1990-01-04",
        umur_suami=None,
        umur_isteri=None,
    )

    validated = validate_record(
        record,
        RUJUK_CELL_RESULTS,
        VALIDATION_CONFIG,
        layout_confidence=1.0,
        record_type="rujuk",
    )

    assert validated.status_review == "OK"
    assert validated.record_type == "RUJUK"


def test_validate_record_rujuk_flags_implausible_age_when_present() -> None:
    record = ExtractedRecord(
        record_type="RUJUK",
        bil="1/90",
        nama_suami="RAZALI BIN HARUN",
        ic_suami="6200510",
        nama_isteri="SARIMAH BT. MOHD AMIN",
        ic_isteri="6042214",
        tarikh_rujuk="1990-01-04",
        umur_suami=5,
    )

    validated = validate_record(
        record,
        RUJUK_CELL_RESULTS,
        VALIDATION_CONFIG,
        layout_confidence=1.0,
        record_type="rujuk",
    )

    assert "invalid husband age" in validated.review_reason
    # Present-but-implausible, not absent -- umur_suami=5 has a value.
    assert "Umur Suami" not in validated.missing_fields


def test_validate_record_nikah_reports_all_missing_fields() -> None:
    record = ExtractedRecord(
        record_type="NIKAH",
        bil="12",
        nama_suami="MOHAMAD BIN YASMIN",
        ic_lama_suami="A 1192345",
        umur_suami=25,
        nama_isteri="SITI BINTI ALI",
        ic_baru_isteri="900101101234",
        umur_isteri=23,
        tarikh_nikah="1994-08-27",
    )

    validated = validate_record(
        record,
        {"bil": OcrResult(text="12", average_confidence=0.9)},
        VALIDATION_CONFIG,
        layout_confidence=1.0,
        record_type="nikah",
    )

    assert set(validated.missing_fields) == {
        "Mas Kahwin",
        "Nama Pendaftar",
        "Alamat Pendaftar",
        "Nama Wali",
        "Hubungan Wali",
        "Saksi 1",
        "Saksi 2",
    }


def test_validate_gemini_only_record_marks_good_record_ok_with_solid_confidence() -> None:
    record = ExtractedRecord(
        bil="12",
        nama_suami="MOHAMAD BIN YASMIN",
        ic_lama_suami="A 1192345",
        umur_suami=25,
        nama_isteri="SITI BINTI ALI",
        ic_baru_isteri="900101101234",
        umur_isteri=23,
        mas_kahwin="RM 80.00",
        mas_kahwin_raw="RM 80.00",
        nama_pendaftar="MOHD SALLEH",
        alamat_pendaftar="KAMPUNG BARU",
        nama_wali="ABDUL RAHMAN",
        hubungan_wali="BAPA",
        saksi_1="AHMAD BIN ALI",
        saksi_2="OSMAN BIN DIN",
        tarikh_nikah="1994-08-27",
    )

    validated = validate_gemini_only_record(
        record,
        field_confidence={"nama_suami": 0.95, "nama_isteri": 0.96, "bil": 0.9},
        uncertain_fields=[],
        validation_config=VALIDATION_CONFIG,
        record_type="nikah",
    )

    assert validated.status_review == "OK"
    assert validated.review_reason == []


def test_validate_gemini_only_record_flags_missing_husband_name_as_critical() -> None:
    record = ExtractedRecord(bil="12", nama_isteri="SITI BINTI ALI", tarikh_nikah="1994-08-27")

    validated = validate_gemini_only_record(
        record,
        field_confidence={"nama_isteri": 0.95},
        uncertain_fields=[],
        validation_config=VALIDATION_CONFIG,
        record_type="nikah",
    )

    assert validated.status_review == "REVIEW"
    assert "missing husband name" in validated.review_reason


def test_validate_gemini_only_record_penalizes_missing_confidence_signal() -> None:
    record = ExtractedRecord(
        bil="12",
        nama_suami="MOHAMAD BIN YASMIN",
        ic_lama_suami="A 1192345",
        umur_suami=25,
        nama_isteri="SITI BINTI ALI",
        ic_baru_isteri="900101101234",
        umur_isteri=23,
        mas_kahwin="RM 80.00",
        nama_pendaftar="MOHD SALLEH",
        nama_wali="ABDUL RAHMAN",
        hubungan_wali="BAPA",
        saksi_1="AHMAD BIN ALI",
        saksi_2="OSMAN BIN DIN",
        tarikh_nikah="1994-08-27",
    )

    validated = validate_gemini_only_record(
        record,
        field_confidence={},
        uncertain_fields=[],
        validation_config=VALIDATION_CONFIG,
        record_type="nikah",
    )

    assert "no Gemini field confidence reported" in validated.review_reason
    assert validated.status_review == "REVIEW"


def test_validate_gemini_only_record_penalizes_uncertain_fields() -> None:
    record = ExtractedRecord(
        bil="12",
        nama_suami="MOHAMAD BIN YASMIN",
        ic_lama_suami="A 1192345",
        umur_suami=25,
        nama_isteri="SITI BINTI ALI",
        ic_baru_isteri="900101101234",
        umur_isteri=23,
        mas_kahwin="RM 80.00",
        nama_pendaftar="MOHD SALLEH",
        nama_wali="ABDUL RAHMAN",
        hubungan_wali="BAPA",
        saksi_1="AHMAD BIN ALI",
        saksi_2="OSMAN BIN DIN",
        tarikh_nikah="1994-08-27",
    )

    validated = validate_gemini_only_record(
        record,
        field_confidence={"nama_suami": 0.95, "nama_isteri": 0.6},
        uncertain_fields=["nama_isteri"],
        validation_config=VALIDATION_CONFIG,
        record_type="nikah",
    )

    assert any("uncertain fields" in reason for reason in validated.review_reason)
    assert validated.status_review == "REVIEW"
