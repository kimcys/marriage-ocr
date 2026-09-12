from marriage_ocr.models import ExtractedRecord
from marriage_ocr.typed.models import ProcessingStatus, RawField, Region
from marriage_ocr.typed.validator import status_for_result, validate_record


def _raw_field(key: str, raw_text: str, *, confidence: float = 0.95) -> RawField:
    return RawField(
        key=key,
        output_name=key,
        page_number=2,
        region=Region(0.0, 0.0, 1.0, 1.0),
        raw_text=raw_text,
        confidence=confidence,
    )


def _valid_record() -> ExtractedRecord:
    return ExtractedRecord(
        bil="04/2009",
        nama_suami="HENDON BIN MARIMIN",
        ic_baru_suami="571018105919",
        umur_suami=52,
        nama_isteri="ABIDAH BINTI HALIDI @ HAJI HALIDI",
        ic_lama_isteri="6057990",
        umur_isteri=49,
        mas_kahwin="RM 80.00",
        nama_pendaftar="USTAZ SHUKRI BIN SHARIF",
        alamat_pendaftar="KAMPUNG PARIT 9 SUNGAI LEMAN, 45400 SEKINCHAN SELANGOR",
        nama_wali="HAJI HALIDI BIN HAJI OSMAN",
        hubungan_wali="BAPA KANDUNG",
        saksi_1="HAMZAH BIN ABAS",
        saksi_2="RAMLI BIN ISMAIL",
        tarikh_nikah="21.09.1984",
    )


def test_strict_record_succeeds_without_tarikh_keluar() -> None:
    summary = validate_record(_valid_record(), {}, word_confidence_threshold=0.75, max_retry_fields=6)
    assert summary.failed_fields == ()
    assert status_for_result(summary, retry_count=0) is ProcessingStatus.SUCCESS


def test_missing_required_field_is_review_required_after_retry() -> None:
    record = _valid_record()
    record.saksi_2 = None
    summary = validate_record(record, {}, word_confidence_threshold=0.75, max_retry_fields=6)
    assert "Saksi 2" in summary.failed_fields
    assert status_for_result(summary, retry_count=1) is ProcessingStatus.REVIEW_REQUIRED


def test_retry_candidates_are_capped_at_six() -> None:
    summary = validate_record(ExtractedRecord(), {}, word_confidence_threshold=0.75, max_retry_fields=6)
    assert len(summary.retry_fields) == 6


def test_document_error_is_failed() -> None:
    summary = validate_record(ExtractedRecord(), {}, word_confidence_threshold=0.75, max_retry_fields=6)
    assert status_for_result(summary, retry_count=0, document_error="Vision unavailable") is ProcessingStatus.FAILED


def test_letter_prefixed_old_ic_is_accepted() -> None:
    # Regression: the validator's old-IC check used to require a bare
    # \d{7,8} fullmatch, rejecting the normal old-IC format (a letter prefix,
    # e.g. "A1192345" or "R/F119395") even though normalize_ic can now
    # actually produce that value.
    record = _valid_record()
    record.ic_lama_isteri = "A1192345"
    summary = validate_record(record, {}, word_confidence_threshold=0.75, max_retry_fields=6)
    assert "IC Isteri" not in summary.failed_fields

    record.ic_lama_isteri = "R/F119395"
    summary = validate_record(record, {}, word_confidence_threshold=0.75, max_retry_fields=6)
    assert "IC Isteri" not in summary.failed_fields


def test_tarikh_nikah_bleed_into_alamat_pendaftar_is_flagged_as_contamination() -> None:
    # Regression: tarikh_nikah/alamat_pendaftar/nama_pendaftar sit with
    # near-zero margin in BORANG_4B_REGIONS, so per-page alignment drift
    # sometimes captures tarikh_nikah's own row text ("Hijrah ... Hari ...
    # Masa ...") into alamat_pendaftar instead of the real Tempat value --
    # confirmed against real batch data. High Vision confidence alone can't
    # catch this (the bled-in text is genuinely, clearly printed), so the
    # contamination-label check is what has to.
    record = _valid_record()
    record.alamat_pendaftar = "Nikah Hijrah 03 J ' AWAL 1430 Hari SABTU Masa 5.00 PTG"
    raw_fields = {"alamat_pendaftar": _raw_field("alamat_pendaftar", record.alamat_pendaftar)}
    summary = validate_record(record, raw_fields, word_confidence_threshold=0.75, max_retry_fields=6)
    assert "Alamat Pendaftar" in summary.failed_fields


def test_contamination_check_does_not_false_positive_on_names_containing_hari() -> None:
    # "HARI" is a real substring of common Malay names (Zahari, Bahari) --
    # the contamination check must word-boundary match, not substring match,
    # or it would wrongly reject genuinely correct extractions.
    record = _valid_record()
    record.nama_pendaftar = "USTAZ ZAHARI BIN BAHARI"
    raw_fields = {"nama_pendaftar": _raw_field("nama_pendaftar", record.nama_pendaftar)}
    summary = validate_record(record, raw_fields, word_confidence_threshold=0.75, max_retry_fields=6)
    assert "Nama Pendaftar" not in summary.failed_fields


def test_tempat_is_not_treated_as_contamination_in_alamat_pendaftar() -> None:
    # alamat_pendaftar's region stands in for the form's "Tempat" line (there
    # is no separate region for it) -- "TEMPAT" appearing in a correct
    # extraction is expected, not a sign the box grabbed the wrong row.
    record = _valid_record()
    record.alamat_pendaftar = "Tempat: PEJABAT AGAMA ISLAM DAERAH SABAK BERNAM"
    raw_fields = {"alamat_pendaftar": _raw_field("alamat_pendaftar", record.alamat_pendaftar)}
    summary = validate_record(record, raw_fields, word_confidence_threshold=0.75, max_retry_fields=6)
    assert "Alamat Pendaftar" not in summary.failed_fields


def test_multiline_age_raw_text_does_not_trigger_false_failure() -> None:
    record = _valid_record()
    raw_fields = {
        "umur_suami": type("Raw", (), {"raw_text": "Umur : 52 Tahun\nBangsa : MELAYU", "confidence": 0.96})(),
        "umur_isteri": type("Raw", (), {"raw_text": "Umur : 49 Tahun\nBangsa : MELAYU", "confidence": 0.96})(),
    }
    summary = validate_record(record, raw_fields, word_confidence_threshold=0.75, max_retry_fields=6)
    assert "Umur Suami" not in summary.failed_fields
    assert "Umur Isteri" not in summary.failed_fields
