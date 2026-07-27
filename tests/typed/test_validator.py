from marriage_ocr.models import ExtractedRecord
from marriage_ocr.typed.models import ProcessingStatus
from marriage_ocr.typed.validator import status_for_result, validate_record


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


def test_multiline_age_raw_text_does_not_trigger_false_failure() -> None:
    record = _valid_record()
    raw_fields = {
        "umur_suami": type("Raw", (), {"raw_text": "Umur : 52 Tahun\nBangsa : MELAYU", "confidence": 0.96})(),
        "umur_isteri": type("Raw", (), {"raw_text": "Umur : 49 Tahun\nBangsa : MELAYU", "confidence": 0.96})(),
    }
    summary = validate_record(record, raw_fields, word_confidence_threshold=0.75, max_retry_fields=6)
    assert "Umur Suami" not in summary.failed_fields
    assert "Umur Isteri" not in summary.failed_fields
