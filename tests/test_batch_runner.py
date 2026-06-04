from marriage_ocr.batch_runner import normalize_record
from marriage_ocr.models import ExtractedRecord


def test_normalize_record_handles_extracted_record_dataclass():
    record = ExtractedRecord(
        bil="12",
        nama_suami="MOHAMAD BIN YASMIN",
        status_review="OK",
        review_reason=[],
    )

    normalized = normalize_record(record)

    assert normalized["bil"] == "12"
    assert normalized["nama_suami"] == "MOHAMAD BIN YASMIN"
    assert normalized["status_review"] == "OK"
