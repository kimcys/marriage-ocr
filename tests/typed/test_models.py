from pathlib import Path

from marriage_ocr.models import ExtractedRecord
from marriage_ocr.typed.models import ProcessingStatus, Region, TypedDocumentResult


def test_region_expand_is_clipped_to_page_bounds() -> None:
    assert Region(0.02, 0.03, 0.98, 0.99).expand(0.05) == Region(0.0, 0.0, 1.0, 1.0)


def test_document_result_builds_csv_diagnostics() -> None:
    result = TypedDocumentResult(
        record=ExtractedRecord(bil="04/2009"),
        source_file="016057990052009.pdf",
        processing_status=ProcessingStatus.REVIEW_REQUIRED,
        failed_fields=("Saksi 2", "Tarikh Nikah"),
        retry_count=2,
        error_message="Required fields remain invalid",
    )

    assert result.review_required is True
    assert result.failed_fields_text == "Saksi 2;Tarikh Nikah"

