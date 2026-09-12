import csv
from pathlib import Path

from marriage_ocr.models import ExtractedRecord
from marriage_ocr.typed.csv_writer import TYPED_CSV_COLUMNS, TypedCsvStore
from marriage_ocr.typed.models import ProcessingStatus, TypedDocumentResult


def _result(source: str, bil: str, status: ProcessingStatus) -> TypedDocumentResult:
    return TypedDocumentResult(
        record=ExtractedRecord(bil=bil, nama_suami="A, BIN B"),
        source_file=source,
        processing_status=status,
    )


def test_csv_columns_match_approved_order() -> None:
    assert TYPED_CSV_COLUMNS[:5] == [
        "Bil",
        "Nama Suami",
        "IC Lama Suami",
        "IC Baru Suami",
        "Umur Suami",
    ]
    assert TYPED_CSV_COLUMNS[-7:] == [
        "Source File",
        "Processing Status",
        "Review Required",
        "Failed Fields",
        "Retry Count",
        "Error Message",
        "Duplicate Of Source File",
    ]


def test_store_replaces_existing_source_and_quotes_commas(tmp_path: Path) -> None:
    output = tmp_path / "typed_records.csv"
    store = TypedCsvStore.load(output, reset_output=True, skip_existing=False)
    store.upsert(_result("record.pdf", "01/2009", ProcessingStatus.REVIEW_REQUIRED))
    store.flush()
    store.upsert(_result("record.pdf", "04/2009", ProcessingStatus.SUCCESS))
    store.flush()

    with output.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["Bil"] == "04/2009"
    assert rows[0]["Nama Suami"] == "A, BIN B"


def _cerai_result(source: str, ic_suami: str, status: ProcessingStatus) -> TypedDocumentResult:
    return TypedDocumentResult(
        record=ExtractedRecord(
            record_type="CERAI",
            bil="536/2010",
            nama_suami="ANUAR BIN SANIKRAM",
            ic_suami=ic_suami,
        ),
        source_file=source,
        processing_status=status,
    )


def test_upsert_flags_same_content_from_a_different_source_file(tmp_path: Path) -> None:
    output = tmp_path / "typed_records.csv"
    store = TypedCsvStore.load(output, reset_output=True, skip_existing=False)

    store.upsert(_cerai_result("scan_a.pdf", "740326145837", ProcessingStatus.SUCCESS))
    store.upsert(_cerai_result("scan_b.pdf", "740326145837", ProcessingStatus.SUCCESS))
    store.flush()

    with output.open(newline="", encoding="utf-8-sig") as handle:
        rows = {row["Source File"]: row for row in csv.DictReader(handle)}

    assert rows["scan_a.pdf"]["Duplicate Of Source File"] == ""
    assert rows["scan_b.pdf"]["Duplicate Of Source File"] == "scan_a.pdf"


def test_upsert_does_not_flag_retry_of_the_same_source_file(tmp_path: Path) -> None:
    output = tmp_path / "typed_records.csv"
    store = TypedCsvStore.load(output, reset_output=True, skip_existing=False)

    store.upsert(_cerai_result("scan_a.pdf", "740326145837", ProcessingStatus.REVIEW_REQUIRED))
    store.upsert(_cerai_result("scan_a.pdf", "740326145837", ProcessingStatus.SUCCESS))
    store.flush()

    with output.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["Duplicate Of Source File"] == ""


def test_content_index_rebuilds_from_a_reloaded_csv(tmp_path: Path) -> None:
    output = tmp_path / "typed_records.csv"
    first_run = TypedCsvStore.load(output, reset_output=True, skip_existing=False)
    first_run.upsert(_cerai_result("scan_a.pdf", "740326145837", ProcessingStatus.SUCCESS))
    first_run.flush()

    second_run = TypedCsvStore.load(output, reset_output=False, skip_existing=False)
    second_run.upsert(_cerai_result("scan_b.pdf", "740326145837", ProcessingStatus.SUCCESS))
    second_run.flush()

    with output.open(newline="", encoding="utf-8-sig") as handle:
        rows = {row["Source File"]: row for row in csv.DictReader(handle)}

    assert rows["scan_b.pdf"]["Duplicate Of Source File"] == "scan_a.pdf"


def _nikah_result(source: str, ic_baru_suami: str, status: ProcessingStatus) -> TypedDocumentResult:
    return TypedDocumentResult(
        record=ExtractedRecord(bil="01/2009", nama_suami="A, BIN B", ic_baru_suami=ic_baru_suami),
        source_file=source,
        processing_status=status,
    )


def test_all_nikah_batch_drops_cerai_only_columns_that_stay_blank(tmp_path: Path) -> None:
    output = tmp_path / "typed_records.csv"
    store = TypedCsvStore.load(output, reset_output=True, skip_existing=False)
    store.upsert(_nikah_result("record.pdf", "561217085317", ProcessingStatus.SUCCESS))
    store.flush()

    with output.open(newline="", encoding="utf-8-sig") as handle:
        header = next(csv.reader(handle))

    assert "IC Suami" not in header
    assert "Bangsa Suami" not in header
    # Nikah's own fields, and run-metadata columns, are never dropped.
    assert "IC Baru Suami" in header
    assert "Record Type" in header
    assert "Source File" in header


def test_a_mixed_batch_keeps_ic_suami_once_any_row_actually_uses_it(tmp_path: Path) -> None:
    output = tmp_path / "typed_records.csv"
    store = TypedCsvStore.load(output, reset_output=True, skip_existing=False)
    store.upsert(_result("nikah.pdf", "01/2009", ProcessingStatus.SUCCESS))
    store.upsert(_cerai_result("cerai.pdf", "740326145837", ProcessingStatus.SUCCESS))
    store.flush()

    with output.open(newline="", encoding="utf-8-sig") as handle:
        rows = {row["Source File"]: row for row in csv.DictReader(handle)}

    assert rows["cerai.pdf"]["IC Suami"] == "740326145837"
    assert rows["nikah.pdf"]["IC Suami"] == ""


def test_skip_existing_only_skips_success_rows(tmp_path: Path) -> None:
    output = tmp_path / "typed_records.csv"
    store = TypedCsvStore.load(output, reset_output=True, skip_existing=False)
    store.upsert(_result("success.pdf", "01/2009", ProcessingStatus.SUCCESS))
    store.upsert(_result("review.pdf", "02/2009", ProcessingStatus.REVIEW_REQUIRED))
    store.flush()

    reloaded = TypedCsvStore.load(output, reset_output=False, skip_existing=True)
    assert reloaded.should_skip("success.pdf") is True
    assert reloaded.should_skip("review.pdf") is False

