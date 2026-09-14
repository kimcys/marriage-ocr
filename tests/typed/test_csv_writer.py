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


def test_typed_csv_drops_nikah_only_duplicate_columns() -> None:
    # Per explicit client request -- these were Nikah-only columns with no
    # other use, cluttering the exported/displayed record with no distinct
    # information (the underlying ic_wali/ic_saksi_1/ic_saksi_2/
    # pemberian_lain ExtractedRecord fields are untouched).
    removed_columns = {"IC Wali", "IC Saksi 1", "IC Saksi 2", "Pemberian Lain"}
    assert removed_columns.isdisjoint(TYPED_CSV_COLUMNS)


def test_typed_csv_blanks_shared_cerai_rujuk_fields_on_nikah_rows(tmp_path: Path) -> None:
    # Tarikh Lahir Suami/Isteri are real, populated columns for typed Cerai/
    # Rujuk (a birthdate printed instead of an age) -- shared in this one
    # column list, so removing the column itself would break Cerai/Rujuk.
    # Per explicit client request, Nikah rows specifically must never show a
    # value in these cells even when the underlying field is populated
    # (Nikah Legacy genuinely sets tarikh_lahir_suami/isteri internally, to
    # decide whether Umur's own TIADA MAKLUMAT placeholder applies).
    output = tmp_path / "typed_records.csv"
    store = TypedCsvStore.load(output, reset_output=True, skip_existing=False)
    store.upsert(
        TypedDocumentResult(
            record=ExtractedRecord(
                bil="01/2009",
                nama_suami="A, BIN B",
                tarikh_lahir_suami="1980",
                tarikh_lahir_isteri="1985",
            ),
            source_file="nikah_legacy.pdf",
            processing_status=ProcessingStatus.SUCCESS,
        )
    )
    store.upsert(
        TypedDocumentResult(
            record=ExtractedRecord(
                record_type="CERAI",
                bil="02/2009",
                nama_suami="C, BIN D",
                tarikh_lahir_suami="1980",
                tarikh_lahir_isteri="1985",
            ),
            source_file="cerai.pdf",
            processing_status=ProcessingStatus.SUCCESS,
        )
    )
    store.flush()

    with output.open(newline="", encoding="utf-8-sig") as handle:
        rows = {row["Source File"]: row for row in csv.DictReader(handle)}

    assert rows["nikah_legacy.pdf"]["Tarikh Lahir Suami"] == ""
    assert rows["nikah_legacy.pdf"]["Tarikh Lahir Isteri"] == ""
    assert rows["cerai.pdf"]["Tarikh Lahir Suami"] == "1980"
    assert rows["cerai.pdf"]["Tarikh Lahir Isteri"] == "1985"


def test_typed_csv_drops_rujuk_only_or_globally_unused_columns() -> None:
    # Per explicit client request -- Tempat Rujuk was Rujuk-only (with no
    # other type using it) and No Kad Perakuan Islam Suami/Isteri were
    # globally unpopulated across every typed record type, so both were
    # removed from the schema entirely rather than blanked per-type.
    removed_columns = {"Tempat Rujuk", "No Kad Perakuan Islam Suami", "No Kad Perakuan Islam Isteri"}
    assert removed_columns.isdisjoint(TYPED_CSV_COLUMNS)


def test_typed_csv_blanks_shared_cerai_fields_on_rujuk_rows(tmp_path: Path) -> None:
    # Pekerjaan Suami-Isteri / Tempat Nikah Daerah-Negeri are real, populated
    # columns for typed Cerai -- shared in this one column list, so removing
    # them outright would break Cerai. Per explicit client request, Rujuk
    # rows specifically must never show a value in these cells even when the
    # underlying field is populated.
    output = tmp_path / "typed_records.csv"
    store = TypedCsvStore.load(output, reset_output=True, skip_existing=False)
    store.upsert(
        TypedDocumentResult(
            record=ExtractedRecord(
                record_type="RUJUK",
                bil="01/2009",
                nama_suami="A, BIN B",
                pekerjaan_suami="PENIAGA",
                pekerjaan_isteri="SURI RUMAH",
                tempat_nikah_daerah="PETALING",
                tempat_nikah_negeri="SELANGOR",
            ),
            source_file="rujuk.pdf",
            processing_status=ProcessingStatus.SUCCESS,
        )
    )
    store.upsert(
        TypedDocumentResult(
            record=ExtractedRecord(
                record_type="CERAI",
                bil="02/2009",
                nama_suami="C, BIN D",
                pekerjaan_suami="PENIAGA",
                pekerjaan_isteri="SURI RUMAH",
                tempat_nikah_daerah="PETALING",
                tempat_nikah_negeri="SELANGOR",
            ),
            source_file="cerai.pdf",
            processing_status=ProcessingStatus.SUCCESS,
        )
    )
    store.flush()

    with output.open(newline="", encoding="utf-8-sig") as handle:
        rows = {row["Source File"]: row for row in csv.DictReader(handle)}

    assert rows["rujuk.pdf"]["Pekerjaan Suami"] == ""
    assert rows["rujuk.pdf"]["Pekerjaan Isteri"] == ""
    assert rows["rujuk.pdf"]["Tempat Nikah Daerah"] == ""
    assert rows["rujuk.pdf"]["Tempat Nikah Negeri"] == ""
    assert rows["cerai.pdf"]["Pekerjaan Suami"] == "PENIAGA"
    assert rows["cerai.pdf"]["Pekerjaan Isteri"] == "SURI RUMAH"
    assert rows["cerai.pdf"]["Tempat Nikah Daerah"] == "PETALING"
    assert rows["cerai.pdf"]["Tempat Nikah Negeri"] == "SELANGOR"


def test_typed_csv_drops_globally_unused_or_cerai_only_columns() -> None:
    # Per explicit client request -- Tempat Bercerai was Cerai-only (Tempat
    # Cerai is the correct column instead) and the other three were globally
    # unpopulated across every typed record type, so all were removed from
    # the schema entirely rather than blanked per-type.
    removed_columns = {
        "Bil Daftar Rujuk Asal",
        "No Sijil Perakuan Nikah Rujuk",
        "No Permohonan Cerai",
        "Tempat Bercerai",
        "Jumlah Bayaran",
    }
    assert removed_columns.isdisjoint(TYPED_CSV_COLUMNS)


def test_typed_csv_blanks_fields_only_real_for_nikah_on_other_record_types(tmp_path: Path) -> None:
    # No Siri / Saksi 1 / Saksi 2 are real, populated columns for typed
    # Nikah alone -- shared in this one column list, so removing them
    # outright would break Nikah. Per explicit client request, Cerai rows
    # specifically must never show a value in these cells even when the
    # underlying field is populated (already blanked for Rujuk previously).
    output = tmp_path / "typed_records.csv"
    store = TypedCsvStore.load(output, reset_output=True, skip_existing=False)
    store.upsert(
        TypedDocumentResult(
            record=ExtractedRecord(
                bil="01/2009",
                nama_suami="A, BIN B",
                no_siri="123456",
                saksi_1="E, BIN F",
                saksi_2="G, BIN H",
            ),
            source_file="nikah.pdf",
            processing_status=ProcessingStatus.SUCCESS,
        )
    )
    store.upsert(
        TypedDocumentResult(
            record=ExtractedRecord(
                record_type="CERAI",
                bil="02/2009",
                nama_suami="C, BIN D",
                no_siri="654321",
                saksi_1="I, BIN J",
                saksi_2="K, BIN L",
            ),
            source_file="cerai.pdf",
            processing_status=ProcessingStatus.SUCCESS,
        )
    )
    store.flush()

    with output.open(newline="", encoding="utf-8-sig") as handle:
        rows = {row["Source File"]: row for row in csv.DictReader(handle)}

    assert rows["nikah.pdf"]["No Siri"] == "123456"
    assert rows["nikah.pdf"]["Saksi 1"] == "E, BIN F"
    assert rows["nikah.pdf"]["Saksi 2"] == "G, BIN H"
    assert rows["cerai.pdf"]["No Siri"] == ""
    assert rows["cerai.pdf"]["Saksi 1"] == ""
    assert rows["cerai.pdf"]["Saksi 2"] == ""


def test_typed_csv_blanks_tarikh_keluar_on_cerai_rows_only(tmp_path: Path) -> None:
    # Tarikh Keluar is real, populated data for typed Nikah and Rujuk alike
    # -- per explicit client request, pure clutter for Cerai specifically.
    output = tmp_path / "typed_records.csv"
    store = TypedCsvStore.load(output, reset_output=True, skip_existing=False)
    store.upsert(
        TypedDocumentResult(
            record=ExtractedRecord(
                record_type="RUJUK",
                bil="01/2009",
                nama_suami="A, BIN B",
                tarikh_keluar="01-01-2020",
            ),
            source_file="rujuk.pdf",
            processing_status=ProcessingStatus.SUCCESS,
        )
    )
    store.upsert(
        TypedDocumentResult(
            record=ExtractedRecord(
                record_type="CERAI",
                bil="02/2009",
                nama_suami="C, BIN D",
                tarikh_keluar="02-02-2020",
            ),
            source_file="cerai.pdf",
            processing_status=ProcessingStatus.SUCCESS,
        )
    )
    store.flush()

    with output.open(newline="", encoding="utf-8-sig") as handle:
        rows = {row["Source File"]: row for row in csv.DictReader(handle)}

    assert rows["rujuk.pdf"]["Tarikh Keluar"] == "01-01-2020"
    assert rows["cerai.pdf"]["Tarikh Keluar"] == ""


def test_blank_umur_shows_tiada_maklumat_on_nikah_modern_rows(tmp_path: Path) -> None:
    # Real client samples print "T. MAKLUMAT" (Tiada Maklumat / "no
    # information") directly on the form for a wali whose age genuinely
    # wasn't recorded -- a blank CSV cell there reads as "not extracted"
    # rather than "the form itself says there's no information", so this
    # placeholder is shown instead per explicit client request.
    output = tmp_path / "typed_records.csv"
    store = TypedCsvStore.load(output, reset_output=True, skip_existing=False)
    store.upsert(
        TypedDocumentResult(
            record=ExtractedRecord(bil="01/2009", nama_suami="A, BIN B"),
            source_file="modern.pdf",
            processing_status=ProcessingStatus.SUCCESS,
        )
    )
    store.flush()

    with output.open(newline="", encoding="utf-8-sig") as handle:
        row = next(csv.DictReader(handle))

    assert row["Umur Suami"] == "TIADA MAKLUMAT"
    assert row["Umur Isteri"] == "TIADA MAKLUMAT"
    assert row["Umur Wali"] == "TIADA MAKLUMAT"


def test_blank_umur_stays_empty_on_nikah_legacy_rows(tmp_path: Path) -> None:
    # Nikah Legacy prints a birthdate instead of an age and has no Umur
    # region at all -- these fields stay blank there because the
    # concept doesn't apply to that layout, not because the form said so,
    # so the Modern-only placeholder above must not appear. tarikh_lahir_
    # suami/isteri only ever get a value on the legacy path. A companion
    # Modern row keeps these columns from being pruned as entirely unused.
    output = tmp_path / "typed_records.csv"
    store = TypedCsvStore.load(output, reset_output=True, skip_existing=False)
    store.upsert(
        TypedDocumentResult(
            record=ExtractedRecord(
                bil="01/2009",
                nama_suami="A, BIN B",
                tarikh_lahir_suami="1980",
                tarikh_lahir_isteri="1985",
            ),
            source_file="legacy.pdf",
            processing_status=ProcessingStatus.SUCCESS,
        )
    )
    store.upsert(
        TypedDocumentResult(
            record=ExtractedRecord(bil="02/2009", nama_suami="C, BIN D", umur_suami=30),
            source_file="modern.pdf",
            processing_status=ProcessingStatus.SUCCESS,
        )
    )
    store.flush()

    with output.open(newline="", encoding="utf-8-sig") as handle:
        rows = {row["Source File"]: row for row in csv.DictReader(handle)}

    assert rows["legacy.pdf"]["Umur Suami"] == ""
    assert rows["legacy.pdf"]["Umur Isteri"] == ""
    assert rows["legacy.pdf"]["Umur Wali"] == ""


def test_blank_umur_stays_empty_on_cerai_rows(tmp_path: Path) -> None:
    # Cerai has no Wali field at all -- these columns stay blank there
    # because the concept doesn't apply, not because the form said so. A
    # companion Nikah row keeps these columns from being pruned as unused.
    output = tmp_path / "typed_records.csv"
    store = TypedCsvStore.load(output, reset_output=True, skip_existing=False)
    store.upsert(_cerai_result("cerai.pdf", "740326145837", ProcessingStatus.SUCCESS))
    store.upsert(
        TypedDocumentResult(
            record=ExtractedRecord(bil="02/2009", nama_suami="C, BIN D", umur_suami=30),
            source_file="modern.pdf",
            processing_status=ProcessingStatus.SUCCESS,
        )
    )
    store.flush()

    with output.open(newline="", encoding="utf-8-sig") as handle:
        rows = {row["Source File"]: row for row in csv.DictReader(handle)}

    assert rows["cerai.pdf"]["Umur Suami"] == ""
    assert rows["cerai.pdf"]["Umur Isteri"] == ""
    assert rows["cerai.pdf"]["Umur Wali"] == ""

