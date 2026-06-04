import importlib
from io import BytesIO
import sys
from pathlib import Path
from zipfile import ZipFile

import streamlit as st

from marriage_ocr.exporter import record_to_export_dict
from marriage_ocr.models import ExtractedRecord

EXPORT_CONFIG = {
    "append": True,
    "dedupe": True,
    "sheet_name": "Records",
}


def test_web_app_import_and_preview_limit(monkeypatch) -> None:
    module = _import_web_app(monkeypatch)
    records = [ExtractedRecord(bil=str(index), status_review="OK") for index in range(250)]

    rows = module._build_preview_rows(records)

    assert hasattr(module, "main")
    assert len(rows) == 200
    assert rows[0]["Bil"] == "0"
    assert rows[-1]["Bil"] == "199"


def test_web_app_prefers_corrected_rows_for_export(monkeypatch, tmp_path: Path) -> None:
    module = _import_web_app(monkeypatch)
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    original_row = record_to_export_dict(_make_record(source_record="record_001"), timestamp="2026-05-25T12:00:00")
    corrected_row = dict(original_row)
    corrected_row["Nama Suami"] = "CORRECTED NAME"

    module._write_export_rows_csv([original_row], output_dir / module.ORIGINAL_CSV_NAME)
    module._write_corrected_rows(output_dir, [corrected_row])

    rows, label, csv_path = module._load_preferred_export_rows(output_dir)

    assert label == "Corrected batch records"
    assert csv_path.name == module.CORRECTED_CSV_NAME
    assert rows[0]["Nama Suami"] == "CORRECTED NAME"


def test_web_app_reset_corrections_preserves_original_rows(monkeypatch, tmp_path: Path) -> None:
    module = _import_web_app(monkeypatch)
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    original_row = record_to_export_dict(_make_record(source_record="record_001"), timestamp="2026-05-25T12:00:00")
    corrected_row = dict(original_row)
    corrected_row["Nama Isteri"] = "CORRECTED SPOUSE"

    original_csv_path = output_dir / module.ORIGINAL_CSV_NAME
    module._write_export_rows_csv([original_row], original_csv_path)
    module._write_corrected_rows(output_dir, [corrected_row])
    (output_dir / f"{module.RECORDS_PART_PREFIX}_001.csv").write_text("placeholder", encoding="utf-8")
    (output_dir / f"{module.RECORDS_PART_PREFIX}_001.xlsx").write_text("placeholder", encoding="utf-8")

    module._reset_corrections(output_dir)

    assert original_csv_path.exists()
    assert not (output_dir / module.CORRECTED_CSV_NAME).exists()
    assert not (output_dir / module.CORRECTED_JSON_NAME).exists()
    assert not list(output_dir.glob(f"{module.RECORDS_PART_PREFIX}_*.csv"))
    assert not list(output_dir.glob(f"{module.RECORDS_PART_PREFIX}_*.xlsx"))


def test_web_app_prepare_original_export_parts_uses_generic_numbered_files(monkeypatch, tmp_path: Path) -> None:
    module = _import_web_app(monkeypatch)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    records = [_make_record(source_record=f"record_{index:03d}", bil=str(index)) for index in range(1, 6)]

    csv_paths, xlsx_paths = module._prepare_original_export_parts(
        records,
        output_dir,
        EXPORT_CONFIG,
        rows_per_file=2,
    )

    assert [path.name for path in csv_paths] == [
        "records_part_001.csv",
        "records_part_002.csv",
        "records_part_003.csv",
    ]
    assert [path.name for path in xlsx_paths] == [
        "records_part_001.xlsx",
        "records_part_002.xlsx",
        "records_part_003.xlsx",
    ]
    assert not (output_dir / "records.xlsx").exists()


def test_web_app_prepare_preferred_export_parts_uses_corrected_rows(monkeypatch, tmp_path: Path) -> None:
    module = _import_web_app(monkeypatch)
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    original_row = record_to_export_dict(_make_record(source_record="record_001"), timestamp="2026-05-25T12:00:00")
    corrected_row = dict(original_row)
    corrected_row["Nama Suami"] = "CORRECTED NAME"

    csv_paths, xlsx_paths = module._prepare_preferred_export_parts(
        [corrected_row],
        output_dir,
        EXPORT_CONFIG,
        rows_per_file=5000,
    )

    assert [path.name for path in csv_paths] == ["records_part_001.csv"]
    assert [path.name for path in xlsx_paths] == ["records_part_001.xlsx"]

    export_rows = module._load_export_rows(csv_paths[0])
    assert export_rows[0]["Nama Suami"] == "CORRECTED NAME"


def test_web_app_build_download_payload_uses_single_file_name(monkeypatch, tmp_path: Path) -> None:
    module = _import_web_app(monkeypatch)
    csv_path = tmp_path / "records_part_001.csv"
    csv_path.write_text("Bil\n1\n", encoding="utf-8")

    file_name, data, mime = module._build_download_payload(
        [csv_path],
        fallback_path=None,
        single_file_name="records.csv",
        archive_name="records_csv.zip",
        mime_type="text/csv",
    )

    assert file_name == "records.csv"
    assert data == csv_path.read_bytes()
    assert mime == "text/csv"


def test_web_app_build_download_payload_zips_multiple_parts(monkeypatch, tmp_path: Path) -> None:
    module = _import_web_app(monkeypatch)
    part_one = tmp_path / "records_part_001.xlsx"
    part_two = tmp_path / "records_part_002.xlsx"
    part_one.write_text("first", encoding="utf-8")
    part_two.write_text("second", encoding="utf-8")

    file_name, data, mime = module._build_download_payload(
        [part_one, part_two],
        fallback_path=None,
        single_file_name="records.xlsx",
        archive_name="records_xlsx.zip",
        mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    assert file_name == "records_xlsx.zip"
    assert mime == "application/zip"

    with ZipFile(BytesIO(data)) as archive:
        assert archive.namelist() == ["records_part_001.xlsx", "records_part_002.xlsx"]
        assert archive.read("records_part_001.xlsx") == b"first"
        assert archive.read("records_part_002.xlsx") == b"second"


def _import_web_app(monkeypatch):
    monkeypatch.setattr(st, "set_page_config", lambda **_: None)
    sys.modules.pop("marriage_ocr.web_app", None)
    return importlib.import_module("marriage_ocr.web_app")


def _make_record(*, source_record: str, bil: str = "1") -> ExtractedRecord:
    return ExtractedRecord(
        bil=bil,
        nama_suami="MOHAMAD BIN YASMIN",
        ic_lama_suami="A.1192345",
        umur_suami=25,
        nama_isteri="SITI BINTI ALI",
        ic_baru_isteri="900101-10-1234",
        umur_isteri=23,
        mas_kahwin="RM 80.00",
        mas_kahwin_raw="RM 8O.OO",
        nama_pendaftar="MOHD SALLEH",
        alamat_pendaftar="KAMPUNG BARU",
        nama_wali="ABDUL RAHMAN",
        hubungan_wali="BAPA",
        saksi_1="AHMAD BIN ALI",
        saksi_2="OSMAN BIN DIN",
        tarikh_nikah="27-08-1994",
        tarikh_nikah_raw="27.8.94",
        tarikh_keluar="02-06-1995",
        tarikh_keluar_raw="2.6.95",
        remarks="TIADA",
        confidence=0.93,
        status_review="OK",
        review_reason=[],
        source_file="sample.jpg",
        source_page=1,
        source_record=source_record,
        crop_folder=f"debug/sample/{source_record}",
        raw_bil=bil,
        raw_suami_isteri="RAW SUAMI ISTERI",
        raw_pendaftar="RAW PENDAFTAR",
        raw_wali="RAW WALI",
        raw_hubungan_wali="RAW HUBUNGAN",
        raw_saksi="RAW SAKSI",
        raw_tarikh_nikah="RAW NIKAH",
        raw_tarikh_keluar="RAW KELUAR",
        raw_remarks="RAW REMARKS",
        raw_ocr_json='{"mock":true}',
    )
