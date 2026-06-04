from marriage_ocr.models import OcrResult
from marriage_ocr.parser import (
    parse_pendaftar_cell,
    parse_record_ocr,
    parse_saksi_cell,
    parse_wali_cells,
)


def test_pendaftar_parser_splits_name_and_address() -> None:
    parsed = parse_pendaftar_cell("MOHD SALLEH\nKAMPUNG BARU\nMELAKA")
    assert parsed.nama == "MOHD SALLEH"
    assert parsed.alamat == "KAMPUNG BARU\nMELAKA"


def test_wali_parser_extracts_relationship() -> None:
    parsed = parse_wali_cells("ABDUL RAHMAN (BAPA)", "")
    assert parsed.nama == "ABDUL RAHMAN"
    assert parsed.hubungan == "BAPA"


def test_saksi_parser_removes_numbering() -> None:
    parsed = parse_saksi_cell("1) AHMAD BIN ALI\n2) OSMAN BIN DIN")
    assert parsed.saksi_1 == "AHMAD BIN ALI"
    assert parsed.saksi_2 == "OSMAN BIN DIN"


def test_record_parser_builds_structured_record() -> None:
    cell_results = {
        "bil": OcrResult(text="12"),
        "suami_isteri": OcrResult(
            text="\n".join(
                [
                    "MOHAMAD BIN YASMIN",
                    "A 1192345 25 TAHUN",
                    "SITI BINTI ALI",
                    "900101101234 23 THN",
                    "RM 8O.OO",
                ]
            )
        ),
        "pendaftar": OcrResult(text="MOHD SALLEH\nKAMPUNG BARU"),
        "wali": OcrResult(text="ABDUL RAHMAN"),
        "hubungan_wali": OcrResult(text="BAPA"),
        "saksi": OcrResult(text="1) AHMAD BIN ALI\n2) OSMAN BIN DIN"),
        "tarikh_nikah": OcrResult(text="27.8.94"),
        "tarikh_keluar": OcrResult(text="2.6.95"),
        "remarks": OcrResult(text="TIADA"),
    }

    record = parse_record_ocr(cell_results, source_record="record_012")

    assert record.bil == "12"
    assert record.nama_suami == "MOHAMAD BIN YASMIN"
    assert record.nama_isteri == "SITI BINTI ALI"
    assert record.mas_kahwin == "RM 80.00"
    assert record.nama_pendaftar == "MOHD SALLEH"
    assert record.hubungan_wali == "BAPA"
    assert record.saksi_2 == "OSMAN BIN DIN"
    assert record.tarikh_nikah == "27-08-1994"
    assert record.tarikh_keluar == "02-06-1995"
    assert record.status_review == "OK"
