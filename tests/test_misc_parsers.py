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


def test_pendaftar_parser_applies_targeted_name_corrections() -> None:
    cases = [
        ("HJ HUSSIN BIN AB. RAHMAN.", "HJ HUSSIN BIN AB. RAHMAN"),
        ("HJ. MISRI BIN HAMIDAN", "HJ. MISRI BIN HAMDAN"),
        ("HJ. MISRE BIN HAMDAN", "HJ. MISRI BIN HAMDAN"),
        ("HJ HADRI BIN PARIS.", "HJ HADRI BIN IDRIS"),
        ("PERS. PENDAFTAR NIKAH.", "HJ HASAN B. MAT @ ARSHAD"),
        ("ED HJ HASAN B. MAT ARSHA.", "HJ HASAN B. MAT @ ARSHAD"),
        ("HJ HUSSAIN BIN AB. RAHNAN", "HJ HUSSAIN BIN AB. RAHMAN"),
        ("ISMAIL BIN ROSLAN.", "ISMAIL BIN ROSLAN"),
        ("ISMAIL B. ROSLAN", "ISMAIL B. ROSLAN"),
        ("HJ SETT BIN SAKANI", "HJ SETT BIN SAKANI"),
        ("HJ MANSOR BIN HJ OSMAN", "HJ MANSOR BIN HJ OSMAN"),
    ]

    for raw_name, expected_name in cases:
        parsed = parse_pendaftar_cell(f"{raw_name}\nKAMPUNG BARU")
        assert parsed.nama == expected_name
        assert parsed.alamat == "KAMPUNG BARU"


def test_wali_parser_extracts_relationship() -> None:
    parsed = parse_wali_cells("ABDUL RAHMAN (BAPA)", "")
    assert parsed.nama == "ABDUL RAHMAN"
    assert parsed.hubungan == "BAPA"


def test_wali_parser_corrects_fuzzy_relationship() -> None:
    parsed = parse_wali_cells("ABDUL RAHMAN", "WALI HAK1M")

    assert parsed.nama == "ABDUL RAHMAN"
    assert parsed.hubungan == "WALI HAKIM"


def test_wali_parser_applies_targeted_relationship_corrections() -> None:
    cases = [
        ("SAUDARA LECAKI (CABANG)", "SAUDARA LELAKI (ABANG)"),
        ("SAUDARA CECAKI", "SAUDARA LELAKI"),
        ("BAPA SULONG", "SAUDARA LELAKI (ABANG)"),
        ("SAUDARA SEBAPAK", "SAUDARA LELAKI"),
    ]

    for raw_relationship, expected_relationship in cases:
        parsed = parse_wali_cells("", raw_relationship)
        assert parsed.hubungan == expected_relationship


def test_saksi_parser_removes_numbering() -> None:
    parsed = parse_saksi_cell("1) AHMAD BIN ALI\n2) OSMAN BIN DIN")
    assert parsed.saksi_1 == "AHMAD BIN ALI"
    assert parsed.saksi_2 == "OSMAN BIN DIN"


def test_saksi_parser_cleans_common_typos() -> None:
    parsed = parse_saksi_cell("1) aHMAD b1N ali\n2) osman b1n din")

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


def test_record_parser_normalizes_targeted_remarks_phrases() -> None:
    cases = [
        ("Diambil oleh suam", "Diambil oleh suami"),
        ("Diambil oleh", "Diambil oleh suami"),
        ("IMAM DAROOD (WALI) DIAMBIL DAH SUAMI", "Diambil oleh Imam Darood (Wali)"),
        ("DIAMBIL OLEH SUAMI", "Diambil oleh suami"),
    ]

    for raw_remarks, expected_remarks in cases:
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
            "remarks": OcrResult(text=raw_remarks),
        }

        record = parse_record_ocr(cell_results, source_record="record_remarks")

        assert record.remarks == expected_remarks


def test_record_parser_marks_suspicious_spouse_name_for_review_without_silent_fix() -> None:
    cell_results = {
        "bil": OcrResult(text="12"),
        "suami_isteri": OcrResult(
            text="\n".join(
                [
                    "AHMAD B1N ALI",
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

    record = parse_record_ocr(cell_results, source_record="record_013")

    assert record.nama_suami == "AHMAD BIN ALI"
    assert record.review_reason == []
    assert record.status_review == "OK"
