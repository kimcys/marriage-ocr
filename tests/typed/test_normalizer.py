import pytest

from marriage_ocr.typed.models import RawField, Region
from marriage_ocr.typed.normalizer import (
    build_extracted_record,
    normalize_age,
    normalize_bil,
    normalize_date_preserving_style,
    normalize_ic,
    normalize_mas_kahwin,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("04/2009 (SALINAN 1: 468/1984 S. BERNAM)", "04/2009"),
        ("330/2009 DAERAH SABAK BERNAM", "330/2009"),
        ("Bilangan Daftar: 32/2009", "32/2009"),
    ],
)
def test_normalize_bil_keeps_primary_number(raw: str, expected: str) -> None:
    assert normalize_bil(raw) == expected


def test_normalize_ic_classifies_old_and_new_numbers() -> None:
    assert normalize_ic("571018-10-5919") == (None, "571018105919")
    assert normalize_ic("6057990") == ("6057990", None)
    assert normalize_ic("12-34") == (None, None)


def test_normalize_ic_keeps_old_ic_letter_prefix() -> None:
    # Regression: old-format ICs are always letter-prefixed (e.g. "A1192345"),
    # never a bare digit run. Before this, normalize_ic only matched
    # \b\d{7,8}\b for the old-IC fallback, which never matches when a letter is
    # directly attached with no separator (the normal old-IC format) -- so a
    # wali's or witness's old IC on a typed form was dropped entirely, not just
    # missing its prefix.
    assert normalize_ic("A1192345") == ("A1192345", None)
    assert normalize_ic("No. Kad Pengenalan: A. 1192345") == ("A1192345", None)
    assert normalize_ic("R/F 119395") == ("R/F119395", None)


def test_normalize_date_normalizes_to_day_month_year() -> None:
    assert normalize_date_preserving_style("21.09.1984") == "21-09-1984"
    assert normalize_date_preserving_style(" 28 / 04 / 2009 ") == "28-04-2009"
    assert normalize_date_preserving_style("31.02.2009") is None
    assert normalize_date_preserving_style("25 ZULHIJJAH 1404\n21.09.1984") == "21-09-1984"


def test_normalize_mas_kahwin_preserves_rm_full_text() -> None:
    assert normalize_mas_kahwin("Mas Kahwin RM80.00") == "RM 80.00"
    assert normalize_mas_kahwin("RM1 500.00") == "RM 1,500.00"


def test_normalize_age_enforces_16_to_120() -> None:
    assert normalize_age("52 Tahun", min_age=16, max_age=120) == 52
    assert normalize_age("9 Tahun", min_age=16, max_age=120) is None
    assert normalize_age("Umur: 52 Tahun\nBangsa: MELAYU", min_age=16, max_age=120) == 52


def _raw(key: str, output_name: str, text: str) -> RawField:
    return RawField(key, output_name, 1, Region(0, 0, 1, 1), text, 0.95)


def test_build_extracted_record_maps_typed_fields() -> None:
    record = build_extracted_record(
        {
            "bil": _raw("bil", "Bil", "04/2009 (SALINAN 1: 468/1984)"),
            "nama_suami": _raw("nama_suami", "Nama Suami", "Nama: HENDON BIN MARIMIN"),
            "id_suami": _raw("id_suami", "IC Suami", "571018-10-5919"),
            "umur_suami": _raw("umur_suami", "Umur Suami", "52 Tahun"),
        }
    )

    assert record.bil == "04/2009"
    assert record.nama_suami == "HENDON BIN MARIMIN"
    assert record.ic_baru_suami == "571018105919"
    assert record.ic_lama_suami is None
    assert record.umur_suami == 52


def test_normalize_name_strips_following_noise_lines() -> None:
    assert (
        build_extracted_record(
            {
                "nama_isteri": _raw(
                    "nama_isteri",
                    "Nama Isteri",
                    "ABIDAH BINTI HALIDI @ HAJI HALIDI No Sin 016173\n: 6057990 Umur : 49 Tahun",
                )
            }
        ).nama_isteri
        == "ABIDAH BINTI HALIDI @ HAJI HALIDI"
    )
