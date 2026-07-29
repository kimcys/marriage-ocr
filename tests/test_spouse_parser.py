import pytest

from marriage_ocr.parser import parse_spouse_cell


def test_spouse_parser_extracts_names_ids_ages_and_money() -> None:
    text = "\n".join(
        [
            "MOHAMAD BIN YASMIN",
            "A 1192345 25 TAHUN",
            "SITI BINTI ALI",
            "900101101234 23 THN",
            "RM 8O.OO",
        ]
    )

    parsed, issues = parse_spouse_cell(text)

    assert parsed["nama_suami"] == "MOHAMAD BIN YASMIN"
    assert parsed["ic_lama_suami"] == "A1192345"
    assert parsed["umur_suami"] == 25
    assert parsed["nama_isteri"] == "SITI BINTI ALI"
    assert parsed["ic_baru_isteri"] == "900101101234"
    assert parsed["umur_isteri"] == 23
    assert parsed["mas_kahwin"] == "RM 80.00"
    assert issues == []


def test_spouse_parser_normalizes_wife_old_ic_in_compact_form() -> None:
    text = "\n".join(
        [
            "MOHAMAD BIN YASMIN",
            "A 1192345 25 TAHUN",
            "SITI BINTI ALI",
            "R/F 119395 23 THN",
            "RM 8O.OO",
        ]
    )

    parsed, issues = parse_spouse_cell(text)

    assert parsed["ic_lama_suami"] == "A1192345"
    assert parsed["ic_lama_isteri"] == "R/F119395"
    assert issues == []


def test_spouse_parser_repairs_common_ocr_confusion_for_wife_age() -> None:
    text = "\n".join(
        [
            "MOHAMAD BIN YASMIN",
            "A 1192345 25 TAHUN",
            "SITI BINTI ALI",
            "900101101234 76 THN",
            "RM 8O.OO",
        ]
    )

    parsed, issues = parse_spouse_cell(text)

    assert parsed["umur_suami"] == 25
    assert parsed["umur_isteri"] == 26
    assert issues == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("SHARIZA BINTI SHAIKH MOHD.", "SHARIZA BINTI SHAIKH MOHD"),
        ("ZAUYAH BINTI MOHAMMAD.", "ZAUYAH BINTI MOHAMMAD"),
        ("SAADIAH BINTI HAJI OSMAN.", "SAADIAH BINTI HAJI OSMAN"),
        ("HAIROON SAH BINTI IMARA DAY", "HAIROON SAH BINTI IMAM DAYOOD"),
        ("SALWA BING AHMED SAMIE", "SALWA BINTI AHMED SAMION"),
        ("NORRAINAH BINTI MOHAMMED", "NORRAINAH BINTI MOHAMMED NOOR"),
    ],
)
def test_spouse_parser_applies_targeted_wife_name_corrections(text: str, expected: str) -> None:
    parsed, issues = parse_spouse_cell(
        "\n".join(
            [
                "MOHAMAD BIN YASMIN",
                "A 1192345 25 TAHUN",
                text,
                "900101101234 23 THN",
                "RM 8O.OO",
            ]
        )
    )

    assert parsed["nama_isteri"] == expected
    assert issues == []


def test_spouse_parser_preserves_uncommon_name_without_dictionary_autocorrect() -> None:
    text = "\n".join(
        [
            "ZULQARNAIN BIN YASMIN",
            "A 1192345 25 TAHUN",
            "SITI BINTI ALI",
            "900101101234 23 THN",
            "RM 8O.OO",
        ]
    )

    parsed, issues = parse_spouse_cell(text)

    assert parsed["nama_suami"] == "ZULQARNAIN BIN YASMIN"
    assert issues == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("AHMAD B1N ALI", "AHMAD BIN ALI"),
        ("NOR AZMI BIN ABDUL KADIR @ AHMAD YUSOF", "NOR AZMI BIN ABDUL KADIR @ AHMAD YUTA"),
        ("HAJA MOHIAREN BIN MOHANGE", "HAJA MOHIAREN BIN MOHAMAD GANI"),
        ("ALS BIN HJ. TAHAR.", "ALI BIN HJ. TAHAR"),
        ("ABDUL RAYMAN BIRI JAKARTA", "ABDUL RAHMAN BIN ZAKARIA"),
    ],
)
def test_spouse_parser_applies_targeted_husband_name_corrections(text: str, expected: str) -> None:
    parsed, issues = parse_spouse_cell(text)

    assert parsed["nama_suami"] == expected
