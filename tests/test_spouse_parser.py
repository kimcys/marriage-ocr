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


def test_spouse_parser_survives_stray_dash_placeholder_line() -> None:
    # Regression: a lone "-" line (marking "no value" in the handwriting) used
    # to desync the strict name/detail/name/detail line sequence, causing the
    # second spouse's real name/IC/age lines to never be consumed at all --
    # silently dropping the whole wife record.
    text = "\n".join(
        [
            "MOHD. REZAM BIN ABDUL RAZA",
            "R/F 119395",
            "-",
            "25 TAHUN.",
            "SITI FATIMAH BINTI NOR SAHAM",
            "A.2938963.",
            "20 TAHUN.",
            "RM. 80.00.",
        ]
    )

    parsed, _issues = parse_spouse_cell(text)

    assert parsed["umur_suami"] == 25
    assert parsed["nama_isteri"] == "SITI FATIMAH BINTI NOR SAHAM"
    assert parsed["ic_lama_isteri"] == "A2938963"
    assert parsed["umur_isteri"] == 20


def test_spouse_parser_recovers_ic_prefix_letter_glued_onto_name() -> None:
    # Regression: real OCR output glued the IC-prefix letter onto the end of
    # the preceding name word with no space ("MANSORA. 2360015" for husband
    # name "MANSOR" + IC "A.2360015"). The name/detail split used to land
    # after that letter, silently dropping it from the IC value.
    text = "\n".join(
        [
            "MOHD. HANIF BIN MANSOR",
            "@MANSORA. 2360015->2 TAHUNE",
            "ZAUYAH BINTI MOHAMMAD.",
            "A.2536101- JI TAHUN.",
            "RM.80.00.",
        ]
    )

    parsed, _issues = parse_spouse_cell(text)

    assert parsed["ic_lama_suami"] == "A2360015"


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
