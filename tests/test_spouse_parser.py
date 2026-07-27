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
    assert parsed["ic_lama_suami"] == "A.1192345"
    assert parsed["umur_suami"] == 25
    assert parsed["nama_isteri"] == "SITI BINTI ALI"
    assert parsed["ic_baru_isteri"] == "900101-10-1234"
    assert parsed["umur_isteri"] == 23
    assert parsed["mas_kahwin"] == "RM 80.00"
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


def test_spouse_parser_does_not_silently_fix_connector_tokens_without_retry_evidence() -> None:
    text = "\n".join(
        [
            "AHMAD B1N ALI",
            "A 1192345 25 TAHUN",
            "SITI BINTI ALI",
            "900101101234 23 THN",
            "RM 8O.OO",
        ]
    )

    parsed, issues = parse_spouse_cell(text)

    assert parsed["nama_suami"] == "AHMAD B1N ALI"
    assert "nama_suami_suspicious" in issues
