from marriage_ocr.parser import parse_identifiers


def test_ic_parser_normalizes_old_ic() -> None:
    parsed = parse_identifiers("A 1192345")
    assert parsed.ic_lama == "A1192345"
    assert parsed.ic_baru is None


def test_ic_parser_normalizes_new_ic() -> None:
    parsed = parse_identifiers("900101101234")
    assert parsed.ic_lama is None
    assert parsed.ic_baru == "900101101234"


def test_ic_parser_keeps_raw_matches() -> None:
    parsed = parse_identifiers("A1192345 / 900101-10-1234")
    assert parsed.raw == "A1192345 / 900101-10-1234"


def test_ic_parser_preserves_prefix_letters_and_repairs_digit_body() -> None:
    parsed = parse_identifiers("B 1192345 / 900101-1O-1234")

    assert parsed.ic_lama == "B1192345"
    assert parsed.ic_baru == "900101101234"


def test_ic_parser_rejects_invalid_birth_date_for_new_ic() -> None:
    parsed = parse_identifiers("991332-10-1234")

    assert parsed.ic_lama is None
    assert parsed.ic_baru is None


def test_ic_parser_normalizes_new_ic_separators_and_leap_day() -> None:
    parsed = parse_identifiers("000229 10 1234")

    assert parsed.ic_lama is None
    assert parsed.ic_baru == "000229101234"


def test_ic_parser_supports_slash_prefixed_legacy_ids() -> None:
    parsed = parse_identifiers("R/F 119395")

    assert parsed.ic_lama == "R/F119395"
    assert parsed.ic_baru is None
