from marriage_ocr.parser import parse_identifiers


def test_ic_parser_normalizes_old_ic() -> None:
    parsed = parse_identifiers("A 1192345")
    assert parsed.ic_lama == "A.1192345"
    assert parsed.ic_baru is None


def test_ic_parser_normalizes_new_ic() -> None:
    parsed = parse_identifiers("900101101234")
    assert parsed.ic_lama is None
    assert parsed.ic_baru == "900101-10-1234"


def test_ic_parser_keeps_raw_matches() -> None:
    parsed = parse_identifiers("A1192345 / 900101-10-1234")
    assert parsed.raw == "A1192345 / 900101-10-1234"


def test_ic_parser_preserves_prefix_letters_and_repairs_digit_body() -> None:
    parsed = parse_identifiers("B 1192345 / 900101-1O-1234")

    assert parsed.ic_lama == "B.1192345"
    assert parsed.ic_baru == "900101-10-1234"
