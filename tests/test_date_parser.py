from marriage_ocr.parser import parse_date


def test_date_parser_normalizes_supported_formats() -> None:
    assert parse_date("27.8.94").normalized == "27-08-1994"
    assert parse_date("2.6.95").normalized == "02-06-1995"
    assert parse_date("27-08-1994").normalized == "27-08-1994"


def test_date_parser_recovers_collapsed_ocr_date() -> None:
    parsed = parse_date("27.894")
    assert parsed.normalized == "27-08-1994"
    assert parsed.needs_review is False
