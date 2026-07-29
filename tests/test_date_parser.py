from marriage_ocr.parser import parse_date


def test_date_parser_normalizes_supported_formats() -> None:
    assert parse_date("27.8.94").normalized == "27-08-1994"
    assert parse_date("2.6.95").normalized == "02-06-1995"
    assert parse_date("27-08-1994").normalized == "27-08-1994"
    assert parse_date("1994-08-26").normalized == "26-08-1994"
    assert parse_date("27/08/1994").normalized == "27-08-1994"


def test_date_parser_recovers_collapsed_ocr_date() -> None:
    parsed = parse_date("27.894")
    assert parsed.normalized == "27-08-1994"
    assert parsed.needs_review is False


def test_date_parser_repairs_common_ocr_digit_confusions() -> None:
    parsed = parse_date("3O.O1.94")

    assert parsed.normalized == "30-01-1994"
    assert parsed.needs_review is False


def test_date_parser_repairs_targeted_date_ocr_errors() -> None:
    assert parse_date("27-08-1944").normalized == "27-08-1994"
    assert parse_date("07-08-1994").normalized == "27-08-1994"


def test_date_parser_rejects_impossible_dates_and_keeps_review_flag() -> None:
    parsed = parse_date("31/02/94")

    assert parsed.normalized is None
    assert parsed.raw == "31/02/94"
    assert parsed.needs_review is True
