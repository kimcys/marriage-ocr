from marriage_ocr.parser import parse_money


def test_money_parser_normalizes_known_examples() -> None:
    cases = [
        ("RM80.00", "RM 80.00", False),
        ("RM.80.00", "RM 80.00", False),
        ("RM 8O.OO", "RM 80.00", False),
        ("Rm 80 oo", "RM 80.00", False),
        ("80.00", "RM 80.00", True),
    ]

    for raw_text, expected_value, expected_review in cases:
        parsed = parse_money(raw_text)
        assert parsed.normalized == expected_value
        assert parsed.needs_review is expected_review
