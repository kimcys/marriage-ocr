from marriage_ocr.parser import parse_ages


def test_age_parser_accepts_practical_age_examples() -> None:
    assert parse_ages("25 TAHUN") == [25]
    assert parse_ages("23 THN") == [23]
    assert parse_ages("34 TAHUN.") == [34]


def test_age_parser_rejects_invalid_examples() -> None:
    assert parse_ages("257AHUN") == []
    assert parse_ages("1832779") == []


def test_age_parser_repairs_common_ocr_confusion_for_seventies() -> None:
    assert parse_ages("76 TAHUN") == [26]
