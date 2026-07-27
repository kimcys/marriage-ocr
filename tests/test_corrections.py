from marriage_ocr.corrections import clean_name, correct_relationship, normalize_numeric_ocr, normalize_ocr_text


def test_normalize_ocr_text_repairs_common_malay_terms() -> None:
    assert normalize_ocr_text("b1nti\nmohd.") == "BINTI\nMOHD"


def test_normalize_numeric_ocr_repairs_date_digits() -> None:
    assert normalize_numeric_ocr("27.8.9O") == "27.8.90"


def test_correct_relationship_handles_fuzzy_matches() -> None:
    assert correct_relationship("WALI HAK1M") == "WALI HAKIM"
    assert correct_relationship("BAPA KANOUNG") == "BAPA KANDUNG"


def test_clean_name_removes_noise() -> None:
    assert clean_name("mohd. salleh 123") == "MOHD SALLEH"
