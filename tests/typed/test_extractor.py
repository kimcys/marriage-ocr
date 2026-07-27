from marriage_ocr.typed.extractor import join_words_in_reading_order, words_in_region
from marriage_ocr.typed.models import PageOcrResult, PositionedWord, Region


def _word(text: str, x: float, y: float, confidence: float = 0.9) -> PositionedWord:
    return PositionedWord(text, confidence, x, y, x + 0.04, y + 0.01, 1)


def test_words_in_region_uses_word_centre_and_tolerance() -> None:
    words = (
        _word("INSIDE", 0.20, 0.30),
        _word("EDGE", 0.39, 0.30),
        _word("OUTSIDE", 0.50, 0.30),
    )
    selected = words_in_region(words, Region(0.20, 0.29, 0.42, 0.33), tolerance=0.01)
    assert [word.text for word in selected] == ["INSIDE", "EDGE"]


def test_join_words_preserves_two_line_address_order() -> None:
    words = (
        _word("SELANGOR", 0.20, 0.42),
        _word("KAMPUNG", 0.20, 0.40),
        _word("LEMAN", 0.33, 0.40),
        _word("SUNGAI", 0.27, 0.40),
    )
    assert join_words_in_reading_order(words) == "KAMPUNG SUNGAI LEMAN\nSELANGOR"


def test_extract_raw_fields_maps_page_one_bil() -> None:
    from marriage_ocr.typed.extractor import extract_raw_fields

    word = PositionedWord("04/2009", 0.97, 0.25, 0.315, 0.34, 0.328, 1)
    page = PageOcrResult("record.pdf", 1, (word,), "04/2009", {})

    fields = extract_raw_fields([page])

    assert fields["bil"].raw_text == "04/2009"
    assert fields["bil"].confidence == 0.97

