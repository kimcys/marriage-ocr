from __future__ import annotations

from marriage_ocr import triage
from marriage_ocr.models import OcrResult
from marriage_ocr.triage import Classification


def test_classify_headers_detects_handwritten_nikah_legacy():
    doc_type, record_type, layout_variant, notes = triage._classify_headers(
        "BIL NAMA SUAMI DAN ISTERI PENDAFTAR DAFTAR PERKAHWINAN ORANG ISLAM"
    )
    assert doc_type == "handwritten"
    assert record_type == "nikah"
    assert layout_variant == "legacy"
    assert notes == []


def test_classify_headers_detects_handwritten_cerai_modern_layout():
    doc_type, record_type, layout_variant, notes = triage._classify_headers(
        "DAFTAR PERCERAIAN ORANG ISLAM BIL DAFTAR TARIKH DAFTAR NO.SIRI CATATAN"
    )
    assert doc_type == "handwritten"
    assert record_type == "cerai"
    assert layout_variant == "modern"


def test_classify_headers_detects_handwritten_rujuk():
    doc_type, record_type, layout_variant, _ = triage._classify_headers("DAFTAR RUJUK ORANG ISLAM")
    assert doc_type == "handwritten"
    assert record_type == "rujuk"
    assert layout_variant == "legacy"


def test_classify_headers_detects_typed_nikah_modern():
    doc_type, record_type, layout_variant, _ = triage._classify_headers(
        "ENAKMEN UNDANG-UNDANG KELUARGA ISLAM (NEGERI SELANGOR) NO. 2 TAHUN 2003\n"
        "BORANG 4B\nSURAT PERAKUAN NIKAH"
    )
    assert doc_type == "typed"
    assert record_type == "nikah"
    assert layout_variant == "modern"


def test_classify_headers_detects_typed_nikah_legacy():
    doc_type, record_type, layout_variant, _ = triage._classify_headers(
        "ENAKMEN UNDANG-UNDANG KELUARGA ISLAM SELANGOR NO. 4 TAHUN 1984\n"
        "BORANG 3A/FORM 3A\nSURAT PERAKUAN NIKAH"
    )
    assert doc_type == "typed"
    assert record_type == "nikah"
    assert layout_variant == "legacy"


def test_classify_headers_detects_typed_cerai_modern():
    doc_type, record_type, layout_variant, _ = triage._classify_headers(
        "ENAKMEN UNDANG-UNDANG KELUARGA ISLAM (NEGERI SELANGOR) NO. 2 TAHUN 2003\n"
        "BORANG 10\nSURAT PERAKUAN CERAI"
    )
    assert doc_type == "typed"
    assert record_type == "cerai"
    assert layout_variant == "modern"


def test_classify_headers_detects_typed_cerai_legacy():
    doc_type, record_type, layout_variant, _ = triage._classify_headers(
        "ENAKMEN UNDANG-UNDANG KELUARGA ISLAM SELANGOR NO. 4 TAHUN 1984\n"
        "BORANG 9/FORM 9\nSURAT PERAKUAN CERAI"
    )
    assert doc_type == "typed"
    assert record_type == "cerai"
    assert layout_variant == "legacy"


def test_classify_headers_detects_typed_rujuk_modern():
    doc_type, record_type, layout_variant, _ = triage._classify_headers(
        "ENAKMEN UNDANG-UNDANG KELUARGA ISLAM (NEGERI SELANGOR) NO. 2 TAHUN 2003\n"
        "BORANG 7B\nSURAT PERAKUAN RUJUK"
    )
    assert doc_type == "typed"
    assert record_type == "rujuk"
    assert layout_variant == "modern"


def test_classify_headers_detects_typed_rujuk_legacy():
    doc_type, record_type, layout_variant, _ = triage._classify_headers(
        "ENAKMEN UNDANG-UNDANG KELUARGA ISLAM SELANGOR NO. 4 TAHUN 1984\n"
        "BORANG 5/FORM 5\nSURAT PERAKUAN RUJUK"
    )
    assert doc_type == "typed"
    assert record_type == "rujuk"
    assert layout_variant == "legacy"


def test_classify_headers_handles_book_spine_split_and_ocr_typo_variants():
    # Regression: a real two-page-spread photo's book-spine gap split the
    # printed title into two lines, and a separate real sample had Vision
    # misread I->J. Both must still classify as Cerai.
    spine_split = "1997:-\nDAFTAR PERCERAIA\nORANG ISLAMA\nJUMLAH\nBIL"
    assert triage._classify_headers(spine_split)[:2] == ("handwritten", "cerai")

    ocr_typo = "DAFTAR PERCERAJAN ORANG ISLAM\nBIL\nNAMA BEKAS SUAMI DAN ISTERI"
    assert triage._classify_headers(ocr_typo)[:2] == ("handwritten", "cerai")


def test_classify_headers_ignores_cross_reference_mentions_outside_title_region():
    # Regression: a real Cerai page's own row-level remarks read "RUJUK BUKU
    # DAFTAR" (a cross-reference note to a different book, not the page's
    # own type) far enough down the page that it must not override the
    # page's actual printed title.
    text = "\n".join(
        [
            "DAFTAR PERCERAIAN ORANG ISLAM",
            "BIL",
            "NAMA BEKAS SUAMI DAN ISTERI",
            "TEMPAT CERAI",
            "PENDAFTAR",
            "KEADAAN TALAK",
            "JUMLAH",
            "TARIKH CERAI",
            "TANDATANGAN",
            "TARIKH KELUAR",
            "HAL-HAL LAIN",
            "05/2009",
            "004960",
            "2.1.09",
            "RUJUK BUKU DAFTAR",
        ]
    )
    doc_type, record_type, _, _ = triage._classify_headers(text)
    assert doc_type == "handwritten"
    assert record_type == "cerai"


def test_classify_headers_ignores_typed_cerai_cross_references_to_nikah_and_rujuk():
    # Regression: a real typed Cerai certificate (Borang 10, input/cerai -
    # typed/01740326145837082010.pdf) cross-references the original marriage
    # and a prior reconciliation via printed lines "Bilangan Daftar Surat
    # Perakuan Nikah" / "...Surat Perakuan Rujuk", both containing the exact
    # substrings "SURAT PERAKUAN NIKAH"/"SURAT PERAKUAN RUJUK". Unlike the
    # handwritten check, the typed check is not header-region-limited (a
    # typed certificate's own title can appear several lines down), so this
    # previously misclassified the page as Nikah before ever reaching the
    # page's own "SURAT PERAKUAN CERAI" title.
    text = "\n".join(
        [
            "No. Siri: 014074",
            "(KETUA PENDAFTAR)",
            "1 . Bilangan Daftar Cerai",
            "(NEGERI SELANGOR NO. 2 TAHUN 2003)",
            "(SUBSEKSYEN 55(g))",
            "BORANG 10",
            "SURAT PERAKUAN CERAI",
            "536/2010",
            "2 . Bilangan Daftar Surat Perakuan Nikah",
            "3 . Bilangan Daftar Surat Perakuan Rujuk",
        ]
    )
    doc_type, record_type, layout_variant, _ = triage._classify_headers(text)
    assert doc_type == "typed"
    assert record_type == "cerai"
    assert layout_variant == "modern"


def test_classify_headers_detects_typed_rujuk_apostrophe_spelling():
    # Regression: a real typed Rujuk certificate (Borang 8B, input/rujuk -
    # typed/03600124085069082010.pdf) prints "SURAT PERAKUAN RUJU'"
    # (apostrophe, no trailing K) rather than "...RUJUK" -- previously fell
    # through to doc_type="unknown" since only the K-ending spelling was
    # matched.
    text = "\n".join(
        [
            "(NEGERI SELANGOR NO. 2 TAHUN 2003)",
            "(SUBSEKSYEN 52(9))",
            "BORANG 8B",
            "SURAT PERAKUAN RUJU'",
        ]
    )
    doc_type, record_type, layout_variant, _ = triage._classify_headers(text)
    assert doc_type == "typed"
    assert record_type == "rujuk"
    assert layout_variant == "modern"


def test_classify_headers_unknown_when_no_keyword_matches():
    doc_type, record_type, layout_variant, notes = triage._classify_headers("SOME UNRELATED PAGE TEXT")
    assert doc_type == "unknown"
    assert record_type is None
    assert layout_variant is None
    assert notes == ["no known header keyword matched"]


def test_count_jawi_chars_counts_only_arabic_range_letters():
    jawi_count, alpha_count = triage._count_jawi_chars("ABC بن 123 .,")
    assert alpha_count == 5  # A, B, C, ب, ن
    assert jawi_count == 2  # ب, ن


def test_assess_jawi_flags_page_that_is_almost_entirely_jawi():
    # ms/en-hinted pass returns near-nothing (as it typically does on a real
    # Jawi page); the ar-hinted pass recovers substantial, purely-Jawi text.
    primary = OcrResult(text=".", average_confidence=0.1)
    jawi = OcrResult(text="بنتورمحمدعبدالله", average_confidence=0.9)

    is_jawi, proportion, notes = triage._assess_jawi(primary, jawi, threshold=0.85)

    assert is_jawi is True
    assert proportion == 1.0
    assert any("ar-hinted pass" in note for note in notes)


def test_assess_jawi_does_not_flag_incidental_jawi_label():
    # A mostly-Rumi page with just one small Jawi label/translation must
    # stay far below the threshold -- per the user's explicit clarification
    # that incidental Jawi is fine, only a wholly-Jawi page should be skipped.
    mostly_rumi_text = "NAMA SUAMI DAN ISTERI TARIKH NIKAH MAS KAHWIN " * 5 + "نكاح"
    primary = OcrResult(text=mostly_rumi_text, average_confidence=0.9)
    jawi = OcrResult(text="نكاح", average_confidence=0.3)

    is_jawi, proportion, _ = triage._assess_jawi(primary, jawi, threshold=0.85)

    assert is_jawi is False
    assert proportion < 0.85


def test_classification_is_a_plain_dataclass_with_expected_fields():
    classification = Classification(
        doc_type="handwritten",
        record_type="nikah",
        layout_variant="legacy",
        is_jawi=False,
        jawi_proportion=0.0,
    )
    assert classification.notes == []
