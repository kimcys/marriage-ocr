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


def test_classify_headers_tolerates_a_dropped_letter_in_perkahwinan():
    # Regression: a real 154-photo handwritten Nikah ledger batch had Vision
    # misread "PERKAHWINAN" as each of these on different pages (varying
    # photo quality/angle) -- an exact substring check missed all of them
    # even though the header was genuinely present and legible to a human,
    # sending roughly 1 in 4 pages of the batch to manual review.
    for garbled in ("PERKAHWNAN", "PERKAHWAN", "PERKAHNAN"):
        doc_type, record_type, layout_variant, _ = triage._classify_headers(
            f"DAFTAR {garbled} ORANG ISLAM\nBIL\nNAMA SUAMI DAN ISTERI"
        )
        assert (doc_type, record_type, layout_variant) == ("handwritten", "nikah", "legacy"), garbled


def test_classify_headers_tolerates_a_stray_space_splitting_perkahwinan():
    # Regression: a real sample read "DAFTAR PERKAH NAN ORANG ISLAM" -- Vision
    # inserted a space where a letter-drop left two short fragments; the
    # fuzzy match must try adjacent word pairs joined, not just single words.
    doc_type, record_type, layout_variant, _ = triage._classify_headers(
        "DAFTAR PERKAH NAN ORANG ISLAM\nBIL\nNAMA SUAMI DAN ISTERI"
    )
    assert (doc_type, record_type, layout_variant) == ("handwritten", "nikah", "legacy")


def test_classify_headers_fuzzy_match_does_not_over_match_unrelated_text():
    # The tolerance must stay tight enough that ordinary row content (names,
    # column headers) never accidentally reads as a title keyword.
    doc_type, record_type, layout_variant, notes = triage._classify_headers(
        "BIL\nNAMA SUAMI DAN ISTERI\n836/2007 MOHD JEFRI AZHAR BIN ARD\nNO: 179914"
    )
    assert doc_type == "unknown"
    assert record_type is None
    assert layout_variant is None
    assert notes == ["no known header keyword matched"]

    # A genuinely unrelated short word must not fuzzy-match the short
    # "RUJUK" keyword just because it happens to be nearby in length.
    doc_type, _, _, _ = triage._classify_headers("SEBUAH BUKU CATATAN LAMA")
    assert doc_type == "unknown"


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


class _FakeVisionEngine:
    """Stands in for GoogleVisionOcrEngine: returns `text` for every call and
    records the image each call was sent."""

    sent: list = []

    def __init__(self, config):
        self._hints = config["language_hints"]

    def read_image_annotated(self, image_path):
        from types import SimpleNamespace

        _FakeVisionEngine.sent.append((self._hints, image_path.suffix))
        word = SimpleNamespace(
            symbols=[SimpleNamespace(text="NIKAH")],
            confidence=0.9,
            bounding_box=SimpleNamespace(vertices=[SimpleNamespace(x=x, y=y) for x, y in [(10, 10), (60, 10), (60, 30), (10, 30)]]),
        )
        page = SimpleNamespace(
            width=2480, height=3509, blocks=[SimpleNamespace(paragraphs=[SimpleNamespace(words=[word])])]
        )
        text = _FakeVisionEngine.text
        return OcrResult(text=text), SimpleNamespace(pages=[page], text=text)

    def read_image(self, image_path):
        return self.read_image_annotated(image_path)[0]


def _classify_pdf(tmp_path, monkeypatch, text):
    import pymupdf

    pdf = tmp_path / "doc.pdf"
    document = pymupdf.open()
    document.new_page(width=595, height=842)
    document.new_page(width=595, height=842)
    document.save(pdf)
    document.close()
    _FakeVisionEngine.sent = []
    _FakeVisionEngine.text = text
    monkeypatch.setattr(triage, "GoogleVisionOcrEngine", _FakeVisionEngine)
    output = tmp_path / "page1.json"
    result = triage.classify_file(pdf, allowed_extensions=[".pdf"], page_ocr_output=output)
    return result, output


def test_classify_saves_page1_ocr_for_a_typed_pdf(tmp_path, monkeypatch):
    from marriage_ocr.typed.page_ocr_cache import load_page_ocr_cache

    result, output = _classify_pdf(
        tmp_path, monkeypatch, "ENAKMEN UNDANG-UNDANG KELUARGA ISLAM NO. 2 TAHUN 2003\nBORANG 4B\nSURAT PERAKUAN NIKAH"
    )

    assert result.doc_type == "typed"
    # One Vision call -- no ar-hinted Jawi pass for a typed form -- on the
    # lossless PNG process-typed itself sends.
    assert _FakeVisionEngine.sent == [(["ms", "en"], ".png")]
    cached = load_page_ocr_cache(output, source_file="source.pdf")
    assert cached is not None
    assert cached.matches(dpi=300, width=2480, height=3509)
    assert [word.text for word in cached.result.words] == ["NIKAH"]


def test_classify_saves_nothing_for_a_handwritten_pdf(tmp_path, monkeypatch):
    result, output = _classify_pdf(tmp_path, monkeypatch, "DAFTAR PERKAHWINAN ORANG ISLAM")

    assert result.doc_type == "handwritten"
    assert not output.exists()
    # Handwritten pages keep the ar-hinted Jawi pass.
    assert [hints for hints, _ in _FakeVisionEngine.sent] == [["ms", "en"], ["ar"]]


def _write_jpg(path, height):
    import cv2
    import numpy as np

    cv2.imwrite(str(path), np.full((height, 400, 3), 255, dtype=np.uint8))


class _FakeStackEngine:
    """Stands in for GoogleVisionOcrEngine on a stacked image: one Vision text
    block per entry of band_blocks[hints], at the given pixel y-range."""

    calls: list = []
    band_blocks: dict = {}

    def __init__(self, config):
        self._hints = tuple(config["language_hints"])

    def read_image_annotated(self, image_path):
        from types import SimpleNamespace as NS

        import cv2

        _FakeStackEngine.calls.append(self._hints)
        height, width = cv2.imread(str(image_path)).shape[:2]
        blocks = []
        for (top, bottom), text in _FakeStackEngine.band_blocks.get(self._hints, []):
            words = []
            for i, w in enumerate(text.split()):
                x = 10 + i * 30
                box = NS(vertices=[NS(x=x, y=top), NS(x=x + 25, y=top), NS(x=x + 25, y=bottom), NS(x=x, y=bottom)])
                words.append(NS(symbols=[NS(text=w)], confidence=0.9, bounding_box=box))
            vertices = [NS(x=0, y=top), NS(x=10, y=top), NS(x=10, y=bottom), NS(x=0, y=bottom)]
            blocks.append(NS(bounding_box=NS(vertices=vertices), paragraphs=[NS(words=words)]))
        return OcrResult(text=""), NS(pages=[NS(width=width, height=height, blocks=blocks)], text="")


def test_stacked_classify_judges_each_page_from_its_own_band(tmp_path, monkeypatch):
    heights = [100, 100, 100]
    files = []
    for index, height in enumerate(heights):
        path = tmp_path / f"page{index}.jpg"
        _write_jpg(path, height)
        files.append(path)
    # Stacked height = 3*100 + 2*80 gap = 460; band centres at y ~ 50, 230, 410.
    _FakeStackEngine.calls = []
    _FakeStackEngine.band_blocks = {
        ("ms", "en"): [
            ((40, 60), "DAFTAR PERKAHWINAN ORANG ISLAM"),
            ((220, 240), "DAFTAR PERCERAIAN ORANG ISLAM CATATAN"),
            ((400, 420), "DAFTAR RUJUK ORANG ISLAM"),
        ],
        ("ar",): [],
    }
    monkeypatch.setattr(triage, "GoogleVisionOcrEngine", _FakeStackEngine)

    results = triage.classify_files_stacked(files, allowed_extensions=[".jpg"])

    assert [(r.doc_type, r.record_type, r.layout_variant, r.is_jawi) for r in results] == [
        ("handwritten", "nikah", "legacy", False),
        ("handwritten", "cerai", "modern", False),
        ("handwritten", "rujuk", "legacy", False),
    ]
    # Two Vision calls for the whole stack -- the same as ONE per-file classify.
    assert _FakeStackEngine.calls == [("ms", "en"), ("ar",)]


def test_stacked_classify_rejects_oversized_stacks(tmp_path, monkeypatch):
    import pytest

    image = tmp_path / "page.jpg"
    _write_jpg(image, 100)
    monkeypatch.setattr(triage, "MAX_STACK_JPEG_BYTES", 10)
    with pytest.raises(triage.StackTooLargeError):
        triage.classify_files_stacked([image], allowed_extensions=[".jpg"])

    with pytest.raises(ValueError, match="1-3"):
        triage.classify_files_stacked([image] * 4, allowed_extensions=[".jpg"])


def test_jawi_threshold_catches_real_jawi_pages_and_never_a_rumi_one():
    """Proportions measured on real samples (see DEFAULT_JAWI_PROPORTION_THRESHOLD):
    Jawi register pages 0.61-0.90, Rumi pages 0.00."""
    def page(jawi_chars, latin_chars):
        return OcrResult(text="ج" * jawi_chars + "a" * latin_chars)

    threshold = triage.DEFAULT_JAWI_PROPORTION_THRESHOLD
    for proportion in (0.61, 0.77, 0.90):
        jawi = int(proportion * 1000)
        is_jawi, measured, _ = triage._assess_jawi(page(0, 10), page(jawi, 1000 - jawi), threshold)
        assert is_jawi, proportion
    is_jawi, measured, _ = triage._assess_jawi(page(0, 800), page(0, 700), threshold)
    assert not is_jawi and measured == 0.0



def test_a_stack_of_only_typed_forms_skips_the_jawi_pass(tmp_path, monkeypatch):
    files = []
    for index in range(3):
        path = tmp_path / f"page{index}.jpg"
        _write_jpg(path, 100)
        files.append(path)
    _FakeStackEngine.calls = []
    _FakeStackEngine.band_blocks = {
        ("ms", "en"): [
            ((40, 60), "SURAT PERAKUAN NIKAH TAHUN 1984"),
            ((220, 240), "SURAT PERAKUAN CERAI"),
            ((400, 420), "SURAT PERAKUAN RUJUK"),
        ],
    }
    monkeypatch.setattr(triage, "GoogleVisionOcrEngine", _FakeStackEngine)

    results = triage.classify_files_stacked(files, allowed_extensions=[".jpg"])

    assert [(r.doc_type, r.record_type, r.layout_variant) for r in results] == [
        ("typed", "nikah", "legacy"),
        ("typed", "cerai", "modern"),
        ("typed", "rujuk", "modern"),
    ]
    assert _FakeStackEngine.calls == [("ms", "en")]  # 1 Vision call for 3 forms



def test_a_stamp_mentioning_another_certificate_does_not_win_over_the_title(tmp_path, monkeypatch):
    """Real Rujuk-modern sample: a 'TELAH BERCERAI ... DAFTAR CERAI' stamp next
    to the 'SURAT PERAKUAN RUJU'' title must not merge into it."""
    import pymupdf

    path = tmp_path / "rujuk.pdf"
    document = pymupdf.open()
    document.new_page(width=595, height=842)
    document.save(path)
    document.close()
    _FakeStackEngine.calls = []
    _FakeStackEngine.band_blocks = {
        ("ms", "en"): [
            ((30, 40), "NO. SIRI SURAT PERAKUAN CERAI / RUJU'"),
            ((60, 70), "SURAT PERAKUAN RUJU'"),
        ],
    }
    monkeypatch.setattr(triage, "GoogleVisionOcrEngine", _FakeStackEngine)

    (result,) = triage.classify_files_stacked([path], allowed_extensions=[".pdf"])

    assert (result.doc_type, result.record_type) == ("typed", "rujuk")


def test_the_standalone_title_wins_over_a_mention_elsewhere():
    # Rujuk form with a registrar's stamp mentioning "Surat Perakuan Cerai".
    rujuk = "ENAKMEN UNDANG-UNDANG KELUARGA ISLAM\nNO. SIRI SURAT PERAKUAN CERAI / RUJU'\nSURAT PERAKUAN RUJU'\nBILANGAN DAFTAR RUJUK"
    assert triage._classify_headers(rujuk)[:2] == ("typed", "rujuk")
    # Cerai form whose field label mentions "Surat Perakuan Rujuk".
    cerai = "SURAT PERAKUAN CERAI\n3. BILANGAN DAFTAR SURAT PERAKUAN RUJUK: -"
    assert triage._classify_headers(cerai)[:2] == ("typed", "cerai")
