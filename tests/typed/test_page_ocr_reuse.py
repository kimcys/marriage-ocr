from pathlib import Path

import pymupdf as fitz

from marriage_ocr.typed.models import PageOcrResult, PositionedWord, ProcessingStatus
from marriage_ocr.typed.page_ocr_cache import load_page_ocr_cache, write_page_ocr_cache
from marriage_ocr.typed.pipeline import process_typed_input
from tests.typed.test_pipeline import _synthetic_complete_page_results, _write_two_page_pdf

_CONFIG = """
ocr:
  google_vision:
    language_hints: [ms, en]
typed:
  pdf_dpi: 300
  pdf_batch_size: 4
  render_workers: 1
  word_confidence_threshold: 0.75
  region_boundary_tolerance: 0.01
  retry:
    api_attempts: 1
    max_fields_per_pdf: 6
    request_batch_size: 16
  validation:
    min_age: 16
    max_age: 120
""".strip()

# A 595x842pt page rendered at 300 dpi.
_WIDTH, _HEIGHT = 2480, 3509


def _page1_from_synthetic(pdf: Path) -> PageOcrResult:
    class _Page:
        source_file = pdf.name
        page_number = 1

    return _synthetic_complete_page_results([_Page()])[0]


def _run(tmp_path: Path, page1_ocr: Path | None, monkeypatch) -> tuple[list[list[int]], object]:
    ocr_calls: list[list[int]] = []

    def fake_ocr(pages, client):
        ocr_calls.append([page.page_number for page in pages])
        return _synthetic_complete_page_results(pages)

    monkeypatch.setattr("marriage_ocr.typed.pipeline._ocr_micro_batch", fake_ocr)
    config = tmp_path / "typed.yaml"
    config.write_text(_CONFIG, encoding="utf-8")
    result = process_typed_input(
        input_path=tmp_path / "doc.pdf",
        output_path=tmp_path / "out.csv",
        debug_path=tmp_path / "debug",
        config_path=config,
        reset_output=True,
        page1_ocr_path=page1_ocr,
    )
    return ocr_calls, result


def test_cache_round_trips_words_and_retargets_source_file(tmp_path: Path) -> None:
    word = PositionedWord("NIKAH", 0.9, 0.1, 0.2, 0.3, 0.4, 1)
    original = PageOcrResult("classify-name.pdf", 1, (word,), "SURAT PERAKUAN NIKAH", {})
    path = tmp_path / "page1.json"
    write_page_ocr_cache(path, original, dpi=300, width=10, height=20)

    cached = load_page_ocr_cache(path, source_file="source.pdf")

    assert cached is not None
    assert cached.result.source_file == "source.pdf"
    assert cached.result.words == (word,)
    assert cached.result.full_text == "SURAT PERAKUAN NIKAH"
    assert cached.matches(dpi=300, width=10, height=20)
    assert not cached.matches(dpi=200, width=10, height=20)


def test_unreadable_cache_is_ignored(tmp_path: Path) -> None:
    path = tmp_path / "page1.json"
    path.write_text("{not json", encoding="utf-8")
    assert load_page_ocr_cache(path, source_file="source.pdf") is None


def test_process_typed_reuses_page1_and_only_ocrs_page2(tmp_path: Path, monkeypatch) -> None:
    pdf = tmp_path / "doc.pdf"
    _write_two_page_pdf(pdf)
    cache = tmp_path / "page1.json"
    write_page_ocr_cache(cache, _page1_from_synthetic(pdf), dpi=300, width=_WIDTH, height=_HEIGHT)

    ocr_calls, result = _run(tmp_path, cache, monkeypatch)

    assert ocr_calls == [[2]]
    assert result.records[0].processing_status is ProcessingStatus.SUCCESS


def test_process_typed_ocrs_both_pages_when_the_render_does_not_match(tmp_path: Path, monkeypatch) -> None:
    pdf = tmp_path / "doc.pdf"
    _write_two_page_pdf(pdf)
    cache = tmp_path / "page1.json"
    write_page_ocr_cache(cache, _page1_from_synthetic(pdf), dpi=200, width=1653, height=2339)

    ocr_calls, result = _run(tmp_path, cache, monkeypatch)

    assert ocr_calls == [[1, 2]]
    assert result.records[0].processing_status is ProcessingStatus.SUCCESS


def test_process_typed_without_cache_ocrs_both_pages(tmp_path: Path, monkeypatch) -> None:
    _write_two_page_pdf(tmp_path / "doc.pdf")
    ocr_calls, _ = _run(tmp_path, None, monkeypatch)
    assert ocr_calls == [[1, 2]]


def test_rendered_page_size_matches_the_constant_used_above(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    _write_two_page_pdf(pdf)
    with fitz.open(pdf) as document:
        pixmap = document.load_page(0).get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72), alpha=False)
    assert (pixmap.width, pixmap.height) == (_WIDTH, _HEIGHT)
