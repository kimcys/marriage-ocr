from pathlib import Path

import pymupdf as fitz
import pytest

from marriage_ocr.typed.loader import discover_typed_pdfs, render_typed_pdf


def _write_pdf(path: Path, page_count: int) -> None:
    document = fitz.open()
    for page_index in range(page_count):
        page = document.new_page(width=595, height=842)
        page.insert_text((72, 72), f"PAGE {page_index + 1}")
    document.save(path)
    document.close()


def test_discover_typed_pdfs_returns_sorted_pdf_files(tmp_path: Path) -> None:
    _write_pdf(tmp_path / "b.pdf", 2)
    _write_pdf(tmp_path / "a.PDF", 2)
    (tmp_path / "ignore.jpg").write_bytes(b"not a pdf")

    assert [path.name for path in discover_typed_pdfs(tmp_path)] == ["a.PDF", "b.pdf"]


def test_render_typed_pdf_writes_two_300_dpi_pages(tmp_path: Path) -> None:
    pdf_path = tmp_path / "record.pdf"
    _write_pdf(pdf_path, 2)

    pages = render_typed_pdf(pdf_path, tmp_path / "debug", dpi=300)

    assert [page.page_number for page in pages] == [1, 2]
    assert all(page.image_path.exists() for page in pages)
    assert all(page.width > 2000 for page in pages)
    assert all(page.height > 3000 for page in pages)


def test_render_typed_pdf_rejects_non_two_page_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "one-page.pdf"
    _write_pdf(pdf_path, 1)

    with pytest.raises(ValueError, match=r"Expected exactly 2 page\(s\), found 1"):
        render_typed_pdf(pdf_path, tmp_path / "debug", dpi=300)


def test_render_typed_pdf_accepts_custom_expected_page_count(tmp_path: Path) -> None:
    # Legacy Cerai/Rujuk typed certs (Borang 9/Borang 5) are genuinely
    # single-page, unlike Borang 4B/modern Cerai/Rujuk's 2 pages.
    pdf_path = tmp_path / "legacy.pdf"
    _write_pdf(pdf_path, 1)

    pages = render_typed_pdf(pdf_path, tmp_path / "debug", dpi=300, expected_pages=1)

    assert [page.page_number for page in pages] == [1]

