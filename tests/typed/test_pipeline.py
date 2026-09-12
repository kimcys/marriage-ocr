from pathlib import Path
from types import SimpleNamespace

import pymupdf as fitz

from marriage_ocr.typed.models import PageOcrResult, PositionedWord, ProcessingStatus, TypedDocumentResult
from marriage_ocr.typed.pipeline import _process_micro_batch, process_typed_input


def _write_two_page_pdf(path: Path) -> None:
    document = fitz.open()
    document.new_page(width=595, height=842)
    document.new_page(width=595, height=842)
    document.save(path)
    document.close()


def _write_one_page_pdf(path: Path) -> None:
    document = fitz.open()
    document.new_page(width=595, height=842)
    document.save(path)
    document.close()


def _word(text: str, x1: float, y1: float, x2: float, y2: float, page_number: int) -> PositionedWord:
    return PositionedWord(text, 0.98, x1, y1, x2, y2, page_number)


def _field_word(page_number: int, region: tuple[float, float, float, float], text: str) -> PositionedWord:
    x1, y1, x2, y2 = region
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return PositionedWord(text, 0.97, cx - 0.01, cy - 0.004, cx + 0.01, cy + 0.004, page_number)


def _synthetic_complete_page_results(pages):
    from marriage_ocr.typed.template import BORANG_4B_ANCHORS, BORANG_4B_REGIONS

    results = []
    for page in pages:
        words = []
        for phrase, (x, y) in BORANG_4B_ANCHORS.get(page.page_number, {}).items():
            x1 = x
            y1 = y
            words.append(_word(phrase.split()[0], x1, y1, x1 + 0.03, y1 + 0.01, page.page_number))
        values = {
            "bil": "04/2009",
            "nama_suami": "HENDON BIN MARIMIN",
            "id_suami": "571018-10-5919",
            "umur_suami": "52 Tahun",
            "nama_isteri": "ABIDAH BINTI HALIDI @ HAJI HALIDI",
            "id_isteri": "6057990",
            "umur_isteri": "49 Tahun",
            "nama_wali": "HAJI HALIDI BIN HAJI OSMAN",
            "hubungan_wali": "BAPA KANDUNG",
            "saksi_1": "HAMZAH BIN ABAS",
            "saksi_2": "RAMLI BIN ISMAIL",
            "tarikh_nikah": "21.09.1984",
            "alamat_pendaftar": "KAMPUNG PARIT 9 SUNGAI LEMAN, 45400 SEKINCHAN SELANGOR",
            "nama_pendaftar": "USTAZ SHUKRI BIN SHARIF",
            "mas_kahwin": "RM 80.00",
        }
        for key, (field_page, region) in BORANG_4B_REGIONS.items():
            if field_page != page.page_number:
                continue
            words.append(_field_word(page.page_number, (region.x1, region.y1, region.x2, region.y2), values[key]))
        results.append(
            PageOcrResult(
                source_file=page.source_file,
                page_number=page.page_number,
                words=tuple(words),
                full_text=" ".join(word.text for word in words),
                raw_response={"words": [word.text for word in words]},
            )
        )
    return tuple(results)


def test_process_typed_input_writes_one_success_row_per_pdf(monkeypatch, tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_two_page_pdf(input_dir / "b.pdf")
    _write_two_page_pdf(input_dir / "a.pdf")

    monkeypatch.setattr("marriage_ocr.typed.pipeline._ocr_micro_batch", lambda pages, client: _synthetic_complete_page_results(pages))

    config_path = tmp_path / "typed.yaml"
    config_path.write_text(
        """
ocr:
  google_vision:
    language_hints: [ms, en]
typed:
  pdf_dpi: 300
  pdf_batch_size: 4
  render_workers: 4
  word_confidence_threshold: 0.75
  region_boundary_tolerance: 0.01
  retry:
    api_attempts: 3
    initial_delay_seconds: 1
    backoff_multiplier: 2
    max_fields_per_pdf: 6
    crop_padding_ratio: 0.05
    request_batch_size: 16
  validation:
    min_age: 16
    max_age: 120
""".strip(),
        encoding="utf-8",
    )

    result = process_typed_input(
        input_path=input_dir,
        output_path=tmp_path / "typed_records.csv",
        debug_path=tmp_path / "debug",
        config_path=config_path,
        reset_output=True,
    )

    assert result.discovered_pdfs == 2
    assert [record.source_file for record in result.records] == ["a.pdf", "b.pdf"]
    assert all(record.processing_status is ProcessingStatus.SUCCESS for record in result.records)


def test_process_micro_batch_passes_each_pdfs_own_rendered_pages(monkeypatch, tmp_path: Path) -> None:
    """Regression: _process_single_pdf's `pages` kwarg must come from THIS
    pdf's own rendered pages, not whichever pdf the earlier
    `for pdf in pdfs: pages = rendered.get(pdf.name)` loop (building
    ordered_pages for the bulk OCR call) happened to leave `pages` bound to
    when it exited. With a real >1-pdf micro-batch (pdf_batch_size=4 is the
    default across every typed config, so this is the common case, not an
    edge case) every pdf but the loop's last one previously got the wrong
    pdf's RenderedPage tuple -- confirmed against a real 5-file typed Nikah
    run where only the last pdf in the first 4-pdf batch came out clean."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_two_page_pdf(input_dir / "a.pdf")
    _write_two_page_pdf(input_dir / "b.pdf")
    _write_two_page_pdf(input_dir / "c.pdf")

    monkeypatch.setattr(
        "marriage_ocr.typed.pipeline._ocr_micro_batch", lambda pages, client: _synthetic_complete_page_results(pages)
    )

    seen_pages: dict[str, tuple] = {}

    def _fake_process_single_pdf(pdf, *, pages, **_kwargs):
        seen_pages[pdf.name] = pages
        return TypedDocumentResult(
            record=SimpleNamespace(),
            source_file=pdf.name,
            processing_status=ProcessingStatus.SUCCESS,
        )

    monkeypatch.setattr("marriage_ocr.typed.pipeline._process_single_pdf", _fake_process_single_pdf)

    pdfs = sorted(input_dir.glob("*.pdf"))
    _process_micro_batch(
        pdfs=pdfs,
        debug_path=tmp_path / "debug",
        client=None,
        typed_cfg={"pdf_dpi": 300, "render_workers": 4},
        retry_cfg={},
        validation_cfg={},
        template_name="borang_4b",
        record_type="NIKAH",
    )

    assert set(seen_pages) == {"a.pdf", "b.pdf", "c.pdf"}
    for pdf in pdfs:
        pages = seen_pages[pdf.name]
        assert pages is not None
        # Every RenderedPage handed to _process_single_pdf for this pdf must
        # actually belong to this pdf, not a different one in the batch.
        assert all(page.source_pdf == pdf for page in pages)


_TYPED_CONFIG_TEXT = """
ocr:
  google_vision:
    language_hints: [ms, en]
typed:
  pdf_dpi: 300
  pdf_batch_size: 4
  render_workers: 4
  word_confidence_threshold: 0.75
  region_boundary_tolerance: 0.01
  retry:
    api_attempts: 3
    initial_delay_seconds: 1
    backoff_multiplier: 2
    max_fields_per_pdf: 6
    crop_padding_ratio: 0.05
    request_batch_size: 16
  validation:
    min_age: 16
    max_age: 120
""".strip()


def test_process_typed_input_does_not_retain_debug_artifacts_by_default(monkeypatch, tmp_path: Path) -> None:
    """Debug PNGs (rendered pages, region overlays) are the overwhelming
    majority of stored bytes for a real import and existed only to eyeball a
    bad extraction -- so unless a caller opts in, they must land in a
    throwaway temp dir, not `debug_path`, and the record must not carry a
    crop_folder pointing at output that won't persist."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_two_page_pdf(input_dir / "a.pdf")

    monkeypatch.setattr(
        "marriage_ocr.typed.pipeline._ocr_micro_batch",
        lambda pages, client: _synthetic_complete_page_results(pages),
    )

    config_path = tmp_path / "typed.yaml"
    config_path.write_text(_TYPED_CONFIG_TEXT, encoding="utf-8")
    debug_path = tmp_path / "debug"

    result = process_typed_input(
        input_path=input_dir,
        output_path=tmp_path / "typed_records.csv",
        debug_path=debug_path,
        config_path=config_path,
        reset_output=True,
        retain_debug_artifacts=False,
    )

    assert result.records[0].processing_status is ProcessingStatus.SUCCESS
    assert result.records[0].record.crop_folder is None
    assert not debug_path.exists()


def test_process_typed_input_retains_debug_artifacts_when_configured(monkeypatch, tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_two_page_pdf(input_dir / "a.pdf")

    monkeypatch.setattr(
        "marriage_ocr.typed.pipeline._ocr_micro_batch",
        lambda pages, client: _synthetic_complete_page_results(pages),
    )

    config_path = tmp_path / "typed.yaml"
    config_path.write_text(_TYPED_CONFIG_TEXT, encoding="utf-8")
    debug_path = tmp_path / "debug"

    result = process_typed_input(
        input_path=input_dir,
        output_path=tmp_path / "typed_records.csv",
        debug_path=debug_path,
        config_path=config_path,
        reset_output=True,
        retain_debug_artifacts=True,
    )

    document_debug_dir = debug_path / "a"
    assert result.records[0].record.crop_folder == str(document_debug_dir)
    assert (document_debug_dir / "page_1_regions.png").exists()
    assert (document_debug_dir / "page_2_regions.png").exists()
    assert (document_debug_dir / "validation.json").exists()


def test_pipeline_marks_failed_row_for_one_page_pdf(monkeypatch, tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_one_page_pdf(input_dir / "broken.pdf")
    config_path = tmp_path / "typed.yaml"
    config_path.write_text("typed: {}\n", encoding="utf-8")

    result = process_typed_input(
        input_path=input_dir,
        output_path=tmp_path / "typed_records.csv",
        debug_path=tmp_path / "debug",
        config_path=config_path,
        reset_output=True,
    )

    assert result.discovered_pdfs == 1
    assert result.records[0].processing_status is ProcessingStatus.FAILED
    assert "Expected exactly 2 page(s)" in result.records[0].error_message

