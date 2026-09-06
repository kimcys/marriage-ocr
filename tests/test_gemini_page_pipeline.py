from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from marriage_ocr import gemini_page_pipeline
from marriage_ocr.models import ExtractedRecord
from marriage_ocr.pipeline import process_input


def _page(name: str = "sample.jpg") -> SimpleNamespace:
    return SimpleNamespace(
        debug_name="page_001",
        image=np.zeros((16, 16, 3), dtype=np.uint8),
        relative_source=Path(name),
        source_page=1,
    )


def _gemini_result(**overrides) -> SimpleNamespace:
    record = ExtractedRecord(
        bil="1/97",
        nama_suami="AHMAD BIN ALI",
        ic_lama_suami="A 1192345",
        umur_suami=25,
        nama_isteri="SITI BINTI ALI",
        ic_baru_isteri="900101101234",
        umur_isteri=23,
        mas_kahwin="RM 80.00",
        mas_kahwin_raw="RM 80.00",
        nama_pendaftar="MOHD SALLEH",
        alamat_pendaftar="KAMPUNG BARU",
        nama_wali="ABDUL RAHMAN",
        hubungan_wali="BAPA",
        saksi_1="AHMAD BIN ALI",
        saksi_2="OSMAN BIN DIN",
        tarikh_nikah="1994-08-27",
    )
    for key, value in overrides.items():
        setattr(record, key, value)
    return SimpleNamespace(
        record=record,
        field_confidence={"nama_suami": 0.95, "nama_isteri": 0.94},
        uncertain_fields=[],
        notes=[],
    )


def _configure(monkeypatch, tmp_path: Path, *, pages, extract_page_impl, output_path=None, retain_artifacts=True):
    monkeypatch.setattr(
        gemini_page_pipeline,
        "load_runtime_config",
        lambda _: SimpleNamespace(
            data={
                "record_type": "nikah",
                "layout_variant": "legacy",
                "input": {"allowed_extensions": []},
                "preprocessing": {},
                "validation": {"ok_confidence_threshold": 0.85, "min_age": 15, "max_age": 100},
                "postprocess": {"bil_sequence": {"enabled": False}},
                "llm": {"model": "gemini-3-flash-preview"},
                "debug": {"retain_artifacts": retain_artifacts},
                "export": {},
            },
            env_file=None,
        ),
    )
    def fake_write_image(path, image, **kw):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake-image-bytes")

    monkeypatch.setattr("marriage_ocr.document_loader.load_document_pages", lambda *a, **kw: pages)
    monkeypatch.setattr("marriage_ocr.document_loader.write_image", fake_write_image)
    monkeypatch.setattr(
        "marriage_ocr.preprocess.preprocess_image",
        lambda image, settings: SimpleNamespace(color=image, binary=np.zeros((16, 16), dtype=np.uint8)),
    )
    monkeypatch.setattr("marriage_ocr.parser.save_parsed_record", lambda *a, **kw: None)
    monkeypatch.setattr("marriage_ocr.postprocess.correct_bil_sequence", lambda records, **kw: records)

    export_calls: list[list[ExtractedRecord]] = []

    def fake_export_xlsx(records, output_path, export_config, *, reset_output, skip_existing):
        export_calls.append(list(records))
        return SimpleNamespace(written_count=len(records), skipped_duplicates=0, total_rows=len(records), output_path=output_path)

    monkeypatch.setattr("marriage_ocr.exporter.export_records_to_xlsx", fake_export_xlsx)
    monkeypatch.setattr(
        "marriage_ocr.exporter.export_records_to_csv",
        lambda records, output_path, export_config, **kw: fake_export_xlsx(records, output_path, export_config, reset_output=kw.get("reset_output", False), skip_existing=kw.get("skip_existing", False)),
    )

    class FakeExtractor:
        def __init__(self, config):
            self.config = config

        def extract_page(self, image_path):
            return extract_page_impl(image_path)

    monkeypatch.setattr(gemini_page_pipeline, "GeminiPageExtractor", FakeExtractor)

    return export_calls


def test_process_input_gemini_page_routes_here_via_pipeline_engine_config(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "marriage_ocr.pipeline.load_runtime_config",
        lambda _: SimpleNamespace(data={"pipeline": {"engine": "gemini_page"}}, env_file=None),
    )

    called = {}

    def fake_process_input_gemini_page(**kwargs):
        called.update(kwargs)
        return "sentinel-result"

    monkeypatch.setattr(
        "marriage_ocr.gemini_page_pipeline.process_input_gemini_page", fake_process_input_gemini_page
    )

    result = process_input(
        input_path=tmp_path / "in",
        output_path=tmp_path / "out.xlsx",
        debug_path=tmp_path / "debug",
        config_path=tmp_path / "config.yaml",
    )

    assert result == "sentinel-result"
    assert called["config_path"] == tmp_path / "config.yaml"


def test_process_input_gemini_page_exports_records_from_one_page_call(monkeypatch, tmp_path: Path) -> None:
    page = _page()
    export_calls = _configure(
        monkeypatch,
        tmp_path,
        pages=[page],
        extract_page_impl=lambda image_path: [_gemini_result(bil="1/97"), _gemini_result(bil="2/97")],
    )

    result = gemini_page_pipeline.process_input_gemini_page(
        input_path=tmp_path / "in",
        output_path=tmp_path / "out.xlsx",
        debug_path=tmp_path / "debug",
        config_path=tmp_path / "config.yaml",
    )

    assert result.total_detected_records == 2
    assert [r.bil for r in export_calls[0]] == ["1/97", "2/97"]
    assert result.gemini_calls == 1
    assert result.refinement_ocr_calls == 0
    assert result.failed_pages == []


def test_process_input_gemini_page_marks_good_record_ok(monkeypatch, tmp_path: Path) -> None:
    page = _page()
    _configure(
        monkeypatch, tmp_path, pages=[page],
        extract_page_impl=lambda image_path: [_gemini_result()],
    )

    result = gemini_page_pipeline.process_input_gemini_page(
        input_path=tmp_path / "in",
        output_path=tmp_path / "out.xlsx",
        debug_path=tmp_path / "debug",
        config_path=tmp_path / "config.yaml",
    )

    assert result.status_counts == {"OK": 1}
    assert result.records[0].source_file == "sample.jpg"
    assert result.records[0].source_page == 1
    assert result.records[0].source_record == "record_001"


def test_process_input_gemini_page_saves_debug_artifacts_with_page_as_crop(monkeypatch, tmp_path: Path) -> None:
    page = _page()
    debug_path = tmp_path / "debug"
    _configure(
        monkeypatch, tmp_path, pages=[page],
        extract_page_impl=lambda image_path: [_gemini_result()],
        retain_artifacts=True,
    )

    gemini_page_pipeline.process_input_gemini_page(
        input_path=tmp_path / "in",
        output_path=tmp_path / "out.xlsx",
        debug_path=debug_path,
        config_path=tmp_path / "config.yaml",
        retain_debug_artifacts=True,
    )

    record_dir = debug_path / "page_001" / "records" / "record_001"
    assert (record_dir / "full_record.jpg").read_bytes() == b"fake-image-bytes"


def test_process_input_gemini_page_isolates_one_failing_page(monkeypatch, tmp_path: Path) -> None:
    good_page = _page("good.jpg")
    bad_page = _page("bad.jpg")

    # Both pages share the same fake debug_name ("page_001"), so key off
    # call order rather than the image path to decide which one fails.
    calls = {"count": 0}

    def extract_page_impl(image_path):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("simulated Gemini failure")
        return [_gemini_result()]

    monkeypatch.setattr(gemini_page_pipeline, "write_error_report", lambda *a, **kw: None)
    export_calls = _configure(
        monkeypatch, tmp_path, pages=[good_page, bad_page], extract_page_impl=extract_page_impl,
    )

    result = gemini_page_pipeline.process_input_gemini_page(
        input_path=tmp_path / "in",
        output_path=tmp_path / "out.xlsx",
        debug_path=tmp_path / "debug",
        config_path=tmp_path / "config.yaml",
    )

    assert result.failed_pages == ["bad.jpg"]
    assert result.total_detected_records == 1
    assert len(export_calls[0]) == 1
