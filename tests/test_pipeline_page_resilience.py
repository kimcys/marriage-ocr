from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

import marriage_ocr.pipeline as pipeline
from marriage_ocr.models import ExtractedRecord, OcrResult


def test_one_page_crash_does_not_lose_other_pages_records(monkeypatch, tmp_path: Path) -> None:
    """Regression: the per-page loop had no exception handling at all, so one
    bad page (a crash in layout detection, preprocessing, an OCR call, etc.)
    would kill the entire run and lose every already-processed page's records
    -- exactly the failure mode the proposal's "small, retryable page jobs"
    design is meant to avoid at 1M-record volume.
    """
    good_page = SimpleNamespace(
        debug_name="page_001",
        image=np.zeros((16, 16, 3), dtype=np.uint8),
        relative_source=Path("source.pdf"),
        source_page=1,
    )
    bad_page = SimpleNamespace(
        debug_name="page_002",
        image=np.zeros((16, 16, 3), dtype=np.uint8),
        relative_source=Path("source.pdf"),
        source_page=2,
    )

    record_dir = tmp_path / "record_001"
    record_dir.mkdir(parents=True, exist_ok=True)
    cell_paths = {"suami_isteri": record_dir / "suami_isteri.jpg"}
    for path in cell_paths.values():
        path.write_bytes(b"crop")
    (record_dir / "full_record.jpg").write_bytes(b"full")
    saved_record = SimpleNamespace(
        record_index=1,
        record_dir=record_dir,
        full_record_path=record_dir / "full_record.jpg",
        cell_paths=cell_paths,
    )
    record_output = SimpleNamespace(
        record_index=1,
        record_dir=record_dir,
        cell_results={"suami_isteri": OcrResult(text="raw spouse", average_confidence=0.85)},
        raw_json_path=None,
    )
    good_record = ExtractedRecord(bil="1", nama_suami="AHMAD BIN ALI", nama_isteri="SITI BINTI ALI")

    export_calls: list[list[ExtractedRecord]] = []

    monkeypatch.setattr(
        pipeline,
        "load_runtime_config",
        lambda _: SimpleNamespace(
            data={
                "input": {"allowed_extensions": []},
                "preprocessing": {},
                "ocr": {
                    "engine": "fake",
                    "mode": "cell_crops",
                    "save_raw_json": False,
                    "min_average_confidence": 0.5,
                    "field_refinement": {"enabled": False},
                },
                "layout": {"min_record_height_px": 80, "max_record_height_px": 280},
                "validation": {"ok_confidence_threshold": 0.85},
                "postprocess": {"bil_sequence": {"enabled": False}},
                "llm": {"enabled": False},
                "debug": {"retain_artifacts": True},
                "export": {},
            },
            env_file=None,
        ),
    )
    monkeypatch.setattr(
        "marriage_ocr.document_loader.load_document_pages", lambda *args, **kwargs: [good_page, bad_page]
    )
    monkeypatch.setattr("marriage_ocr.document_loader.write_image", lambda *args, **kwargs: None)

    def fake_preprocess_image(image, settings):
        if image is bad_page.image:
            raise RuntimeError("simulated crash processing this page")
        return SimpleNamespace(color=image, binary=np.zeros((16, 16), dtype=np.uint8))

    monkeypatch.setattr("marriage_ocr.preprocess.preprocess_image", fake_preprocess_image)
    monkeypatch.setattr(
        "marriage_ocr.layout.detect_layout",
        lambda *args, **kwargs: SimpleNamespace(
            records=[
                SimpleNamespace(
                    index=1,
                    marker_box=object(),
                    cells={"suami_isteri": object()},
                    box=SimpleNamespace(height=120),
                )
            ],
            ocr_ready_color=np.zeros((16, 16, 3), dtype=np.uint8),
        ),
    )
    monkeypatch.setattr(
        "marriage_ocr.layout.create_table_overlay", lambda *args, **kwargs: np.zeros((16, 16, 3), dtype=np.uint8)
    )
    monkeypatch.setattr(
        "marriage_ocr.layout.create_record_overlay", lambda *args, **kwargs: np.zeros((16, 16, 3), dtype=np.uint8)
    )
    monkeypatch.setattr("marriage_ocr.cropper.save_record_crops", lambda *args, **kwargs: [saved_record])

    class FakeOcrEngine:
        name = "fake"

    monkeypatch.setattr("marriage_ocr.ocr.build_ocr_engine", lambda cfg: FakeOcrEngine())
    monkeypatch.setattr("marriage_ocr.ocr.run_ocr_on_record_crops", lambda *args, **kwargs: [record_output])
    monkeypatch.setattr("marriage_ocr.parser.parse_record_ocr_output", lambda *args, **kwargs: good_record)
    monkeypatch.setattr("marriage_ocr.parser.save_parsed_record", lambda *args, **kwargs: None)
    monkeypatch.setattr("marriage_ocr.postprocess.correct_bil_sequence", lambda records, **kwargs: records)
    monkeypatch.setattr("marriage_ocr.validation.estimate_layout_confidence", lambda **kwargs: 0.92)
    monkeypatch.setattr(pipeline, "validate_record", lambda record, *args, **kwargs: record)

    def fake_export_records_to_xlsx(records, output_path, export_cfg, **kwargs):
        export_calls.append(list(records))
        return SimpleNamespace(written_count=len(records), skipped_duplicates=0, output_path=output_path)

    monkeypatch.setattr("marriage_ocr.exporter.export_records_to_xlsx", fake_export_records_to_xlsx)

    result = pipeline.process_input(
        input_path=tmp_path / "input.pdf",
        output_path=tmp_path / "output.xlsx",
        debug_path=tmp_path / "debug",
        config_path=tmp_path / "config.yaml",
        retain_debug_artifacts=True,
    )

    # The crash on page 2 must not prevent page 1's already-processed record
    # from being exported.
    assert len(result.records) == 1
    assert result.records[0].nama_suami == "AHMAD BIN ALI"
    assert export_calls and len(export_calls[0]) == 1

    # The failure must be visible, not silently swallowed.
    assert result.failed_pages == ["source.pdf"]
