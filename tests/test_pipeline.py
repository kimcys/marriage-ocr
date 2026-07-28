from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from typer.testing import CliRunner

import marriage_ocr.pipeline as pipeline
from marriage_ocr.cli import app
from marriage_ocr.models import ExtractedRecord, OcrResult
from marriage_ocr.refinement.models import FieldCandidate, FieldRefinementDecision


runner = CliRunner()


class FakeOcrEngine:
    name = "fake"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def read_image(self, image_path: str | Path) -> OcrResult:
        self.calls.append(Path(image_path).name)
        return OcrResult(text="retry", average_confidence=0.91)


def _base_record(*, bil: str) -> ExtractedRecord:
    return ExtractedRecord(
        bil=bil,
        nama_suami="AHMAD B1N ALI",
        ic_baru_suami="900101011234",
        umur_suami=30,
        nama_isteri="SITI BINTI ALI",
        ic_baru_isteri="900101-01-1234",
        umur_isteri=28,
        mas_kahwin="RM100",
        mas_kahwin_raw="RM100",
        nama_pendaftar="PENDAFTAR",
        alamat_pendaftar="ALAMAT",
        nama_wali="WALI",
        hubungan_wali="BAPA",
        saksi_1="SAKSI SATU",
        saksi_2="SAKSI DUA",
        tarikh_nikah="2024-13-40",
    )


def _candidate(value: str, *, attempt: int | None = None) -> FieldCandidate:
    metadata: dict[str, object] = {}
    if attempt is not None:
        metadata["retry_attempt"] = attempt
    return FieldCandidate(
        value=value,
        source="retry_thresholded" if attempt is not None else "original_ocr",
        validity_score=1.0,
        ocr_confidence=0.91,
        plausibility_score=1.0,
        similarity_score=1.0,
        substitutions=0,
        metadata=metadata,
    )


def _decision(
    field_name: str,
    original_value: str | None,
    selected_value: str | None,
    *,
    attempts: int = 0,
    requires_review: bool = False,
    reason: str = "accepted_after_retry",
) -> FieldRefinementDecision:
    candidates = [_candidate(original_value or "", attempt=None)]
    for attempt in range(1, attempts + 1):
        candidates.append(_candidate(selected_value or original_value or "", attempt=attempt))
    selected_candidate = candidates[-1] if candidates else None
    return FieldRefinementDecision(
        field_name=field_name,
        original_value=original_value,
        selected_value=selected_value,
        candidates=tuple(candidates),
        selected_candidate=selected_candidate,
        requires_review=requires_review,
        reason=reason,
    )


def _configure_process(
    monkeypatch,
    tmp_path: Path,
    *,
    parsed_records: list[ExtractedRecord],
    refine_impl,
    validate_impl=None,
    llm_enabled: bool = False,
    field_refinement_enabled: bool = True,
    max_variants_per_field: int = 3,
):
    page = SimpleNamespace(
        debug_name="page_001",
        image=np.zeros((16, 16, 3), dtype=np.uint8),
        relative_source=Path("source.pdf"),
        source_page=1,
    )
    record_outputs = []
    saved_records = []
    for index, _ in enumerate(parsed_records, start=1):
        record_dir = tmp_path / f"record_{index:03d}"
        record_dir.mkdir(parents=True, exist_ok=True)
        cell_paths = {
            "suami_isteri": record_dir / "suami_isteri.jpg",
            "tarikh_nikah": record_dir / "tarikh_nikah.jpg",
        }
        for path in cell_paths.values():
            path.write_bytes(b"crop")
        (record_dir / "full_record.jpg").write_bytes(b"full")
        saved_records.append(
            SimpleNamespace(
                record_index=index,
                record_dir=record_dir,
                full_record_path=record_dir / "full_record.jpg",
                cell_paths=cell_paths,
            )
        )
        record_outputs.append(
            SimpleNamespace(
                record_index=index,
                record_dir=record_dir,
                cell_results={
                    "suami_isteri": OcrResult(text="raw spouse", average_confidence=0.85),
                    "tarikh_nikah": OcrResult(text="raw date", average_confidence=0.84),
                },
                raw_json_path=None,
            )
        )

    parsed_iter = iter(parsed_records)
    export_calls: list[list[ExtractedRecord]] = []
    validate_inputs: list[ExtractedRecord] = []

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
                    "field_refinement": {
                        "enabled": field_refinement_enabled,
                        "max_variants_per_field": max_variants_per_field,
                        "minimum_candidate_score": 0.75,
                        "minimum_score_improvement": 0.12,
                    },
                },
                "layout": {"min_record_height_px": 80, "max_record_height_px": 280},
                "validation": {"ok_confidence_threshold": 0.85},
                "postprocess": {"bil_sequence": {"enabled": False}},
                "llm": {"enabled": llm_enabled, "provider": "gemini"},
                "debug": {"retain_artifacts": True},
                "export": {},
            },
            env_file=None,
        ),
    )
    monkeypatch.setattr("marriage_ocr.document_loader.load_document_pages", lambda *args, **kwargs: [page])
    monkeypatch.setattr("marriage_ocr.document_loader.write_image", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "marriage_ocr.preprocess.preprocess_image",
        lambda image, settings: SimpleNamespace(color=image, binary=np.zeros((16, 16), dtype=np.uint8)),
    )
    monkeypatch.setattr(
        "marriage_ocr.layout.detect_layout",
        lambda *args, **kwargs: SimpleNamespace(
            records=[
                SimpleNamespace(
                    index=index,
                    marker_box=object(),
                    cells={"suami_isteri": object(), "tarikh_nikah": object()},
                    box=SimpleNamespace(height=120),
                )
                for index in range(1, len(parsed_records) + 1)
            ],
            ocr_ready_color=np.zeros((16, 16, 3), dtype=np.uint8),
        ),
    )
    monkeypatch.setattr("marriage_ocr.layout.create_table_overlay", lambda *args, **kwargs: np.zeros((16, 16, 3), dtype=np.uint8))
    monkeypatch.setattr("marriage_ocr.layout.create_record_overlay", lambda *args, **kwargs: np.zeros((16, 16, 3), dtype=np.uint8))
    monkeypatch.setattr("marriage_ocr.cropper.save_record_crops", lambda *args, **kwargs: saved_records)
    monkeypatch.setattr("marriage_ocr.ocr.build_ocr_engine", lambda cfg: FakeOcrEngine())
    monkeypatch.setattr("marriage_ocr.ocr.run_ocr_on_record_crops", lambda *args, **kwargs: record_outputs)
    monkeypatch.setattr("marriage_ocr.parser.parse_record_ocr_output", lambda *args, **kwargs: next(parsed_iter))
    monkeypatch.setattr("marriage_ocr.parser.save_parsed_record", lambda *args, **kwargs: None)
    monkeypatch.setattr("marriage_ocr.postprocess.correct_bil_sequence", lambda records, **kwargs: records)
    monkeypatch.setattr("marriage_ocr.validation.estimate_layout_confidence", lambda **kwargs: 0.92)
    monkeypatch.setattr(
        "marriage_ocr.exporter.export_records_to_xlsx",
        lambda records, output_path, export_cfg, **kwargs: export_calls.append(list(records))
        or SimpleNamespace(written_count=len(records), skipped_duplicates=0, output_path=output_path),
    )
    monkeypatch.setattr(pipeline, "refine_field", refine_impl, raising=False)

    if llm_enabled:
        monkeypatch.setattr(
            pipeline,
            "_build_gemini_record_processor",
            lambda *args, **kwargs: validate_impl,
        )
    else:
        def fake_validate(record, cell_results, validation_config, *, layout_confidence=1.0, layout_ok=True):
            validate_inputs.append(record)
            if validate_impl is not None:
                return validate_impl(record, cell_results, validation_config, layout_confidence=layout_confidence, layout_ok=layout_ok)
            return record

        monkeypatch.setattr(pipeline, "validate_record", fake_validate)

    result = pipeline.process_input(
        input_path=tmp_path / "input.pdf",
        output_path=tmp_path / "output.xlsx",
        debug_path=tmp_path / "debug",
        config_path=tmp_path / "config.yaml",
        retain_debug_artifacts=True,
    )

    return result, validate_inputs, export_calls


def test_process_input_refines_before_validation(monkeypatch, tmp_path: Path) -> None:
    def fake_refine_field(field_name, original_value, parsed_value=None, **kwargs):
        ocr_engine = kwargs["ocr_engine"]
        crop_path = kwargs["crop_path"]
        if field_name == "nama_suami":
            ocr_engine.read_image(crop_path)
            ocr_engine.read_image(crop_path)
            return _decision(field_name, original_value, "AHMAD BIN ALI", attempts=2)
        if field_name == "tarikh_nikah":
            ocr_engine.read_image(crop_path)
            return _decision(field_name, original_value, "2024-01-03", attempts=1)
        return _decision(field_name, original_value, original_value, reason="accepted_without_retry")

    result, validate_inputs, export_calls = _configure_process(
        monkeypatch,
        tmp_path,
        parsed_records=[_base_record(bil="1")],
        refine_impl=fake_refine_field,
    )

    assert validate_inputs[0].nama_suami == "AHMAD BIN ALI"
    assert validate_inputs[0].tarikh_nikah == "2024-01-03"
    assert result.refinement_ocr_calls == 3
    assert export_calls[0][0].nama_suami == "AHMAD BIN ALI"


def test_process_input_skips_retry_stage_when_refinement_disabled(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_refine_field(field_name, original_value, parsed_value=None, **kwargs):
        calls.append(field_name)
        return _decision(field_name, original_value, "SHOULD NOT APPLY", attempts=1)

    result, validate_inputs, _ = _configure_process(
        monkeypatch,
        tmp_path,
        parsed_records=[_base_record(bil="1")],
        refine_impl=fake_refine_field,
        field_refinement_enabled=False,
    )

    assert calls == []
    assert validate_inputs[0].nama_suami == "AHMAD B1N ALI"
    assert result.refinement_ocr_calls == 0


def test_process_input_honors_max_retry_count_and_safe_fallback(monkeypatch, tmp_path: Path) -> None:
    def fake_refine_field(field_name, original_value, parsed_value=None, *, settings, ocr_engine, crop_path, **kwargs):
        assert settings.max_variants_per_field == 2
        if field_name == "nama_suami":
            ocr_engine.read_image(crop_path)
            ocr_engine.read_image(crop_path)
            return _decision(field_name, original_value, original_value, attempts=2, requires_review=True, reason="retry_ocr_failed")
        return _decision(field_name, original_value, original_value, reason="accepted_without_retry")

    result, validate_inputs, _ = _configure_process(
        monkeypatch,
        tmp_path,
        parsed_records=[_base_record(bil="1")],
        refine_impl=fake_refine_field,
        max_variants_per_field=2,
    )

    assert validate_inputs[0].nama_suami == "AHMAD B1N ALI"
    assert result.refinement_ocr_calls == 2


def test_process_input_continues_when_one_field_refinement_fails(monkeypatch, tmp_path: Path) -> None:
    seen: list[tuple[str, str | None]] = []

    def fake_refine_field(field_name, original_value, parsed_value=None, **kwargs):
        seen.append((field_name, original_value))
        if field_name == "nama_suami" and original_value == "AHMAD B1N ALI":
            raise RuntimeError("retry crashed")
        if field_name == "nama_suami" and original_value == "KAMAL B1N OSMAN":
            return _decision(field_name, original_value, "KAMAL BIN OSMAN", attempts=1)
        return _decision(field_name, original_value, original_value, reason="accepted_without_retry")

    second_record = _base_record(bil="2")
    second_record.nama_suami = "KAMAL B1N OSMAN"

    result, _, export_calls = _configure_process(
        monkeypatch,
        tmp_path,
        parsed_records=[_base_record(bil="1"), second_record],
        refine_impl=fake_refine_field,
    )

    assert len(result.records) == 2
    assert result.records[0].nama_suami == "AHMAD B1N ALI"
    assert result.records[1].nama_suami == "KAMAL BIN OSMAN"
    assert export_calls[0][1].nama_suami == "KAMAL BIN OSMAN"
    assert ("nama_suami", "AHMAD B1N ALI") in seen
    assert ("nama_suami", "KAMAL B1N OSMAN") in seen


def test_process_input_passes_refined_record_into_gemini_path(monkeypatch, tmp_path: Path) -> None:
    def fake_refine_field(field_name, original_value, parsed_value=None, **kwargs):
        if field_name == "nama_suami":
            return _decision(field_name, original_value, "AHMAD BIN ALI", attempts=1)
        return _decision(field_name, original_value, original_value, reason="accepted_without_retry")

    seen: list[str | None] = []

    def fake_gemini_processor(record, record_output, *, layout_confidence):
        seen.append(record.nama_suami)
        return ExtractedRecord(
            **{
                **record.to_dict(),
                "status_review": "OK",
                "review_reason": [],
                "confidence": 0.95,
            }
        )

    result, _, _ = _configure_process(
        monkeypatch,
        tmp_path,
        parsed_records=[_base_record(bil="1")],
        refine_impl=fake_refine_field,
        validate_impl=fake_gemini_processor,
        llm_enabled=True,
    )

    assert seen == ["AHMAD BIN ALI"]
    assert result.records[0].nama_suami == "AHMAD BIN ALI"


def test_process_command_prints_refinement_summary(monkeypatch, tmp_path: Path) -> None:
    input_path = tmp_path / "input.pdf"
    input_path.write_bytes(b"pdf")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        "marriage_ocr.cli._load_command_runtime",
        lambda command_name, config_path: (
            {"debug": {"retain_artifacts": False}, "ocr": {"engine": "fake"}},
            SimpleNamespace(data={}, env_file=None),
            SimpleNamespace(log_path=tmp_path / "process.log"),
        ),
    )
    monkeypatch.setattr(
        "marriage_ocr.cli.process_input",
        lambda **kwargs: pipeline.ProcessResult(
            records=[],
            total_pages=1,
            total_detected_records=1,
            total_parsed_records=1,
            status_counts={"OK": 1},
            output_path=tmp_path / "output.xlsx",
            debug_path=tmp_path / "debug",
            refinement_ocr_calls=4,
        ),
    )

    result = runner.invoke(
        app,
        [
            "process",
            "--input",
            str(input_path),
            "--output",
            str(tmp_path / "output.xlsx"),
            "--config",
            str(config_path),
        ],
    )

    assert result.exit_code == 0
    assert "Refinement retry OCR calls: 4" in result.stdout
