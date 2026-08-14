from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from marriage_ocr.models import ExtractedRecord, OcrResult
from marriage_ocr import pipeline


@dataclass
class _FakeRecordOutput:
    record_dir: Path
    cell_results: dict[str, OcrResult]


def test_build_gemini_record_processor_returns_none_when_disabled() -> None:
    assert pipeline._build_gemini_record_processor({"enabled": False}, validation_config={}) is None


def test_build_gemini_record_processor_uses_full_record_and_merge(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    class FakeGeminiExtractor:
        def __init__(self, config):
            calls["config"] = dict(config)

        def extract_record(self, *, record_crop_path, ocr_cells):
            calls["record_crop_path"] = Path(record_crop_path)
            calls["ocr_cells"] = dict(ocr_cells)
            return object()

    def fake_merge_parser_and_gemini(
        *,
        parser_record,
        gemini_result,
        cell_results,
        validation_config,
        layout_confidence,
        prefer_gemini_threshold,
        review_below_field_confidence,
    ):
        calls["merge"] = {
            "parser_record": parser_record,
            "gemini_result": gemini_result,
            "cell_results": dict(cell_results),
            "validation_config": dict(validation_config),
            "layout_confidence": layout_confidence,
            "prefer_gemini_threshold": prefer_gemini_threshold,
            "review_below_field_confidence": review_below_field_confidence,
        }
        return ExtractedRecord(
            bil="1",
            confidence=0.91,
            status_review="OK",
            review_reason=["merged"],
        )

    monkeypatch.setattr(pipeline, "GeminiRecordExtractor", FakeGeminiExtractor)
    monkeypatch.setattr(pipeline, "merge_parser_and_gemini", fake_merge_parser_and_gemini)

    record_dir = tmp_path / "page_001" / "records" / "record_001"
    record_dir.mkdir(parents=True)
    (record_dir / "full_record.jpg").write_bytes(b"fake-image")

    processor = pipeline._build_gemini_record_processor(
        {
            "enabled": True,
            "provider": "gemini",
            "model": "gemini-2.5-flash",
            "prefer_gemini_threshold": 0.7,
            "review_below_field_confidence": 0.8,
        },
        validation_config={"ok_confidence_threshold": 0.9},
    )

    assert processor is not None
    parser_record = ExtractedRecord(bil="1")
    record_output = _FakeRecordOutput(
        record_dir=record_dir,
        cell_results={"bil": OcrResult(text="1", average_confidence=0.95)},
    )

    merged = processor(parser_record, record_output, layout_confidence=0.88)

    assert merged.confidence == 0.91
    assert merged.status_review == "OK"
    assert calls["config"]["model"] == "gemini-2.5-flash"
    assert calls["record_crop_path"] == record_dir / "full_record.jpg"
    assert calls["ocr_cells"]["bil"].text == "1"
    assert calls["merge"]["layout_confidence"] == 0.88
    assert calls["merge"]["prefer_gemini_threshold"] == 0.7
    assert calls["merge"]["review_below_field_confidence"] == 0.8


def test_build_gemini_record_processor_rejects_unsupported_provider(monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "GeminiRecordExtractor", object)

    try:
        pipeline._build_gemini_record_processor(
            {"enabled": True, "provider": "ollama"},
            validation_config={},
        )
    except ValueError as error:
        assert "Unsupported llm.provider value" in str(error)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected ValueError for unsupported provider")


def test_validate_record_with_optional_gemini_falls_back_when_gemini_fails() -> None:
    class FakeLogger:
        def __init__(self) -> None:
            self.messages: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def warning(self, *args: object, **kwargs: object) -> None:
            self.messages.append((args, kwargs))

    def failing_processor(*args, **kwargs):
        raise RuntimeError("leaked api key")

    parsed_record = ExtractedRecord(
        bil="1",
        nama_suami="ALI BIN ABU",
        ic_lama_suami="A.1234567",
        umur_suami=30,
        nama_isteri="SITI BINTI ALI",
        ic_baru_isteri="900101-01-1234",
        umur_isteri=28,
        mas_kahwin="RM100",
        mas_kahwin_raw="RM100",
        nama_pendaftar="ABDUL",
        alamat_pendaftar="KUALA LUMPUR",
        nama_wali="AHMAD",
        hubungan_wali="BAPA",
        saksi_1="HASHIM",
        saksi_2="RAHMAN",
        tarikh_nikah="2024-01-01",
    )
    record_output = _FakeRecordOutput(
        record_dir=Path("."),
        cell_results={"bil": OcrResult(text="1", average_confidence=0.95)},
    )

    validated = pipeline._validate_record_with_optional_gemini(
        parsed_record=parsed_record,
        record_output=record_output,
        layout_confidence=1.0,
        gemini_processor=failing_processor,
        gemini_state={},
        validation_config={"ok_confidence_threshold": 0.85, "min_average_confidence": 0.5},
        logger=FakeLogger(),
        source_file="input.pdf",
        source_page=1,
    )

    assert validated.status_review == "REVIEW"
    assert "Gemini unavailable: RuntimeError" in validated.review_reason


def test_validate_record_with_optional_gemini_disables_gemini_after_leaked_key() -> None:
    class FakeLogger:
        def __init__(self) -> None:
            self.messages: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def warning(self, *args: object, **kwargs: object) -> None:
            self.messages.append((args, kwargs))

    calls = {"count": 0}

    def failing_processor(*args, **kwargs):
        calls["count"] += 1
        raise RuntimeError("403 PERMISSION_DENIED. Your API key was reported as leaked.")

    parsed_record = ExtractedRecord(
        bil="1",
        nama_suami="ALI BIN ABU",
        ic_lama_suami="A1234567",
        umur_suami=30,
        nama_isteri="SITI BINTI ALI",
        ic_baru_isteri="900101-01-1234",
        umur_isteri=28,
        mas_kahwin="RM100",
        mas_kahwin_raw="RM100",
        nama_pendaftar="ABDUL",
        alamat_pendaftar="KUALA LUMPUR",
        nama_wali="AHMAD",
        hubungan_wali="BAPA",
        saksi_1="HASHIM",
        saksi_2="RAHMAN",
        tarikh_nikah="2024-01-01",
    )
    record_output = _FakeRecordOutput(
        record_dir=Path("."),
        cell_results={"bil": OcrResult(text="1", average_confidence=0.95)},
    )
    gemini_state: dict[str, bool] = {"disabled": False}
    logger = FakeLogger()

    first = pipeline._validate_record_with_optional_gemini(
        parsed_record=parsed_record,
        record_output=record_output,
        layout_confidence=1.0,
        gemini_processor=failing_processor,
        gemini_state=gemini_state,
        validation_config={"ok_confidence_threshold": 0.85, "min_average_confidence": 0.5},
        logger=logger,
        source_file="input.pdf",
        source_page=1,
    )
    second = pipeline._validate_record_with_optional_gemini(
        parsed_record=parsed_record,
        record_output=record_output,
        layout_confidence=1.0,
        gemini_processor=failing_processor,
        gemini_state=gemini_state,
        validation_config={"ok_confidence_threshold": 0.85, "min_average_confidence": 0.5},
        logger=logger,
        source_file="input.pdf",
        source_page=1,
    )

    assert calls["count"] == 1
    assert gemini_state["disabled"] is True
    assert first.status_review == "REVIEW"
    assert second.status_review == "OK"
    assert any("Gemini disabled for the remainder of this run" in str(args[0]) for args, _ in logger.messages)


def test_validate_record_with_optional_gemini_does_not_disable_run_on_rate_limit() -> None:
    """Regression: a single transient 429/RESOURCE_EXHAUSTED used to be
    lumped in with genuinely permanent errors (revoked key, no permission)
    in _should_disable_gemini_for_run, permanently disabling Gemini for
    every subsequent record in the run. GeminiRecordExtractor now retries
    transient errors on its own (see gemini_extractor.py); by the time an
    error reaches here, retrying already failed for *that* record, but a
    rate-limit blip says nothing about whether the *next* record's call
    will succeed. Gemini must stay enabled so later records still get a
    chance at higher-accuracy extraction.
    """
    class FakeLogger:
        def warning(self, *args: object, **kwargs: object) -> None:
            pass

    calls = {"count": 0}

    def failing_processor(*args, **kwargs):
        calls["count"] += 1
        raise RuntimeError("429 RESOURCE_EXHAUSTED. Quota exceeded, too many requests.")

    parsed_record = ExtractedRecord(
        bil="1",
        nama_suami="ALI BIN ABU",
        ic_lama_suami="A1234567",
        umur_suami=30,
        nama_isteri="SITI BINTI ALI",
        ic_baru_isteri="900101-01-1234",
        umur_isteri=28,
        mas_kahwin="RM100",
        mas_kahwin_raw="RM100",
        nama_pendaftar="ABDUL",
        alamat_pendaftar="KUALA LUMPUR",
        nama_wali="AHMAD",
        hubungan_wali="BAPA",
        saksi_1="HASHIM",
        saksi_2="RAHMAN",
        tarikh_nikah="2024-01-01",
    )
    record_output = _FakeRecordOutput(
        record_dir=Path("."),
        cell_results={"bil": OcrResult(text="1", average_confidence=0.95)},
    )
    gemini_state: dict[str, bool] = {"disabled": False}

    first = pipeline._validate_record_with_optional_gemini(
        parsed_record=parsed_record,
        record_output=record_output,
        layout_confidence=1.0,
        gemini_processor=failing_processor,
        gemini_state=gemini_state,
        validation_config={"ok_confidence_threshold": 0.85, "min_average_confidence": 0.5},
        logger=FakeLogger(),
        source_file="input.pdf",
        source_page=1,
    )
    second = pipeline._validate_record_with_optional_gemini(
        parsed_record=parsed_record,
        record_output=record_output,
        layout_confidence=1.0,
        gemini_processor=failing_processor,
        gemini_state=gemini_state,
        validation_config={"ok_confidence_threshold": 0.85, "min_average_confidence": 0.5},
        logger=FakeLogger(),
        source_file="input.pdf",
        source_page=1,
    )

    assert calls["count"] == 2
    assert gemini_state["disabled"] is False
    assert first.status_review == "REVIEW"
    assert second.status_review == "REVIEW"


def test_validate_record_with_optional_gemini_receives_refined_record() -> None:
    seen: list[str | None] = []

    def gemini_processor(parsed_record, record_output, *, layout_confidence):
        seen.append(parsed_record.nama_suami)
        return ExtractedRecord(
            **{
                **parsed_record.to_dict(),
                "status_review": "OK",
                "review_reason": [],
                "confidence": 0.97,
            }
        )

    refined_record = ExtractedRecord(
        bil="1",
        nama_suami="AHMAD BIN ALI",
        ic_baru_suami="900101-01-1234",
        umur_suami=30,
        nama_isteri="SITI BINTI ALI",
        ic_baru_isteri="900101-01-5678",
        umur_isteri=28,
        mas_kahwin="RM100",
        mas_kahwin_raw="RM100",
        nama_pendaftar="ABDUL",
        alamat_pendaftar="KUALA LUMPUR",
        nama_wali="AHMAD",
        hubungan_wali="BAPA",
        saksi_1="HASHIM",
        saksi_2="RAHMAN",
        tarikh_nikah="2024-01-03",
    )
    record_output = _FakeRecordOutput(
        record_dir=Path("."),
        cell_results={"bil": OcrResult(text="1", average_confidence=0.95)},
    )

    validated = pipeline._validate_record_with_optional_gemini(
        parsed_record=refined_record,
        record_output=record_output,
        layout_confidence=1.0,
        gemini_processor=gemini_processor,
        gemini_state={},
        validation_config={"ok_confidence_threshold": 0.85, "min_average_confidence": 0.5},
        logger=object(),
        source_file="input.pdf",
        source_page=1,
    )

    assert seen == ["AHMAD BIN ALI"]
    assert validated.nama_suami == "AHMAD BIN ALI"


def test_validate_record_with_optional_gemini_accepts_refined_record() -> None:
    class FakeLogger:
        def warning(self, *args: object, **kwargs: object) -> None:
            pass

    def failing_processor(parsed_record, record_output, *, layout_confidence):
        assert parsed_record.nama_suami == "AHMAD BIN ALI"
        raise RuntimeError("leaked api key")

    parsed_record = ExtractedRecord(
        bil="1",
        nama_suami="AHMAD BIN ALI",
        ic_lama_suami="A.1234567",
        umur_suami=30,
        nama_isteri="SITI BINTI ALI",
        ic_baru_isteri="900101-01-1234",
        umur_isteri=28,
        mas_kahwin="RM100",
        mas_kahwin_raw="RM100",
        nama_pendaftar="ABDUL",
        alamat_pendaftar="KUALA LUMPUR",
        nama_wali="AHMAD",
        hubungan_wali="BAPA",
        saksi_1="HASHIM",
        saksi_2="RAHMAN",
        tarikh_nikah="2024-01-01",
    )
    record_output = _FakeRecordOutput(
        record_dir=Path("."),
        cell_results={"bil": OcrResult(text="1", average_confidence=0.95)},
    )

    validated = pipeline._validate_record_with_optional_gemini(
        parsed_record=parsed_record,
        record_output=record_output,
        layout_confidence=1.0,
        gemini_processor=failing_processor,
        gemini_state={},
        validation_config={"ok_confidence_threshold": 0.85, "min_average_confidence": 0.5},
        logger=FakeLogger(),
        source_file="input.pdf",
        source_page=1,
    )

    assert validated.status_review == "REVIEW"
    assert "Gemini unavailable: RuntimeError" in validated.review_reason
