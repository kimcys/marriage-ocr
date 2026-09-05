from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from marriage_ocr.models import ExtractedRecord
from llm.gemini_batch_extractor import (
    BatchRecordItem,
    GeminiBatchRecordExtractor,
    discover_batch_items,
    run_batch_extraction,
)


def _write_record(
    record_dir: Path,
    *,
    bil: str = "1",
    include_parsed: bool = True,
) -> None:
    record_dir.mkdir(parents=True, exist_ok=True)
    (record_dir / "full_record.jpg").write_bytes(b"fake-image")
    (record_dir / "raw_ocr.json").write_text(
        json.dumps(
            {
                "engine": "google_vision",
                "cells": {
                    "bil": {"text": bil, "average_confidence": 0.9, "lines": []},
                    "suami_isteri": {"text": "AHMAD BIN ALI", "average_confidence": 0.8, "lines": []},
                },
            }
        ),
        encoding="utf-8",
    )
    if include_parsed:
        record = ExtractedRecord(bil=bil, nama_suami="AHMAD BIN ALI")
        (record_dir / "parsed_record.json").write_text(
            json.dumps(record.to_dict(), ensure_ascii=False), encoding="utf-8"
        )


def test_discover_batch_items_finds_only_complete_record_dirs(tmp_path: Path) -> None:
    _write_record(tmp_path / "page_001" / "record_001")
    _write_record(tmp_path / "page_001" / "record_002")
    (tmp_path / "page_001" / "record_003").mkdir(parents=True)  # missing artifacts

    items = discover_batch_items(tmp_path)

    assert sorted(item.key for item in items) == [
        "page_001/record_001",
        "page_001/record_002",
    ]
    assert items[0].ocr_cells["bil"].text == "1"
    assert items[0].ocr_cells["suami_isteri"].text == "AHMAD BIN ALI"


class FakePart:
    @staticmethod
    def from_bytes(*, data: bytes, mime_type: str) -> dict[str, object]:
        return {"data": data, "mime_type": mime_type}


class FakeGenerateContentConfig:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class FakeInlinedRequest:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class FakeCreateBatchJobConfig:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class FakeTypes:
    Part = FakePart
    GenerateContentConfig = FakeGenerateContentConfig
    InlinedRequest = FakeInlinedRequest
    CreateBatchJobConfig = FakeCreateBatchJobConfig


def _make_extractor(**overrides) -> GeminiBatchRecordExtractor:
    extractor = GeminiBatchRecordExtractor.__new__(GeminiBatchRecordExtractor)
    extractor.model = "gemini-2.5-flash"
    extractor.temperature = 0.0
    extractor.max_output_tokens = 4096
    extractor.record_type = "nikah"
    extractor._types = FakeTypes()
    extractor._build_prompt = lambda ocr_cells: "prompt"
    for key, value in overrides.items():
        setattr(extractor, key, value)
    return extractor


def test_build_request_wires_prompt_image_config_and_key_metadata(tmp_path: Path) -> None:
    record_dir = tmp_path / "record_001"
    _write_record(record_dir)
    item = BatchRecordItem(key="record_001", record_dir=record_dir, ocr_cells={})

    extractor = _make_extractor()
    request = extractor.build_request(item)

    assert request.kwargs["contents"] == ["prompt", {"data": b"fake-image", "mime_type": "image/jpeg"}]
    assert request.kwargs["metadata"] == {"key": "record_001"}
    assert isinstance(request.kwargs["config"], FakeGenerateContentConfig)


def test_submit_calls_batches_create_with_model_and_requests(tmp_path: Path) -> None:
    record_dir = tmp_path / "record_001"
    _write_record(record_dir)
    item = BatchRecordItem(key="record_001", record_dir=record_dir, ocr_cells={})

    calls = []

    class FakeBatches:
        def create(self, *, model, src, config):
            calls.append({"model": model, "src": src, "config": config})
            return SimpleNamespace(name="batches/123")

    extractor = _make_extractor(_client=SimpleNamespace(batches=FakeBatches()))
    job = extractor.submit([item], display_name="test-batch")

    assert job.name == "batches/123"
    assert calls[0]["model"] == "gemini-2.5-flash"
    assert len(calls[0]["src"]) == 1
    assert calls[0]["config"].kwargs["display_name"] == "test-batch"


def test_wait_polls_until_terminal_state_then_stops() -> None:
    states = iter(["JOB_STATE_QUEUED", "JOB_STATE_RUNNING", "JOB_STATE_SUCCEEDED"])
    get_calls = []

    class FakeBatches:
        def get(self, *, name):
            get_calls.append(name)
            return SimpleNamespace(name=name, state=next(states))

    sleeps: list[float] = []
    extractor = _make_extractor(_client=SimpleNamespace(batches=FakeBatches()))

    job = extractor.wait(
        SimpleNamespace(name="batches/123"), poll_seconds=5.0, sleep=sleeps.append
    )

    assert str(job.state) == "JOB_STATE_SUCCEEDED"
    assert len(get_calls) == 3
    assert sleeps == [5.0, 5.0]


def test_wait_raises_timeout_error_when_never_terminal() -> None:
    class FakeBatches:
        def get(self, *, name):
            return SimpleNamespace(name=name, state="JOB_STATE_RUNNING")

    extractor = _make_extractor(_client=SimpleNamespace(batches=FakeBatches()))

    with pytest.raises(TimeoutError):
        extractor.wait(
            SimpleNamespace(name="batches/123"),
            poll_seconds=10.0,
            timeout_seconds=15.0,
            sleep=lambda _seconds: None,
        )


def test_collect_results_parses_success_and_records_per_item_error() -> None:
    ok_response = SimpleNamespace(text=json.dumps({"bil": "1"}), parsed=None)
    job = SimpleNamespace(
        name="batches/123",
        state="JOB_STATE_PARTIALLY_SUCCEEDED",
        error=None,
        dest=SimpleNamespace(
            inlined_responses=[
                SimpleNamespace(metadata={"key": "record_001"}, error=None, response=ok_response),
                SimpleNamespace(
                    metadata={"key": "record_002"},
                    error="RESOURCE_EXHAUSTED",
                    response=None,
                ),
            ]
        ),
    )

    extractor = _make_extractor()
    results = extractor.collect_results(job)

    assert results["record_001"].record.bil == "1"
    assert isinstance(results["record_002"], Exception)
    assert "RESOURCE_EXHAUSTED" in str(results["record_002"])


def test_collect_results_raises_on_failed_job_state() -> None:
    job = SimpleNamespace(name="batches/123", state="JOB_STATE_FAILED", error="quota exceeded", dest=None)
    extractor = _make_extractor()

    with pytest.raises(RuntimeError, match="quota exceeded"):
        extractor.collect_results(job)


def test_run_batch_extraction_merges_results_and_writes_validated_record(tmp_path: Path) -> None:
    record_dir = tmp_path / "record_001"
    _write_record(record_dir, bil="1")

    gemini_payload = {
        "bil": "1",
        "nama_suami": "AHMAD BIN ALI",
        "field_confidence": [{"field": "bil", "confidence": 0.95}],
        "uncertain_fields": [],
        "notes": [],
    }
    ok_response = SimpleNamespace(text=json.dumps(gemini_payload), parsed=None)
    job = SimpleNamespace(
        name="batches/123",
        state="JOB_STATE_SUCCEEDED",
        error=None,
        dest=SimpleNamespace(
            inlined_responses=[
                SimpleNamespace(metadata={"key": "record_001"}, error=None, response=ok_response),
            ]
        ),
    )

    extractor = _make_extractor()
    extractor.submit = lambda items, display_name=None: job
    extractor.wait = lambda job, **kwargs: job

    summary = run_batch_extraction(
        tmp_path,
        llm_config={},
        validation_config={},
        extractor=extractor,
    )

    assert summary.submitted == 1
    assert summary.merged == 1
    assert not summary.failed
    validated_path = record_dir / "validated_record.json"
    assert validated_path.exists()
    saved = json.loads(validated_path.read_text(encoding="utf-8"))
    assert saved["bil"] == "1"
    assert saved["nama_suami"] == "AHMAD BIN ALI"


def test_run_batch_extraction_skips_records_without_parsed_record(tmp_path: Path) -> None:
    _write_record(tmp_path / "record_001", include_parsed=False)

    extractor = _make_extractor()
    extractor.submit = lambda items, display_name=None: (_ for _ in ()).throw(
        AssertionError("submit should not be called when nothing is submittable")
    )

    summary = run_batch_extraction(
        tmp_path,
        llm_config={},
        validation_config={},
        extractor=extractor,
    )

    assert summary.submitted == 1
    assert summary.skipped_no_parsed_record == ["record_001"]
    assert summary.merged == 0


def test_run_batch_extraction_skips_parser_confident_nikah_records(tmp_path: Path, monkeypatch) -> None:
    record_dir = tmp_path / "record_001"
    _write_record(record_dir, bil="1")

    monkeypatch.setattr(
        "llm.gemini_batch_extractor.validate_record",
        lambda *args, **kwargs: ExtractedRecord(bil="1", status_review="OK", confidence=0.95),
    )

    extractor = _make_extractor()
    extractor.submit = lambda items, display_name=None: (_ for _ in ()).throw(
        AssertionError("submit should not be called when the parser already clears the threshold")
    )

    summary = run_batch_extraction(
        tmp_path,
        llm_config={},
        validation_config={},
        record_type="nikah",
        extractor=extractor,
        skip_gemini_when_parser_ok=True,
        skip_gemini_min_confidence=0.90,
    )

    assert summary.submitted == 1
    assert summary.skipped_parser_confident == ["record_001"]
    assert summary.merged == 0
    assert not summary.failed
    saved = json.loads((record_dir / "validated_record.json").read_text(encoding="utf-8"))
    assert saved["status_review"] == "OK"
    assert saved["confidence"] == 0.95


def test_run_batch_extraction_falls_back_to_parser_only_on_item_error(tmp_path: Path) -> None:
    record_dir = tmp_path / "record_001"
    _write_record(record_dir, bil="1")

    job = SimpleNamespace(
        name="batches/123",
        state="JOB_STATE_PARTIALLY_SUCCEEDED",
        error=None,
        dest=SimpleNamespace(
            inlined_responses=[
                SimpleNamespace(metadata={"key": "record_001"}, error="rate limited", response=None),
            ]
        ),
    )

    extractor = _make_extractor()
    extractor.submit = lambda items, display_name=None: job
    extractor.wait = lambda job, **kwargs: job

    summary = run_batch_extraction(
        tmp_path,
        llm_config={},
        validation_config={},
        extractor=extractor,
    )

    assert summary.merged == 0
    assert "record_001" in summary.failed
    saved = json.loads((record_dir / "validated_record.json").read_text(encoding="utf-8"))
    assert saved["bil"] == "1"
    assert any("Gemini batch unavailable" in reason for reason in saved["review_reason"])
