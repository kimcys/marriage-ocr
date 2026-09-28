from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from marriage_ocr import gemini_page_batch, gemini_page_pipeline
from marriage_ocr.models import ExtractedRecord

_GEMINI_PAGE_CONFIG = {
    "pipeline": {"engine": "gemini_page"},
    "record_type": "nikah",
    "layout_variant": "legacy",
    "input": {"allowed_extensions": []},
    "preprocessing": {},
    "validation": {"ok_confidence_threshold": 0.85, "min_age": 15, "max_age": 100},
    "postprocess": {"bil_sequence": {"enabled": False}},
    "llm": {"model": "gemini-3.5-flash-lite"},
    "debug": {"retain_artifacts": False},
    "export": {},
}


def _pages():
    return [
        SimpleNamespace(
            debug_name=f"page_{n:03d}",
            image=np.zeros((16, 16, 3), dtype=np.uint8),
            relative_source=Path("ledger.jpg"),
            source_page=n,
        )
        for n in (1, 2)
    ]


class FakeExtractor:
    model = "gemini-3.5-flash-lite"
    temperature = 0.0
    max_output_tokens = 4096

    def __init__(self, config):
        self.config = config

    def _page_prompt(self):
        return "PAGE PROMPT"

    def _page_schema(self):
        return {"type": "OBJECT", "properties": {"records": {"type": "ARRAY"}}}

    def extract_page(self, image_path):
        raise AssertionError("batch mode must never make a synchronous Gemini call")

    def _payload_to_result(self, payload):
        return SimpleNamespace(
            record=ExtractedRecord(bil=payload["bil"], nama_suami=payload["nama_suami"]),
            field_confidence={},
            uncertain_fields=[],
            notes=[],
        )


@pytest.fixture
def pipeline(monkeypatch):
    config = SimpleNamespace(data=_GEMINI_PAGE_CONFIG, env_file=None)
    monkeypatch.setattr(gemini_page_pipeline, "load_runtime_config", lambda _: config)
    monkeypatch.setattr(gemini_page_batch, "load_runtime_config", lambda _: config)
    monkeypatch.setattr(gemini_page_pipeline, "GeminiPageExtractor", FakeExtractor)

    def fake_write_image(path, image, **kw):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"jpeg-bytes-" + path.parent.name.encode())

    monkeypatch.setattr("marriage_ocr.document_loader.load_document_pages", lambda *a, **kw: _pages())
    monkeypatch.setattr("marriage_ocr.document_loader.write_image", fake_write_image)
    monkeypatch.setattr(
        "marriage_ocr.preprocess.preprocess_image",
        lambda image, settings: SimpleNamespace(color=image, binary=np.zeros((16, 16), dtype=np.uint8)),
    )
    monkeypatch.setattr("marriage_ocr.parser.save_parsed_record", lambda *a, **kw: None)
    exported: list[list[ExtractedRecord]] = []

    def fake_export(records, output_path, export_config, *, reset_output, skip_existing):
        exported.append(list(records))
        return SimpleNamespace(written_count=len(records), skipped_duplicates=0, output_path=output_path)

    monkeypatch.setattr("marriage_ocr.exporter.export_records_to_xlsx", fake_export)
    return exported


def test_prepare_saves_each_preprocessed_page_and_the_exact_request(pipeline, tmp_path: Path) -> None:
    out = tmp_path / "prepared"
    manifest = gemini_page_batch.prepare(tmp_path / "ledger.jpg", tmp_path / "cfg.yaml", out)

    assert [page["index"] for page in manifest["pages"]] == [1, 2]
    assert (out / "page_1.jpg").read_bytes() == b"jpeg-bytes-page_001"
    assert (out / "page_2.jpg").read_bytes() == b"jpeg-bytes-page_002"
    assert manifest["model"] == "gemini-3.5-flash-lite"
    request = manifest["request"]
    assert request["contents"][0]["parts"] == [{"text": "PAGE PROMPT"}]
    assert request["generation_config"] == {
        "temperature": 0.0,
        "max_output_tokens": 20480,  # max(4096 * 5, 16384), as extract_page uses
        "response_mime_type": "application/json",
        "response_schema": {"type": "OBJECT", "properties": {"records": {"type": "ARRAY"}}},
    }
    assert json.loads((out / "manifest.json").read_text())["pages"] == manifest["pages"]


def test_prepare_refuses_non_gemini_page_configs(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        gemini_page_batch,
        "load_runtime_config",
        lambda _: SimpleNamespace(data={"pipeline": {"engine": "vision"}}, env_file=None),
    )
    with pytest.raises(gemini_page_batch.BatchNotSupported):
        gemini_page_batch.prepare(tmp_path / "x.jpg", tmp_path / "cfg.yaml", tmp_path / "out")


def _prepared(pipeline_fixture, tmp_path: Path) -> Path:
    out = tmp_path / "prepared"
    gemini_page_batch.prepare(tmp_path / "ledger.jpg", tmp_path / "cfg.yaml", out)
    return out


def test_finish_runs_the_pipeline_from_batch_payloads(pipeline, tmp_path: Path) -> None:
    prepared = _prepared(pipeline, tmp_path)
    payloads = tmp_path / "payloads"
    payloads.mkdir()
    for index, name in ((1, "AHMAD BIN ALI"), (2, "OSMAN BIN DIN")):
        (payloads / f"job-1__{index}.json").write_text(
            json.dumps({"records": [{"bil": f"{index}/94", "nama_suami": name}]})
        )

    result = gemini_page_batch.finish(
        input_path=tmp_path / "ledger.jpg",
        config_path=tmp_path / "cfg.yaml",
        prepared_dir=prepared,
        payload_dir=payloads,
        key_prefix="job-1",
        output_path=tmp_path / "result.xlsx",
        debug_path=tmp_path / "debug",
    )

    assert [r.nama_suami for r in result.records] == ["AHMAD BIN ALI", "OSMAN BIN DIN"]
    assert [[r.nama_suami for r in call] for call in pipeline] == [["AHMAD BIN ALI", "OSMAN BIN DIN"]]


def test_finish_refuses_when_a_page_has_no_result(pipeline, tmp_path: Path) -> None:
    prepared = _prepared(pipeline, tmp_path)
    payloads = tmp_path / "payloads"
    payloads.mkdir()
    (payloads / "job-1__1.json").write_text(json.dumps({"records": []}))

    with pytest.raises(gemini_page_batch.MissingBatchResult):
        gemini_page_batch.finish(
            input_path=tmp_path / "ledger.jpg",
            config_path=tmp_path / "cfg.yaml",
            prepared_dir=prepared,
            payload_dir=payloads,
            key_prefix="job-1",
            output_path=tmp_path / "result.xlsx",
            debug_path=tmp_path / "debug",
        )
    assert pipeline == []  # nothing exported


class FakeClient:
    def __init__(self, state="JOB_STATE_SUCCEEDED", result_lines=()):
        self.uploaded: list[tuple[str, object]] = []
        self.created: list[dict] = []
        self._state = state
        self._result_lines = result_lines
        outer = self

        class Files:
            def upload(self, *, file, config):
                outer.uploaded.append((file, config))
                n = len(outer.uploaded)
                return SimpleNamespace(uri=f"https://files/{n}", name=f"files/{n}")

            def download(self, *, file):
                assert file == "files/results"
                return "\n".join(json.dumps(line) for line in outer._result_lines).encode()

        class Batches:
            def create(self, *, model, src, config):
                outer.created.append({"model": model, "src": src, "config": config})
                return SimpleNamespace(name="batches/abc")

            def get(self, *, name):
                return SimpleNamespace(
                    name=name, state=SimpleNamespace(name=outer._state), dest=SimpleNamespace(file_name="files/results")
                )

        self.files = Files()
        self.batches = Batches()


def test_submit_writes_one_jsonl_line_per_page_referencing_its_uploaded_image(pipeline, tmp_path: Path) -> None:
    prepared = _prepared(pipeline, tmp_path)
    client = FakeClient()

    batches = gemini_page_batch.submit(
        [gemini_page_batch.SubmitItem("job-1", prepared)], display_name="test", client=client
    )

    assert batches == [{"name": "batches/abc", "model": "gemini-3.5-flash-lite", "keys": ["job-1"]}]
    images = [Path(f).name for f, _ in client.uploaded[:2]]
    assert images == ["page_1.jpg", "page_2.jpg"]
    jsonl_path, jsonl_config = client.uploaded[2]
    assert client.created == [{"model": "gemini-3.5-flash-lite", "src": "files/3", "config": {"display_name": "test"}}]
    assert jsonl_config.mime_type == "jsonl"


def test_collect_writes_payloads_and_errors_per_page(tmp_path: Path) -> None:
    ok_response = {"candidates": [{"content": {"parts": [{"text": json.dumps({"records": [{"bil": "1"}]})}]}}]}
    client = FakeClient(
        result_lines=[
            {"key": "job-1__1", "response": ok_response},
            {"key": "job-1__2", "error": {"code": 400, "message": "bad image"}},
        ]
    )

    summary = gemini_page_batch.collect("batches/abc", tmp_path, client=client)

    assert summary["done"] and summary["ok"] == ["job-1__1"]
    assert "bad image" in summary["errors"]["job-1__2"]
    assert json.loads((tmp_path / "job-1__1.json").read_text()) == {"records": [{"bil": "1"}]}
    assert (tmp_path / "job-1__2.error").exists()


def test_collect_does_nothing_while_the_batch_is_still_running(tmp_path: Path) -> None:
    summary = gemini_page_batch.collect("batches/abc", tmp_path / "out", client=FakeClient(state="JOB_STATE_RUNNING"))
    assert summary == {"state": "JOB_STATE_RUNNING", "done": False, "ok": [], "errors": {}}
    assert not (tmp_path / "out").exists()
