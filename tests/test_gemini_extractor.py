from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from marriage_ocr.models import OcrResult
from llm.gemini_extractor import GeminiRecordExtractor
from llm.gemini_extractor import _resolve_api_key_source


def test_payload_to_result_accepts_field_confidence_entries() -> None:
    extractor = GeminiRecordExtractor.__new__(GeminiRecordExtractor)

    result = GeminiRecordExtractor._payload_to_result(
        extractor,
        {
            "bil": "1",
            "field_confidence": [
                {"field": "bil", "confidence": 0.91},
                {"field": "nama_suami", "confidence": 0.82},
            ],
            "uncertain_fields": ["nama_suami"],
            "notes": ["checked"],
        },
    )

    assert result.record.bil == "1"
    assert result.field_confidence == {"bil": 0.91, "nama_suami": 0.82}
    assert result.record.confidence == 0.865
    assert result.uncertain_fields == ["nama_suami"]
    assert result.notes == ["checked"]


def test_resolve_api_key_source_prefers_config_then_env(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert _resolve_api_key_source({}) == (None, None)

    monkeypatch.setenv("GEMINI_API_KEY", "env-key")
    assert _resolve_api_key_source({}) == ("GEMINI_API_KEY", "env-key")
    assert _resolve_api_key_source({"api_key": "config-key"}) == ("config.api_key", "config-key")


def test_extract_record_uses_single_client(tmp_path: Path) -> None:
    class FakePart:
        @staticmethod
        def from_bytes(*, data: bytes, mime_type: str) -> dict[str, object]:
            return {"data": data, "mime_type": mime_type}

    class FakeGenerateContentConfig:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class FakeTypes:
        Part = FakePart
        GenerateContentConfig = FakeGenerateContentConfig

    class Models:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def generate_content(self, *, model: str, contents: list[object], config: object) -> SimpleNamespace:
            self.calls.append({"model": model, "contents": contents, "config": config})
            return SimpleNamespace(text=json.dumps({"bil": "7"}))

    class FakeClient:
        def __init__(self, models: object) -> None:
            self.models = models

    models = Models()

    extractor = GeminiRecordExtractor.__new__(GeminiRecordExtractor)
    extractor.model = "gemini-2.5-flash"
    extractor.temperature = 0.0
    extractor.max_output_tokens = 4096
    extractor.save_raw_json = False
    extractor._types = FakeTypes()
    extractor._client = FakeClient(models)
    extractor._api_key_source = "GEMINI_API_KEY"
    extractor._build_prompt = lambda ocr_cells: "prompt"

    crop_path = tmp_path / "record.jpg"
    crop_path.write_bytes(b"fake-image")

    result = extractor.extract_record(
        record_crop_path=crop_path,
        ocr_cells={"bil": OcrResult(text="7", average_confidence=0.95)},
    )

    assert result.record.bil == "7"
    assert len(models.calls) == 1
    assert models.calls[0]["model"] == "gemini-2.5-flash"
