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

    class FakeHttpOptions:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class FakeTypes:
        Part = FakePart
        GenerateContentConfig = FakeGenerateContentConfig
        HttpOptions = FakeHttpOptions

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


def test_extract_record_retries_transient_rate_limit_before_succeeding(tmp_path: Path) -> None:
    """Regression: GeminiRecordExtractor._generate_content used to call
    generate_content once with no retry. A single transient 429
    (RESOURCE_EXHAUSTED) propagated straight to pipeline.py, which not only
    fell back to parser-only validation for that record but (previously)
    permanently disabled Gemini for the rest of the run -- see
    pipeline.py::_should_disable_gemini_for_run. Retrying the call here
    first means a routine rate-limit blip no longer costs an entire run's
    worth of Gemini corroboration.
    """
    from google.genai import errors as genai_errors

    class FakePart:
        @staticmethod
        def from_bytes(*, data: bytes, mime_type: str) -> dict[str, object]:
            return {"data": data, "mime_type": mime_type}

    class FakeGenerateContentConfig:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class FakeHttpOptions:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class FakeTypes:
        Part = FakePart
        GenerateContentConfig = FakeGenerateContentConfig
        HttpOptions = FakeHttpOptions

    call_count = {"n": 0}
    seen_configs: list[object] = []

    class Models:
        def generate_content(self, *, model: str, contents: list[object], config: object) -> SimpleNamespace:
            call_count["n"] += 1
            seen_configs.append(config)
            if call_count["n"] < 3:
                raise genai_errors.APIError(429, {"error": {"message": "rate limited"}})
            return SimpleNamespace(text=json.dumps({"bil": "7"}))

    class FakeClient:
        def __init__(self, models: object) -> None:
            self.models = models

    extractor = GeminiRecordExtractor.__new__(GeminiRecordExtractor)
    extractor.model = "gemini-2.5-flash"
    extractor.temperature = 0.0
    extractor.max_output_tokens = 4096
    extractor.save_raw_json = False
    extractor._types = FakeTypes()
    extractor._client = FakeClient(Models())
    extractor._api_key_source = "GEMINI_API_KEY"
    extractor._build_prompt = lambda ocr_cells: "prompt"
    extractor._api_attempts = 3
    extractor._initial_delay_seconds = 0.0
    extractor._backoff_multiplier = 1.0
    extractor._request_timeout_seconds = 45.0

    crop_path = tmp_path / "record.jpg"
    crop_path.write_bytes(b"fake-image")

    result = extractor.extract_record(
        record_crop_path=crop_path,
        ocr_cells={"bil": OcrResult(text="7", average_confidence=0.95)},
    )

    assert result.record.bil == "7"
    assert call_count["n"] == 3
    # A stalled connection must not hang a worker indefinitely -- every
    # attempt must carry an explicit HTTP timeout (in ms).
    assert all(config.kwargs["http_options"].kwargs["timeout"] == 45000 for config in seen_configs)


def test_extract_record_gives_up_after_exhausting_retries_on_rate_limit(tmp_path: Path) -> None:
    from google.genai import errors as genai_errors

    class FakePart:
        @staticmethod
        def from_bytes(*, data: bytes, mime_type: str) -> dict[str, object]:
            return {"data": data, "mime_type": mime_type}

    class FakeGenerateContentConfig:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class FakeHttpOptions:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class FakeTypes:
        Part = FakePart
        GenerateContentConfig = FakeGenerateContentConfig
        HttpOptions = FakeHttpOptions

    call_count = {"n": 0}

    class Models:
        def generate_content(self, *, model: str, contents: list[object], config: object) -> SimpleNamespace:
            call_count["n"] += 1
            raise genai_errors.APIError(429, {"error": {"message": "rate limited"}})

    class FakeClient:
        def __init__(self, models: object) -> None:
            self.models = models

    extractor = GeminiRecordExtractor.__new__(GeminiRecordExtractor)
    extractor.model = "gemini-2.5-flash"
    extractor.temperature = 0.0
    extractor.max_output_tokens = 4096
    extractor.save_raw_json = False
    extractor._types = FakeTypes()
    extractor._client = FakeClient(Models())
    extractor._api_key_source = "GEMINI_API_KEY"
    extractor._build_prompt = lambda ocr_cells: "prompt"
    extractor._api_attempts = 2
    extractor._initial_delay_seconds = 0.0
    extractor._backoff_multiplier = 1.0

    crop_path = tmp_path / "record.jpg"
    crop_path.write_bytes(b"fake-image")

    try:
        extractor.extract_record(
            record_crop_path=crop_path,
            ocr_cells={"bil": OcrResult(text="7", average_confidence=0.95)},
        )
        assert False, "expected APIError to propagate after exhausting attempts"
    except genai_errors.APIError:
        pass

    assert call_count["n"] == 2


def test_build_prompt_uses_aggressive_handwritten_mode() -> None:
    extractor = GeminiRecordExtractor.__new__(GeminiRecordExtractor)
    extractor.config = {"prompt_mode": "handwritten_aggressive"}

    prompt = GeminiRecordExtractor._build_prompt(
        extractor,
        {"bil": OcrResult(text="7", average_confidence=0.95)},
    )

    assert "Correct OCR errors aggressively across all fields" in prompt
    assert "Prioritize semantic correctness over literal transcription" in prompt


def test_extract_record_prefers_sdk_parsed_payload(tmp_path: Path) -> None:
    class FakePart:
        @staticmethod
        def from_bytes(*, data: bytes, mime_type: str) -> dict[str, object]:
            return {"data": data, "mime_type": mime_type}

    class FakeGenerateContentConfig:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class FakeHttpOptions:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class FakeTypes:
        Part = FakePart
        GenerateContentConfig = FakeGenerateContentConfig
        HttpOptions = FakeHttpOptions

    class Models:
        def generate_content(self, *, model: str, contents: list[object], config: object) -> SimpleNamespace:
            return SimpleNamespace(
                parsed={"bil": "9", "nama_suami": "TEST SUAMI"},
                text='{"bil": "broken"',
            )

    class FakeClient:
        def __init__(self) -> None:
            self.models = Models()

    extractor = GeminiRecordExtractor.__new__(GeminiRecordExtractor)
    extractor.model = "gemini-2.5-flash"
    extractor.temperature = 0.0
    extractor.max_output_tokens = 4096
    extractor.save_raw_json = False
    extractor._types = FakeTypes()
    extractor._client = FakeClient()
    extractor._api_key_source = "GEMINI_API_KEY"
    extractor._build_prompt = lambda ocr_cells: "prompt"

    crop_path = tmp_path / "record.jpg"
    crop_path.write_bytes(b"fake-image")

    result = extractor.extract_record(
        record_crop_path=crop_path,
        ocr_cells={"bil": OcrResult(text="9", average_confidence=0.95)},
    )

    assert result.record.bil == "9"
    assert result.record.nama_suami == "TEST SUAMI"


def test_parse_response_text_accepts_raw_newlines_in_strings() -> None:
    extractor = GeminiRecordExtractor.__new__(GeminiRecordExtractor)

    payload = GeminiRecordExtractor._parse_response_text(
        extractor,
        '{"bil":"1","alamat_pendaftar":"LINE 1\nLINE 2"}',
    )

    assert payload["bil"] == "1"
    assert payload["alamat_pendaftar"] == "LINE 1\nLINE 2"


def test_parse_response_text_extracts_wrapped_json_object() -> None:
    extractor = GeminiRecordExtractor.__new__(GeminiRecordExtractor)

    payload = GeminiRecordExtractor._parse_response_text(
        extractor,
        'Here is the result:\n{"bil":"1","alamat_pendaftar":"LINE 1",}\nThanks.',
    )

    assert payload["bil"] == "1"
    assert payload["alamat_pendaftar"] == "LINE 1"


def test_parse_response_text_recovers_truncated_string_field() -> None:
    extractor = GeminiRecordExtractor.__new__(GeminiRecordExtractor)

    payload = GeminiRecordExtractor._parse_response_text(
        extractor,
        '{"bil":"463/94","nama_suami":"HASNOOR AZWAN BIN JAMIL","id_isteri_raw":"A.1857',
    )

    assert payload["bil"] == "463/94"
    assert payload["nama_suami"] == "HASNOOR AZWAN BIN JAMIL"
    assert payload["id_isteri_raw"] == "A.1857"


def test_parse_response_text_drops_dangling_trailing_field() -> None:
    extractor = GeminiRecordExtractor.__new__(GeminiRecordExtractor)

    payload = GeminiRecordExtractor._parse_response_text(
        extractor,
        '{"bil":"470/94","nama_suami":"ABDUL RAHMAN BIN JAKARIA","nama_wali":',
    )

    assert payload["bil"] == "470/94"
    assert payload["nama_suami"] == "ABDUL RAHMAN BIN JAKARIA"
    assert "nama_wali" not in payload
