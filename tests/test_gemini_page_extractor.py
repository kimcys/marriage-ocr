from __future__ import annotations

import json
from pathlib import Path

from llm.gemini_page_extractor import GeminiPageExtractor


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


def _make_extractor(*, record_type: str = "nikah", **overrides) -> GeminiPageExtractor:
    extractor = GeminiPageExtractor.__new__(GeminiPageExtractor)
    extractor.model = "gemini-3-flash-preview"
    extractor.temperature = 0.0
    extractor.max_output_tokens = 4096
    extractor.record_type = record_type
    extractor.layout_variant = "legacy"
    extractor.config = {}
    extractor._types = FakeTypes()
    extractor._api_attempts = 1
    extractor._initial_delay_seconds = 0.0
    extractor._backoff_multiplier = 2.0
    extractor._request_timeout_seconds = 60.0
    extractor.save_raw_json = False
    for key, value in overrides.items():
        setattr(extractor, key, value)
    return extractor


def test_page_schema_wraps_per_record_schema_in_records_array() -> None:
    extractor = _make_extractor(record_type="nikah")
    schema = extractor._page_schema()

    assert schema["type"] == "OBJECT"
    assert schema["required"] == ["records"]
    assert schema["properties"]["records"]["type"] == "ARRAY"
    assert schema["properties"]["records"]["items"] == extractor._response_schema()


def test_page_prompt_for_nikah_reuses_single_record_rules_with_page_framing() -> None:
    extractor = _make_extractor(record_type="nikah", config={"prompt_mode": "handwritten_aggressive"})
    prompt = extractor._page_prompt()

    assert "Daftar Perkahwinan Orang Islam" in prompt
    assert "ALL handwritten rows (records) visible on this ONE PAGE" in prompt
    assert '"records":' in prompt
    assert "Correct OCR errors aggressively" in prompt  # from the reused handwritten_aggressive rules
    assert "OCR cell hints" not in prompt  # no Vision hints exist in this pipeline


def test_page_prompt_for_cerai_uses_record_schemas_field_notes() -> None:
    extractor = _make_extractor(record_type="cerai")
    prompt = extractor._page_prompt()

    assert "Daftar Perceraian Orang Islam" in prompt
    assert "no_rujukan" in prompt  # from CERAI_FIELD_NOTES
    assert "ALL handwritten rows (records) visible on this ONE PAGE" in prompt


def test_extract_page_parses_multiple_records_from_one_response(tmp_path: Path) -> None:
    image_path = tmp_path / "page.jpg"
    image_path.write_bytes(b"fake-page-image")

    payload = {
        "records": [
            {"bil": "1/97", "nama_suami": "ABDULLAH B. ABD HAMID"},
            {"bil": "2/97", "nama_suami": "MOHD. HAMIDI B. SALLEH"},
        ]
    }

    class Models:
        def generate_content(self, *, model, contents, config):
            return type("Response", (), {"text": json.dumps(payload), "parsed": None})()

    extractor = _make_extractor(record_type="nikah", _client=type("Client", (), {"models": Models()})())

    results = extractor.extract_page(image_path)

    assert [r.record.bil for r in results] == ["1/97", "2/97"]
    assert [r.record.nama_suami for r in results] == ["ABDULLAH B. ABD HAMID", "MOHD. HAMIDI B. SALLEH"]


def test_extract_page_raises_for_missing_image(tmp_path: Path) -> None:
    extractor = _make_extractor(record_type="nikah")
    missing = tmp_path / "does_not_exist.jpg"

    try:
        extractor.extract_page(missing)
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass
