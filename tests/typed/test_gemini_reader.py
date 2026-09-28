import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from marriage_ocr.typed.gemini_reader import FIELD_LABELS, GeminiTypedReader, build_request, raw_fields_from_payload
from marriage_ocr.typed.models import ProcessingStatus
from marriage_ocr.typed.pipeline import process_typed_input
from marriage_ocr.typed.template import TEMPLATES
from tests.typed.test_pipeline import _write_two_page_pdf

_CONFIG = """
record_type: NIKAH
ocr:
  google_vision:
    language_hints: [ms, en]
typed:
  template: nikah_legacy
  pdf_dpi: 300
  pdf_batch_size: 4
  render_workers: 1
  word_confidence_threshold: 0.75
  retry:
    api_attempts: 1
    max_fields_per_pdf: 6
    request_batch_size: 16
  validation:
    min_age: 16
    max_age: 120
""".strip()


def test_every_template_region_has_a_label() -> None:
    for template, labels in FIELD_LABELS.items():
        assert set(labels) == set(TEMPLATES[template]["regions"]), template


def test_request_lists_every_region_and_the_struck_through_rule() -> None:
    request = build_request("cerai_modern", "CERAI", 2)
    assert "Surat Perakuan Cerai" in request["prompt"]
    assert "CROSSED OUT do not apply" in request["prompt"]
    for key in TEMPLATES["cerai_modern"]["regions"]:
        assert f"- {key}:" in request["prompt"]
    schema = request["generation_config"]["response_schema"]
    assert set(schema["properties"]) == set(TEMPLATES["cerai_modern"]["regions"])
    assert request["generation_config"]["temperature"] == 0.0


def test_payload_becomes_rawfields_like_the_vision_extractor() -> None:
    fields = raw_fields_from_payload({"nama_suami": " AZEMI BIN ABDULLAH ", "bil": None}, "nikah_legacy")
    assert set(fields) == set(TEMPLATES["nikah_legacy"]["regions"])
    assert (fields["nama_suami"].raw_text, fields["nama_suami"].confidence) == ("AZEMI BIN ABDULLAH", 1.0)
    assert (fields["bil"].raw_text, fields["bil"].confidence) == ("", 0.0)
    assert fields["nama_suami"].region == TEMPLATES["nikah_legacy"]["regions"]["nama_suami"][1]


def _run(tmp_path: Path, provider, monkeypatch, *, ocr_allowed: bool):
    pdf = tmp_path / "doc.pdf"
    _write_two_page_pdf(pdf)
    config = tmp_path / "typed.yaml"
    config.write_text(_CONFIG, encoding="utf-8")
    calls = []

    def fake_ocr(pages, client):
        if not ocr_allowed:
            raise AssertionError("Vision OCR must not run for a Gemini-read PDF")
        calls.append(len(pages))
        from tests.typed.test_pipeline import _synthetic_complete_page_results

        return _synthetic_complete_page_results(pages)

    monkeypatch.setattr("marriage_ocr.typed.pipeline._ocr_micro_batch", fake_ocr)
    result = process_typed_input(
        input_path=pdf,
        output_path=tmp_path / "out.csv",
        debug_path=tmp_path / "debug",
        config_path=config,
        reset_output=True,
        raw_fields_provider=provider,
    )
    return result, calls


def test_a_gemini_read_pdf_goes_through_the_normalizers_without_any_vision_call(tmp_path, monkeypatch) -> None:
    payload = {
        "bil": "504/2000",
        "nama_suami": "AZEMI BIN ABDULLAH",
        "nama_isteri": "FARAHANIM BINTI AHMAD",
        "tarikh_nikah": "2 RABIULAKHIR 1421 / 4.7.2000",
    }
    result, _ = _run(
        tmp_path, lambda pdf, template, rt: raw_fields_from_payload(payload, template), monkeypatch, ocr_allowed=False
    )

    record = result.records[0]
    assert record.record.nama_suami == "AZEMI BIN ABDULLAH"
    assert record.record.nama_isteri == "FARAHANIM BINTI AHMAD"
    assert record.retry_count == 0
    assert record.processing_status is not ProcessingStatus.FAILED
    assert (tmp_path / "out.csv").exists()


def test_a_pdf_gemini_could_not_read_falls_back_to_vision(tmp_path, monkeypatch) -> None:
    result, calls = _run(tmp_path, lambda pdf, template, rt: None, monkeypatch, ocr_allowed=True)
    assert calls == [2]
    assert len(result.records) == 1


def test_live_reader_returns_none_on_any_error(tmp_path) -> None:
    pdf = tmp_path / "doc.pdf"
    _write_two_page_pdf(pdf)

    class Broken:
        class models:
            @staticmethod
            def generate_content(**kwargs):
                raise RuntimeError("503 UNAVAILABLE")

    assert GeminiTypedReader(client=Broken()).read(pdf, "nikah_legacy", "NIKAH") is None


def test_live_reader_parses_the_json_answer(tmp_path) -> None:
    pdf = tmp_path / "doc.pdf"
    _write_two_page_pdf(pdf)
    sent = {}

    class Client:
        class models:
            @staticmethod
            def generate_content(*, model, contents, config):
                sent.update(model=model, parts=len(contents))
                return SimpleNamespace(text=json.dumps({"nama_suami": "AHMAD BIN ALI"}))

    fields = GeminiTypedReader(client=Client()).read(pdf, "nikah_legacy", "NIKAH")
    assert fields["nama_suami"].raw_text == "AHMAD BIN ALI"
    assert sent == {"model": "gemini-3.5-flash-lite", "parts": 3}  # prompt + 2 page images


def test_crossed_out_choice_fields_are_never_taken_from_gemini() -> None:
    fields = raw_fields_from_payload({"keadaan_talak": "Bain Sughra", "cerai_dalam_keadaan": "SUCI"}, "cerai_modern")
    assert "check which options are crossed out" in fields["keadaan_talak"].raw_text
    assert fields["keadaan_talak"].confidence == 0.0
    assert fields["cerai_dalam_keadaan"].raw_text == "SUCI"  # a typed answer is still read


def test_a_cerai_modern_record_is_always_flagged_for_review(tmp_path, monkeypatch) -> None:
    import pymupdf

    config = tmp_path / "typed.yaml"
    config.write_text(
        "record_type: CERAI\ntyped:\n  template: cerai_modern\n  validation:\n    min_age: 16\n    max_age: 120\n",
        encoding="utf-8",
    )
    pdf = tmp_path / "cert.pdf"
    document = pymupdf.open()
    document.new_page(width=595, height=842)
    document.new_page(width=595, height=842)
    document.save(pdf)
    document.close()
    payload = {"bil": "199/2020", "nama_suami": "CHAN GEE KUAN", "nama_isteri": "CHE NORHAYATI BINTI CHE MAT"}

    result = process_typed_input(
        input_path=pdf,
        output_path=tmp_path / "out.csv",
        debug_path=tmp_path / "debug",
        config_path=config,
        reset_output=True,
        raw_fields_provider=lambda p, template, rt: raw_fields_from_payload(payload, template),
    )

    record = result.records[0]
    assert record.processing_status is ProcessingStatus.REVIEW_REQUIRED
    assert {"keadaan_talak", "tempat_cerai"} <= set(record.failed_fields)


def test_live_reader_retries_googles_busy_errors_before_giving_up(tmp_path) -> None:
    pdf = tmp_path / "doc.pdf"
    _write_two_page_pdf(pdf)
    calls = []

    class Busy(Exception):
        code = 503

    class FlakyClient:
        class models:
            @staticmethod
            def generate_content(**kwargs):
                calls.append(1)
                if len(calls) < 3:
                    raise Busy("503 UNAVAILABLE")
                return SimpleNamespace(text=json.dumps({"nama_suami": "AHMAD BIN ALI"}))

    slept = []
    fields = GeminiTypedReader(client=FlakyClient(), sleep=slept.append).read(pdf, "nikah_legacy", "NIKAH")

    assert fields["nama_suami"].raw_text == "AHMAD BIN ALI"
    assert len(calls) == 3 and slept == [2.0, 4.0]
