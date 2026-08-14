from __future__ import annotations

import logging
import json
import mimetypes
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

from google.genai import errors as genai_errors

from marriage_ocr.models import ExtractedRecord, OcrResult


LOGGER = logging.getLogger(__name__)

TRANSIENT_GEMINI_HTTP_CODES = frozenset({429, 500, 502, 503, 504})


@dataclass(frozen=True)
class GeminiRecordResult:
    """Gemini structured extraction result for a single marriage-register row."""

    record: ExtractedRecord
    field_confidence: dict[str, float] = field(default_factory=dict)
    uncertain_fields: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    raw_response: dict[str, Any] = field(default_factory=dict)


class GeminiRecordExtractor:
    """Use Gemini as a semantic record extractor after Google Vision OCR.

    Google Vision remains the layout/OCR anchor. Gemini receives the full row crop
    plus the OCR cell text as hints and returns one strict JSON object.
    """

    # Class-level fallbacks so tests that bypass __init__ via
    # GeminiRecordExtractor.__new__(...) still get single-attempt behavior
    # instead of an AttributeError.
    _api_attempts = 1
    _initial_delay_seconds = 0.0
    _backoff_multiplier = 2.0
    _request_timeout_seconds = 60.0

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.model = str(self.config.get("model", "gemini-2.5-flash"))
        self.temperature = float(self.config.get("temperature", 0.0))
        self.max_output_tokens = int(self.config.get("max_output_tokens", 4096))
        self.save_raw_json = bool(self.config.get("save_raw_json", True))
        self._api_attempts = int(self.config.get("api_attempts", 3))
        self._initial_delay_seconds = float(self.config.get("initial_delay_seconds", 1.0))
        self._backoff_multiplier = float(self.config.get("backoff_multiplier", 2.0))
        self._request_timeout_seconds = float(self.config.get("request_timeout_seconds", 60.0))
        if self._api_attempts < 1:
            raise ValueError("api_attempts must be at least 1")

        try:
            from google import genai
            from google.genai import types
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Gemini extraction requires the `google-genai` package. "
                "Install it with: pip install google-genai"
            ) from exc

        self._types = types
        self._api_key_source, api_key = _resolve_api_key_source(self.config)
        self._client = genai.Client(api_key=api_key) if api_key else genai.Client()
        if self._api_key_source is not None:
            LOGGER.info("Gemini API key source %s is active", self._api_key_source)

    def extract_record(
        self,
        *,
        record_crop_path: str | Path,
        ocr_cells: Mapping[str, OcrResult],
    ) -> GeminiRecordResult:
        crop_path = Path(record_crop_path)
        if not crop_path.exists():
            raise FileNotFoundError(f"Record crop not found: {crop_path}")

        image_bytes = crop_path.read_bytes()
        image_part = self._types.Part.from_bytes(data=image_bytes, mime_type=_guess_mime_type(crop_path))
        prompt = self._build_prompt(ocr_cells)

        response = self._generate_content(prompt, image_part)

        payload = self._extract_response_payload(response)
        if self.save_raw_json:
            raw_path = crop_path.parent / "gemini_record.json"
            raw_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        return self._payload_to_result(payload)

    def _generate_content(self, prompt: str, image_part: Any) -> Any:
        """Call Gemini with retry-with-backoff on transient errors.

        Without this, a single 429/RESOURCE_EXHAUSTED or 5xx propagated
        straight to pipeline.py, which fell back to parser-only validation
        for that record AND (previously) permanently disabled Gemini for
        every subsequent record in the run -- see
        pipeline.py::_should_disable_gemini_for_run. At 1M-record scale with
        many parallel workers hitting the same Gemini quota, a transient
        rate-limit response is the routine case, not the exception; retrying
        the single call first, before falling back at all, keeps far more
        records on the higher-accuracy Gemini path.
        """
        config = self._types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            response_mime_type="application/json",
            response_schema=GEMINI_RECORD_SCHEMA,
            http_options=self._types.HttpOptions(timeout=int(self._request_timeout_seconds * 1000)),
        )
        delay = self._initial_delay_seconds
        for attempt in range(1, self._api_attempts + 1):
            try:
                return self._client.models.generate_content(
                    model=self.model,
                    contents=[prompt, image_part],
                    config=config,
                )
            except genai_errors.APIError as error:
                if attempt == self._api_attempts or error.code not in TRANSIENT_GEMINI_HTTP_CODES:
                    raise
                time.sleep(delay)
                delay *= self._backoff_multiplier
        raise AssertionError("Gemini retry loop exhausted unexpectedly")

    def _extract_response_payload(self, response: Any) -> dict[str, Any]:
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, Mapping):
            return dict(parsed)
        if hasattr(parsed, "model_dump"):
            dumped = parsed.model_dump()
            if isinstance(dumped, dict):
                return dumped
        return self._parse_response_text(getattr(response, "text", ""))

    def _build_prompt(self, ocr_cells: Mapping[str, OcrResult]) -> str:
        cell_hints = {
            name: {
                "text": result.text,
                "average_confidence": result.average_confidence,
            }
            for name, result in ocr_cells.items()
        }

        prompt_mode = str(self.config.get("prompt_mode", "")).strip().lower()
        if prompt_mode == "handwritten_aggressive":
            instructions = """
You are extracting ONE handwritten row from a Malay Islamic marriage register,
Daftar Perkahwinan Orang Islam.

Use the image as the primary source. Use the Google Vision OCR cell hints only as
secondary evidence. If the OCR hint conflicts with the image, prefer the image.

Return only JSON matching the schema. Do not include markdown.

Rules:
- Correct OCR errors aggressively across all fields when the image supports the correction.
- Prioritize semantic correctness over literal transcription for every column.
- Normalize obvious OCR artifacts, broken spacing, and punctuation noise.
- Do not preserve misspellings that are clearly OCR mistakes.
- Preserve names as written only when the image truly supports the original spelling.
- Common corrections: 4J/ITJ/14J -> HJ.; BIR/8IN -> BIN; BINT!/BINT1 -> BINTI; BAP9/B4PA -> BAPA.
- Dates: put the original visible text in *_raw; normalize to YYYY-MM-DD only if clear.
  Never guess a plausible-looking date when the handwriting is unclear or partially
  illegible -- set the normalized date field to null and put whatever you can
  actually make out in *_raw instead. A wrong date is worse than a missing one, and
  a missing date field with no other parser evidence is always sent for human review,
  so there is no benefit to guessing.
- IC values: split old/new IC only when obvious; otherwise keep uncertain text in id_*_raw.
- mas_kahwin should usually contain RM and a numeric amount when visible.
- hubungan_wali should be a relationship such as BAPA, ABANG, ADIK-BERADIK, WALI HAKIM, etc.
- saksi_1 and saksi_2 are the two marriage witnesses when visible.
- Use the strongest plausible spelling for addresses, remarks, and other free-text columns instead of leaving OCR noise unchanged.
        - Return field_confidence as an array of objects with `field` and `confidence`.
        - Put fields below 0.70 confidence into uncertain_fields.
""".strip()
        else:
            instructions = """
You are extracting ONE handwritten row from a Malay Islamic marriage register,
Daftar Perkahwinan Orang Islam.

Use the image as the primary source. Use the Google Vision OCR cell hints only as
secondary evidence. If the OCR hint conflicts with the image, prefer the image.

Return only JSON matching the schema. Do not include markdown.

Rules:
- Do not invent missing values. Use null if unreadable.
- Preserve names as written, but normalize obvious OCR artifacts only when clear.
- Common corrections: 4J/ITJ/14J -> HJ.; BIR/8IN -> BIN; BINT!/BINT1 -> BINTI; BAP9/B4PA -> BAPA.
- Dates: put the original visible text in *_raw; normalize to YYYY-MM-DD only if clear.
  Never guess a plausible-looking date when the handwriting is unclear or partially
  illegible -- set the normalized date field to null and put whatever you can
  actually make out in *_raw instead. A wrong date is worse than a missing one, and
  a missing date field with no other parser evidence is always sent for human review,
  so there is no benefit to guessing.
- IC values: split old/new IC only when obvious; otherwise keep uncertain text in id_*_raw.
- mas_kahwin should usually contain RM and a numeric amount when visible.
- hubungan_wali should be a relationship such as BAPA, ABANG, ADIK-BERADIK, WALI HAKIM, etc.
- saksi_1 and saksi_2 are the two marriage witnesses when visible.
        - Return field_confidence as an array of objects with `field` and `confidence`.
        - Put fields below 0.70 confidence into uncertain_fields.
""".strip()
        return f"""
{instructions}

Google Vision OCR cell hints:
{json.dumps(cell_hints, ensure_ascii=False, indent=2)}
""".strip()

    def _parse_response_text(self, text: str) -> dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
            cleaned = re.sub(r"```$", "", cleaned).strip()
        cleaned = _extract_json_object_text(cleaned)
        payload = _load_json_payload(cleaned)
        if payload is None:
            raise ValueError(f"Gemini returned invalid JSON: {cleaned[:500]}")
        if not isinstance(payload, dict):
            raise ValueError("Gemini returned JSON, but not an object")
        return payload

    def _payload_to_result(self, payload: Mapping[str, Any]) -> GeminiRecordResult:
        record = ExtractedRecord(
            bil=_clean_str(payload.get("bil")),
            nama_suami=_clean_name(payload.get("nama_suami")),
            ic_lama_suami=_clean_str(payload.get("ic_lama_suami")),
            ic_baru_suami=_clean_str(payload.get("ic_baru_suami")),
            id_suami_raw=_clean_str(payload.get("id_suami_raw")),
            umur_suami=_clean_int(payload.get("umur_suami")),
            nama_isteri=_clean_name(payload.get("nama_isteri")),
            ic_lama_isteri=_clean_str(payload.get("ic_lama_isteri")),
            ic_baru_isteri=_clean_str(payload.get("ic_baru_isteri")),
            id_isteri_raw=_clean_str(payload.get("id_isteri_raw")),
            umur_isteri=_clean_int(payload.get("umur_isteri")),
            mas_kahwin=_clean_str(payload.get("mas_kahwin")),
            mas_kahwin_raw=_clean_str(payload.get("mas_kahwin_raw")),
            nama_pendaftar=_clean_name(payload.get("nama_pendaftar")),
            alamat_pendaftar=_clean_str(payload.get("alamat_pendaftar")),
            nama_wali=_clean_name(payload.get("nama_wali")),
            hubungan_wali=_clean_relation(payload.get("hubungan_wali")),
            saksi_1=_clean_name(payload.get("saksi_1")),
            saksi_2=_clean_name(payload.get("saksi_2")),
            tarikh_nikah=_clean_str(payload.get("tarikh_nikah")),
            tarikh_nikah_raw=_clean_str(payload.get("tarikh_nikah_raw")),
            tarikh_keluar=_clean_str(payload.get("tarikh_keluar")),
            tarikh_keluar_raw=_clean_str(payload.get("tarikh_keluar_raw")),
            remarks=_clean_str(payload.get("remarks")),
        )

        field_confidence = _normalize_field_confidence(payload.get("field_confidence"))
        uncertain_fields = [str(v) for v in payload.get("uncertain_fields") or []]
        notes = [str(v) for v in payload.get("notes") or []]

        if field_confidence:
            record.confidence = round(mean(field_confidence.values()), 4)

        return GeminiRecordResult(
            record=record,
            field_confidence=field_confidence,
            uncertain_fields=uncertain_fields,
            notes=notes,
            raw_response=dict(payload),
        )


GEMINI_RECORD_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "bil": {"type": "STRING", "nullable": True},
        "nama_suami": {"type": "STRING", "nullable": True},
        "ic_lama_suami": {"type": "STRING", "nullable": True},
        "ic_baru_suami": {"type": "STRING", "nullable": True},
        "id_suami_raw": {"type": "STRING", "nullable": True},
        "umur_suami": {"type": "INTEGER", "nullable": True},
        "nama_isteri": {"type": "STRING", "nullable": True},
        "ic_lama_isteri": {"type": "STRING", "nullable": True},
        "ic_baru_isteri": {"type": "STRING", "nullable": True},
        "id_isteri_raw": {"type": "STRING", "nullable": True},
        "umur_isteri": {"type": "INTEGER", "nullable": True},
        "mas_kahwin": {"type": "STRING", "nullable": True},
        "mas_kahwin_raw": {"type": "STRING", "nullable": True},
        "nama_pendaftar": {"type": "STRING", "nullable": True},
        "alamat_pendaftar": {"type": "STRING", "nullable": True},
        "nama_wali": {"type": "STRING", "nullable": True},
        "hubungan_wali": {"type": "STRING", "nullable": True},
        "saksi_1": {"type": "STRING", "nullable": True},
        "saksi_2": {"type": "STRING", "nullable": True},
        "tarikh_nikah": {"type": "STRING", "nullable": True},
        "tarikh_nikah_raw": {"type": "STRING", "nullable": True},
        "tarikh_keluar": {"type": "STRING", "nullable": True},
        "tarikh_keluar_raw": {"type": "STRING", "nullable": True},
        "remarks": {"type": "STRING", "nullable": True},
        "field_confidence": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "field": {"type": "STRING", "nullable": False},
                    "confidence": {"type": "NUMBER", "nullable": False},
                },
                "required": ["field", "confidence"],
            },
            "nullable": True,
        },
        "uncertain_fields": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "nullable": True,
        },
        "notes": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "nullable": True,
        },
    },
    "required": [
        "bil", "nama_suami", "ic_lama_suami", "ic_baru_suami", "id_suami_raw", "umur_suami",
        "nama_isteri", "ic_lama_isteri", "ic_baru_isteri", "id_isteri_raw", "umur_isteri",
        "mas_kahwin", "mas_kahwin_raw", "nama_pendaftar", "alamat_pendaftar", "nama_wali",
        "hubungan_wali", "saksi_1", "saksi_2", "tarikh_nikah", "tarikh_nikah_raw",
        "tarikh_keluar", "tarikh_keluar_raw", "remarks", "field_confidence", "uncertain_fields", "notes",
    ],
}


def _resolve_api_key_source(config: Mapping[str, Any]) -> tuple[str | None, str | None]:
    api_key = _clean_str(config.get("api_key"))
    if api_key is not None:
        return "config.api_key", api_key

    api_key = _clean_str(os.getenv("GEMINI_API_KEY"))
    if api_key is not None:
        return "GEMINI_API_KEY", api_key

    return None, None


def _load_json_payload(text: str) -> dict[str, Any] | None:
    for candidate in _iter_json_parse_candidates(text):
        try:
            payload = json.loads(candidate, strict=False)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _iter_json_parse_candidates(text: str) -> list[str]:
    normalized = re.sub(r",(\s*[}\]])", r"\1", text)
    candidates: list[str] = [normalized]

    repaired = _close_open_json_structures(normalized)
    if repaired != normalized:
        candidates.append(repaired)

    for comma_index in reversed(_top_level_comma_positions(normalized)):
        truncated = normalized[:comma_index]
        repaired_truncated = _close_open_json_structures(truncated)
        if repaired_truncated not in candidates:
            candidates.append(repaired_truncated)

    return candidates


def _close_open_json_structures(text: str) -> str:
    trimmed = text.rstrip()
    if not trimmed:
        return trimmed

    stack: list[str] = []
    in_string = False
    escape = False

    for char in trimmed:
        if escape:
            escape = False
            continue
        if in_string:
            if char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            stack.append("}")
        elif char == "[":
            stack.append("]")
        elif char in "}]":
            if stack and stack[-1] == char:
                stack.pop()

    repaired = trimmed
    if in_string:
        repaired += '"'

    repaired = repaired.rstrip()
    if repaired.endswith(":"):
        repaired = repaired[:-1].rstrip()

    repaired = re.sub(r",\s*$", "", repaired)
    repaired += "".join(reversed(stack))
    repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
    return repaired


def _top_level_comma_positions(text: str) -> list[int]:
    positions: list[int] = []
    depth = 0
    in_string = False
    escape = False

    for index, char in enumerate(text):
        if escape:
            escape = False
            continue
        if in_string:
            if char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            depth += 1
        elif char in "}]":
            depth = max(0, depth - 1)
        elif char == "," and depth == 1:
            positions.append(index)

    return positions


def _clean_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return re.sub(r"\s+", " ", text)


def _clean_name(value: Any) -> str | None:
    text = _clean_str(value)
    if text is None:
        return None
    fixes = {
        "  ": " ",
        " BIR ": " BIN ",
        " 8IN ": " BIN ",
        " BINT! ": " BINTI ",
        " BINT1 ": " BINTI ",
        "4J.": "HJ.",
        "ITJ.": "HJ.",
        "14J.": "HJ.",
    }
    out = f" {text.upper()} "
    for wrong, right in fixes.items():
        out = out.replace(wrong, right)
    return re.sub(r"\s+", " ", out.strip())


def _clean_relation(value: Any) -> str | None:
    text = _clean_name(value)
    if text is None:
        return None
    return text.replace("BAP9", "BAPA").replace("B4PA", "BAPA")


def _clean_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        match = re.search(r"\d+", str(value))
        return int(match.group(0)) if match else None


def _clamp_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    return max(0.0, min(1.0, round(number, 4)))


def _normalize_field_confidence(value: Any) -> dict[str, float]:
    if isinstance(value, Mapping):
        return {str(key): _clamp_float(item) for key, item in value.items()}

    if not isinstance(value, list):
        return {}

    normalized: dict[str, float] = {}
    for item in value:
        if not isinstance(item, Mapping):
            continue
        field_name = str(item.get("field", "")).strip()
        if not field_name:
            continue
        normalized[field_name] = _clamp_float(item.get("confidence"))
    return normalized


def _guess_mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    return mime_type or "image/jpeg"


def _extract_json_object_text(text: str) -> str:
    start = text.find("{")
    if start == -1:
        return text

    depth = 0
    in_string = False
    escape = False

    for index in range(start, len(text)):
        character = text[index]
        if in_string:
            if escape:
                escape = False
            elif character == "\\":
                escape = True
            elif character == '"':
                in_string = False
            continue

        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    return text[start:]
