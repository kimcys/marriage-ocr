"""Read one whole page image and extract every record on it in a single
Gemini call -- no Vision OCR, no per-cell crops, no deterministic parser.

This is the "full-Gemini-per-page" approach validated by hand against real
scans before this module existed: cheaper than Vision's cell_crops mode
(Vision is ~76% of the current handwritten cost, entirely eliminated here),
and its record segmentation held up better than the current layout
detector's in every real-scan comparison, including two pages where the
layout detector was confirmed to miscount records (a real, pre-existing bug
independent of this change -- see src/marriage_ocr/layout.py's
_optimize_record_starts docstring).

What this does NOT have, by design, since there's no second extraction path
to compare against: the parser/Gemini disagreement check that currently
drives review-flagging. See marriage_ocr.validation.validate_gemini_only_record
for what replaces it -- objective field-value sanity checks (IC format, age
range, date validity), not just trusting Gemini's own reported confidence
(which was found to vary too little across records to reliably rank them).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .gemini_extractor import GeminiRecordExtractor, GeminiRecordResult, _guess_mime_type
from . import record_schemas

_PAGE_DOCUMENT_DESCRIPTIONS: dict[str, str] = {
    "nikah": "a Malay Islamic marriage register, Daftar Perkahwinan Orang Islam",
    "cerai": "a Malay Islamic divorce register, Daftar Perceraian Orang Islam",
    "rujuk": "a Malay Islamic reconciliation register, Daftar Rujuk Orang Islam",
}

_PAGE_FIELD_NOTES: dict[str, str] = {
    "cerai": record_schemas.CERAI_FIELD_NOTES,
    "rujuk": record_schemas.RUJUK_FIELD_NOTES,
}

def _rules_only(instructions: str) -> str:
    """Both _nikah_instructions() and record_schemas._SHARED_RULES open with
    a "you're extracting ONE row" framing sentence that also tells Gemini to
    treat "Google Vision OCR cell hints" as secondary evidence -- accurate
    for the single-record path those strings were written for, but this
    pipeline has no Vision OCR and no hints to fall back on, so keeping that
    sentence would just be a confusing, wrong instruction. Cut everything
    before the "Rules:" bullet list, which is Vision-agnostic, and let
    _page_prompt's own framing (written for this pipeline) stand in for the
    rest.
    """
    marker = "Rules:"
    index = instructions.find(marker)
    return instructions[index:] if index != -1 else instructions


# A page can hold several records with many fields each; the single-record
# max_output_tokens (tuned for one row) truncates a multi-record response
# well before every row's JSON fits -- confirmed empirically against a
# 5-record Cerai page. Floor this generously rather than tune per record
# count, since a too-small budget fails the whole page, not just one row.
_MIN_PAGE_OUTPUT_TOKENS = 16384


class GeminiPageExtractor(GeminiRecordExtractor):
    """Same config/client/model as GeminiRecordExtractor, and reuses its
    schema (_response_schema) and response parsing (_payload_to_result)
    unchanged -- only the prompt and the "one call covers many records"
    shape are new.
    """

    def _page_schema(self) -> dict[str, Any]:
        return {
            "type": "OBJECT",
            "properties": {"records": {"type": "ARRAY", "items": self._response_schema()}},
            "required": ["records"],
        }

    def _page_prompt(self) -> str:
        document_description = _PAGE_DOCUMENT_DESCRIPTIONS[self.record_type]
        if self.record_type == "nikah":
            rules = _rules_only(self._nikah_instructions())
        else:
            rules = "\n\n".join([_rules_only(record_schemas._SHARED_RULES), _PAGE_FIELD_NOTES[self.record_type]])

        layout_note = record_schemas._layout_note(self.layout_variant)

        return f"""
You are extracting ALL handwritten rows (records) visible on this ONE PAGE of
{document_description}. Each horizontal row on the page is a separate record.

Identify every record row on the page, strictly top to bottom, and extract the
same set of fields for EACH row as its own separate JSON object. Do not merge
information from different rows into the same record, and do not let a name,
IC number, or date from one row bleed into an adjacent row's record -- keep
each row's data strictly isolated to that row. A row with no legible content
(a blank ruled row with nothing filled in) is not a record -- do not invent
one for it.

Return only JSON: {{"records": [ ... one object per row, top to bottom ... ]}}.
Do not include markdown.

{layout_note}

{rules}
""".strip()

    def extract_page(self, page_image_path: str | Path) -> list[GeminiRecordResult]:
        image_path = Path(page_image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Page image not found: {image_path}")

        image_bytes = image_path.read_bytes()
        image_part = self._types.Part.from_bytes(data=image_bytes, mime_type=_guess_mime_type(image_path))

        response = self._generate_content(
            self._page_prompt(),
            image_part,
            schema=self._page_schema(),
            max_output_tokens=max(self.max_output_tokens * 5, _MIN_PAGE_OUTPUT_TOKENS),
        )

        payload = self._extract_response_payload(response)
        if self.save_raw_json:
            raw_path = image_path.parent / "gemini_page_response.json"
            raw_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        records_payload = payload.get("records", [])
        return [self._payload_to_result(record_payload) for record_payload in records_payload]
