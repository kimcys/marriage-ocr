"""Gemini Batch Mode extraction -- ~50% cheaper than the synchronous
`GeminiRecordExtractor` path, at the cost of asynchronous turnaround (minutes
to hours; Google's batch SLA is up to 24h) instead of an immediate response.

This is deliberately a standalone tool, not wired into the real-time
`process`/`process-typed`/`batch_runner` flow: those call
`GeminiRecordExtractor` synchronously per record because they need an answer
immediately (duplicate checks, DB insert, export happen right after). Batch
Mode can't fit that shape at all -- you must collect many requests, submit
them together, and wait once for the whole job.

Intended usage: run a normal batch through `process`/`process-typed`/
`batch_runner` with `debug.retain_artifacts: true` and `llm.enabled: false`
(so Gemini is never called synchronously, keeping it out of the hot path),
which leaves every record's `full_record.jpg` + `raw_ocr.json` +
`parsed_record.json` on disk. Then point `run_batch_extraction` at that debug
root: it submits every record crop as one Gemini batch job, waits for it,
merges each result into its parser record exactly like the synchronous path
does (via `merge_parser_and_gemini`), and overwrites `validated_record.json`
-- so the result is a drop-in, review-UI-compatible upgrade of an
already-completed run, at half the Gemini cost.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from marriage_ocr.models import ExtractedRecord, OcrLine, OcrResult
from marriage_ocr.parser import save_parsed_record
from marriage_ocr.validation import parser_confidence_clears_threshold, validate_record

from .gemini_extractor import GeminiRecordExtractor, GeminiRecordResult, _guess_mime_type
from .record_merge import merge_parser_and_gemini

LOGGER = logging.getLogger(__name__)

# JOB_STATE_* values a batch job stops changing at on its own -- see
# google.genai.types.JobState. PARTIALLY_SUCCEEDED still has to be walked
# per-item: some requests came back fine, others carry an error on their own
# InlinedResponse.
TERMINAL_JOB_STATES = frozenset(
    {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
        "JOB_STATE_PARTIALLY_SUCCEEDED",
    }
)
FAILED_JOB_STATES = frozenset({"JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"})


@dataclass(frozen=True)
class BatchRecordItem:
    """One record queued for batch extraction, addressed by its record_dir."""

    key: str
    record_dir: Path
    ocr_cells: Mapping[str, OcrResult]


@dataclass
class BatchExtractionSummary:
    submitted: int = 0
    merged: int = 0
    skipped_no_parsed_record: list[str] = field(default_factory=list)
    skipped_parser_confident: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    validated_records: list[ExtractedRecord] = field(default_factory=list)


def discover_batch_items(debug_root: str | Path) -> list[BatchRecordItem]:
    """Find every `record_*/` folder with both `full_record.jpg` and
    `raw_ocr.json` under debug_root (recursively), reconstructing the same
    `ocr_cells` mapping `GeminiRecordExtractor.extract_record` receives live,
    so batch results are directly comparable to a synchronous run.

    Requires a prior run with `debug.retain_artifacts: true` (for
    `full_record.jpg`) and OCR's `save_raw_json` on (also gated by
    `retain_artifacts`, see pipeline.py) -- both are on by default whenever
    `retain_artifacts` is enabled.
    """
    debug_root = Path(debug_root)
    items: list[BatchRecordItem] = []
    for record_dir in sorted(debug_root.rglob("record_*")):
        if not record_dir.is_dir():
            continue
        image_path = record_dir / "full_record.jpg"
        raw_json_path = record_dir / "raw_ocr.json"
        if not image_path.exists() or not raw_json_path.exists():
            continue
        ocr_cells = _load_ocr_cells(raw_json_path)
        key = str(record_dir.relative_to(debug_root))
        items.append(BatchRecordItem(key=key, record_dir=record_dir, ocr_cells=ocr_cells))
    return items


def load_parsed_record(record_dir: Path) -> ExtractedRecord | None:
    path = record_dir / "parsed_record.json"
    if not path.exists():
        return None
    return ExtractedRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _load_ocr_cells(raw_json_path: Path) -> dict[str, OcrResult]:
    payload = json.loads(raw_json_path.read_text(encoding="utf-8"))
    cells: dict[str, OcrResult] = {}
    for cell_name, cell_payload in payload.get("cells", {}).items():
        cells[cell_name] = OcrResult(
            text=cell_payload.get("text", ""),
            average_confidence=cell_payload.get("average_confidence", 0.0),
            lines=[
                OcrLine(
                    text=line.get("text", ""),
                    confidence=line.get("confidence", 0.0),
                    bbox=line.get("bbox"),
                )
                for line in cell_payload.get("lines", [])
            ],
        )
    return cells


class GeminiBatchRecordExtractor(GeminiRecordExtractor):
    """Reuses `GeminiRecordExtractor`'s prompt-building, schema, and response
    parsing unchanged (identical output shape to the synchronous path) but
    submits every record as one Gemini Batch Mode job instead of N
    synchronous calls.
    """

    def build_request(self, item: BatchRecordItem) -> Any:
        image_path = item.record_dir / "full_record.jpg"
        image_part = self._types.Part.from_bytes(
            data=image_path.read_bytes(), mime_type=_guess_mime_type(image_path)
        )
        prompt = self._build_prompt(item.ocr_cells)
        config = self._types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            response_mime_type="application/json",
            response_schema=self._response_schema(),
        )
        return self._types.InlinedRequest(
            contents=[prompt, image_part],
            config=config,
            metadata={"key": item.key},
        )

    def submit(self, items: Sequence[BatchRecordItem], *, display_name: str | None = None) -> Any:
        requests = [self.build_request(item) for item in items]
        return self._client.batches.create(
            model=self.model,
            src=requests,
            config=self._types.CreateBatchJobConfig(display_name=display_name),
        )

    def wait(
        self,
        job: Any,
        *,
        poll_seconds: float = 30.0,
        timeout_seconds: float | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> Any:
        elapsed = 0.0
        name = job.name
        while True:
            job = self._client.batches.get(name=name)
            state = str(job.state)
            if state in TERMINAL_JOB_STATES:
                return job
            if timeout_seconds is not None and elapsed >= timeout_seconds:
                raise TimeoutError(f"Gemini batch job {name} still {state} after {elapsed:.0f}s")
            sleep(poll_seconds)
            elapsed += poll_seconds

    def collect_results(self, job: Any) -> dict[str, GeminiRecordResult | Exception]:
        state = str(job.state)
        if state in FAILED_JOB_STATES:
            raise RuntimeError(f"Gemini batch job {job.name} ended in {state}: {job.error}")

        responses = job.dest.inlined_responses if job.dest is not None else None
        if responses is None:
            raise RuntimeError(f"Gemini batch job {job.name} ({state}) has no inlined responses in dest={job.dest!r}")

        results: dict[str, GeminiRecordResult | Exception] = {}
        for inlined in responses:
            key = (inlined.metadata or {}).get("key")
            if key is None:
                LOGGER.warning("Gemini batch response with no 'key' metadata; skipping")
                continue
            if inlined.error is not None:
                results[key] = RuntimeError(f"Gemini batch item {key} failed: {inlined.error}")
                continue
            try:
                payload = self._extract_response_payload(inlined.response)
                results[key] = self._payload_to_result(payload)
            except Exception as exc:  # noqa: BLE001 -- recorded per-item, not fatal to the batch
                results[key] = exc
        return results


def run_batch_extraction(
    debug_root: str | Path,
    *,
    llm_config: Mapping[str, Any],
    validation_config: Mapping[str, Any],
    record_type: str = "nikah",
    prefer_gemini_threshold: float = 0.70,
    review_below_field_confidence: float = 0.80,
    layout_confidence: float = 1.0,
    display_name: str | None = None,
    poll_seconds: float = 30.0,
    timeout_seconds: float | None = None,
    sleep: Callable[[float], None] = time.sleep,
    extractor: GeminiBatchRecordExtractor | None = None,
    skip_gemini_when_parser_ok: bool = False,
    skip_gemini_min_confidence: float = 0.90,
) -> BatchExtractionSummary:
    """End-to-end: discover records under debug_root, submit them as one
    Gemini batch job, wait for it, merge results the same way the
    synchronous pipeline does, and overwrite each record's
    `validated_record.json`.

    `layout_confidence` defaults to 1.0 (same default `merge_parser_and_gemini`
    uses) because layout geometry isn't reconstructable from record crops
    alone here -- it only affects one review-flagging heuristic
    (validation.py's `layout_confidence < 0.75` check), never the extracted
    field values themselves.

    `extractor` is exposed mainly for tests to inject a fake Gemini client;
    real callers can leave it out and it's built from `llm_config`.

    `skip_gemini_when_parser_ok` mirrors pipeline.py's synchronous
    skip-Gemini-when-parser-confident check (see
    validation.parser_confidence_clears_threshold) -- without it, a debug
    root produced with that setting enabled would still submit every
    record here, including ones the live run already decided didn't need
    Gemini at all, silently burning batch-mode spend on records that were
    never meant to reach Gemini in the first place.
    """
    if extractor is None:
        extractor = GeminiBatchRecordExtractor(llm_config)
    items = discover_batch_items(debug_root)
    summary = BatchExtractionSummary(submitted=len(items))
    if not items:
        return summary

    parsed_records: dict[str, ExtractedRecord] = {}
    submittable: list[BatchRecordItem] = []
    for item in items:
        parsed_record = load_parsed_record(item.record_dir)
        if parsed_record is None:
            summary.skipped_no_parsed_record.append(item.key)
            continue

        if record_type == "nikah" and skip_gemini_when_parser_ok:
            parser_validated = validate_record(
                parsed_record,
                item.ocr_cells,
                validation_config,
                layout_confidence=layout_confidence,
                record_type=record_type,
            )
            if parser_confidence_clears_threshold(parser_validated, min_confidence=skip_gemini_min_confidence):
                save_parsed_record(parser_validated, item.record_dir / "validated_record.json")
                summary.validated_records.append(parser_validated)
                summary.skipped_parser_confident.append(item.key)
                continue

        parsed_records[item.key] = parsed_record
        submittable.append(item)

    if not submittable:
        return summary

    job = extractor.submit(submittable, display_name=display_name)
    job = extractor.wait(job, poll_seconds=poll_seconds, timeout_seconds=timeout_seconds, sleep=sleep)
    results = extractor.collect_results(job)

    items_by_key = {item.key: item for item in submittable}
    for key, parser_record in parsed_records.items():
        item = items_by_key[key]
        result = results.get(key)

        if isinstance(result, Exception) or result is None:
            error_text = str(result) if result is not None else "no response returned for this item"
            summary.failed[key] = error_text
            validated_record = validate_record(
                parser_record,
                item.ocr_cells,
                validation_config,
                layout_confidence=layout_confidence,
                record_type=record_type,
            )
            reasons = list(validated_record.review_reason or [])
            reasons.append(f"Gemini batch unavailable: {error_text}")
            validated_record.review_reason = reasons
            if validated_record.status_review == "OK":
                validated_record.status_review = "REVIEW"
        else:
            validated_record = merge_parser_and_gemini(
                parser_record=parser_record,
                gemini_result=result,
                cell_results=item.ocr_cells,
                validation_config=validation_config,
                layout_confidence=layout_confidence,
                prefer_gemini_threshold=prefer_gemini_threshold,
                review_below_field_confidence=review_below_field_confidence,
                record_type=record_type,
            )
            summary.merged += 1

        save_parsed_record(validated_record, item.record_dir / "validated_record.json")
        summary.validated_records.append(validated_record)

    return summary
