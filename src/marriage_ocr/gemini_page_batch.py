"""Gemini Batch Mode for the handwritten `gemini_page` pipeline -- same model,
prompt, image and response schema as the synchronous call, at half the price,
with results arriving asynchronously (Google targets under 24h; usually much
sooner).

Three steps, each a CLI command so a caller (marriage-be) can run them from
different processes/machines with files in between:

1. prepare  -- runs gemini_page_pipeline's own preprocessing for an input and,
   instead of calling Gemini, saves each page's preprocessed image plus the
   exact request (prompt, schema, generation config) it would have sent.
2. submit   -- uploads many prepared inputs' images, writes one JSONL request
   per page, and starts one Gemini batch job per model.
3. collect  -- once a batch job has finished, downloads its results and writes
   each page's JSON payload (or error) to disk.
4. finish   -- runs the same pipeline again for one input, taking each page's
   Gemini result from the collected payloads, so validation, Bil sequencing
   and export are exactly the synchronous path's.

Nothing here changes the synchronous `process` command.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from marriage_ocr.config import load_runtime_config

MANIFEST_VERSION = 1
MANIFEST_NAME = "manifest.json"
_MIN_PAGE_OUTPUT_TOKENS = 16384  # mirrors gemini_page_extractor.extract_page
TERMINAL_STATES = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}


class BatchNotSupported(ValueError):
    """The config isn't a gemini_page config -- run it synchronously."""


def _require_gemini_page(config_path: Path) -> None:
    cfg = load_runtime_config(config_path).data
    engine = str(cfg.get("pipeline", {}).get("engine", "")).strip().lower()
    if engine != "gemini_page":
        raise BatchNotSupported(f"{config_path} is not a gemini_page config (engine={engine!r})")


def prepare(input_path: Path, config_path: Path, out_dir: Path) -> dict[str, Any]:
    """Write out_dir/manifest.json plus one preprocessed page image per page."""
    from marriage_ocr.gemini_page_pipeline import process_input_gemini_page

    _require_gemini_page(config_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"version": MANIFEST_VERSION, "pages": []}

    def capture(extractor, page_index: int, page_image_path: Path):
        image_name = f"page_{page_index}{page_image_path.suffix or '.jpg'}"
        shutil.copyfile(page_image_path, out_dir / image_name)
        if "request" not in manifest:
            from llm.gemini_extractor import _guess_mime_type

            manifest["model"] = extractor.model
            manifest["mime_type"] = _guess_mime_type(page_image_path)
            manifest["request"] = {
                "contents": [{"role": "user", "parts": [{"text": extractor._page_prompt()}]}],
                "generation_config": {
                    "temperature": extractor.temperature,
                    "max_output_tokens": max(extractor.max_output_tokens * 5, _MIN_PAGE_OUTPUT_TOKENS),
                    "response_mime_type": "application/json",
                    "response_schema": extractor._page_schema(),
                },
            }
        manifest["pages"].append({"index": page_index, "image": image_name})
        return []

    with tempfile.TemporaryDirectory(prefix="marriage-ocr-batch-prepare-") as tmp:
        result = process_input_gemini_page(
            input_path=input_path,
            output_path=None,
            debug_path=Path(tmp),
            config_path=config_path,
            retain_debug_artifacts=False,
            page_results_provider=capture,
        )
    if result.failed_pages or not manifest["pages"]:
        raise RuntimeError(f"could not prepare {input_path} for Gemini batch (failed pages: {result.failed_pages})")
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return manifest


def page_key(prefix: str, page_index: int) -> str:
    return f"{prefix}__{page_index}"


@dataclass(frozen=True)
class SubmitItem:
    key_prefix: str
    prepared_dir: Path


def _client():
    import os

    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    return genai.Client(api_key=api_key) if api_key else genai.Client()


def submit(items: Sequence[SubmitItem], *, display_name: str, client: Any = None) -> list[dict[str, Any]]:
    """Upload every prepared page and start one batch job per model.
    Returns [{"name": batch_job_name, "model": ..., "keys": [key_prefix, ...]}]."""
    from google.genai import types

    client = client or _client()
    by_model: dict[str, list[tuple[SubmitItem, dict[str, Any]]]] = {}
    for item in items:
        manifest = json.loads((item.prepared_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
        by_model.setdefault(manifest["model"], []).append((item, manifest))

    batches = []
    for model, entries in by_model.items():
        with tempfile.TemporaryDirectory(prefix="marriage-ocr-batch-submit-") as tmp:
            jsonl_path = Path(tmp) / "requests.jsonl"
            with jsonl_path.open("w", encoding="utf-8") as jsonl:
                for item, manifest in entries:
                    for page in manifest["pages"]:
                        uploaded = client.files.upload(
                            file=str(item.prepared_dir / page["image"]),
                            config=types.UploadFileConfig(mime_type=manifest["mime_type"]),
                        )
                        request = json.loads(json.dumps(manifest["request"]))
                        request["contents"][0]["parts"].append(
                            {"file_data": {"file_uri": uploaded.uri, "mime_type": manifest["mime_type"]}}
                        )
                        line = {"key": page_key(item.key_prefix, page["index"]), "request": request}
                        jsonl.write(json.dumps(line, ensure_ascii=False) + "\n")
            uploaded_jsonl = client.files.upload(
                file=str(jsonl_path), config=types.UploadFileConfig(display_name=display_name, mime_type="jsonl")
            )
        job = client.batches.create(model=model, src=uploaded_jsonl.name, config={"display_name": display_name})
        batches.append({"name": job.name, "model": model, "keys": [item.key_prefix for item, _ in entries]})
    return batches


def _state_name(job: Any) -> str:
    state = getattr(job, "state", None)
    return str(getattr(state, "name", state))


def _payload_from_response(response: dict[str, Any]) -> dict[str, Any]:
    candidates = response.get("candidates") or []
    if not candidates:
        raise ValueError("no candidates in response")
    parts = (candidates[0].get("content") or {}).get("parts") or []
    text = "".join(part.get("text", "") for part in parts)
    return json.loads(text)


def collect(batch_name: str, out_dir: Path, *, client: Any = None) -> dict[str, Any]:
    """If the batch job has finished, write out_dir/<key>.json (the page's
    JSON payload) or out_dir/<key>.error for every page. Returns
    {"state": ..., "done": bool, "ok": [...keys], "errors": {key: message}}."""
    client = client or _client()
    job = client.batches.get(name=batch_name)
    state = _state_name(job)
    summary: dict[str, Any] = {"state": state, "done": state in TERMINAL_STATES, "ok": [], "errors": {}}
    if state != "JOB_STATE_SUCCEEDED":
        return summary

    out_dir.mkdir(parents=True, exist_ok=True)
    content = client.files.download(file=job.dest.file_name)
    text = content.decode("utf-8") if isinstance(content, (bytes, bytearray)) else str(content)
    for line in text.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = str(row.get("key", ""))
        try:
            if row.get("error"):
                raise ValueError(json.dumps(row["error"], ensure_ascii=False)[:500])
            payload = _payload_from_response(row.get("response") or {})
            (out_dir / f"{key}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            summary["ok"].append(key)
        except Exception as error:  # a bad page never blocks the rest of the batch
            (out_dir / f"{key}.error").write_text(str(error), encoding="utf-8")
            summary["errors"][key] = str(error)
    return summary


class MissingBatchResult(RuntimeError):
    pass


def finish(
    *,
    input_path: Path,
    config_path: Path,
    prepared_dir: Path,
    payload_dir: Path,
    key_prefix: str,
    output_path: Path,
    debug_path: Path,
    reset_output: bool = True,
):
    """Run the gemini_page pipeline for `input_path` with each page's Gemini
    result read from `payload_dir` instead of a live call. Raises
    MissingBatchResult (before writing anything) if any page lacks one."""
    from marriage_ocr.gemini_page_pipeline import process_input_gemini_page

    _require_gemini_page(config_path)
    manifest = json.loads((prepared_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    missing = [
        page["index"]
        for page in manifest["pages"]
        if not (payload_dir / f"{page_key(key_prefix, page['index'])}.json").is_file()
    ]
    if missing:
        raise MissingBatchResult(f"no batch result for page(s) {missing} of {input_path}")

    def from_payload(extractor, page_index: int, page_image_path: Path):
        payload = json.loads((payload_dir / f"{page_key(key_prefix, page_index)}.json").read_text(encoding="utf-8"))
        return [extractor._payload_to_result(record) for record in payload.get("records", [])]

    result = process_input_gemini_page(
        input_path=input_path,
        output_path=output_path,
        debug_path=debug_path,
        config_path=config_path,
        reset_output=reset_output,
        page_results_provider=from_payload,
    )
    if result.failed_pages:
        # The synchronous path would have retried a live call; here the
        # caller should rerun this input synchronously instead of keeping a
        # partial result.
        raise RuntimeError(f"page(s) failed while finishing from batch results: {result.failed_pages}")
    return result
