"""A saved page-1 Vision result that `classify` hands to `process-typed`.

`classify` already runs a `ms/en`-hinted DOCUMENT_TEXT_DETECTION over a
file's first page -- for a typed PDF, byte-for-byte the same request
`process-typed` would then make again for page 1 (same PyMuPDF render at
the same dpi, same PNG encoding, same feature and language hints). Saving
that result and reusing it saves one Vision call per typed PDF.

Only reused when the render it came from provably matches the typed
pipeline's own (dpi and pixel dimensions); anything else falls back to a
normal OCR call, so a stale or mismatched file costs a call, never accuracy.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from marriage_ocr.typed.models import PageOcrResult, PositionedWord

PAGE_OCR_CACHE_VERSION = 1


@dataclass(frozen=True)
class CachedPageOcr:
    result: PageOcrResult
    dpi: int
    width: int
    height: int

    def matches(self, *, dpi: int, width: int, height: int) -> bool:
        return (self.dpi, self.width, self.height) == (dpi, width, height)


def write_page_ocr_cache(path: Path, result: PageOcrResult, *, dpi: int, width: int, height: int) -> None:
    payload = {
        "version": PAGE_OCR_CACHE_VERSION,
        "page_number": result.page_number,
        "dpi": dpi,
        "width": width,
        "height": height,
        "full_text": result.full_text,
        "words": [
            {"text": word.text, "confidence": word.confidence, "bbox": [word.x1, word.y1, word.x2, word.y2]}
            for word in result.words
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def load_page_ocr_cache(path: Path, *, source_file: str) -> CachedPageOcr | None:
    """None for anything unreadable or from an incompatible version -- the
    caller then just OCRs the page normally."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != PAGE_OCR_CACHE_VERSION:
            return None
        page_number = int(payload["page_number"])
        words = tuple(
            PositionedWord(
                text=str(word["text"]),
                confidence=float(word["confidence"]),
                x1=float(word["bbox"][0]),
                y1=float(word["bbox"][1]),
                x2=float(word["bbox"][2]),
                y2=float(word["bbox"][3]),
                page_number=page_number,
            )
            for word in payload["words"]
        )
        full_text = str(payload["full_text"])
        return CachedPageOcr(
            result=PageOcrResult(
                source_file=source_file,
                page_number=page_number,
                words=words,
                full_text=full_text,
                raw_response={
                    "source_file": source_file,
                    "page_number": page_number,
                    "full_text": full_text,
                    "words": payload["words"],
                    "reused_from_classify": True,
                },
            ),
            dpi=int(payload["dpi"]),
            width=int(payload["width"]),
            height=int(payload["height"]),
        )
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return None
