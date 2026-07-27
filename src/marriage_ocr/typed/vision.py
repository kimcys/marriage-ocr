from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

from google.api_core import exceptions as google_exceptions

from marriage_ocr.typed.models import PageOcrResult, PositionedWord, RenderedPage


TRANSIENT_VISION_EXCEPTIONS = (
    google_exceptions.DeadlineExceeded,
    google_exceptions.InternalServerError,
    google_exceptions.ResourceExhausted,
    google_exceptions.ServiceUnavailable,
    google_exceptions.TooManyRequests,
)


class _UnavailableVisionClient:
    def __init__(self, error: Exception) -> None:
        self._error = error

    def batch_annotate_images(self, *, requests: Sequence[object]) -> object:
        raise RuntimeError("Google Vision client is unavailable") from self._error


def _vertex_coord(vertex: object, axis: str) -> float:
    value = getattr(vertex, axis, 0) or 0
    return float(value)


def annotation_to_page_result(
    annotation: object,
    *,
    source_file: str,
    page_number: int,
) -> PageOcrResult:
    positioned: list[PositionedWord] = []
    annotation_pages = list(getattr(annotation, "pages", ()) or ())
    for annotation_page in annotation_pages:
        raw_width = int(getattr(annotation_page, "width", 0) or 0)
        raw_height = int(getattr(annotation_page, "height", 0) or 0)
        fallback_x = 0.0
        fallback_y = 0.0
        page_words: list[tuple[str, float, list[object]]] = []
        for block in getattr(annotation_page, "blocks", ()) or ():
            for paragraph in getattr(block, "paragraphs", ()) or ():
                for word in getattr(paragraph, "words", ()) or ():
                    text = "".join(getattr(symbol, "text", "") for symbol in getattr(word, "symbols", ())).strip()
                    vertices = list(getattr(getattr(word, "bounding_box", None), "vertices", ()) or ())
                    if text and len(vertices) >= 4:
                        page_words.append((text, float(getattr(word, "confidence", 0.0) or 0.0), vertices))
                        fallback_x = max(fallback_x, max(_vertex_coord(vertex, "x") for vertex in vertices))
                        fallback_y = max(fallback_y, max(_vertex_coord(vertex, "y") for vertex in vertices))

        width = raw_width or int(fallback_x) or 1
        height = raw_height or int(fallback_y) or 1
        width = max(1, width)
        height = max(1, height)
        for text, confidence, vertices in page_words:
            xs = [_vertex_coord(vertex, "x") for vertex in vertices]
            ys = [_vertex_coord(vertex, "y") for vertex in vertices]
            positioned.append(
                PositionedWord(
                    text=text,
                    confidence=confidence,
                    x1=min(xs) / width,
                    y1=min(ys) / height,
                    x2=max(xs) / width,
                    y2=max(ys) / height,
                    page_number=page_number,
                )
            )

    return PageOcrResult(
        source_file=source_file,
        page_number=page_number,
        words=tuple(positioned),
        full_text=str(getattr(annotation, "text", "") or ""),
        raw_response={
            "source_file": source_file,
            "page_number": page_number,
            "full_text": str(getattr(annotation, "text", "") or ""),
            "words": [
                {
                    "text": word.text,
                    "confidence": word.confidence,
                    "bbox": [word.x1, word.y1, word.x2, word.y2],
                }
                for word in positioned
            ],
        },
    )


class TypedVisionClient:
    def __init__(
        self,
        *,
        client: object | None = None,
        vision_module: object | None = None,
        language_hints: Sequence[str] = ("ms", "en"),
        api_attempts: int = 3,
        initial_delay_seconds: float = 1.0,
        backoff_multiplier: float = 2.0,
        request_batch_size: int = 16,
    ) -> None:
        if request_batch_size < 1 or request_batch_size > 16:
            raise ValueError("Vision request_batch_size must be between 1 and 16")
        if api_attempts < 1:
            raise ValueError("api_attempts must be at least 1")
        if vision_module is None:
            from google.cloud import vision as google_vision

            vision_module = google_vision
        self._vision = vision_module
        if client is not None:
            self._client = client
        else:
            try:
                self._client = vision_module.ImageAnnotatorClient()
            except Exception as error:  # pragma: no cover - exercised in environments without credentials
                self._client = _UnavailableVisionClient(error)
        self._language_hints = tuple(language_hints)
        self._api_attempts = api_attempts
        self._initial_delay_seconds = initial_delay_seconds
        self._backoff_multiplier = backoff_multiplier
        self._request_batch_size = request_batch_size

    def _build_request(self, image_path: Path) -> object:
        image = self._vision.Image(content=image_path.read_bytes())
        feature = self._vision.Feature(type_=self._vision.Feature.Type.DOCUMENT_TEXT_DETECTION)
        context = self._vision.ImageContext(language_hints=list(self._language_hints))
        return self._vision.AnnotateImageRequest(
            image=image,
            features=[feature],
            image_context=context,
        )

    def _call_batch(self, requests: Sequence[object]) -> object:
        delay = self._initial_delay_seconds
        for attempt in range(1, self._api_attempts + 1):
            try:
                return self._client.batch_annotate_images(requests=list(requests))
            except TRANSIENT_VISION_EXCEPTIONS:
                if attempt == self._api_attempts:
                    raise
                time.sleep(delay)
                delay *= self._backoff_multiplier
        raise AssertionError("Vision retry loop exhausted unexpectedly")

    def annotate_pages(self, pages: Sequence[RenderedPage]) -> tuple[PageOcrResult, ...]:
        items = [(page.source_file, page.page_number, page.image_path) for page in pages]
        return self.annotate_image_paths(items)

    def annotate_image_paths(
        self,
        items: Sequence[tuple[str, int, Path]],
    ) -> tuple[PageOcrResult, ...]:
        results: list[PageOcrResult] = []
        for start in range(0, len(items), self._request_batch_size):
            chunk = items[start : start + self._request_batch_size]
            response = self._call_batch([self._build_request(path) for _, _, path in chunk])
            responses = list(getattr(response, "responses", ()))
            if len(responses) != len(chunk):
                raise RuntimeError(
                    f"Vision returned {len(responses)} responses for {len(chunk)} images"
                )
            for (source_file, page_number, _), item_response in zip(chunk, responses, strict=True):
                message = str(getattr(getattr(item_response, "error", None), "message", "") or "")
                if message:
                    results.append(
                        PageOcrResult(
                            source_file=source_file,
                            page_number=page_number,
                            words=(),
                            full_text="",
                            raw_response={
                                "source_file": source_file,
                                "page_number": page_number,
                                "error_message": message,
                                "words": [],
                            },
                            error_message=message,
                        )
                    )
                    continue
                results.append(
                    annotation_to_page_result(
                        item_response.full_text_annotation,
                        source_file=source_file,
                        page_number=page_number,
                    )
                )
        return tuple(results)
