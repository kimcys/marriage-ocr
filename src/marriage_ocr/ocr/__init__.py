from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import inspect
import json
from pathlib import Path
import platform
from statistics import mean
import sys
from typing import Any, Mapping, Sequence

import cv2

from marriage_ocr.models import OcrLine, OcrResult


class OcrEngine(ABC):
    name: str

    @abstractmethod
    def read_image(self, image_path: str | Path) -> OcrResult:
        raise NotImplementedError


class MockOcrEngine(OcrEngine):
    name = "mock"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self._config = dict(config or {})
        self._prefix = str(self._config.get("prefix", "MOCK_OCR"))
        self._base_confidence = float(self._config.get("base_confidence", 0.82))

    def read_image(self, image_path: str | Path) -> OcrResult:
        path = Path(image_path)
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load OCR image: {path}")

        height, width = image.shape[:2]
        darkness_ratio = max(0.0, min(1.0, 1.0 - (float(image.mean()) / 255.0)))
        confidence = max(0.01, min(0.99, self._base_confidence + (darkness_ratio * 0.12)))
        record_name = path.parent.name.upper()
        cell_name = path.stem.upper()
        text = f"{self._prefix}[{record_name}:{cell_name}]"

        return OcrResult(
            text=text,
            lines=[
                OcrLine(
                    text=text,
                    confidence=confidence,
                    bbox=[0.0, 0.0, float(width), float(height)],
                )
            ],
            average_confidence=confidence,
        )


class PaddleOcrEngine(OcrEngine):
    name = "paddle"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        engine_config = dict(config or {})
        try:
            from paddleocr import PaddleOCR
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "OCR engine 'paddle' requires the `paddleocr` package. "
                f"{_build_paddle_install_guidance()}"
            ) from exc

        try:
            import paddle  # noqa: F401
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "OCR engine 'paddle' requires the `paddlepaddle` runtime package in addition "
                f"to `paddleocr`. {_build_paddle_install_guidance()}"
            ) from exc

        self._use_angle_cls = bool(engine_config.pop("use_angle_cls", True))
        show_log = bool(engine_config.pop("show_log", False))
        lang = str(engine_config.pop("lang", "en"))

        constructor_kwargs = _build_paddle_constructor_kwargs(
            paddle_ocr_class=PaddleOCR,
            lang=lang,
            use_angle_cls=self._use_angle_cls,
            show_log=show_log,
            extra_config=engine_config,
        )
        self._engine = PaddleOCR(**constructor_kwargs)

    def read_image(self, image_path: str | Path) -> OcrResult:
        path = Path(image_path)
        raw_result = _run_paddle_inference(self._engine, str(path), use_angle_cls=self._use_angle_cls)
        candidates = _normalize_paddle_output(raw_result)

        lines: list[OcrLine] = []
        for item in candidates:
            text = str(item["text"]).strip()
            confidence = float(item["confidence"])
            if not text:
                continue

            lines.append(
                OcrLine(
                    text=text,
                    confidence=confidence,
                    bbox=item.get("bbox"),
                )
            )

        average_confidence = mean([line.confidence for line in lines]) if lines else 0.0
        text = "\n".join(line.text for line in lines)
        return OcrResult(text=text, lines=lines, average_confidence=average_confidence)


class GoogleVisionOcrEngine(OcrEngine):
    name = "google_vision"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        engine_config = dict(config or {})

        try:
            from google.cloud import vision
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "OCR engine 'google_vision' requires `google-cloud-vision`. "
                "Install it with: pip install google-cloud-vision"
            ) from exc

        self._vision = vision
        self._language_hints = list(engine_config.get("language_hints", ["ms", "en"]))

        try:
            self._client = vision.ImageAnnotatorClient()
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize Google Vision OCR client. "
                "Ensure GOOGLE_APPLICATION_CREDENTIALS points to a valid service-account JSON file."
            ) from exc

    def read_image(self, image_path: str | Path) -> OcrResult:
        path = Path(image_path)
        content = path.read_bytes()
        image = self._vision.Image(content=content)
        image_context = self._vision.ImageContext(language_hints=self._language_hints)

        response = self._client.document_text_detection(
            image=image,
            image_context=image_context,
        )

        if response.error.message:
            raise RuntimeError(f"Google Vision OCR failed for {path.name}: {response.error.message}")

        return _google_vision_annotation_to_word_result(response.full_text_annotation)


def build_ocr_engine(config: Mapping[str, Any]) -> OcrEngine:
    engine_name = str(config.get("engine", "mock")).strip().lower()
    if engine_name == "mock":
        return MockOcrEngine(config.get("mock", {}))
    if engine_name in {"paddle", "paddleocr"}:
        return PaddleOcrEngine(config.get("paddle", {}))
    if engine_name in {"google_vision", "googlevision", "vision"}:
        return GoogleVisionOcrEngine(config.get("google_vision", {}))
    raise ValueError(f"Unsupported OCR engine: {engine_name}")


def read_ocr_images(
    engine: OcrEngine,
    image_paths: Sequence[str | Path],
) -> list[tuple[Path, OcrResult]]:
    return [(Path(image_path), engine.read_image(image_path)) for image_path in image_paths]


def run_ocr_on_record_crops(
    records: Sequence[RecordCropPaths],
    engine: OcrEngine,
    *,
    save_raw_json: bool = True,
) -> list[RecordOcrOutput]:
    outputs: list[RecordOcrOutput] = []
    for record in records:
        cell_results: dict[str, OcrResult] = {}
        for cell_name, cell_path in record.cell_paths.items():
            cell_results[cell_name] = engine.read_image(cell_path)

        raw_json_path: Path | None = None
        if save_raw_json:
            raw_json_path = record.record_dir / "raw_ocr.json"
            payload = {
                "engine": engine.name,
                "record": f"record_{record.record_index:03d}",
                "cells": {cell_name: result.to_dict() for cell_name, result in cell_results.items()},
            }
            raw_json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

        outputs.append(
            RecordOcrOutput(
                record_index=record.record_index,
                record_dir=record.record_dir,
                cell_results=cell_results,
                raw_json_path=raw_json_path,
            )
        )

    return outputs



def run_ocr_on_page_layout(
    page_image_path: str | Path,
    layout: Any,
    records: Sequence[RecordCropPaths],
    engine: OcrEngine,
    *,
    save_raw_json: bool = True,
) -> list[RecordOcrOutput]:
    """Run OCR once on the full page, then assign OCR words into detected cells.

    This is the production path for Google Vision. It avoids OCRing every crop and
    keeps handwriting context, while still returning the same RecordOcrOutput
    shape used by the parser/exporter.
    """

    page_result = engine.read_image(page_image_path)
    page_words = [line for line in page_result.lines if line.bbox is not None and line.text.strip()]

    outputs: list[RecordOcrOutput] = []
    for record_crop, record_layout in zip(records, layout.records, strict=True):
        cell_results: dict[str, OcrResult] = {}
        for cell_name, cell_box in record_layout.cells.items():
            # Signatures are not useful OCR fields and often pollute nearby dates.
            if cell_name == "tandatangan":
                continue
            assigned = _assign_words_to_box(page_words, cell_box)
            cell_result = _words_to_ocr_result(assigned)

            if _needs_crop_fallback(cell_name, cell_result):
                crop_path = record_crop.cell_paths.get(cell_name)
                if crop_path is not None and crop_path.exists():
                    crop_result = engine.read_image(crop_path)
                    cell_result = _choose_better_cell_result(cell_result, crop_result)

            cell_results[cell_name] = cell_result

        raw_json_path: Path | None = None
        if save_raw_json:
            raw_json_path = record_crop.record_dir / "raw_ocr.json"
            payload = {
                "engine": f"{engine.name}:full_page",
                "record": f"record_{record_crop.record_index:03d}",
                "page_image": str(page_image_path),
                "cells": {cell_name: result.to_dict() for cell_name, result in cell_results.items()},
            }
            raw_json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

        outputs.append(
            RecordOcrOutput(
                record_index=record_crop.record_index,
                record_dir=record_crop.record_dir,
                cell_results=cell_results,
                raw_json_path=raw_json_path,
            )
        )

    if save_raw_json:
        page_json_path = Path(page_image_path).with_suffix(".ocr.json")
        page_json_path.write_text(
            json.dumps(
                {
                    "engine": engine.name,
                    "page_image": str(page_image_path),
                    "average_confidence": page_result.average_confidence,
                    "text": page_result.text,
                    "lines": [line.__dict__ for line in page_result.lines],
                },
                indent=2,
                ensure_ascii=True,
            ),
            encoding="utf-8",
        )

    return outputs


def _needs_crop_fallback(cell_name: str, result: OcrResult) -> bool:
    if cell_name == "tandatangan":
        return False

    important_fields = {
        "bil",
        "suami_isteri",
        "pendaftar",
        "wali",
        "hubungan_wali",
        "saksi",
        "tarikh_nikah",
        "tarikh_keluar",
    }

    if cell_name not in important_fields:
        return False

    text = (result.text or "").strip()

    if not text:
        return True

    if result.average_confidence < 0.60:
        return True

    if cell_name in {"tarikh_nikah", "tarikh_keluar"} and len(text) < 4:
        return True

    if cell_name == "bil" and len(text) < 2:
        return True

    return False


def _choose_better_cell_result(full_page_result: OcrResult, crop_result: OcrResult) -> OcrResult:
    full_text = (full_page_result.text or "").strip()
    crop_text = (crop_result.text or "").strip()

    if crop_text and not full_text:
        return crop_result

    if crop_result.average_confidence >= full_page_result.average_confidence + 0.10:
        return crop_result

    if len(crop_text) > len(full_text) * 1.5 and crop_result.average_confidence >= 0.50:
        return crop_result

    return full_page_result


def _assign_words_to_box(words: Sequence[OcrLine], box: Any) -> list[OcrLine]:
    assigned: list[OcrLine] = []
    for word in words:
        if word.bbox is None:
            continue
        overlap = _bbox_overlap_ratio(word.bbox, [box.x, box.y, box.right, box.bottom])
        center_x = (word.bbox[0] + word.bbox[2]) / 2.0
        center_y = (word.bbox[1] + word.bbox[3]) / 2.0
        center_inside = box.x <= center_x <= box.right and box.y <= center_y <= box.bottom
        if center_inside or overlap >= 0.45:
            assigned.append(word)
    return assigned


def _words_to_ocr_result(words: Sequence[OcrLine]) -> OcrResult:
    if not words:
        return OcrResult(text="", lines=[], average_confidence=0.0)

    sorted_words = sorted(words, key=lambda line: ((line.bbox or [0, 0, 0, 0])[1], (line.bbox or [0, 0, 0, 0])[0]))
    heights = [max(1.0, (line.bbox or [0, 0, 0, 0])[3] - (line.bbox or [0, 0, 0, 0])[1]) for line in sorted_words]
    line_gap = max(10.0, mean(heights) * 0.75) if heights else 12.0

    grouped: list[list[OcrLine]] = []
    for word in sorted_words:
        cy = ((word.bbox or [0, 0, 0, 0])[1] + (word.bbox or [0, 0, 0, 0])[3]) / 2.0
        if not grouped:
            grouped.append([word])
            continue
        last_group = grouped[-1]
        last_cy = mean([((item.bbox or [0, 0, 0, 0])[1] + (item.bbox or [0, 0, 0, 0])[3]) / 2.0 for item in last_group])
        if abs(cy - last_cy) <= line_gap:
            last_group.append(word)
        else:
            grouped.append([word])

    output_lines: list[OcrLine] = []
    for group in grouped:
        group = sorted(group, key=lambda line: (line.bbox or [0, 0, 0, 0])[0])
        text = " ".join(word.text for word in group if word.text.strip()).strip()
        if not text:
            continue
        xs = [coord for word in group for coord in ((word.bbox or [0, 0, 0, 0])[0], (word.bbox or [0, 0, 0, 0])[2])]
        ys = [coord for word in group for coord in ((word.bbox or [0, 0, 0, 0])[1], (word.bbox or [0, 0, 0, 0])[3])]
        output_lines.append(
            OcrLine(
                text=text,
                confidence=mean([word.confidence for word in group]) if group else 0.0,
                bbox=[min(xs), min(ys), max(xs), max(ys)] if xs and ys else None,
            )
        )

    return OcrResult(
        text="\n".join(line.text for line in output_lines),
        lines=output_lines,
        average_confidence=mean([line.confidence for line in output_lines]) if output_lines else 0.0,
    )


def _bbox_overlap_ratio(a: Sequence[float], b: Sequence[float]) -> float:
    left = max(float(a[0]), float(b[0]))
    top = max(float(a[1]), float(b[1]))
    right = min(float(a[2]), float(b[2]))
    bottom = min(float(a[3]), float(b[3]))
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    area_a = max(1.0, (float(a[2]) - float(a[0])) * (float(a[3]) - float(a[1])))
    return intersection / area_a


def _normalize_paddle_output(raw_result: Any) -> list[Any]:
    if raw_result is None:
        return []
    normalized = _normalize_modern_paddle_output(raw_result)
    if normalized is not None:
        return normalized
    if isinstance(raw_result, list) and raw_result and isinstance(raw_result[0], list):
        first = raw_result[0]
        if first and isinstance(first[0], list) and len(first[0]) >= 2:
            return _normalize_legacy_paddle_lines(first)
    if isinstance(raw_result, list):
        return _normalize_legacy_paddle_lines(raw_result)
    return []


def _google_vision_annotation_to_result(annotation: Any) -> OcrResult:
    if annotation is None:
        return OcrResult(text="", lines=[], average_confidence=0.0)

    lines: list[OcrLine] = []
    for page in getattr(annotation, "pages", []) or []:
        for block in getattr(page, "blocks", []) or []:
            for paragraph in getattr(block, "paragraphs", []) or []:
                words: list[str] = []
                confidences: list[float] = []
                vertices_x: list[float] = []
                vertices_y: list[float] = []

                for word in getattr(paragraph, "words", []) or []:
                    word_text = "".join(
                        str(getattr(symbol, "text", ""))
                        for symbol in (getattr(word, "symbols", []) or [])
                    ).strip()
                    if not word_text:
                        continue

                    words.append(word_text)
                    confidences.append(float(getattr(word, "confidence", 0.0) or 0.0))

                    for vertex in getattr(getattr(word, "bounding_box", None), "vertices", []) or []:
                        vertices_x.append(float(getattr(vertex, "x", 0.0) or 0.0))
                        vertices_y.append(float(getattr(vertex, "y", 0.0) or 0.0))

                paragraph_text = " ".join(words).strip()
                if not paragraph_text:
                    continue

                confidence = mean(confidences) if confidences else float(getattr(paragraph, "confidence", 0.0) or 0.0)
                bbox = None
                if vertices_x and vertices_y:
                    bbox = [min(vertices_x), min(vertices_y), max(vertices_x), max(vertices_y)]

                lines.append(OcrLine(text=paragraph_text, confidence=confidence, bbox=bbox))

    average_confidence = mean([line.confidence for line in lines]) if lines else 0.0
    text = str(getattr(annotation, "text", "") or "\n".join(line.text for line in lines))
    return OcrResult(text=text, lines=lines, average_confidence=average_confidence)


def _google_vision_annotation_to_word_result(annotation: Any) -> OcrResult:
    if annotation is None:
        return OcrResult(text="", lines=[], average_confidence=0.0)

    lines: list[OcrLine] = []
    for page in getattr(annotation, "pages", []) or []:
        for block in getattr(page, "blocks", []) or []:
            for paragraph in getattr(block, "paragraphs", []) or []:
                for word in getattr(paragraph, "words", []) or []:
                    word_text = "".join(
                        str(getattr(symbol, "text", ""))
                        for symbol in (getattr(word, "symbols", []) or [])
                    ).strip()
                    if not word_text:
                        continue

                    vertices_x: list[float] = []
                    vertices_y: list[float] = []
                    for vertex in getattr(getattr(word, "bounding_box", None), "vertices", []) or []:
                        vertices_x.append(float(getattr(vertex, "x", 0.0) or 0.0))
                        vertices_y.append(float(getattr(vertex, "y", 0.0) or 0.0))

                    bbox = None
                    if vertices_x and vertices_y:
                        bbox = [min(vertices_x), min(vertices_y), max(vertices_x), max(vertices_y)]

                    lines.append(OcrLine(text=word_text, confidence=float(getattr(word, "confidence", 0.0) or 0.0), bbox=bbox))

    average_confidence = mean([line.confidence for line in lines]) if lines else 0.0
    text = str(getattr(annotation, "text", "") or "\n".join(line.text for line in lines))
    return OcrResult(text=text, lines=lines, average_confidence=average_confidence)


def _polygon_to_bbox(points: Any) -> list[float] | None:
    if hasattr(points, "tolist"):
        points = points.tolist()

    if not isinstance(points, (list, tuple)) or not points:
        return None
    if len(points) == 4 and not isinstance(points[0], (list, tuple)):
        try:
            return [float(points[0]), float(points[1]), float(points[2]), float(points[3])]
        except (TypeError, ValueError):
            return None

    xs: list[float] = []
    ys: list[float] = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        xs.append(float(point[0]))
        ys.append(float(point[1]))

    if not xs or not ys:
        return None
    return [min(xs), min(ys), max(xs), max(ys)]


def _build_paddle_install_guidance(
    *,
    python_version: tuple[int, int] | None = None,
    system: str | None = None,
    machine: str | None = None,
) -> str:
    current_python = python_version or sys.version_info[:2]
    current_system = (system or platform.system()).strip()
    current_machine = (machine or platform.machine()).strip().lower()

    parts = [
        "Install PaddleOCR and its runtime dependencies, or switch `ocr.engine` to `mock`."
    ]

    if current_python < (3, 9) or current_python > (3, 13):
        parts.append(
            f"Current Python is {current_python[0]}.{current_python[1]}; "
            "the current PaddlePaddle installation guide documents Python 3.9-3.13."
        )

    if current_system == "Darwin" and current_machine in {"arm64", "aarch64"}:
        parts.append(
            "Current platform is macOS arm64; use the macOS CPU installation path for "
            "`paddlepaddle` in a supported Python environment."
        )

    return " ".join(parts)


def _build_paddle_constructor_kwargs(
    *,
    paddle_ocr_class: type[Any],
    lang: str,
    use_angle_cls: bool,
    show_log: bool,
    extra_config: Mapping[str, Any],
) -> dict[str, Any]:
    signature = inspect.signature(paddle_ocr_class)
    accepted = set(signature.parameters)
    kwargs: dict[str, Any] = {}

    if "lang" in accepted:
        kwargs["lang"] = lang
    if "use_angle_cls" in accepted:
        kwargs["use_angle_cls"] = use_angle_cls
    elif "use_textline_orientation" in accepted:
        kwargs["use_textline_orientation"] = use_angle_cls
    if "show_log" in accepted:
        kwargs["show_log"] = show_log

    for key, value in extra_config.items():
        if key in accepted:
            kwargs[key] = value

    return kwargs


def _run_paddle_inference(engine: Any, image_path: str, *, use_angle_cls: bool) -> Any:
    predict_method = getattr(engine, "predict", None)
    if callable(predict_method):
        signature = inspect.signature(predict_method)
        kwargs: dict[str, Any] = {}
        if "use_textline_orientation" in signature.parameters:
            kwargs["use_textline_orientation"] = use_angle_cls
        return predict_method(image_path, **kwargs)

    ocr_method = getattr(engine, "ocr", None)
    if not callable(ocr_method):
        raise RuntimeError("Paddle OCR engine does not expose `predict` or `ocr`.")

    signature = inspect.signature(ocr_method)
    if "cls" in signature.parameters:
        return ocr_method(image_path, cls=use_angle_cls)
    return ocr_method(image_path)


def _normalize_modern_paddle_output(raw_result: Any) -> list[dict[str, Any]] | None:
    if not isinstance(raw_result, list) or not raw_result:
        return None

    normalized: list[dict[str, Any]] = []
    recognized = False
    for item in raw_result:
        texts = _read_paddle_mapping_value(item, "rec_texts")
        if not isinstance(texts, list):
            continue

        recognized = True
        scores = _read_paddle_mapping_value(item, "rec_scores")
        boxes = _read_paddle_mapping_value(item, "rec_boxes")
        if boxes is None:
            boxes = _read_paddle_mapping_value(item, "rec_polys")
        if boxes is None:
            boxes = _read_paddle_mapping_value(item, "dt_polys")

        for index, text in enumerate(texts):
            box = boxes[index] if isinstance(boxes, list) and index < len(boxes) else None
            score = scores[index] if isinstance(scores, list) and index < len(scores) else 0.0
            normalized.append(
                {
                    "text": str(text),
                    "confidence": float(score),
                    "bbox": _polygon_to_bbox(box),
                }
            )

    if recognized:
        return normalized
    return None


def _normalize_legacy_paddle_lines(raw_lines: list[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in raw_lines:
        if not item or len(item) < 2:
            continue

        bbox_points = item[0]
        prediction = item[1]
        if not isinstance(prediction, (list, tuple)) or len(prediction) < 2:
            continue

        normalized.append(
            {
                "text": str(prediction[0]),
                "confidence": float(prediction[1]),
                "bbox": _polygon_to_bbox(bbox_points),
            }
        )

    return normalized


def _read_paddle_mapping_value(item: Any, key: str) -> Any:
    if hasattr(item, "tolist"):
        item = item.tolist()
    if isinstance(item, Mapping):
        value = item.get(key)
    else:
        getter = getattr(item, "get", None)
        value = getter(key) if callable(getter) else getattr(item, key, None)

    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


@dataclass(frozen=True)
class RecordCropPaths:
    record_index: int
    record_dir: Path
    full_record_path: Path
    cell_paths: dict[str, Path]


@dataclass(frozen=True)
class RecordOcrOutput:
    record_index: int
    record_dir: Path
    cell_results: dict[str, OcrResult]
    raw_json_path: Path | None
