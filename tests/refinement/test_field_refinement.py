from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from marriage_ocr.models import OcrLine, OcrResult
from marriage_ocr.refinement.field_refinement import refine_field
from marriage_ocr.refinement.models import FieldRefinementSettings


def _write_crop(path: Path) -> None:
    image = np.full((24, 48, 3), 255, dtype=np.uint8)
    cv2.putText(image, "ALI", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA)
    assert cv2.imwrite(str(path), image)


class FakeEngine:
    name = "fake"

    def __init__(self, results: dict[str, OcrResult] | None = None, *, error_on: set[str] | None = None) -> None:
        self._results = dict(results or {})
        self._error_on = set(error_on or set())
        self.calls: list[str] = []

    def read_image(self, image_path: str | Path) -> OcrResult:
        path = Path(image_path)
        self.calls.append(path.name)
        if path.name in self._error_on:
            raise RuntimeError(f"boom:{path.name}")
        return self._results.get(
            path.name,
            OcrResult(
                text="",
                lines=[],
                average_confidence=0.0,
            ),
        )


def _ocr_result(text: str, confidence: float = 0.9) -> OcrResult:
    return OcrResult(
        text=text,
        lines=[OcrLine(text=text, confidence=confidence, bbox=[0.0, 0.0, 10.0, 10.0])],
        average_confidence=confidence,
    )


def test_valid_name_does_not_trigger_retry_ocr(tmp_path: Path) -> None:
    crop_path = tmp_path / "nama_suami.png"
    _write_crop(crop_path)
    engine = FakeEngine()

    decision = refine_field(
        "nama_suami",
        "SITI BINTI ALI",
        crop_path=crop_path,
        engine=engine,
    )

    assert decision.selected_value == "SITI BINTI ALI"
    assert decision.requires_review is False
    assert engine.calls == []


def test_suspicious_name_triggers_retry_ocr(tmp_path: Path) -> None:
    crop_path = tmp_path / "nama_suami.png"
    _write_crop(crop_path)
    engine = FakeEngine(
        {
            "original.png": _ocr_result("AHMAD B1N ALI", 0.62),
            "grayscale.png": _ocr_result("AHMAD BIN ALI", 0.93),
            "thresholded.png": _ocr_result("AHMAD BIN ALI", 0.88),
        }
    )

    decision = refine_field(
        "nama_suami",
        "AHMAD B1N ALI",
        crop_path=crop_path,
        engine=engine,
    )

    assert decision.selected_value == "AHMAD BIN ALI"
    assert decision.requires_review is False
    assert engine.calls == ["original.png", "grayscale.png", "thresholded.png"]


def test_ocr_failure_falls_back_to_original_value(tmp_path: Path) -> None:
    crop_path = tmp_path / "nama_suami.png"
    _write_crop(crop_path)
    engine = FakeEngine(error_on={"original.png", "grayscale.png", "thresholded.png"})

    decision = refine_field(
        "nama_suami",
        "AHMAD B1N ALI",
        crop_path=crop_path,
        engine=engine,
    )

    assert decision.selected_value == "AHMAD B1N ALI"
    assert decision.requires_review is True
    assert decision.reason == "retry_ocr_failed"


def test_maximum_retry_count_is_honored(tmp_path: Path) -> None:
    crop_path = tmp_path / "nama_suami.png"
    _write_crop(crop_path)
    engine = FakeEngine(
        {
            "original.png": _ocr_result("AHMAD B1N ALI", 0.61),
            "grayscale.png": _ocr_result("AHMAD BIN ALI", 0.86),
            "thresholded.png": _ocr_result("AHMAD BIN ALI", 0.95),
        }
    )

    decision = refine_field(
        "nama_suami",
        "AHMAD B1N ALI",
        crop_path=crop_path,
        engine=engine,
        settings=FieldRefinementSettings(max_variants_per_field=2),
    )

    assert decision.selected_value == "AHMAD BIN ALI"
    assert engine.calls == ["original.png", "grayscale.png"]
