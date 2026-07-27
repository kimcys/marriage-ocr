from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from marriage_ocr.refinement.preprocess import build_retry_variants


def _write_crop(path: Path) -> None:
    image = np.full((24, 48, 3), 255, dtype=np.uint8)
    cv2.line(image, (1, 2), (1, 21), (0, 0, 0), 2)
    cv2.putText(image, "ALI", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
    assert cv2.imwrite(str(path), image)


def test_build_retry_variants_includes_expected_variants(tmp_path: Path) -> None:
    crop_path = tmp_path / "name.png"
    _write_crop(crop_path)

    variants = build_retry_variants(crop_path, save_dir=tmp_path / "variants")

    assert [variant.name for variant in variants] == ["original", "grayscale", "thresholded"]
    assert all(variant.image_path.exists() for variant in variants)

    original = cv2.imread(str(variants[0].image_path), cv2.IMREAD_COLOR)
    grayscale = cv2.imread(str(variants[1].image_path), cv2.IMREAD_GRAYSCALE)
    thresholded = cv2.imread(str(variants[2].image_path), cv2.IMREAD_GRAYSCALE)

    assert original is not None
    assert grayscale is not None
    assert thresholded is not None
    assert original.shape[0] > 24
    assert original.shape[1] > 48
    assert grayscale.shape[0] > original.shape[0]
    assert grayscale.shape[1] > original.shape[1]
    assert set(np.unique(thresholded)).issubset({0, 255})


def test_build_retry_variants_is_deterministic(tmp_path: Path) -> None:
    crop_path = tmp_path / "name.png"
    _write_crop(crop_path)

    first = build_retry_variants(crop_path, save_dir=tmp_path / "first")
    second = build_retry_variants(crop_path, save_dir=tmp_path / "second")

    assert [variant.name for variant in first] == [variant.name for variant in second]
    assert [variant.image_path.read_bytes() for variant in first] == [
        variant.image_path.read_bytes() for variant in second
    ]
