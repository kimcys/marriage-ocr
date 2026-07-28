from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile

import cv2


@dataclass(frozen=True)
class RetryVariant:
    name: str
    source: str
    image_path: Path
    temporary: bool = False


def build_retry_variants(
    crop_path: Path,
    *,
    padding_ratio: float = 0.05,
    save_dir: Path | None = None,
) -> list[RetryVariant]:
    image = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to load retry crop: {crop_path}")

    output_dir, temporary = _prepare_output_dir(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    padded = _pad_image(image, padding_ratio=padding_ratio)
    grayscale = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)
    grayscale = cv2.resize(grayscale, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    grayscale = cv2.convertScaleAbs(grayscale, alpha=1.2, beta=8.0)
    _, thresholded = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    variants = [
        RetryVariant("original", "retry_original", output_dir / "original.png", temporary=temporary),
        RetryVariant("grayscale", "retry_grayscale", output_dir / "grayscale.png", temporary=temporary),
        RetryVariant("thresholded", "retry_thresholded", output_dir / "thresholded.png", temporary=temporary),
    ]

    if not cv2.imwrite(str(variants[0].image_path), padded):
        raise ValueError(f"Failed to write retry variant: {variants[0].image_path}")
    if not cv2.imwrite(str(variants[1].image_path), grayscale):
        raise ValueError(f"Failed to write retry variant: {variants[1].image_path}")
    if not cv2.imwrite(str(variants[2].image_path), thresholded):
        raise ValueError(f"Failed to write retry variant: {variants[2].image_path}")

    return variants[:3]


def cleanup_retry_variants(variants: list[RetryVariant]) -> None:
    temporary_dirs = {variant.image_path.parent for variant in variants if variant.temporary}
    for directory in sorted(temporary_dirs):
        shutil.rmtree(directory, ignore_errors=True)


def _prepare_output_dir(save_dir: Path | None) -> tuple[Path, bool]:
    if save_dir is not None:
        return save_dir, False
    return Path(tempfile.mkdtemp(prefix="marriage-ocr-retry-")), True


def _pad_image(image, *, padding_ratio: float) -> object:
    height, width = image.shape[:2]
    base = max(height, width)
    padding = max(1, int(round(base * max(0.0, padding_ratio))))
    return cv2.copyMakeBorder(
        image,
        padding,
        padding,
        padding,
        padding,
        borderType=cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )
