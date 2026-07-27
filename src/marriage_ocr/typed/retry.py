from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import cv2

from marriage_ocr.typed.extractor import join_words_in_reading_order
from marriage_ocr.typed.models import PageOcrResult, RawField, RenderedPage, RetryCrop
from marriage_ocr.typed.template import get_region
from marriage_ocr.typed.vision import TypedVisionClient


def create_retry_crops(
    *,
    pages: Sequence[RenderedPage],
    field_keys: Sequence[str],
    retry_dir: Path,
    padding_ratio: float = 0.05,
) -> tuple[RetryCrop, ...]:
    retry_crops: list[RetryCrop] = []
    page_lookup = {page.page_number: page for page in pages}
    retry_dir.mkdir(parents=True, exist_ok=True)
    for field_key in field_keys:
        page_number, region = get_region(field_key)
        page = page_lookup.get(page_number)
        if page is None:
            raise ValueError(f"Retry field {field_key} has no rendered page {page_number}")
        image = cv2.imread(str(page.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read rendered page: {page.image_path}")
        expanded = region.expand(padding_ratio)
        x1 = max(0, int(expanded.x1 * page.width))
        y1 = max(0, int(expanded.y1 * page.height))
        x2 = min(page.width, int(expanded.x2 * page.width + 0.9999))
        y2 = min(page.height, int(expanded.y2 * page.height + 0.9999))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Empty retry crop for field {field_key}")
        crop = image[y1:y2, x1:x2]
        field_dir = retry_dir / "retries"
        field_dir.mkdir(parents=True, exist_ok=True)
        crop_path = field_dir / f"{field_key}.png"
        if not cv2.imwrite(str(crop_path), crop):
            raise OSError(f"Failed to write retry crop: {crop_path}")
        retry_crops.append(
            RetryCrop(
                source_file=page.source_file,
                field_key=field_key,
                page_number=page_number,
                crop_path=crop_path,
                region=expanded,
            )
        )
    return tuple(retry_crops)


def extract_retry_raw_fields(
    retry_crops: Sequence[RetryCrop],
    client: TypedVisionClient,
) -> dict[str, RawField]:
    items = [(crop.source_file, crop.page_number, crop.crop_path) for crop in retry_crops]
    results = client.annotate_image_paths(items)
    extracted: dict[str, RawField] = {}
    for crop, result in zip(retry_crops, results, strict=True):
        extracted[crop.field_key] = RawField(
            key=crop.field_key,
            output_name=crop.field_key.replace("_", " ").title(),
            page_number=crop.page_number,
            region=crop.region,
            raw_text=result.full_text or join_words_in_reading_order(result.words),
            confidence=max((word.confidence for word in result.words), default=0.0),
            words=result.words,
        )
    return extracted


def prefer_retry_value(
    original: RawField,
    retried: RawField,
    *,
    original_valid: bool,
    retry_valid: bool,
) -> RawField:
    if retry_valid and not original_valid:
        return retried
    if retry_valid and retried.confidence >= original.confidence + 0.05:
        return retried
    return original
