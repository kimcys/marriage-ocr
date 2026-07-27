from pathlib import Path

import cv2
import numpy as np

from marriage_ocr.typed.models import RawField, Region, RenderedPage
from marriage_ocr.typed.retry import create_retry_crops, prefer_retry_value


def test_create_retry_crops_expands_region_and_writes_image(tmp_path: Path) -> None:
    image = np.full((1000, 800, 3), 255, dtype=np.uint8)
    image_path = tmp_path / "page_1.png"
    cv2.imwrite(str(image_path), image)
    page = RenderedPage(tmp_path / "record.pdf", "record.pdf", 1, image_path, 800, 1000)

    crops = create_retry_crops(
        pages=(page,),
        field_keys=("bil",),
        retry_dir=tmp_path / "retries",
        padding_ratio=0.05,
    )

    assert len(crops) == 1
    assert crops[0].crop_path.exists()
    crop = cv2.imread(str(crops[0].crop_path))
    assert crop.shape[0] > int((0.337 - 0.307) * 1000)


def test_prefer_retry_value_requires_valid_improvement() -> None:
    original = RawField("bil", "Bil", 1, Region(0, 0, 1, 1), "Bilangan Daftar", 0.40)
    retried = RawField("bil", "Bil", 1, Region(0, 0, 1, 1), "04/2009", 0.90)
    assert prefer_retry_value(original, retried, original_valid=False, retry_valid=True) == retried
    assert prefer_retry_value(original, retried, original_valid=True, retry_valid=False) == original

