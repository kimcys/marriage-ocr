import numpy as np
import pytest

from marriage_ocr import preprocess as preprocess_module
from marriage_ocr.preprocess import PreprocessSettings, preprocess_image


def test_preprocess_supports_otsu_threshold() -> None:
    image = np.full((32, 32, 3), 255, dtype=np.uint8)
    image[:, 16:] = 0

    result = preprocess_image(
        image,
        PreprocessSettings(
            processing_width=0,
            deskew_enabled=False,
            threshold_method="otsu",
        ),
    )

    assert result.binary.shape == (32, 32)
    assert set(np.unique(result.binary).tolist()).issubset({0, 255})


def test_preprocess_rejects_unknown_threshold_method() -> None:
    image = np.full((8, 8, 3), 255, dtype=np.uint8)

    with pytest.raises(ValueError, match="Unsupported threshold method"):
        preprocess_image(
            image,
            PreprocessSettings(
                processing_width=0,
                deskew_enabled=False,
                threshold_method="invalid",
            ),
        )


@pytest.mark.parametrize(
    ("hough_lines",),
    [
        (np.array([[[0, 0, 10, 0]]], dtype=np.int32),),
        (np.array([[0, 0, 10, 0]], dtype=np.int32),),
    ],
)
def test_preprocess_handles_both_hough_line_shapes(monkeypatch: pytest.MonkeyPatch, hough_lines: np.ndarray) -> None:
    image = np.full((32, 32, 3), 255, dtype=np.uint8)

    monkeypatch.setattr(preprocess_module.cv2, "HoughLinesP", lambda *args, **kwargs: hough_lines)

    result = preprocess_image(
        image,
        PreprocessSettings(
            processing_width=0,
            threshold_method="otsu",
        ),
    )

    assert result.color.shape == image.shape
    assert result.rotation_applied == 0.0
