from marriage_ocr.typed.models import PositionedWord, Region
from marriage_ocr.typed.template import apply_transform, estimate_transform, get_region


def _word(text: str, x: float, y: float) -> PositionedWord:
    return PositionedWord(text, 0.99, x, y, x + 0.03, y + 0.01, 1)


def test_get_region_returns_approved_bil_coordinates() -> None:
    page_number, region = get_region("bil")
    assert page_number == 1
    assert region == Region(0.205, 0.307, 0.600, 0.337)


def test_apply_transform_moves_and_scales_region() -> None:
    transformed = apply_transform(
        Region(0.2, 0.3, 0.4, 0.5),
        transform=type("Transform", (), {"dx": 0.01, "dy": -0.02, "scale_x": 1.02, "scale_y": 0.98})(),
    )
    assert 0.21 <= transformed.x1 <= 0.215
    assert 0.27 <= transformed.y1 <= 0.28


def test_estimate_transform_rejects_extreme_anchor_shift() -> None:
    words = (
        _word("A.", 0.60, 0.70),
        _word("MAKLUMAT", 0.64, 0.70),
        _word("PASANGAN", 0.72, 0.70),
    )
    transform = estimate_transform(words, page_number=1)
    assert transform.safe is False
    assert "unsafe transform" in " ".join(transform.diagnostics).lower()


def test_estimate_transform_clamps_noisy_scale_estimate() -> None:
    words = (
        _word("A", 0.105, 0.425),
        _word("MAKLUMAT", 0.140, 0.425),
        _word("PASANGAN", 0.220, 0.425),
        _word("SUAMI", 0.105, 0.450),
        _word("ISTERI", 0.105, 0.585),
        _word("B", 0.105, 0.751),
        _word("MAKLUMAT", 0.140, 0.751),
        _word("WALI", 0.220, 0.751),
    )
    transform = estimate_transform(words, page_number=1)
    assert transform.safe is True
    assert transform.scale_y == 1.0
