from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class PreprocessSettings:
    processing_width: int = 2200
    expected_landscape: bool = True
    denoise_kernel_size: int = 5
    threshold_method: str = "adaptive"
    adaptive_block_size: int = 31
    adaptive_c: int = 15
    deskew_enabled: bool = True
    deskew_max_angle: float = 7.5
    min_rotation_for_deskew: float = 0.15
    hough_threshold: int = 150
    hough_min_line_length_ratio: float = 0.35
    hough_max_line_gap: int = 20


@dataclass(frozen=True)
class PreprocessResult:
    color: np.ndarray
    grayscale: np.ndarray
    binary: np.ndarray
    rotation_applied: float
    original_size: tuple[int, int]
    processed_size: tuple[int, int]


def preprocess_image(image: np.ndarray, settings: PreprocessSettings) -> PreprocessResult:
    if image.size == 0:
        raise ValueError("Cannot preprocess an empty image")

    normalized = _ensure_landscape(image, settings.expected_landscape)
    resized = _resize_to_width(normalized, settings.processing_width)

    rotation_applied = 0.0
    if settings.deskew_enabled:
        estimated_skew = _estimate_skew_angle(resized, settings)
        if abs(estimated_skew) >= settings.min_rotation_for_deskew:
            resized = _rotate_image(resized, -estimated_skew)
            rotation_applied = -estimated_skew

    grayscale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    denoised = cv2.GaussianBlur(grayscale, (_odd(settings.denoise_kernel_size), _odd(settings.denoise_kernel_size)), 0)
    binary = _threshold_image(denoised, settings)

    return PreprocessResult(
        color=resized,
        grayscale=grayscale,
        binary=binary,
        rotation_applied=rotation_applied,
        original_size=(image.shape[1], image.shape[0]),
        processed_size=(resized.shape[1], resized.shape[0]),
    )


def _ensure_landscape(image: np.ndarray, expected_landscape: bool) -> np.ndarray:
    if expected_landscape and image.shape[0] > image.shape[1]:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image


def _resize_to_width(image: np.ndarray, processing_width: int) -> np.ndarray:
    if processing_width <= 0:
        return image

    height, width = image.shape[:2]
    if width == processing_width:
        return image

    scale = processing_width / float(width)
    resized_height = max(1, int(round(height * scale)))
    return cv2.resize(image, (processing_width, resized_height), interpolation=cv2.INTER_CUBIC)


def _threshold_image(image: np.ndarray, settings: PreprocessSettings) -> np.ndarray:
    method = settings.threshold_method.strip().lower()
    if method == "otsu":
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    if method != "adaptive":
        raise ValueError(f"Unsupported threshold method: {settings.threshold_method}")

    return cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        _threshold_block_size(settings.adaptive_block_size),
        settings.adaptive_c,
    )


def _estimate_skew_angle(image: np.ndarray, settings: PreprocessSettings) -> float:
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.GaussianBlur(grayscale, (_odd(settings.denoise_kernel_size), _odd(settings.denoise_kernel_size)), 0)
    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        _threshold_block_size(settings.adaptive_block_size),
        settings.adaptive_c,
    )

    min_line_length = max(100, int(binary.shape[1] * settings.hough_min_line_length_ratio))
    lines = cv2.HoughLinesP(
        binary,
        rho=1,
        theta=np.pi / 180.0,
        threshold=settings.hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=settings.hough_max_line_gap,
    )

    if lines is None:
        return 0.0

    angles: list[float] = []
    for raw_line in lines:
        x1, y1, x2, y2 = _extract_hough_line_points(raw_line)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        normalized = _normalize_line_angle(angle, settings.deskew_max_angle)
        if normalized is not None:
            angles.append(normalized)

    if not angles:
        return 0.0

    return float(np.median(np.array(angles, dtype=np.float32)))


def _normalize_line_angle(angle: float, max_angle: float) -> float | None:
    folded = _fold_angle(angle)
    for candidate in (folded, folded - 90.0, folded + 90.0):
        if abs(candidate) <= max_angle:
            return candidate
    return None


def _fold_angle(angle: float) -> float:
    while angle <= -90.0:
        angle += 180.0
    while angle > 90.0:
        angle -= 180.0
    return angle


def _extract_hough_line_points(raw_line: np.ndarray) -> tuple[int, int, int, int]:
    points = np.asarray(raw_line).reshape(-1)
    if points.size != 4:
        raise ValueError(f"Unexpected Hough line shape: {np.asarray(raw_line).shape}")
    x1, y1, x2, y2 = (int(value) for value in points.tolist())
    return x1, y1, x2, y2


def _rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos_value = abs(rotation_matrix[0, 0])
    sin_value = abs(rotation_matrix[0, 1])
    new_width = int((height * sin_value) + (width * cos_value))
    new_height = int((height * cos_value) + (width * sin_value))

    rotation_matrix[0, 2] += (new_width / 2.0) - center[0]
    rotation_matrix[1, 2] += (new_height / 2.0) - center[1]

    return cv2.warpAffine(
        image,
        rotation_matrix,
        (new_width, new_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def _odd(value: int) -> int:
    normalized = max(1, int(value))
    if normalized % 2 == 0:
        normalized += 1
    return normalized


def _threshold_block_size(value: int) -> int:
    return max(3, _odd(value))
