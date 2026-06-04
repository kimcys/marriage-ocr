from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Mapping

import cv2
import numpy as np


DEFAULT_COLUMN_ORDER = [
    "bil",
    "suami_isteri",
    "pendaftar",
    "wali",
    "hubungan_wali",
    "saksi",
    "tarikh_nikah",
    "tandatangan",
    "tarikh_keluar",
    "remarks",
]


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    @property
    def center_y(self) -> int:
        return self.y + (self.height // 2)

    @classmethod
    def from_bounds(cls, left: int, top: int, right: int, bottom: int) -> Box:
        return cls(
            x=int(left),
            y=int(top),
            width=max(1, int(right) - int(left)),
            height=max(1, int(bottom) - int(top)),
        )

    def inset(self, padding: int) -> Box:
        if padding <= 0:
            return self

        horizontal_padding = min(padding, max(0, (self.width - 1) // 2))
        vertical_padding = min(padding, max(0, (self.height - 1) // 2))
        return Box.from_bounds(
            self.x + horizontal_padding,
            self.y + vertical_padding,
            self.right - horizontal_padding,
            self.bottom - vertical_padding,
        )

    def clamp(self, max_width: int, max_height: int) -> Box:
        left = min(max(self.x, 0), max_width - 1)
        top = min(max(self.y, 0), max_height - 1)
        right = min(max(self.right, left + 1), max_width)
        bottom = min(max(self.bottom, top + 1), max_height)
        return Box.from_bounds(left, top, right, bottom)

    def as_rect(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)

    def slices(self) -> tuple[slice, slice]:
        return slice(self.y, self.bottom), slice(self.x, self.right)


@dataclass(frozen=True)
class LineCandidate:
    y: int
    coverage: int


@dataclass
class RecordLayout:
    index: int
    box: Box
    marker_box: Box | None = None
    cells: dict[str, Box] = field(default_factory=dict)


@dataclass
class TableLayout:
    table_box: Box
    column_order: list[str]
    column_edges: list[int]
    vertical_line_positions: list[int]
    horizontal_line_positions: list[int]
    horizontal_line_candidates: list[LineCandidate]
    records: list[RecordLayout]
    vertical_mask: np.ndarray
    horizontal_mask: np.ndarray
    line_mask: np.ndarray
    ocr_ready_color: np.ndarray


def detect_layout(
    color_image: np.ndarray,
    binary_image: np.ndarray,
    layout_config: Mapping[str, Any],
) -> TableLayout:
    if color_image.shape[:2] != binary_image.shape[:2]:
        raise ValueError("Color and binary images must have the same dimensions")

    inverse_binary = cv2.bitwise_not(binary_image)
    vertical_mask, horizontal_mask = _extract_line_masks(inverse_binary)
    line_mask = cv2.bitwise_or(vertical_mask, horizontal_mask)

    table_box = _detect_table_box(vertical_mask, horizontal_mask, binary_image)
    fallback_columns = _load_fallback_column_spans(layout_config)
    column_order = list(fallback_columns.keys()) or list(DEFAULT_COLUMN_ORDER)
    column_edges, vertical_positions = _detect_column_edges(
        vertical_mask=vertical_mask,
        table_box=table_box,
        fallback_columns=fallback_columns,
    )
    line_candidates = _detect_horizontal_lines(horizontal_mask, table_box)
    marker_boxes = _detect_bil_markers(
        binary_image=binary_image,
        vertical_mask=vertical_mask,
        horizontal_mask=horizontal_mask,
        table_box=table_box,
        bil_right=column_edges[1],
    )
    record_boxes = _detect_record_boxes(
        table_box=table_box,
        marker_boxes=marker_boxes,
        line_candidates=line_candidates,
        min_record_height=int(layout_config.get("min_record_height_px", 80)),
        max_record_height=int(layout_config.get("max_record_height_px", 280)),
    )
    record_layouts = _build_record_layouts(
        record_boxes=record_boxes,
        marker_boxes=marker_boxes,
        column_order=column_order,
        column_edges=column_edges,
        image_shape=binary_image.shape,
    )

    return TableLayout(
        table_box=table_box,
        column_order=column_order,
        column_edges=column_edges,
        vertical_line_positions=vertical_positions,
        horizontal_line_positions=[candidate.y for candidate in line_candidates],
        horizontal_line_candidates=line_candidates,
        records=record_layouts,
        vertical_mask=vertical_mask,
        horizontal_mask=horizontal_mask,
        line_mask=line_mask,
        ocr_ready_color=_remove_table_lines(color_image, line_mask),
    )


def create_table_overlay(color_image: np.ndarray, layout: TableLayout) -> np.ndarray:
    overlay = color_image.copy()
    box = layout.table_box
    cv2.rectangle(overlay, (box.x, box.y), (box.right, box.bottom), (40, 220, 40), 3)

    for index, x in enumerate(layout.column_edges):
        cv2.line(overlay, (x, box.y), (x, box.bottom), (255, 120, 0), 2)
        if 0 < index < len(layout.column_edges):
            label = layout.column_order[index - 1]
            cv2.putText(
                overlay,
                label,
                (max(8, x - 18), max(24, box.y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 120, 0),
                1,
                cv2.LINE_AA,
            )

    return overlay


def create_record_overlay(color_image: np.ndarray, layout: TableLayout) -> np.ndarray:
    overlay = color_image.copy()

    for record in layout.records:
        box = record.box
        cv2.rectangle(overlay, (box.x, box.y), (box.right, box.bottom), (0, 200, 255), 3)
        cv2.putText(
            overlay,
            f"record_{record.index:03d}",
            (box.x + 6, max(22, box.y + 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )

        if record.marker_box is not None:
            marker = record.marker_box
            cv2.rectangle(overlay, (marker.x, marker.y), (marker.right, marker.bottom), (0, 80, 255), 2)

    return overlay


def _extract_line_masks(inverse_binary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = inverse_binary.shape
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(25, height // 25)))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, width // 25), 1))

    vertical_mask = cv2.morphologyEx(inverse_binary, cv2.MORPH_OPEN, vertical_kernel)
    horizontal_mask = cv2.morphologyEx(inverse_binary, cv2.MORPH_OPEN, horizontal_kernel)
    return vertical_mask, horizontal_mask


def _detect_table_box(
    vertical_mask: np.ndarray,
    horizontal_mask: np.ndarray,
    binary_image: np.ndarray,
) -> Box:
    combined = cv2.add(vertical_mask, horizontal_mask)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, close_kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        image_height, image_width = binary_image.shape
        image_area = image_height * image_width
        candidates: list[tuple[int, Box]] = []
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            area = width * height
            if width < image_width * 0.50 or height < image_height * 0.35:
                continue
            if area < image_area * 0.15:
                continue
            candidates.append((area, Box(x=x, y=y, width=width, height=height)))

        if candidates:
            return max(candidates, key=lambda item: item[0])[1]

    return Box(x=0, y=0, width=binary_image.shape[1], height=binary_image.shape[0])


def _load_fallback_column_spans(layout_config: Mapping[str, Any]) -> dict[str, tuple[float, float]]:
    fallback = layout_config.get("fallback_column_ratios", {})
    if not isinstance(fallback, Mapping):
        return {
            name: span
            for name, span in zip(
                DEFAULT_COLUMN_ORDER,
                [
                    (0.00, 0.08),
                    (0.08, 0.25),
                    (0.25, 0.43),
                    (0.43, 0.50),
                    (0.50, 0.58),
                    (0.58, 0.70),
                    (0.70, 0.77),
                    (0.77, 0.84),
                    (0.84, 0.91),
                    (0.91, 1.00),
                ],
            )
        }

    normalized: dict[str, tuple[float, float]] = {}
    for name, value in fallback.items():
        if isinstance(value, (list, tuple)) and len(value) == 2:
            normalized[str(name)] = (float(value[0]), float(value[1]))
    return normalized


def _detect_column_edges(
    vertical_mask: np.ndarray,
    table_box: Box,
    fallback_columns: Mapping[str, tuple[float, float]],
) -> tuple[list[int], list[int]]:
    contours, _ = cv2.findContours(vertical_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_positions: list[int] = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if height < table_box.height * 0.15:
            continue
        if x < table_box.x - 30 or x > table_box.right + 30:
            continue
        if y > table_box.bottom or y + height < table_box.y:
            continue
        detected_positions.append(x + (width // 2))

    detected_positions = sorted(set(detected_positions))
    expected_edges = [table_box.x]
    for _, (_, end_ratio) in fallback_columns.items():
        expected_edges.append(table_box.x + int(round(table_box.width * end_ratio)))
    expected_edges[-1] = table_box.right

    tolerance = max(24, int(table_box.width * 0.03))
    snapped = [expected_edges[0]]
    for expected in expected_edges[1:-1]:
        nearby = [position for position in detected_positions if abs(position - expected) <= tolerance]
        snapped.append(min(nearby, key=lambda position: abs(position - expected)) if nearby else expected)
    snapped.append(expected_edges[-1])

    edges = _normalize_edges(snapped, minimum_gap=max(18, int(table_box.width * 0.02)))
    return edges, detected_positions


def _normalize_edges(edges: list[int], minimum_gap: int) -> list[int]:
    if not edges:
        return []

    normalized = [int(edges[0])]
    for edge in edges[1:]:
        coerced = max(normalized[-1] + minimum_gap, int(edge))
        normalized.append(coerced)
    return normalized


def _detect_horizontal_lines(horizontal_mask: np.ndarray, table_box: Box) -> list[LineCandidate]:
    row_slice, column_slice = table_box.slices()
    table_rows = horizontal_mask[row_slice, column_slice]
    coverage = (table_rows > 0).sum(axis=1)

    threshold = max(120, int(table_box.width * 0.25))
    candidates: list[LineCandidate] = []
    in_run = False
    run_start = 0
    best_index = 0
    best_value = 0

    for index, value in enumerate(coverage):
        value_int = int(value)
        if value_int >= threshold:
            if not in_run:
                in_run = True
                run_start = index
                best_index = index
                best_value = value_int
            elif value_int > best_value:
                best_index = index
                best_value = value_int
        elif in_run:
            candidates.append(LineCandidate(y=table_box.y + best_index, coverage=best_value))
            in_run = False

    if in_run:
        candidates.append(LineCandidate(y=table_box.y + best_index, coverage=best_value))

    return candidates


def _detect_bil_markers(
    binary_image: np.ndarray,
    vertical_mask: np.ndarray,
    horizontal_mask: np.ndarray,
    table_box: Box,
    bil_right: int,
) -> list[Box]:
    bil_left = min(max(table_box.x + 2, 0), bil_right - 1)
    bil_right = max(bil_left + 1, bil_right - 2)

    inverse_binary = cv2.bitwise_not(binary_image)
    text_mask = cv2.subtract(inverse_binary, cv2.bitwise_or(vertical_mask, horizontal_mask))
    row_slice = slice(table_box.y, table_box.bottom)
    column_slice = slice(bil_left, bil_right)
    region = text_mask[row_slice, column_slice]
    row_density = (region > 0).sum(axis=1)
    threshold = max(18, int(region.shape[1] * 0.12))

    markers: list[Box] = []
    in_run = False
    run_start = 0
    run_end = 0
    for index, value in enumerate(row_density):
        if int(value) >= threshold:
            if not in_run:
                in_run = True
                run_start = index
            run_end = index
        elif in_run:
            marker = _build_marker_box(region, bil_left, table_box.y, run_start, run_end)
            if marker is not None:
                markers.append(marker)
            in_run = False

    if in_run:
        marker = _build_marker_box(region, bil_left, table_box.y, run_start, run_end)
        if marker is not None:
            markers.append(marker)

    return markers


def _build_marker_box(
    region: np.ndarray,
    absolute_left: int,
    absolute_top: int,
    run_start: int,
    run_end: int,
) -> Box | None:
    run_length = run_end - run_start + 1
    # Real red BIL markers in the sample are around 22-34 px high at the
    # configured processing width. Tiny 8-10 px blobs near the bottom border were
    # creating empty phantom records.
    if run_length < 14:
        return None

    run_region = region[run_start : run_end + 1]
    nonzero_points = cv2.findNonZero(run_region)
    if nonzero_points is None:
        return None

    x, y, width, height = cv2.boundingRect(nonzero_points)
    if height < 14 or width < 35:
        return None
    return Box(
        x=absolute_left + x,
        y=absolute_top + run_start + y,
        width=width,
        height=height,
    )


def _detect_record_boxes(
    table_box: Box,
    marker_boxes: list[Box],
    line_candidates: list[LineCandidate],
    min_record_height: int,
    max_record_height: int,
) -> list[Box]:
    if marker_boxes:
        marker_boxes = _merge_close_marker_boxes(
            marker_boxes,
            min_gap=max(80, int(min_record_height * 0.85)),
        )
        starts = _select_record_starts(
            table_box=table_box,
            marker_boxes=marker_boxes,
            line_candidates=line_candidates,
            min_record_height=min_record_height,
            max_record_height=max_record_height,
        )
        boxes: list[Box] = []
        for index, start in enumerate(starts):
            end = starts[index + 1] if index + 1 < len(starts) else table_box.bottom
            if end - start < max(30, min_record_height // 2):
                continue
            boxes.append(Box.from_bounds(table_box.x, start, table_box.right, end))
        if boxes:
            return boxes

    return [table_box]

def _merge_close_marker_boxes(marker_boxes: list[Box], min_gap: int) -> list[Box]:
    if not marker_boxes:
        return []

    sorted_markers = sorted(marker_boxes, key=lambda box: box.y)
    merged: list[Box] = []
    group: list[Box] = []

    for marker in sorted_markers:
        if not group or marker.y - group[-1].y <= min_gap:
            group.append(marker)
            continue

        merged.append(_best_marker_in_group(group))
        group = [marker]

    if group:
        merged.append(_best_marker_in_group(group))

    return merged


def _best_marker_in_group(group: list[Box]) -> Box:
    return max(group, key=lambda box: (box.width * box.height, box.y))

def _select_record_starts(
    table_box: Box,
    marker_boxes: list[Box],
    line_candidates: list[LineCandidate],
    min_record_height: int,
    max_record_height: int,
) -> list[int]:
    if len(marker_boxes) == 1:
        return [max(table_box.y, marker_boxes[0].y - 20)]

    candidate_map = {candidate.y: candidate.coverage for candidate in line_candidates}
    search_window = max(140, min_record_height)
    candidates_per_marker: list[list[int]] = []

    for index, marker in enumerate(marker_boxes):
        eligible = [
            candidate.y
            for candidate in line_candidates
            if candidate.y < marker.y and marker.y - candidate.y <= search_window
        ]
        eligible = eligible[-5:]

        fallback = max(table_box.y, marker.y - 20)
        marker_candidates = list(eligible) if eligible else [fallback]
        if index == 0 and not eligible:
            marker_candidates.append(table_box.y)
        marker_candidates = sorted(set(marker_candidates))
        candidates_per_marker.append(marker_candidates)

    if len(marker_boxes) <= 8:
        starts = _optimize_record_starts(
            table_box=table_box,
            marker_boxes=marker_boxes,
            candidates_per_marker=candidates_per_marker,
            candidate_map=candidate_map,
            min_record_height=min_record_height,
            max_record_height=max_record_height,
        )
        if starts:
            return starts

    starts: list[int] = []
    for marker, candidates in zip(marker_boxes, candidates_per_marker, strict=True):
        strong_candidates = [candidate for candidate in candidates if candidate in candidate_map]
        start = strong_candidates[-1] if strong_candidates else candidates[-1]
        if starts and start <= starts[-1] + 20:
            start = max(starts[-1] + max(20, min_record_height // 2), start)
        starts.append(start)

    return starts


def _optimize_record_starts(
    table_box: Box,
    marker_boxes: list[Box],
    candidates_per_marker: list[list[int]],
    candidate_map: Mapping[int, int],
    min_record_height: int,
    max_record_height: int,
) -> list[int]:
    expected_height = table_box.height / max(1, len(marker_boxes))
    allowed_max = max(max_record_height * 1.5, expected_height * 1.25)
    target_offset = 20
    best_cost: float | None = None
    best_sequence: list[int] | None = None

    for sequence in product(*candidates_per_marker):
        starts = list(sequence)
        if any(starts[index] >= marker_boxes[index].y for index in range(len(starts))):
            continue
        if any(starts[index] <= starts[index - 1] + 20 for index in range(1, len(starts))):
            continue

        ends = starts[1:] + [table_box.bottom]
        cost = 0.0
        valid = True

        for index, (marker, start, end) in enumerate(zip(marker_boxes, starts, ends, strict=True)):
            height = end - start
            if height <= 0:
                valid = False
                break

            offset = marker.y - start
            cost += abs(offset - target_offset) * 2.0
            cost += abs(height - expected_height) * 2.0

            if height < min_record_height * 0.75:
                cost += 250.0 + (min_record_height - height) * 4.0
            if height > allowed_max:
                cost += 120.0 + (height - allowed_max) * 2.5

            coverage = candidate_map.get(start)
            if coverage is not None:
                cost -= coverage / 120.0
            else:
                cost += 25.0

        if not valid:
            continue
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_sequence = starts

    return best_sequence or []


def _build_record_layouts(
    record_boxes: list[Box],
    marker_boxes: list[Box],
    column_order: list[str],
    column_edges: list[int],
    image_shape: tuple[int, int],
) -> list[RecordLayout]:
    image_height, image_width = image_shape
    marker_by_center = sorted(marker_boxes, key=lambda marker: marker.center_y)
    used_markers: set[int] = set()
    layouts: list[RecordLayout] = []

    for index, record_box in enumerate(record_boxes, start=1):
        marker_box = _assign_marker_box(record_box, marker_by_center, used_markers)
        cells: dict[str, Box] = {}

        for column_index, column_name in enumerate(column_order):
            left = column_edges[column_index]
            right = column_edges[column_index + 1]
            cell_box = Box.from_bounds(left, record_box.y, right, record_box.bottom)
            padding = 4 if column_name != "bil" else 3
            cells[column_name] = cell_box.inset(padding).clamp(image_width, image_height)

        layouts.append(
            RecordLayout(
                index=index,
                box=record_box.clamp(image_width, image_height),
                marker_box=marker_box,
                cells=cells,
            )
        )

    return layouts


def _assign_marker_box(
    record_box: Box,
    marker_boxes: list[Box],
    used_markers: set[int],
) -> Box | None:
    for index, marker_box in enumerate(marker_boxes):
        if index in used_markers:
            continue
        if record_box.y <= marker_box.center_y < record_box.bottom:
            used_markers.add(index)
            return marker_box
    return None


def _remove_table_lines(color_image: np.ndarray, line_mask: np.ndarray) -> np.ndarray:
    inpaint_mask = cv2.dilate(line_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    return cv2.inpaint(color_image, inpaint_mask, 3, cv2.INPAINT_TELEA)
