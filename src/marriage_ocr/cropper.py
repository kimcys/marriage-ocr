from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from marriage_ocr.layout import Box, TableLayout
from marriage_ocr.ocr import RecordCropPaths


DEBUG_CELL_ORDER = [
    "bil",
    "suami_isteri",
    "pendaftar",
    "wali",
    "hubungan_wali",
    "saksi",
    "tarikh_nikah",
    "tarikh_keluar",
    "remarks",
]

CELL_CROP_PADDING = 12


ImageWriter = Callable[[Path, np.ndarray], None]


def save_record_crops(
    page_debug_dir: Path,
    layout: TableLayout,
    processed_color: np.ndarray,
    write_image: ImageWriter,
) -> list[RecordCropPaths]:
    records_dir = page_debug_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    saved_records: list[RecordCropPaths] = []
    column_bounds = _column_horizontal_bounds(layout)

    for record in layout.records:
        record_dir = records_dir / f"record_{record.index:03d}"
        record_dir.mkdir(parents=True, exist_ok=True)

        record_rows, record_columns = record.box.slices()
        full_record_path = record_dir / "full_record.jpg"
        write_image(full_record_path, processed_color[record_rows, record_columns])
        cell_paths: dict[str, Path] = {}

        for cell_name in DEBUG_CELL_ORDER:
            cell_box = record.cells.get(cell_name)
            if cell_box is None:
                continue
            column_left, column_right = column_bounds.get(cell_name, (record.box.x, record.box.right))
            clamp_box = Box.from_bounds(column_left, record.box.y, column_right, record.box.bottom)
            cell_rows, cell_columns = _padded_slices(
                cell_box, CELL_CROP_PADDING, layout.ocr_ready_color.shape, clamp_box=clamp_box
            )
            cell_path = record_dir / f"{cell_name}.jpg"
            write_image(cell_path, layout.ocr_ready_color[cell_rows, cell_columns])
            cell_paths[cell_name] = cell_path

        saved_records.append(
            RecordCropPaths(
                record_index=record.index,
                record_dir=record_dir,
                full_record_path=full_record_path,
                cell_paths=cell_paths,
            )
        )

    return saved_records


def _column_horizontal_bounds(layout: TableLayout) -> dict[str, tuple[int, int]]:
    """Map each cell name to its own column's (left, right) edges.

    Cells are built from these exact edges (see layout._build_record_layouts:
    Box.from_bounds(column_edges[i], ..., column_edges[i + 1], ...)), so they
    are also the correct horizontal clamp for that cell's crop -- see
    _padded_slices for why the clamp is needed at all.
    """
    bounds: dict[str, tuple[int, int]] = {}
    for index, column_name in enumerate(layout.column_order):
        if index + 1 >= len(layout.column_edges):
            break
        bounds[column_name] = (layout.column_edges[index], layout.column_edges[index + 1])
    return bounds


def _padded_slices(
    box,
    padding: int,
    image_shape: tuple[int, int, int] | tuple[int, int],
    *,
    clamp_box=None,
) -> tuple[slice, slice]:
    """Pad a cell's box outward for the crop, without bleeding past its own
    record or its own column.

    Cells are built as (record_box, column_edges).inset(4) (see
    layout._build_record_layouts), then padded back out by
    CELL_CROP_PADDING=12 here -- a net 8px overshoot past the cell's own
    edges. Vertically, adjacent records share an exact boundary with no gap
    (_detect_record_boxes), so that overshoot bled into the next person's
    row. Horizontally, it bled into the neighboring column/field of the same
    person. Left uncapped, a fallback crop-OCR read could pick up a sliver of
    the wrong person's row or the wrong field. clamp_box (the cell's own
    record-and-column bounds) caps the padding so it can never cross either
    boundary.
    """
    image_height, image_width = image_shape[:2]
    top = max(0, box.y - padding)
    bottom = min(image_height, box.bottom + padding)
    left = max(0, box.x - padding)
    right = min(image_width, box.right + padding)
    if clamp_box is not None:
        top = max(top, clamp_box.y)
        bottom = min(bottom, clamp_box.bottom)
        left = max(left, clamp_box.x)
        right = min(right, clamp_box.right)
    return slice(top, bottom), slice(left, right)
