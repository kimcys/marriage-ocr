from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from marriage_ocr.layout import TableLayout
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
            cell_rows, cell_columns = _padded_slices(cell_box, CELL_CROP_PADDING, layout.ocr_ready_color.shape)
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


def _padded_slices(box, padding: int, image_shape: tuple[int, int, int] | tuple[int, int]) -> tuple[slice, slice]:
    image_height, image_width = image_shape[:2]
    top = max(0, box.y - padding)
    bottom = min(image_height, box.bottom + padding)
    left = max(0, box.x - padding)
    right = min(image_width, box.right + padding)
    return slice(top, bottom), slice(left, right)
