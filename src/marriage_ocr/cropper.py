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
            cell_rows, cell_columns = cell_box.slices()
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
