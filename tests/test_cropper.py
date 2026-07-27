from pathlib import Path

import numpy as np

from marriage_ocr.cropper import save_record_crops
from marriage_ocr.layout import Box, LineCandidate, RecordLayout, TableLayout


def test_save_record_crops_adds_padding_to_cell_crops(tmp_path: Path) -> None:
    processed_color = np.zeros((100, 100, 3), dtype=np.uint8)
    layout = TableLayout(
        table_box=Box.from_bounds(0, 0, 100, 100),
        column_order=["bil"],
        column_edges=[0, 100],
        vertical_line_positions=[],
        horizontal_line_positions=[],
        horizontal_line_candidates=[LineCandidate(y=0, coverage=0)],
        records=[
            RecordLayout(
                index=1,
                box=Box.from_bounds(10, 10, 60, 60),
                cells={"bil": Box.from_bounds(30, 30, 50, 50)},
            )
        ],
        vertical_mask=np.zeros((100, 100), dtype=np.uint8),
        horizontal_mask=np.zeros((100, 100), dtype=np.uint8),
        line_mask=np.zeros((100, 100), dtype=np.uint8),
        ocr_ready_color=processed_color.copy(),
    )

    written: dict[Path, np.ndarray] = {}

    def fake_write_image(path: Path, image: np.ndarray) -> None:
        written[path] = image.copy()

    save_record_crops(tmp_path, layout, processed_color, fake_write_image)

    bil_crop = written[tmp_path / "records" / "record_001" / "bil.jpg"]
    assert bil_crop.shape[:2] == (44, 44)
