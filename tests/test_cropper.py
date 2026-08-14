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
                # Record box is deliberately much larger than the cell on every
                # side, so the crop padding never approaches the record's own
                # boundary -- this test is only about padding, not clamping
                # (see test_save_record_crops_clamps_padding_to_record_box for
                # that).
                box=Box.from_bounds(0, 0, 100, 100),
                cells={"bil": Box.from_bounds(40, 40, 60, 60)},
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


def test_save_record_crops_clamps_padding_to_record_box(tmp_path: Path) -> None:
    """Regression: cells are built as record_box.inset(4) (see
    layout._build_record_layouts), then padded back out by
    CELL_CROP_PADDING=12 -- a net 8px overshoot past the record's own
    top/bottom edge. Since adjacent records share an exact boundary with no
    gap (_detect_record_boxes), that overshoot used to bleed into the next
    record's row, risking a fallback crop-OCR read mixing in a different
    person's text. The padded crop must never cross the record's own box.
    """
    processed_color = np.zeros((300, 100, 3), dtype=np.uint8)
    record1_box = Box.from_bounds(0, 100, 100, 200)
    record2_box = Box.from_bounds(0, 200, 100, 300)
    cell_box = Box.from_bounds(10, record1_box.y, 90, record1_box.bottom).inset(4)

    layout = TableLayout(
        table_box=Box.from_bounds(0, 0, 100, 300),
        column_order=["bil"],
        column_edges=[0, 100],
        vertical_line_positions=[],
        horizontal_line_positions=[],
        horizontal_line_candidates=[LineCandidate(y=0, coverage=0)],
        records=[
            RecordLayout(index=1, box=record1_box, cells={"bil": cell_box}),
            RecordLayout(index=2, box=record2_box, cells={"bil": Box.from_bounds(10, record2_box.y, 90, record2_box.bottom).inset(4)}),
        ],
        vertical_mask=np.zeros((300, 100), dtype=np.uint8),
        horizontal_mask=np.zeros((300, 100), dtype=np.uint8),
        line_mask=np.zeros((300, 100), dtype=np.uint8),
        ocr_ready_color=processed_color.copy(),
    )

    written: dict[Path, np.ndarray] = {}

    def fake_write_image(path: Path, image: np.ndarray) -> None:
        written[path] = image.copy()

    save_record_crops(tmp_path, layout, processed_color, fake_write_image)

    bil_crop = written[tmp_path / "records" / "record_001" / "bil.jpg"]
    # Crop height must not exceed record1's own height (100px): no bleed
    # into record2's rows below y=200.
    assert bil_crop.shape[0] <= record1_box.height


def test_save_record_crops_clamps_padding_to_column_bounds(tmp_path: Path) -> None:
    """Regression: the same net 8px overshoot from CELL_CROP_PADDING=12 vs.
    the record_box.inset(4) used to build cells also bled *sideways* into
    the neighboring column of the same record, since adjacent columns share
    an exact boundary with no gap. A fallback crop-OCR read for one field
    (e.g. "wali") could pick up a sliver of the adjacent field (e.g.
    "hubungan_wali") in the same row. The padded crop must never cross the
    cell's own column edges.
    """
    processed_color = np.zeros((100, 300, 3), dtype=np.uint8)
    record_box = Box.from_bounds(0, 0, 300, 100)
    wali_box = Box.from_bounds(100, 0, 200, 100)
    wali_cell = wali_box.inset(4)

    layout = TableLayout(
        table_box=Box.from_bounds(0, 0, 300, 100),
        column_order=["bil", "wali", "hubungan_wali"],
        column_edges=[0, 100, 200, 300],
        vertical_line_positions=[],
        horizontal_line_positions=[],
        horizontal_line_candidates=[LineCandidate(y=0, coverage=0)],
        records=[
            RecordLayout(
                index=1,
                box=record_box,
                cells={"wali": wali_cell},
            )
        ],
        vertical_mask=np.zeros((100, 300), dtype=np.uint8),
        horizontal_mask=np.zeros((100, 300), dtype=np.uint8),
        line_mask=np.zeros((100, 300), dtype=np.uint8),
        ocr_ready_color=processed_color.copy(),
    )

    written: dict[Path, np.ndarray] = {}

    def fake_write_image(path: Path, image: np.ndarray) -> None:
        written[path] = image.copy()

    save_record_crops(tmp_path, layout, processed_color, fake_write_image)

    wali_crop = written[tmp_path / "records" / "record_001" / "wali.jpg"]
    # Crop width must not exceed the wali column's own width (100px): no
    # bleed into the adjacent bil (x<100) or hubungan_wali (x>=200) columns.
    assert wali_crop.shape[1] <= 100
