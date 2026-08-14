from __future__ import annotations

from marriage_ocr.layout import (
    Box,
    LineCandidate,
    _detect_record_boxes,
    _select_record_starts,
)


def test_tight_but_valid_marker_spacing_does_not_drop_a_record() -> None:
    """Regression: _optimize_record_starts accepted any gap > 20px between
    record starts, but _detect_record_boxes (which turns those starts into
    boxes) drops any record shorter than max(30, min_record_height // 2) --
    40px for the default min_record_height=80. A 21-39px gap used to pass the
    optimizer's own check and then get silently deleted downstream: the
    marker still existed, but its record vanished entirely from the final
    layout -- no box, no error, no review flag, just a missing marriage
    record on export.

    3 markers here are spaced far enough apart that they are NOT merged as
    duplicates (see _merge_close_marker_boxes's min_gap), but the available
    horizontal-line candidates force a tight 25px gap between the first two
    record starts -- previously enough to pass _select_record_starts but not
    _detect_record_boxes.
    """
    table_box = Box.from_bounds(0, 0, 1000, 500)
    marker_boxes = [
        Box.from_bounds(10, 50, 100, 80),
        Box.from_bounds(10, 150, 100, 180),
        Box.from_bounds(10, 400, 100, 430),
    ]
    line_candidates = [
        LineCandidate(y=10, coverage=500),
        LineCandidate(y=35, coverage=500),  # only 25px after y=10
        LineCandidate(y=375, coverage=500),
    ]

    starts = _select_record_starts(
        table_box=table_box,
        marker_boxes=marker_boxes,
        line_candidates=line_candidates,
        min_record_height=80,
        max_record_height=280,
    )
    boxes = _detect_record_boxes(
        table_box=table_box,
        marker_boxes=marker_boxes,
        line_candidates=line_candidates,
        min_record_height=80,
        max_record_height=280,
    )

    assert len(starts) == len(marker_boxes) == 3
    assert len(boxes) == 3, f"expected one box per marker, got {len(boxes)} boxes for {len(marker_boxes)} markers"

    # Every gap between consecutive starts must satisfy the same minimum the
    # box-builder will apply -- otherwise this test would pass today and
    # still let a future edit reopen the inconsistency.
    min_gap = max(30, 80 // 2)
    for previous, current in zip(starts, starts[1:]):
        assert current - previous >= min_gap
