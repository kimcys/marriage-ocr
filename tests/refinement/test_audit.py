from __future__ import annotations

import csv
import json
from pathlib import Path

from marriage_ocr.refinement.audit import (
    load_refinement_audit_sidecar,
    save_refinement_audit_sidecar,
    write_refinement_audit,
)
from marriage_ocr.refinement.models import FieldRefinementAuditRow


def test_write_refinement_audit_has_exact_column_order_and_quotes(tmp_path: Path) -> None:
    output_path = tmp_path / "refinement_audit.csv"
    rows = [
        FieldRefinementAuditRow(
            source_file="sample.pdf",
            page_number=1,
            record_index=1,
            field_name="nama_suami",
            original_value="AHMAD B1N ALI",
            selected_value="AHMAD BIN ALI",
            original_score=0.61,
            selected_score=0.86,
            correction_type="connector_typo",
            candidate_source="retry_grayscale",
            reason="Connector correction supported by retry OCR",
            requires_review=False,
            crop_path="crops/page_1/record_1/nama_suami.png",
            retry_count=2,
        ),
        FieldRefinementAuditRow(
            source_file="sample.pdf",
            page_number=1,
            record_index=1,
            field_name="nama_wali",
            original_value="SITI, O'CONNOR",
            selected_value="SITI O'CONNOR",
            original_score=0.52,
            selected_score=0.77,
            correction_type="safe_normalisation",
            candidate_source="original_ocr",
            reason="Whitespace normalized",
            requires_review=True,
            crop_path=None,
            retry_count=0,
        ),
    ]

    write_refinement_audit(rows, output_path)

    with output_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        first_row = next(reader)

    assert header == [
        "source_file",
        "page_number",
        "record_index",
        "field_name",
        "original_value",
        "selected_value",
        "original_score",
        "selected_score",
        "correction_type",
        "candidate_source",
        "reason",
        "requires_review",
        "crop_path",
        "retry_count",
    ]
    assert first_row[4] == "AHMAD B1N ALI"
    assert first_row[11] == "false"

    raw_csv = output_path.read_text(encoding="utf-8")
    assert "SITI, O'CONNOR" in raw_csv


def test_refinement_audit_sidecar_round_trips_rows(tmp_path: Path) -> None:
    record_dir = tmp_path / "page_001" / "records" / "record_001"
    record_dir.mkdir(parents=True)
    rows = [
        FieldRefinementAuditRow(
            source_file="sample.pdf",
            page_number=1,
            record_index=1,
            field_name="tarikh_nikah",
            original_value="31/02/94",
            selected_value="31/02/94",
            original_score=0.30,
            selected_score=0.30,
            correction_type="no_safe_candidate",
            candidate_source="original_ocr",
            reason="All generated dates were invalid",
            requires_review=True,
            crop_path="crops/page_1/record_1/tarikh_nikah.png",
            retry_count=3,
        )
    ]

    sidecar_path = save_refinement_audit_sidecar(record_dir, rows)
    restored = load_refinement_audit_sidecar(record_dir)

    assert sidecar_path.name == "refinement_audit.json"
    assert restored == rows
    assert json.loads(sidecar_path.read_text(encoding="utf-8"))["rows"][0]["field_name"] == "tarikh_nikah"

