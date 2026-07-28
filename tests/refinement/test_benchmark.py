from __future__ import annotations

import json
from pathlib import Path

from marriage_ocr.models import ExtractedRecord
from marriage_ocr.refinement.audit import save_refinement_audit_sidecar
from marriage_ocr.refinement.benchmark import build_refinement_baseline
from marriage_ocr.refinement.models import FieldRefinementAuditRow
from marriage_ocr.review_store import save_corrected_record


def test_first_25_baseline_uses_reviewed_records_in_order(tmp_path: Path) -> None:
    _create_reviewed_record(
        tmp_path,
        source_record="record_001",
        bil="1",
        verified=True,
        refinement_rows=[
            _audit_row("nama_suami", selected_value="AHMAD BIN ALI"),
            _audit_row("ic_baru_isteri", selected_value="900101-10-0000"),
            _audit_row("tarikh_nikah", selected_value="27-08-1994"),
        ],
    )
    _create_reviewed_record(
        tmp_path,
        source_record="record_002",
        bil="2",
        verified=False,
        refinement_rows=[
            _audit_row("nama_isteri", selected_value="SITI BINTI ALI"),
            _audit_row("ic_lama_suami", selected_value="A.1192345"),
            _audit_row("tarikh_keluar", selected_value="02-06-1995"),
        ],
    )
    _create_reviewed_record(
        tmp_path,
        source_record="record_003",
        bil="3",
        page_name="page_b",
        verified=True,
        refinement_rows=[
            _audit_row("nama_isteri", selected_value="SITI BINTI ABU"),
            _audit_row("ic_lama_suami", selected_value="A.1192345"),
            _audit_row("tarikh_keluar", selected_value="03-06-1995"),
        ],
    )

    metrics = build_refinement_baseline(tmp_path, limit=1)

    assert metrics.record_count == 1
    assert metrics.name_exact_match_count == 1
    assert metrics.ic_exact_match_count == 0
    assert metrics.date_exact_match_count == 1


def test_first_25_baseline_counts_exact_matches_from_verified_review_bundles(tmp_path: Path) -> None:
    _create_reviewed_record(
        tmp_path,
        source_record="record_001",
        bil="1",
        verified=True,
        refinement_rows=[
            _audit_row("nama_suami", selected_value="AHMAD BIN ALI"),
            _audit_row("ic_baru_isteri", selected_value="900101-10-0000"),
            _audit_row("tarikh_nikah", selected_value="27-08-1994"),
        ],
    )
    _create_reviewed_record(
        tmp_path,
        source_record="record_002",
        bil="2",
        verified=False,
        refinement_rows=[
            _audit_row("nama_isteri", selected_value="SITI BINTI ALI"),
            _audit_row("ic_lama_suami", selected_value="A.1192345"),
            _audit_row("tarikh_keluar", selected_value="02-06-1995"),
        ],
    )
    _create_reviewed_record(
        tmp_path,
        source_record="record_003",
        bil="3",
        page_name="page_b",
        verified=True,
        refinement_rows=[
            _audit_row("nama_isteri", selected_value="SITI BINTI ABU"),
            _audit_row("ic_lama_suami", selected_value="A.1192345"),
            _audit_row("tarikh_keluar", selected_value="03-06-1995"),
        ],
    )

    metrics = build_refinement_baseline(tmp_path, limit=25)

    assert metrics.record_count == 2
    assert metrics.name_exact_match_count == 1
    assert metrics.ic_exact_match_count == 1
    assert metrics.date_exact_match_count == 1


def _create_reviewed_record(
    root: Path,
    *,
    source_record: str,
    bil: str,
    verified: bool,
    refinement_rows: list[FieldRefinementAuditRow],
    page_name: str = "page_a",
) -> Path:
    record_dir = root / page_name / "records" / source_record
    record_dir.mkdir(parents=True)

    validated_record = _make_record(source_record=source_record, bil=bil)
    (record_dir / "validated_record.json").write_text(
        json.dumps(validated_record.to_dict(), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    (record_dir / "parsed_record.json").write_text(
        json.dumps(validated_record.to_dict(), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    (record_dir / "raw_ocr.json").write_text(
        json.dumps({"cells": {"bil": {"text": bil}}}, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    save_corrected_record(
        record_dir,
        validated_record,
        verified=verified,
        reviewed_by="QA User",
        review_notes="Reviewed for benchmark baseline",
    )
    save_refinement_audit_sidecar(record_dir, refinement_rows)
    return record_dir


def _audit_row(field_name: str, *, selected_value: str) -> FieldRefinementAuditRow:
    return FieldRefinementAuditRow(
        source_file="sample.jpg",
        page_number=1,
        record_index=1,
        field_name=field_name,
        original_value=None,
        selected_value=selected_value,
        original_score=0.60,
        selected_score=0.90,
        correction_type="benchmark",
        candidate_source="retry_grayscale",
        reason="benchmark",
        requires_review=False,
        crop_path=None,
        retry_count=1,
    )


def _make_record(*, source_record: str, bil: str) -> ExtractedRecord:
    return ExtractedRecord(
        bil=bil,
        nama_suami="AHMAD BIN ALI",
        ic_lama_suami="A.1192345",
        nama_isteri="SITI BINTI ALI",
        ic_baru_isteri="900101-10-1234",
        tarikh_nikah="27-08-1994",
        tarikh_keluar="02-06-1995",
        confidence=0.93,
        status_review="OK",
        review_reason=[],
        source_file="sample.jpg",
        source_page=1,
        source_record=source_record,
        crop_folder=f"debug/sample/{source_record}",
    )
