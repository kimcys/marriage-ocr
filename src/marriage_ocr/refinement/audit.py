from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from marriage_ocr.refinement.models import FieldRefinementAuditRow


REFINEMENT_AUDIT_COLUMNS = [
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

REFINEMENT_AUDIT_SIDECAR_NAME = "refinement_audit.json"


def write_refinement_audit(rows: list[FieldRefinementAuditRow], output_path: str | Path) -> Path:
    resolved_path = Path(output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REFINEMENT_AUDIT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(_audit_row_to_csv_dict(row))
    return resolved_path


def save_refinement_audit_sidecar(record_dir: str | Path, rows: list[FieldRefinementAuditRow]) -> Path:
    resolved_dir = Path(record_dir)
    resolved_dir.mkdir(parents=True, exist_ok=True)
    output_path = resolved_dir / REFINEMENT_AUDIT_SIDECAR_NAME
    payload = {
        "rows": [_audit_row_to_json_dict(row) for row in rows],
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return output_path


def load_refinement_audit_sidecar(record_dir: str | Path) -> list[FieldRefinementAuditRow]:
    sidecar_path = Path(record_dir) / REFINEMENT_AUDIT_SIDECAR_NAME
    if not sidecar_path.exists():
        return []
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        return []
    return [_audit_row_from_dict(row) for row in rows if isinstance(row, dict)]


def _audit_row_to_csv_dict(row: FieldRefinementAuditRow) -> dict[str, Any]:
    data = _audit_row_to_json_dict(row)
    data["requires_review"] = "true" if row.requires_review else "false"
    return data


def _audit_row_to_json_dict(row: FieldRefinementAuditRow) -> dict[str, Any]:
    return {
        "source_file": row.source_file,
        "page_number": row.page_number,
        "record_index": row.record_index,
        "field_name": row.field_name,
        "original_value": row.original_value,
        "selected_value": row.selected_value,
        "original_score": row.original_score,
        "selected_score": row.selected_score,
        "correction_type": row.correction_type,
        "candidate_source": row.candidate_source,
        "reason": row.reason,
        "requires_review": row.requires_review,
        "crop_path": row.crop_path,
        "retry_count": row.retry_count,
    }


def _audit_row_from_dict(data: dict[str, Any]) -> FieldRefinementAuditRow:
    return FieldRefinementAuditRow(
        source_file=str(data.get("source_file", "")),
        page_number=int(data.get("page_number", 0)),
        record_index=int(data.get("record_index", 0)),
        field_name=str(data.get("field_name", "")),
        original_value=_optional_text(data.get("original_value")),
        selected_value=_optional_text(data.get("selected_value")),
        original_score=float(data.get("original_score", 0.0)),
        selected_score=float(data.get("selected_score", 0.0)),
        correction_type=str(data.get("correction_type", "")),
        candidate_source=str(data.get("candidate_source", "")),
        reason=str(data.get("reason", "")),
        requires_review=bool(data.get("requires_review", False)),
        crop_path=_optional_text(data.get("crop_path")),
        retry_count=int(data.get("retry_count", 0)),
    )


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None
