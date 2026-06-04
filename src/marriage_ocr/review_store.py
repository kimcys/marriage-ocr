from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Iterable

from marriage_ocr.exporter import ExportSummary, export_records_to_xlsx
from marriage_ocr.models import ExtractedRecord


ALLOWED_REVIEW_STATUSES = [
    "OK",
    "REVIEW",
    "FAILED_LAYOUT",
    "FAILED_OCR",
    "DUPLICATE",
]


@dataclass(frozen=True)
class ReviewBundle:
    record_dir: Path
    validated_record: ExtractedRecord
    active_record: ExtractedRecord
    corrected_record: ExtractedRecord | None
    parsed_record: ExtractedRecord | None
    raw_ocr: dict[str, Any]
    active_cell_labels: dict[str, str]
    corrected_cells: dict[str, str]
    full_record_path: Path | None
    cell_paths: dict[str, Path]
    verified: bool
    reviewed_at: str | None
    reviewed_by: str | None
    review_notes: str | None

    @property
    def page_dir(self) -> Path:
        return self.record_dir.parent.parent

    @property
    def display_name(self) -> str:
        record = self.active_record
        source_file = record.source_file or self.page_dir.name
        source_page = record.source_page or "?"
        return f"{source_file} | page {source_page} | {self.record_dir.name}"


@dataclass(frozen=True)
class CorrectedRecordPayload:
    record: ExtractedRecord
    verified: bool
    reviewed_at: str | None
    reviewed_by: str | None
    review_notes: str | None
    corrected_cells: dict[str, str]


def discover_record_directories(debug_root: str | Path) -> list[Path]:
    root = Path(debug_root)
    if not root.exists():
        return []

    record_dirs = [path for path in root.glob("**/records/record_*") if path.is_dir()]
    return sorted(record_dirs, key=_record_sort_key)


def load_review_bundles(debug_root: str | Path) -> list[ReviewBundle]:
    return [load_review_bundle(record_dir) for record_dir in discover_record_directories(debug_root)]


def load_review_bundle(record_dir: str | Path) -> ReviewBundle:
    resolved_dir = Path(record_dir)
    validated_record = _load_record_json(resolved_dir / "validated_record.json")
    parsed_record = _load_optional_record_json(resolved_dir / "parsed_record.json")
    raw_ocr = _load_optional_json_dict(resolved_dir / "raw_ocr.json")
    corrected_payload = _load_corrected_payload(resolved_dir / "corrected_record.json")
    active_record = corrected_payload.record if corrected_payload is not None else validated_record
    corrected_cells = corrected_payload.corrected_cells if corrected_payload is not None else {}

    return ReviewBundle(
        record_dir=resolved_dir,
        validated_record=validated_record,
        active_record=active_record,
        corrected_record=corrected_payload.record if corrected_payload is not None else None,
        parsed_record=parsed_record,
        raw_ocr=raw_ocr,
        active_cell_labels=_build_active_cell_labels(raw_ocr, corrected_cells, active_record),
        corrected_cells=corrected_cells,
        full_record_path=_existing_path(resolved_dir / "full_record.jpg"),
        cell_paths=_load_cell_paths(resolved_dir),
        verified=corrected_payload.verified if corrected_payload is not None else False,
        reviewed_at=corrected_payload.reviewed_at if corrected_payload is not None else None,
        reviewed_by=corrected_payload.reviewed_by if corrected_payload is not None else None,
        review_notes=corrected_payload.review_notes if corrected_payload is not None else None,
    )


def save_corrected_record(
    record_dir: str | Path,
    record: ExtractedRecord,
    *,
    verified: bool,
    reviewed_by: str | None = None,
    review_notes: str | None = None,
    corrected_cells: Mapping[str, str] | None = None,
) -> Path:
    payload = {
        "record": record.to_dict(),
        "verified": bool(verified),
        "reviewed_at": datetime.now().isoformat(timespec="seconds"),
        "reviewed_by": _normalize_text_value(reviewed_by),
        "review_notes": _normalize_text_value(review_notes),
        "corrected_cells": _normalize_cell_labels(corrected_cells or {}),
    }

    output_path = Path(record_dir) / "corrected_record.json"
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return output_path


def export_reviewed_records(
    debug_root: str | Path,
    output_path: str | Path,
    export_config: Mapping[str, Any],
    *,
    verified_only: bool = False,
    reset_output: bool = True,
) -> ExportSummary:
    records = list(iter_export_records(debug_root, verified_only=verified_only))
    return export_records_to_xlsx(
        records,
        Path(output_path),
        export_config,
        reset_output=reset_output,
        skip_existing=False,
    )


def iter_export_records(debug_root: str | Path, *, verified_only: bool = False) -> Iterable[ExtractedRecord]:
    for bundle in load_review_bundles(debug_root):
        if verified_only and not bundle.verified:
            continue
        yield bundle.active_record


def _load_record_json(path: Path) -> ExtractedRecord:
    if not path.exists():
        raise FileNotFoundError(f"Required review artifact not found: {path}")
    return ExtractedRecord.from_dict(_load_json_dict(path))


def _load_optional_record_json(path: Path) -> ExtractedRecord | None:
    if not path.exists():
        return None
    return ExtractedRecord.from_dict(_load_json_dict(path))


def _load_corrected_payload(path: Path) -> CorrectedRecordPayload | None:
    if not path.exists():
        return None

    payload = _load_json_dict(path)
    if "record" in payload and isinstance(payload["record"], dict):
        return CorrectedRecordPayload(
            record=ExtractedRecord.from_dict(payload["record"]),
            verified=bool(payload.get("verified", False)),
            reviewed_at=_normalize_text_value(payload.get("reviewed_at")),
            reviewed_by=_normalize_text_value(payload.get("reviewed_by")),
            review_notes=_normalize_text_value(payload.get("review_notes")),
            corrected_cells=_normalize_cell_labels(payload.get("corrected_cells", {})),
        )

    return CorrectedRecordPayload(
        record=ExtractedRecord.from_dict(payload),
        verified=False,
        reviewed_at=None,
        reviewed_by=None,
        review_notes=None,
        corrected_cells={},
    )


def _load_json_dict(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_optional_json_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _load_json_dict(path)


def _load_cell_paths(record_dir: Path) -> dict[str, Path]:
    image_paths = {}
    for path in sorted(record_dir.glob("*.jpg")):
        if path.name == "full_record.jpg":
            continue
        image_paths[path.stem] = path
    return image_paths


def _existing_path(path: Path) -> Path | None:
    return path if path.exists() else None


def _record_sort_key(path: Path) -> tuple[str, int, str]:
    page_dir = path.parent.parent
    source_record = path.name
    source_page = 0

    validated_path = path / "validated_record.json"
    if validated_path.exists():
        record = ExtractedRecord.from_dict(_load_json_dict(validated_path))
        source_page = int(record.source_page or 0)
        source_record = record.source_record or source_record
        source_file = record.source_file or page_dir.name
    else:
        source_file = page_dir.name

    return (source_file, source_page, source_record)


def _normalize_text_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_cell_labels(values: Any) -> dict[str, str]:
    if not isinstance(values, Mapping):
        return {}
    normalized: dict[str, str] = {}
    for key, value in values.items():
        text = _normalize_training_label(value)
        if text is None:
            continue
        normalized[str(key)] = text
    return normalized


def _build_active_cell_labels(
    raw_ocr: Mapping[str, Any],
    corrected_cells: Mapping[str, str],
    record: ExtractedRecord,
) -> dict[str, str]:
    labels = _default_cell_labels_from_raw_ocr(raw_ocr)
    for cell_name, fallback in _fallback_cell_labels_from_record(record).items():
        labels.setdefault(cell_name, fallback)
    labels.update(_normalize_cell_labels(corrected_cells))
    return labels


def _default_cell_labels_from_raw_ocr(raw_ocr: Mapping[str, Any]) -> dict[str, str]:
    labels: dict[str, str] = {}
    cells = raw_ocr.get("cells", {}) if isinstance(raw_ocr, Mapping) else {}
    if not isinstance(cells, Mapping):
        return labels

    for cell_name, payload in cells.items():
        if not isinstance(payload, Mapping):
            continue
        text = _normalize_training_label(payload.get("text"))
        if text is not None:
            labels[str(cell_name)] = text
    return labels


def _fallback_cell_labels_from_record(record: ExtractedRecord) -> dict[str, str]:
    fallbacks = {
        "bil": record.raw_bil or record.bil,
        "suami_isteri": record.raw_suami_isteri,
        "pendaftar": record.raw_pendaftar,
        "wali": record.raw_wali,
        "hubungan_wali": record.raw_hubungan_wali,
        "saksi": record.raw_saksi,
        "tarikh_nikah": record.raw_tarikh_nikah or record.tarikh_nikah_raw or record.tarikh_nikah,
        "tarikh_keluar": record.raw_tarikh_keluar or record.tarikh_keluar_raw or record.tarikh_keluar,
        "remarks": record.raw_remarks or record.remarks,
    }
    return {
        key: normalized
        for key, value in fallbacks.items()
        if (normalized := _normalize_training_label(value)) is not None
    }


def _normalize_training_label(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n")).strip()
    return text or None
