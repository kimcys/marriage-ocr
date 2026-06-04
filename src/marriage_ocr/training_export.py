from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

from marriage_ocr.review_store import ReviewBundle, load_review_bundles


@dataclass(frozen=True)
class TrainingExample:
    bundle: ReviewBundle
    cell_name: str
    source_image_path: Path
    label_text: str
    used_corrected_label: bool


@dataclass(frozen=True)
class TrainingExportSummary:
    total_examples: int
    train_examples: int
    validation_examples: int
    skipped_unverified_records: int
    skipped_empty_labels: int
    output_dir: Path
    labels_path: Path
    train_path: Path
    validation_path: Path
    manifest_path: Path
    stats_path: Path


def export_training_dataset(
    debug_root: str | Path,
    output_dir: str | Path,
    export_config: Mapping[str, Any],
    *,
    verified_only: bool | None = None,
    reset_output: bool = True,
) -> TrainingExportSummary:
    resolved_output_dir = Path(output_dir)
    config_verified_only = bool(export_config.get("verified_only", True))
    require_verified = config_verified_only if verified_only is None else bool(verified_only)
    validation_ratio = float(export_config.get("validation_ratio", 0.20))

    labels_path = resolved_output_dir / str(export_config.get("labels_file", "labels.tsv"))
    train_path = resolved_output_dir / str(export_config.get("train_file", "train.tsv"))
    validation_path = resolved_output_dir / str(export_config.get("validation_file", "val.tsv"))
    manifest_path = resolved_output_dir / str(export_config.get("manifest_file", "manifest.jsonl"))
    stats_path = resolved_output_dir / str(export_config.get("stats_file", "stats.json"))
    crops_root = resolved_output_dir / str(export_config.get("crops_dir", "training_crops"))

    if reset_output:
        _reset_output(resolved_output_dir, crops_root, [labels_path, train_path, validation_path, manifest_path, stats_path])

    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    crops_root.mkdir(parents=True, exist_ok=True)

    bundles = load_review_bundles(debug_root)
    examples: list[TrainingExample] = []
    skipped_unverified_records = 0
    skipped_empty_labels = 0

    for bundle in bundles:
        if require_verified and not bundle.verified:
            skipped_unverified_records += 1
            continue

        bundle_examples, empty_count = _examples_from_bundle(bundle)
        examples.extend(bundle_examples)
        skipped_empty_labels += empty_count

    label_lines: list[str] = []
    train_lines: list[str] = []
    validation_lines: list[str] = []
    manifest_lines: list[str] = []

    train_examples = 0
    validation_examples = 0
    cell_counts: dict[str, int] = {}

    for index, example in enumerate(examples, start=1):
        cell_dir = crops_root / example.cell_name
        cell_dir.mkdir(parents=True, exist_ok=True)
        suffix = example.source_image_path.suffix.lower() or ".jpg"
        destination_name = f"{index:06d}{suffix}"
        destination_path = cell_dir / destination_name
        shutil.copy2(example.source_image_path, destination_path)

        relative_image_path = destination_path.relative_to(resolved_output_dir).as_posix()
        label_line = f"{relative_image_path}\t{_escape_label_for_tsv(example.label_text)}"
        label_lines.append(label_line)

        split = _assign_split(example.bundle, validation_ratio)
        if split == "val":
            validation_lines.append(label_line)
            validation_examples += 1
        else:
            train_lines.append(label_line)
            train_examples += 1

        cell_counts[example.cell_name] = cell_counts.get(example.cell_name, 0) + 1
        manifest_lines.append(
            json.dumps(
                {
                    "image_path": relative_image_path,
                    "label_text": example.label_text,
                    "cell_name": example.cell_name,
                    "dataset_split": split,
                    "verified": example.bundle.verified,
                    "used_corrected_label": example.used_corrected_label,
                    "reviewed_at": example.bundle.reviewed_at,
                    "reviewed_by": example.bundle.reviewed_by,
                    "review_notes": example.bundle.review_notes,
                    "source_file": example.bundle.active_record.source_file,
                    "source_page": example.bundle.active_record.source_page,
                    "source_record": example.bundle.active_record.source_record,
                    "record_dir": str(example.bundle.record_dir),
                },
                ensure_ascii=True,
            )
        )

    labels_path.write_text(_join_lines(label_lines), encoding="utf-8")
    train_path.write_text(_join_lines(train_lines), encoding="utf-8")
    validation_path.write_text(_join_lines(validation_lines), encoding="utf-8")
    manifest_path.write_text(_join_lines(manifest_lines), encoding="utf-8")
    stats_path.write_text(
        json.dumps(
            {
                "total_examples": len(examples),
                "train_examples": train_examples,
                "validation_examples": validation_examples,
                "skipped_unverified_records": skipped_unverified_records,
                "skipped_empty_labels": skipped_empty_labels,
                "verified_only": require_verified,
                "validation_ratio": validation_ratio,
                "cell_counts": cell_counts,
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return TrainingExportSummary(
        total_examples=len(examples),
        train_examples=train_examples,
        validation_examples=validation_examples,
        skipped_unverified_records=skipped_unverified_records,
        skipped_empty_labels=skipped_empty_labels,
        output_dir=resolved_output_dir,
        labels_path=labels_path,
        train_path=train_path,
        validation_path=validation_path,
        manifest_path=manifest_path,
        stats_path=stats_path,
    )


def _examples_from_bundle(bundle: ReviewBundle) -> tuple[list[TrainingExample], int]:
    examples: list[TrainingExample] = []
    skipped_empty_labels = 0

    for cell_name, image_path in bundle.cell_paths.items():
        label_text = bundle.active_cell_labels.get(cell_name, "").strip()
        if not label_text:
            skipped_empty_labels += 1
            continue
        examples.append(
            TrainingExample(
                bundle=bundle,
                cell_name=cell_name,
                source_image_path=image_path,
                label_text=label_text,
                used_corrected_label=cell_name in bundle.corrected_cells,
            )
        )

    return examples, skipped_empty_labels


def _assign_split(bundle: ReviewBundle, validation_ratio: float) -> str:
    ratio = max(0.0, min(1.0, validation_ratio))
    if ratio <= 0.0:
        return "train"
    if ratio >= 1.0:
        return "val"

    record = bundle.active_record
    key = "|".join(
        [
            record.source_file or "",
            str(record.source_page or ""),
            record.source_record or bundle.record_dir.name,
        ]
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "val" if bucket < ratio else "train"


def _reset_output(output_dir: Path, crops_root: Path, files: list[Path]) -> None:
    if crops_root.exists():
        shutil.rmtree(crops_root)

    for path in files:
        if path.exists():
            path.unlink()

    if output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)


def _escape_label_for_tsv(text: str) -> str:
    return text.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n")


def _join_lines(lines: list[str]) -> str:
    return "".join(f"{line}\n" for line in lines)
