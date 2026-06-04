from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from marriage_ocr.exporter import XLSX_COLUMNS, export_records_to_xlsx, record_to_export_dict
from marriage_ocr.models import ExtractedRecord


def export_records_to_csv(records: Iterable[ExtractedRecord], output_path: Path) -> Path:
    record_list = list(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    return write_csv_rows([record_to_export_dict(record) for record in record_list], output_path)


def export_records_to_csv_parts(
    records: Iterable[ExtractedRecord],
    output_dir: Path,
    *,
    rows_per_file: int = 5000,
    filename_prefix: str = "records_part",
) -> list[Path]:
    record_list = [record_to_export_dict(record) for record in records]
    return export_csv_rows_to_parts(
        record_list,
        output_dir,
        rows_per_file=rows_per_file,
        filename_prefix=filename_prefix,
    )


def export_records_to_xlsx_parts(
    records: Iterable[ExtractedRecord],
    output_dir: Path,
    export_config: Mapping[str, Any],
    *,
    rows_per_file: int = 5000,
    filename_prefix: str = "records_part",
) -> list[Path]:
    if rows_per_file <= 0:
        raise ValueError(f"rows_per_file must be positive, got {rows_per_file}")
    if not filename_prefix:
        raise ValueError("filename_prefix must not be empty")

    record_list = list(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for part_index, start in enumerate(range(0, len(record_list), rows_per_file), start=1):
        chunk = record_list[start : start + rows_per_file]
        part_path = output_dir / f"{filename_prefix}_{part_index:03d}.xlsx"
        export_records_to_xlsx(
            chunk,
            part_path,
            export_config,
            reset_output=True,
            skip_existing=False,
        )
        paths.append(part_path)

    return paths


def write_csv_rows(rows: Sequence[Mapping[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=XLSX_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in XLSX_COLUMNS})

    return output_path


def export_csv_rows_to_parts(
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    rows_per_file: int = 5000,
    filename_prefix: str = "records_part",
) -> list[Path]:
    if rows_per_file <= 0:
        raise ValueError(f"rows_per_file must be positive, got {rows_per_file}")
    if not filename_prefix:
        raise ValueError("filename_prefix must not be empty")

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for part_index, start in enumerate(range(0, len(rows), rows_per_file), start=1):
        chunk = rows[start : start + rows_per_file]
        part_path = output_dir / f"{filename_prefix}_{part_index:03d}.csv"
        write_csv_rows(chunk, part_path)
        paths.append(part_path)

    return paths
