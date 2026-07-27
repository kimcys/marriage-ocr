from __future__ import annotations

import csv
import os
from dataclasses import replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable

from marriage_ocr.models import ExtractedRecord
from marriage_ocr.typed.models import ProcessingStatus, TypedDocumentResult


TYPED_CSV_COLUMNS = [
    "Bil",
    "Nama Suami",
    "IC Lama Suami",
    "IC Baru Suami",
    "Umur Suami",
    "Nama Isteri",
    "IC Lama Isteri",
    "IC Baru Isteri",
    "Umur Isteri",
    "Mas Kahwin",
    "Nama Pendaftar",
    "Alamat Pendaftar",
    "Nama Wali",
    "Hubungan Wali",
    "Saksi 1",
    "Saksi 2",
    "Tarikh Nikah",
    "Tarikh Keluar",
    "Source File",
    "Processing Status",
    "Review Required",
    "Failed Fields",
    "Retry Count",
    "Error Message",
]


def _value(value: object | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _record_to_row(result: TypedDocumentResult) -> dict[str, str]:
    record = result.record
    return {
        "Bil": _value(record.bil),
        "Nama Suami": _value(record.nama_suami),
        "IC Lama Suami": _value(record.ic_lama_suami),
        "IC Baru Suami": _value(record.ic_baru_suami),
        "Umur Suami": _value(record.umur_suami),
        "Nama Isteri": _value(record.nama_isteri),
        "IC Lama Isteri": _value(record.ic_lama_isteri),
        "IC Baru Isteri": _value(record.ic_baru_isteri),
        "Umur Isteri": _value(record.umur_isteri),
        "Mas Kahwin": _value(record.mas_kahwin),
        "Nama Pendaftar": _value(record.nama_pendaftar),
        "Alamat Pendaftar": _value(record.alamat_pendaftar),
        "Nama Wali": _value(record.nama_wali),
        "Hubungan Wali": _value(record.hubungan_wali),
        "Saksi 1": _value(record.saksi_1),
        "Saksi 2": _value(record.saksi_2),
        "Tarikh Nikah": _value(record.tarikh_nikah),
        "Tarikh Keluar": _value(record.tarikh_keluar),
        "Source File": result.source_file,
        "Processing Status": result.processing_status.value,
        "Review Required": _value(result.review_required),
        "Failed Fields": result.failed_fields_text,
        "Retry Count": _value(result.retry_count),
        "Error Message": _value(result.error_message),
    }


class TypedCsvStore:
    def __init__(self, output_path: Path, *, skip_existing: bool = False) -> None:
        self.output_path = output_path
        self.skip_existing = skip_existing
        self._rows: dict[str, dict[str, str]] = {}
        self._statuses: dict[str, str] = {}

    @classmethod
    def load(
        cls,
        output_path: Path,
        *,
        reset_output: bool = False,
        skip_existing: bool = False,
    ) -> "TypedCsvStore":
        store = cls(output_path, skip_existing=skip_existing)
        if not reset_output and output_path.exists():
            with output_path.open(newline="", encoding="utf-8-sig") as handle:
                for row in csv.DictReader(handle):
                    source = row.get("Source File", "")
                    if not source:
                        continue
                    store._rows[source] = {column: row.get(column, "") or "" for column in TYPED_CSV_COLUMNS}
                    store._statuses[source] = store._rows[source]["Processing Status"]
        return store

    def should_skip(self, source_file: str) -> bool:
        if not self.skip_existing:
            return False
        return self._statuses.get(source_file) in {
            ProcessingStatus.SUCCESS.value,
            ProcessingStatus.SUCCESS_WITH_RETRY.value,
        }

    def upsert(self, result: TypedDocumentResult) -> None:
        row = _record_to_row(result)
        self._rows[result.source_file] = row
        self._statuses[result.source_file] = row["Processing Status"]

    def flush(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path: Path | None = None
        try:
            with NamedTemporaryFile(
                "w",
                newline="",
                encoding="utf-8-sig",
                delete=False,
                dir=self.output_path.parent,
                prefix=f".{self.output_path.name}.",
                suffix=".tmp",
            ) as handle:
                tmp_path = Path(handle.name)
                writer = csv.DictWriter(handle, fieldnames=TYPED_CSV_COLUMNS, extrasaction="ignore")
                writer.writeheader()
                for source_file in sorted(self._rows, key=str.casefold):
                    writer.writerow(self._rows[source_file])
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self.output_path)
        except Exception:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise

