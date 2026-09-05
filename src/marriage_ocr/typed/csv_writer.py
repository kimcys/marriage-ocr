from __future__ import annotations

import csv
import os
from dataclasses import replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable, Mapping

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
    "Tarikh Nikah Hijri",
    "Tarikh Keluar",
    # Cerai/Rujuk typed-certificate fields -- see src/marriage_ocr/typed/template.py
    "Record Type",
    "IC Suami",
    "IC Isteri",
    "Bangsa Suami",
    "Bangsa Isteri",
    "Tarikh Lahir Suami",
    "Tarikh Lahir Isteri",
    "Warganegara Suami",
    "Warganegara Isteri",
    "Alamat Suami",
    "Alamat Isteri",
    "Alamat Pejabat Suami",
    "Alamat Pejabat Isteri",
    "Pekerjaan Suami",
    "Pekerjaan Isteri",
    "Tarikh Masuk Islam Suami",
    "Tarikh Masuk Islam Isteri",
    "No Kad Perakuan Islam Suami",
    "No Kad Perakuan Islam Isteri",
    "Bil Daftar Nikah",
    "Bil Daftar Rujuk Asal",
    "Tarikh Rujuk",
    "Tarikh Rujuk Hijri",
    "Tarikh Cerai",
    "Tarikh Cerai Hijri",
    "Tarikh Daftar",
    "Tarikh Daftar Hijri",
    "Bilangan Kes Mal",
    "Tempat Nikah Daerah",
    "Tempat Nikah Negeri",
    "Talak Kali Ke",
    "Jumlah Talak",
    "Keadaan Talak",
    "Bayaran Tebus Talak",
    "Tempat Cerai",
    "Tempat Bercerai",
    "Cerai Dalam Keadaan",
    "No Sijil Perakuan Nikah Rujuk",
    "No Permohonan Cerai",
    "Tempat Rujuk",
    "Bil Cerai",
    "Rujuk Kali",
    "Jawatan Pendaftar",
    "Jumlah Bayaran",
    "Hal Hal Lain",
    "Source File",
    "Processing Status",
    "Review Required",
    "Failed Fields",
    "Retry Count",
    "Error Message",
    "Duplicate Of Source File",
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
        "Record Type": _value(record.record_type),
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
        "Tarikh Nikah Hijri": _value(record.tarikh_nikah_hijri),
        "Tarikh Keluar": _value(record.tarikh_keluar),
        "IC Suami": _value(record.ic_suami),
        "IC Isteri": _value(record.ic_isteri),
        "Bangsa Suami": _value(record.bangsa_suami),
        "Bangsa Isteri": _value(record.bangsa_isteri),
        "Tarikh Lahir Suami": _value(record.tarikh_lahir_suami),
        "Tarikh Lahir Isteri": _value(record.tarikh_lahir_isteri),
        "Warganegara Suami": _value(record.warganegara_suami),
        "Warganegara Isteri": _value(record.warganegara_isteri),
        "Alamat Suami": _value(record.alamat_suami),
        "Alamat Isteri": _value(record.alamat_isteri),
        "Alamat Pejabat Suami": _value(record.alamat_pejabat_suami),
        "Alamat Pejabat Isteri": _value(record.alamat_pejabat_isteri),
        "Pekerjaan Suami": _value(record.pekerjaan_suami),
        "Pekerjaan Isteri": _value(record.pekerjaan_isteri),
        "Tarikh Masuk Islam Suami": _value(record.tarikh_masuk_islam_suami),
        "Tarikh Masuk Islam Isteri": _value(record.tarikh_masuk_islam_isteri),
        "No Kad Perakuan Islam Suami": _value(record.no_kad_perakuan_islam_suami),
        "No Kad Perakuan Islam Isteri": _value(record.no_kad_perakuan_islam_isteri),
        "Bil Daftar Nikah": _value(record.bil_daftar_nikah),
        "Bil Daftar Rujuk Asal": _value(record.bil_daftar_rujuk_asal),
        "Tarikh Rujuk": _value(record.tarikh_rujuk),
        "Tarikh Rujuk Hijri": _value(record.tarikh_rujuk_hijri),
        "Tarikh Cerai": _value(record.tarikh_cerai),
        "Tarikh Cerai Hijri": _value(record.tarikh_cerai_hijri),
        "Tarikh Daftar": _value(record.tarikh_daftar),
        "Tarikh Daftar Hijri": _value(record.tarikh_daftar_hijri),
        "Bilangan Kes Mal": _value(record.bilangan_kes_mal),
        "Tempat Nikah Daerah": _value(record.tempat_nikah_daerah),
        "Tempat Nikah Negeri": _value(record.tempat_nikah_negeri),
        "Talak Kali Ke": _value(record.talak_kali_ke),
        "Jumlah Talak": _value(record.jumlah_talak),
        "Keadaan Talak": _value(record.keadaan_talak),
        "Bayaran Tebus Talak": _value(record.bayaran_tebus_talak),
        "Tempat Cerai": _value(record.tempat_cerai),
        "Tempat Bercerai": _value(record.tempat_bercerai),
        "Cerai Dalam Keadaan": _value(record.cerai_dalam_keadaan),
        "No Sijil Perakuan Nikah Rujuk": _value(record.no_sijil_perakuan_nikah_rujuk),
        "No Permohonan Cerai": _value(record.no_permohonan_cerai),
        "Tempat Rujuk": _value(record.tempat_rujuk),
        "Bil Cerai": _value(record.bil_cerai),
        "Rujuk Kali": _value(record.rujuk_kali),
        "Jawatan Pendaftar": _value(record.jawatan_pendaftar),
        "Jumlah Bayaran": _value(record.jumlah_bayaran),
        "Hal Hal Lain": _value(record.hal_hal_lain),
        "Source File": result.source_file,
        "Processing Status": result.processing_status.value,
        "Review Required": _value(result.review_required),
        "Failed Fields": result.failed_fields_text,
        "Retry Count": _value(result.retry_count),
        "Error Message": _value(result.error_message),
        "Duplicate Of Source File": "",
    }


def _content_key(row: Mapping[str, str]) -> tuple[str, str, str, str] | None:
    """Content-identity key for cross-file dedup: same record_type + Bil +
    at least one IC in common as a row already in the store, but from a
    *different* source file -- catches the same real register entry
    scanned/photographed twice under different filenames (byte-different,
    so the batch-runner's file-hash check can't catch it). Bil alone is not
    a safe key -- it repeats across different registrar offices/books --
    so this only fires when an IC is also present.
    """
    record_type = row.get("Record Type") or ""
    bil = row.get("Bil") or ""
    ic_suami = row.get("IC Suami") or row.get("IC Baru Suami") or ""
    ic_isteri = row.get("IC Isteri") or row.get("IC Baru Isteri") or ""
    if not bil or not (ic_suami or ic_isteri):
        return None
    return (record_type, bil, ic_suami, ic_isteri)


class TypedCsvStore:
    def __init__(self, output_path: Path, *, skip_existing: bool = False) -> None:
        self.output_path = output_path
        self.skip_existing = skip_existing
        self._rows: dict[str, dict[str, str]] = {}
        self._statuses: dict[str, str] = {}
        self._content_index: dict[tuple[str, str, str, str], str] = {}

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
                    loaded_row = {column: row.get(column, "") or "" for column in TYPED_CSV_COLUMNS}
                    store._rows[source] = loaded_row
                    store._statuses[source] = loaded_row["Processing Status"]
                    if not loaded_row.get("Duplicate Of Source File"):
                        key = _content_key(loaded_row)
                        if key is not None:
                            store._content_index.setdefault(key, source)
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
        key = _content_key(row)
        if key is not None:
            existing_source = self._content_index.get(key)
            if existing_source is not None and existing_source != result.source_file:
                row["Duplicate Of Source File"] = existing_source
            else:
                self._content_index[key] = result.source_file
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

