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
    # Nikah typed-certificate fields (nikah_legacy/nikah_modern) -- see
    # src/marriage_ocr/typed/template.py's NIKAH_LEGACY/MODERN_REGIONS.
    "No Siri",
    "Hari Nikah",
    "Masa Nikah",
    "Tempat Nikah",
    "Pernikahan Kali",
    "Isteri Ke",
    "Belanja Hantaran",
    "Umur Wali",
    "Alamat Wali",
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


# Real client samples print this exact value ("T. MAKLUMAT" / "T.MAKLUMAT",
# an abbreviation of "Tiada Maklumat") directly on the form for a wali whose
# age/IC genuinely was not recorded -- normalize_age/normalize_ic correctly
# decline to parse that as a number/IC and fall back to None, but a blank
# CSV cell then reads as "not extracted" rather than "the form itself says
# there is no information", which is what a human reviewer needs to see.
# Applied as a display-layer fallback for ANY blank Umur/IC Wali (not only
# when the source is provably "T. MAKLUMAT") per explicit client request:
# a confident wrong value is worse than this placeholder either way. `applies`
# gates this to records where the field is structurally expected at all --
# see the caller for why a bare None isn't enough on its own.
def _value_or_no_info(value: object | None, *, applies: bool) -> str:
    if applies and value is None:
        return "TIADA MAKLUMAT"
    return _value(value)


# Jumlah Bayaran / Tarikh Lahir Suami / Tarikh Lahir Isteri are real, populated
# fields for typed Cerai and Rujuk (a Legacy-era registration fee; a birthdate
# printed instead of an age) -- shared across all three typed record types in
# this one row-builder and column list, so the field itself can't simply be
# dropped without breaking Cerai/Rujuk. For Nikah specifically, per explicit
# client request, these are pure clutter (Nikah's own age/fee concepts are
# Umur Suami/Isteri and Mas Kahwin, already separate columns) -- this blanks
# the cell for Nikah rows while leaving Cerai/Rujuk untouched. Nikah's own
# ExtractedRecord.tarikh_lahir_suami/isteri stay populated internally (see
# umur_ic_wali_applies above, which depends on them still being set) --
# this only affects what reaches the exported cell, not the underlying field.
def _value_unless_nikah(value: object | None, *, record_type: str) -> str:
    if record_type == "NIKAH":
        return ""
    return _value(value)


# No Siri / Pekerjaan Suami-Isteri / Tempat Nikah Daerah-Negeri are real,
# populated fields for other typed record types (No Siri for Nikah's legacy
# cert stamp; the rest for Cerai) -- per explicit client request, this
# blanks them for Rujuk specifically without dropping the column entirely
# and losing that other data.
def _value_unless_rujuk(value: object | None, *, record_type: str) -> str:
    if record_type == "RUJUK":
        return ""
    return _value(value)


# Jumlah Bayaran is real for typed Cerai but, per explicit client request,
# pure clutter for both Nikah (see _value_unless_nikah's own docstring) and
# Rujuk.
def _value_unless_nikah_or_rujuk(value: object | None, *, record_type: str) -> str:
    if record_type in ("NIKAH", "RUJUK"):
        return ""
    return _value(value)


def _record_to_row(result: TypedDocumentResult) -> dict[str, str]:
    record = result.record
    # Umur Suami/Isteri/Wali and IC Wali only structurally exist on typed
    # Nikah Modern records: Cerai/Rujuk have no Wali field at all, and Nikah
    # Legacy prints a birthdate instead of an age for Suami/Isteri and has
    # no Umur/IC Wali region defined (see _build_nikah_typed_record's
    # docstring) -- there, these fields stay None because the concept
    # doesn't apply to that layout, not because the form said so. Legacy
    # is the only path that ever populates tarikh_lahir_suami/isteri, so
    # its presence reliably rules out the Modern-only "TIADA MAKLUMAT"
    # placeholder below.
    umur_ic_wali_applies = (
        record.record_type == "NIKAH"
        and record.tarikh_lahir_suami is None
        and record.tarikh_lahir_isteri is None
    )
    return {
        "Record Type": _value(record.record_type),
        "Bil": _value(record.bil),
        "Nama Suami": _value(record.nama_suami),
        "IC Lama Suami": _value(record.ic_lama_suami),
        "IC Baru Suami": _value(record.ic_baru_suami),
        "Umur Suami": _value_or_no_info(record.umur_suami, applies=umur_ic_wali_applies),
        "Nama Isteri": _value(record.nama_isteri),
        "IC Lama Isteri": _value(record.ic_lama_isteri),
        "IC Baru Isteri": _value(record.ic_baru_isteri),
        "Umur Isteri": _value_or_no_info(record.umur_isteri, applies=umur_ic_wali_applies),
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
        "No Siri": _value_unless_rujuk(record.no_siri, record_type=record.record_type),
        "Hari Nikah": _value(record.hari_nikah),
        "Masa Nikah": _value(record.masa_nikah),
        "Tempat Nikah": _value(record.tempat_nikah),
        "Pernikahan Kali": _value(record.pernikahan_kali),
        "Isteri Ke": _value(record.isteri_ke),
        "Belanja Hantaran": _value(record.belanja_hantaran),
        "Umur Wali": _value_or_no_info(record.umur_wali, applies=umur_ic_wali_applies),
        "Alamat Wali": _value(record.alamat_wali),
        "IC Suami": _value(record.ic_suami),
        "IC Isteri": _value(record.ic_isteri),
        "Bangsa Suami": _value(record.bangsa_suami),
        "Bangsa Isteri": _value(record.bangsa_isteri),
        "Tarikh Lahir Suami": _value_unless_nikah(record.tarikh_lahir_suami, record_type=record.record_type),
        "Tarikh Lahir Isteri": _value_unless_nikah(record.tarikh_lahir_isteri, record_type=record.record_type),
        "Warganegara Suami": _value(record.warganegara_suami),
        "Warganegara Isteri": _value(record.warganegara_isteri),
        "Alamat Suami": _value(record.alamat_suami),
        "Alamat Isteri": _value(record.alamat_isteri),
        "Alamat Pejabat Suami": _value(record.alamat_pejabat_suami),
        "Alamat Pejabat Isteri": _value(record.alamat_pejabat_isteri),
        "Pekerjaan Suami": _value_unless_rujuk(record.pekerjaan_suami, record_type=record.record_type),
        "Pekerjaan Isteri": _value_unless_rujuk(record.pekerjaan_isteri, record_type=record.record_type),
        "Tarikh Masuk Islam Suami": _value(record.tarikh_masuk_islam_suami),
        "Tarikh Masuk Islam Isteri": _value(record.tarikh_masuk_islam_isteri),
        "Bil Daftar Nikah": _value(record.bil_daftar_nikah),
        "Bil Daftar Rujuk Asal": _value(record.bil_daftar_rujuk_asal),
        "Tarikh Rujuk": _value(record.tarikh_rujuk),
        "Tarikh Rujuk Hijri": _value(record.tarikh_rujuk_hijri),
        "Tarikh Cerai": _value(record.tarikh_cerai),
        "Tarikh Cerai Hijri": _value(record.tarikh_cerai_hijri),
        "Tarikh Daftar": _value(record.tarikh_daftar),
        "Tarikh Daftar Hijri": _value(record.tarikh_daftar_hijri),
        "Bilangan Kes Mal": _value(record.bilangan_kes_mal),
        "Tempat Nikah Daerah": _value_unless_rujuk(record.tempat_nikah_daerah, record_type=record.record_type),
        "Tempat Nikah Negeri": _value_unless_rujuk(record.tempat_nikah_negeri, record_type=record.record_type),
        "Talak Kali Ke": _value(record.talak_kali_ke),
        "Jumlah Talak": _value(record.jumlah_talak),
        "Keadaan Talak": _value(record.keadaan_talak),
        "Bayaran Tebus Talak": _value(record.bayaran_tebus_talak),
        "Tempat Cerai": _value(record.tempat_cerai),
        "Tempat Bercerai": _value(record.tempat_bercerai),
        "Cerai Dalam Keadaan": _value(record.cerai_dalam_keadaan),
        "No Sijil Perakuan Nikah Rujuk": _value(record.no_sijil_perakuan_nikah_rujuk),
        "No Permohonan Cerai": _value(record.no_permohonan_cerai),
        "Bil Cerai": _value(record.bil_cerai),
        "Rujuk Kali": _value(record.rujuk_kali),
        "Jawatan Pendaftar": _value(record.jawatan_pendaftar),
        "Jumlah Bayaran": _value_unless_nikah_or_rujuk(record.jumlah_bayaran, record_type=record.record_type),
        "Hal Hal Lain": _value(record.hal_hal_lain),
        "Source File": result.source_file,
        "Processing Status": result.processing_status.value,
        "Review Required": _value(result.review_required),
        "Failed Fields": result.failed_fields_text,
        "Retry Count": _value(result.retry_count),
        "Error Message": _value(result.error_message),
        "Duplicate Of Source File": "",
    }


# Written for every row regardless of record type -- run metadata, not
# OCR-extracted data, so keeping them even when blank is never clutter.
_ALWAYS_WRITTEN_COLUMNS = {
    "Record Type",
    "Source File",
    "Processing Status",
    "Review Required",
    "Failed Fields",
    "Retry Count",
    "Error Message",
    "Duplicate Of Source File",
}


def _active_columns(rows: Mapping[str, Mapping[str, str]]) -> list[str]:
    """TYPED_CSV_COLUMNS is one shared schema across Nikah/Cerai/Rujuk, so a
    single-type batch (e.g. an all-Nikah run) would otherwise always carry
    every Cerai/Rujuk-only column -- "IC Suami" included -- as a
    permanently-blank column sitting next to Nikah's own IC Lama/Baru
    Suami split. Dropping any data column that's empty across *every* row
    currently in the file removes that clutter automatically, and stays
    correct the moment a batch actually mixes in a Cerai/Rujuk document --
    recomputed fresh on every flush(), never cached, so it can't go stale.
    """
    return [
        column
        for column in TYPED_CSV_COLUMNS
        if column in _ALWAYS_WRITTEN_COLUMNS or any(row.get(column) for row in rows.values())
    ]


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
            active_columns = _active_columns(self._rows)
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
                writer = csv.DictWriter(handle, fieldnames=active_columns, extrasaction="ignore")
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

