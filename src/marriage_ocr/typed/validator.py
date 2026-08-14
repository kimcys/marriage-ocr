from __future__ import annotations

import re
from typing import Mapping, Sequence

from marriage_ocr.models import ExtractedRecord
from marriage_ocr.refinement.text_corrections import is_valid_malaysian_ic
from marriage_ocr.typed.models import FieldDiagnostic, ProcessingStatus, RawField, ValidationSummary
from marriage_ocr.typed.normalizer import BIL_PATTERN, DATE_PATTERN, MAS_KAHWIN_PATTERN


RETRY_PRIORITY = (
    "bil",
    "nama_suami",
    "id_suami",
    "umur_suami",
    "nama_isteri",
    "id_isteri",
    "umur_isteri",
    "mas_kahwin",
    "nama_pendaftar",
    "alamat_pendaftar",
    "nama_wali",
    "hubungan_wali",
    "saksi_1",
    "saksi_2",
    "tarikh_nikah",
)

_STRICT_OUTPUT_FIELDS = {
    "bil": "Bil",
    "nama_suami": "Nama Suami",
    "id_suami": "IC Suami",
    "umur_suami": "Umur Suami",
    "nama_isteri": "Nama Isteri",
    "id_isteri": "IC Isteri",
    "umur_isteri": "Umur Isteri",
    "mas_kahwin": "Mas Kahwin",
    "nama_pendaftar": "Nama Pendaftar",
    "alamat_pendaftar": "Alamat Pendaftar",
    "nama_wali": "Nama Wali",
    "hubungan_wali": "Hubungan Wali",
    "saksi_1": "Saksi 1",
    "saksi_2": "Saksi 2",
    "tarikh_nikah": "Tarikh Nikah",
}

_CONTAMINATION_LABELS = ("WARGANEGARA", "BANGSA", "ALAMAT", "SAKSI KEDUA", "BELANJA HANTARAN")


def _text(value: object | None) -> str:
    return "" if value is None else str(value).strip()


def _valid_ic(value: str | None) -> bool:
    if not value:
        return False
    return bool(re.fullmatch(r"\d{7,8}|\d{12}", value))


def _valid_date(value: str | None) -> bool:
    if not value:
        return False
    return bool(DATE_PATTERN.search(value.replace(" ", "")))


def _valid_mas_kahwin(value: str | None) -> bool:
    if not value:
        return False
    return bool(MAS_KAHWIN_PATTERN.search(value))


def _non_label_text(value: str, label: str) -> bool:
    normalized = re.sub(r"[:\s]+", " ", value).strip().upper()
    return bool(normalized and normalized != label.upper())


def validate_record(
    record: ExtractedRecord,
    raw_fields: Mapping[str, RawField],
    *,
    word_confidence_threshold: float,
    min_age: int = 16,
    max_age: int = 120,
    max_retry_fields: int = 6,
) -> ValidationSummary:
    diagnostics: dict[str, FieldDiagnostic] = {}
    retry_candidates: list[str] = []
    failed_fields: list[str] = []

    def mark(
        key: str,
        *,
        value: str | None,
        output_name: str,
        valid: bool,
        confidence: float,
        issues: Sequence[str] = (),
        retryable: bool = True,
    ) -> None:
        diagnostics[key] = FieldDiagnostic(
            key=key,
            output_name=output_name,
            valid=valid,
            confidence=confidence,
            issues=tuple(issues),
        )
        if not valid:
            failed_fields.append(output_name)
            if retryable and key not in retry_candidates:
                retry_candidates.append(key)

    for key, output_name in _STRICT_OUTPUT_FIELDS.items():
        raw = raw_fields.get(key)
        raw_text = _text(raw.raw_text if raw else None)
        check_text = raw_text.splitlines()[0].strip() if key in {"umur_suami", "umur_isteri"} and raw_text else raw_text
        confidence = float(raw.confidence if raw else 0.0)
        issues: list[str] = []
        valid = True

        if key == "bil":
            valid = bool(record.bil and BIL_PATTERN.fullmatch(record.bil))
        elif key == "nama_suami":
            valid = bool(record.nama_suami and _non_label_text(record.nama_suami, output_name))
        elif key == "id_suami":
            valid = bool(record.ic_lama_suami or record.ic_baru_suami)
            if record.ic_lama_suami and not is_valid_malaysian_ic(record.ic_lama_suami):
                valid = False
            if record.ic_baru_suami and not re.fullmatch(r"\d{12}", record.ic_baru_suami):
                valid = False
        elif key == "umur_suami":
            valid = bool(isinstance(record.umur_suami, int) and min_age <= record.umur_suami <= max_age)
        elif key == "nama_isteri":
            valid = bool(record.nama_isteri and _non_label_text(record.nama_isteri, output_name))
        elif key == "id_isteri":
            valid = bool(record.ic_lama_isteri or record.ic_baru_isteri)
            if record.ic_lama_isteri and not is_valid_malaysian_ic(record.ic_lama_isteri):
                valid = False
            if record.ic_baru_isteri and not re.fullmatch(r"\d{12}", record.ic_baru_isteri):
                valid = False
        elif key == "umur_isteri":
            valid = bool(isinstance(record.umur_isteri, int) and min_age <= record.umur_isteri <= max_age)
        elif key == "mas_kahwin":
            valid = bool(record.mas_kahwin and _valid_mas_kahwin(record.mas_kahwin))
        elif key == "nama_pendaftar":
            valid = bool(record.nama_pendaftar and _non_label_text(record.nama_pendaftar, output_name))
        elif key == "alamat_pendaftar":
            valid = bool(record.alamat_pendaftar and _non_label_text(record.alamat_pendaftar, output_name))
        elif key == "nama_wali":
            valid = bool(record.nama_wali and _non_label_text(record.nama_wali, output_name))
        elif key == "hubungan_wali":
            valid = bool(record.hubungan_wali and _non_label_text(record.hubungan_wali, output_name))
        elif key == "saksi_1":
            valid = bool(record.saksi_1 and _non_label_text(record.saksi_1, output_name))
        elif key == "saksi_2":
            valid = bool(record.saksi_2 and _non_label_text(record.saksi_2, output_name))
        elif key == "tarikh_nikah":
            valid = bool(record.tarikh_nikah and _valid_date(record.tarikh_nikah))

        if check_text and confidence < word_confidence_threshold:
            valid = False
            issues.append(f"confidence below threshold: {confidence:.3f}")

        if check_text and key not in {"umur_suami", "umur_isteri", "id_suami", "id_isteri"}:
            contamination = [label for label in _CONTAMINATION_LABELS if label in check_text.upper()]
            if contamination:
                valid = False
                issues.append(f"contamination: {', '.join(contamination)}")

        if not check_text and key in {
            "bil",
            "nama_suami",
            "id_suami",
            "umur_suami",
            "nama_isteri",
            "id_isteri",
            "umur_isteri",
            "mas_kahwin",
            "nama_pendaftar",
            "alamat_pendaftar",
            "nama_wali",
            "hubungan_wali",
            "saksi_1",
            "saksi_2",
            "tarikh_nikah",
        }:
            issues.append("missing field")

        mark(
            key,
            value=raw_text,
            output_name=output_name,
            valid=valid,
            confidence=confidence,
            issues=issues,
        )

    meaningful_field_count = sum(
        1
        for value in [
            record.bil,
            record.nama_suami,
            record.ic_lama_suami or record.ic_baru_suami,
            record.umur_suami,
            record.nama_isteri,
            record.ic_lama_isteri or record.ic_baru_isteri,
            record.umur_isteri,
            record.mas_kahwin,
            record.nama_pendaftar,
            record.alamat_pendaftar,
            record.nama_wali,
            record.hubungan_wali,
            record.saksi_1,
            record.saksi_2,
            record.tarikh_nikah,
        ]
        if value not in {None, ""}
    )

    ordered_retry_fields = tuple(key for key in RETRY_PRIORITY if key in retry_candidates)[:max_retry_fields]
    ordered_failed_fields = tuple(failed_fields)
    return ValidationSummary(
        diagnostics=diagnostics,
        retry_fields=ordered_retry_fields,
        failed_fields=ordered_failed_fields,
        meaningful_field_count=meaningful_field_count,
    )


def status_for_result(
    summary: ValidationSummary,
    retry_count: int,
    document_error: str | None = None,
) -> ProcessingStatus:
    if document_error:
        return ProcessingStatus.FAILED
    if summary.meaningful_field_count <= 0:
        return ProcessingStatus.FAILED
    if summary.failed_fields:
        return ProcessingStatus.REVIEW_REQUIRED
    if retry_count > 0:
        return ProcessingStatus.SUCCESS_WITH_RETRY
    return ProcessingStatus.SUCCESS
