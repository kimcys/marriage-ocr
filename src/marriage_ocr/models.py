from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any


@dataclass
class OcrLine:
    text: str
    confidence: float = 0.0
    bbox: list[float] | None = None


@dataclass
class OcrResult:
    text: str = ""
    lines: list[OcrLine] = field(default_factory=list)
    average_confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "average_confidence": self.average_confidence,
            "lines": [
                {
                    "text": line.text,
                    "confidence": line.confidence,
                    "bbox": line.bbox,
                }
                for line in self.lines
            ],
        }


@dataclass
class ExtractedRecord:
    bil: str | None = None

    nama_suami: str | None = None
    ic_lama_suami: str | None = None
    ic_baru_suami: str | None = None
    id_suami_raw: str | None = None
    umur_suami: int | None = None

    nama_isteri: str | None = None
    ic_lama_isteri: str | None = None
    ic_baru_isteri: str | None = None
    id_isteri_raw: str | None = None
    umur_isteri: int | None = None

    mas_kahwin: str | None = None
    mas_kahwin_raw: str | None = None

    nama_pendaftar: str | None = None
    alamat_pendaftar: str | None = None

    nama_wali: str | None = None
    hubungan_wali: str | None = None

    saksi_1: str | None = None
    saksi_2: str | None = None

    tarikh_nikah: str | None = None
    tarikh_nikah_raw: str | None = None
    tarikh_keluar: str | None = None
    tarikh_keluar_raw: str | None = None

    remarks: str | None = None

    confidence: float = 0.0
    status_review: str = "REVIEW"
    review_reason: list[str] = field(default_factory=list)

    source_file: str | None = None
    source_page: int | None = None
    source_record: str | None = None
    crop_folder: str | None = None

    raw_bil: str | None = None
    raw_suami_isteri: str | None = None
    raw_pendaftar: str | None = None
    raw_wali: str | None = None
    raw_hubungan_wali: str | None = None
    raw_saksi: str | None = None
    raw_tarikh_nikah: str | None = None
    raw_tarikh_keluar: str | None = None
    raw_remarks: str | None = None
    raw_ocr_json: str | None = None

    created_at: str | None = None
    updated_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExtractedRecord:
        allowed = {field_.name for field_ in fields(cls)}
        filtered = {key: value for key, value in data.items() if key in allowed}
        return cls(**filtered)
