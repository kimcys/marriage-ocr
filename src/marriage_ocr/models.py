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
    record_type: str = "NIKAH"

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
    tarikh_nikah_hijri: str | None = None
    tarikh_keluar: str | None = None
    tarikh_keluar_raw: str | None = None

    remarks: str | None = None

    # Shared across Cerai/Rujuk (and, for no_rujukan/no_siri/tarikh_daftar,
    # potentially Nikah too) -- see src/llm/record_schemas.py for the real
    # sample images these are grounded in.
    no_rujukan: str | None = None
    no_siri: str | None = None
    tarikh_daftar: str | None = None

    # Cerai/Rujuk use a single IC/passport field per person rather than
    # Nikah's split ic_lama_*/ic_baru_*.
    ic_suami: str | None = None
    ic_suami_raw: str | None = None
    ic_isteri: str | None = None
    ic_isteri_raw: str | None = None

    # Cerai-specific.
    tempat_cerai: str | None = None
    keadaan_talak: str | None = None
    jumlah_talak: str | None = None
    tarikh_cerai: str | None = None
    tarikh_cerai_raw: str | None = None
    bil_daftar_rujukan: str | None = None

    # Rujuk-specific.
    tempat_rujuk: str | None = None
    bil_cerai: str | None = None
    rujuk_kali: str | None = None
    tarikh_rujuk: str | None = None
    tarikh_rujuk_raw: str | None = None

    # Modern (2020s-era) Cerai/Rujuk layout folds several fields above into
    # one free-text "Catatan" column -- catatan_raw preserves it verbatim
    # even when field-level parsing misses something. hal_hal_lain is the
    # Cerai/Rujuk equivalent of Nikah's `remarks`.
    catatan_raw: str | None = None
    hal_hal_lain: str | None = None

    # Typed Cerai (Borang 8/9/10) and Rujuk (Borang 5/7B/8B) certificates
    # carry richer per-spouse personal details than any handwritten ledger
    # does -- see src/marriage_ocr/typed/template.py for the real samples
    # these are grounded in. Shared by both typed record types.
    bangsa_suami: str | None = None
    bangsa_isteri: str | None = None
    tarikh_lahir_suami: str | None = None
    tarikh_lahir_isteri: str | None = None
    warganegara_suami: str | None = None
    warganegara_isteri: str | None = None
    alamat_suami: str | None = None
    alamat_isteri: str | None = None
    # Legacy-only (Borang 5/9, 1984 enactment): home vs. office address kept
    # as two separate fields rather than the one "Alamat" the modern forms use.
    alamat_pejabat_suami: str | None = None
    alamat_pejabat_isteri: str | None = None
    pekerjaan_suami: str | None = None
    pekerjaan_isteri: str | None = None
    tarikh_masuk_islam_suami: str | None = None
    tarikh_masuk_islam_isteri: str | None = None
    no_kad_perakuan_islam_suami: str | None = None
    no_kad_perakuan_islam_isteri: str | None = None

    # Cross-reference bil numbers and their Hijri date pairs (every Masihi
    # date on these typed forms has a paired Hijrah date; tarikh_nikah_hijri
    # already existed as precedent for this pattern).
    bil_daftar_nikah: str | None = None
    bil_daftar_rujuk_asal: str | None = None
    tarikh_rujuk_hijri: str | None = None
    tarikh_cerai_hijri: str | None = None
    tarikh_daftar_hijri: str | None = None

    # Typed Cerai-specific.
    bilangan_kes_mal: str | None = None
    tempat_nikah_daerah: str | None = None
    tempat_nikah_negeri: str | None = None
    talak_kali_ke: str | None = None
    bayaran_tebus_talak: str | None = None
    # tempat_cerai (above) already holds the "Cara Bercerai" court-permission
    # value, matching the handwritten Cerai ledger's own confusingly-named
    # "Tempat Cerai" column; tempat_bercerai is the genuinely place-shaped
    # field the typed form separately has (e.g. "DALAM MAHKAMAH").
    tempat_bercerai: str | None = None
    cerai_dalam_keadaan: str | None = None
    # Legacy-only (Borang 9): a single combined reference field, a case
    # number, and two named witnesses (a typed Cerai cert having witnesses
    # at all is specific to the 1984-enactment form).
    no_sijil_perakuan_nikah_rujuk: str | None = None
    no_permohonan_cerai: str | None = None

    # Typed Rujuk-specific.
    jawatan_pendaftar: str | None = None

    # Legacy-only (Borang 5/9): a small registration fee, printed in the
    # pre-decimalisation "$" notation on these older forms.
    jumlah_bayaran: str | None = None

    confidence: float = 0.0
    status_review: str = "REVIEW"
    review_reason: list[str] = field(default_factory=list)
    # Which business fields are actually absent (as opposed to present-but-
    # suspicious/invalid) -- a subset of review_reason's causes, named after
    # exporter.py's XLSX column so a caller can map straight to "this column
    # needs a value" instead of parsing free-text reasons.
    missing_fields: list[str] = field(default_factory=list)

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

    # Set by batch_runner.py's content-identity dedup check (same
    # record_type + Bil + an IC in common as an already-inserted record,
    # from a *different* source file) -- not raised by any OCR/normalizer
    # code path.
    is_duplicate: bool = False
    duplicate_of_record_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExtractedRecord:
        allowed = {field_.name for field_ in fields(cls)}
        filtered = {key: value for key, value in data.items() if key in allowed}
        return cls(**filtered)
