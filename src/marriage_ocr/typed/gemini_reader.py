"""Read a typed certificate's template regions with Gemini instead of Vision
OCR + region cropping. Only the *reading* step changes: the result is the same
`RawField` per template region that typed/extractor.py produces from Vision,
so build_extracted_record's normalizers and validate_record run unchanged.

Why: on 79 real certificates across all six templates, Gemini read the typed
values far more reliably than region cropping (which pulls in printed labels
and neighbouring lines, and misfiles values that drift out of their box) --
spot-checked disagreements favoured Gemini ~29 to 1 -- at about a third of the
cost per certificate in Batch Mode. Its one observed weakness, picking a
struck-through option ("*Raj'ie/~~Bain~~"), is handled explicitly in the
prompt.

Labels below are the form's own printed wording for each region, written
against real samples of every template (see typed/template.py for regions).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from marriage_ocr.logging_config import get_logger
from marriage_ocr.typed.models import RawField
from marriage_ocr.typed.template import TEMPLATES

LOGGER = get_logger(__name__)

DEFAULT_MODEL = "gemini-3.5-flash-lite"
_TRANSIENT_CODES = {429, 500, 502, 503, 504}
RENDER_DPI = 300
_MAX_OUTPUT_TOKENS = 4096

_TITLES = {"nikah": "Surat Perakuan Nikah", "cerai": "Surat Perakuan Cerai", "rujuk": "Surat Perakuan Rujuk"}

# template -> region key -> the form's printed label for that value.
FIELD_LABELS: dict[str, dict[str, str]] = {
    "nikah_legacy": {
        "no_siri": "serial number printed top right after 'No' (e.g. 006064)",
        "bil": "1. Bilangan Daftar Nikah",
        "tarikh_nikah": "2. Tarikh Akad Nikah H/M (copy the whole line, Hijri and Masihi)",
        "tarikh_daftar": "4. Tarikh di daftar pada H/M (copy the whole line)",
        "nama_suami": "5. Nama Suami",
        "id_suami": "Nama Suami's No. Kad Pengenalan",
        "tarikh_lahir_suami": "Nama Suami's Tarikh Lahir",
        "alamat_suami": "Nama Suami's Alamat",
        "nama_isteri": "6. Nama Isteri",
        "id_isteri": "Nama Isteri's No. Kad Pengenalan",
        "tarikh_lahir_isteri": "Nama Isteri's Tarikh Lahir",
        "alamat_isteri": "Nama Isteri's Alamat",
        "nama_wali": "7. Nama Wali",
        "id_wali": "Nama Wali's No. Kad Pengenalan",
        "alamat_wali": "Nama Wali's Alamat",
        "hubungan_wali": "Perhubungan (the wali's relationship)",
        "saksi_1": "8. Nama Saksi (i)",
        "id_saksi_1": "Saksi (i)'s No. Kad Pengenalan",
        "saksi_2": "Nama Saksi (ii)",
        "id_saksi_2": "Saksi (ii)'s No. Kad Pengenalan",
        "mas_kahwin": "Mas Kahwin",
        "belanja_hantaran": "Belanja Hantaran",
        "pemberian_lain": "Pemberian lain (jika ada)",
        "jumlah_bayaran": "Jumlah Bayaran",
    },
    "nikah_modern": {
        "no_siri": "No. Siri (top right)",
        "bil": "Bilangan Daftar (all of it, including any ASAL / SALINAN lines)",
        "tarikh_daftar": "Tarikh Daftar (Hijri line and Masihi line)",
        "nama_suami": "SUAMI - Nama",
        "id_suami": "SUAMI - No. Kad Pengenalan/ Pasport",
        "umur_suami": "SUAMI - Umur (Tahun)",
        "warganegara_suami": "SUAMI - Warganegara",
        "bangsa_suami": "SUAMI - Bangsa",
        "alamat_suami": "SUAMI - Alamat",
        "nama_isteri": "ISTERI - Nama",
        "id_isteri": "ISTERI - No. Kad Pengenalan/ Pasport",
        "umur_isteri": "ISTERI - Umur (Tahun)",
        "warganegara_isteri": "ISTERI - Warganegara",
        "bangsa_isteri": "ISTERI - Bangsa",
        "alamat_isteri": "ISTERI - Alamat",
        "nama_wali": "B. MAKLUMAT WALI - Nama Wali",
        "id_wali": "WALI - No. Kad Pengenalan/ Pasport",
        "umur_wali": "WALI - Umur (Tahun)",
        "hubungan_wali": "WALI - Hubungan",
        "alamat_wali": "WALI - Alamat",
        "saksi_1": "SAKSI PERTAMA - Nama",
        "id_saksi_1": "SAKSI PERTAMA - No. Kad Pengenalan/ Pasport",
        "saksi_2": "SAKSI KEDUA - Nama",
        "id_saksi_2": "SAKSI KEDUA - No. Kad Pengenalan/ Pasport",
        "tarikh_nikah_hijri": "D. Tarikh Nikah - Hijrah",
        "tarikh_nikah": "D. Tarikh Nikah - Masihi",
        "hari_nikah": "Hari",
        "masa_nikah": "Masa",
        "tempat_nikah": "Tempat",
        "nama_pendaftar": "Nama Pendaftar / Pen. Pendaftar",
        "pernikahan_kali": "Pernikahan Kali",
        "isteri_ke": "Isteri ke",
        "mas_kahwin": "Mas Kahwin",
        "belanja_hantaran": "Belanja Hantaran",
        "pemberian_lain": "Pemberian Lain (Jika Ada)",
    },
    "cerai_legacy": {
        "bil": "Bilangan Daftar Cerai",
        "no_sijil_perakuan_nikah_rujuk": "No. Sijil Perakuan Nikah/Rujuk",
        "no_permohonan_cerai": "No. Permohonan Cerai",
        "nama_suami": "Nama Suami",
        "id_suami": "Nama Suami's No. Kad Pengenalan",
        "tarikh_lahir_suami": "Nama Suami's Tarikh Lahir",
        "pekerjaan_suami": "Nama Suami's Pekerjaan",
        "alamat_suami": "Nama Suami's Alamat",
        "nama_isteri": "Nama Isteri",
        "id_isteri": "Nama Isteri's No. Kad Pengenalan",
        "tarikh_lahir_isteri": "Nama Isteri's Tarikh Lahir",
        "pekerjaan_isteri": "Nama Isteri's Pekerjaan",
        "alamat_isteri": "Nama Isteri's Alamat",
        "tarikh_nikah": "Tarikh Perkahwinan/Rujuk H/M",
        "saksi_1": "(1) Nama Saksi",
        "saksi_2": "(2) Nama Saksi",
        "tempat_cerai": "Cara Bercerai",
        "keadaan_talak": "Talaq Kali (all of it, including a continuation line)",
        "bayaran_tebus_talak": "Bayaran tebus talaq (jika ada)",
        "tempat_bercerai": "Tempat Bercerai (all of it, including a continuation line)",
        "tarikh_cerai": "Tarikh Bercerai H/M",
        "cerai_dalam_keadaan": "Cerai dalam *Suci/Haidh/Mengandung",
        "hal_hal_lain": "Nafkah dalam iddah / Mut'ah / Penyelesaian harta, dsb. (any typed text on those lines)",
        "jumlah_bayaran": "Jumlah Bayaran",
        "tarikh_daftar": "Tarikh H/M at the bottom (the registration date)",
    },
    "cerai_modern": {
        "bil": "1. Bilangan Daftar Cerai",
        "bil_daftar_nikah": "2. Bilangan Daftar Surat Perakuan Nikah",
        "bil_daftar_rujuk_asal": "3. Bilangan Daftar Surat Perakuan Rujuk",
        "bilangan_kes_mal": "4. Bilangan Kes Mal",
        "tarikh_nikah_hijri": "5. Tarikh Akad Nikah - Hijrah",
        "tarikh_nikah": "5. Tarikh Akad Nikah - Masihi",
        "tarikh_rujuk_hijri": "6. Tarikh Rujuk - Hijrah",
        "tarikh_rujuk": "6. Tarikh Rujuk - Masihi",
        "tempat_nikah_daerah": "7. Tempat Pernikahan: Daerah",
        "tempat_nikah_negeri": "7. Tempat Pernikahan: Negeri/Negara",
        "nama_suami": "8. Nama Suami",
        "id_suami": "8. No. Kad Pengenalan/Pasport (suami)",
        "bangsa_suami": "8. Bangsa (suami)",
        "tarikh_lahir_suami": "8. Tarikh Lahir (suami)",
        "warganegara_suami": "8. Warganegara (suami)",
        "alamat_suami": "8. Alamat (Rumah) (suami, all lines)",
        "pekerjaan_suami": "8. Pekerjaan (suami)",
        "nama_isteri": "9. Nama Isteri",
        "id_isteri": "9. No. Kad Pengenalan/Pasport (isteri)",
        "bangsa_isteri": "9. Bangsa (isteri)",
        "tarikh_lahir_isteri": "9. Tarikh Lahir (isteri)",
        "warganegara_isteri": "9. Warganegara (isteri)",
        "alamat_isteri": "9. Alamat (Rumah) (isteri, all lines)",
        "pekerjaan_isteri": "9. Pekerjaan (isteri)",
        "keadaan_talak": "10. Jenis Perceraian (*Raj'ie/Bain(Bain Sughra/Bain Kubra/Bain Muabbad))",
        "talak_kali_ke": "Talaq kali ke",
        "jumlah_talak": "Jumlah Talaq",
        "bayaran_tebus_talak": "Bayaran Tebus Talaq (Jika Ada)",
        "tempat_cerai": "Cara Bercerai (*Dengan Kebenaran Mahkamah/Tanpa Kebenaran Mahkamah)",
        "tempat_bercerai": "Tempat Bercerai",
        "tarikh_cerai_hijri": "Tarikh Bercerai - Hijrah",
        "tarikh_cerai": "Tarikh Bercerai - Masihi",
        "cerai_dalam_keadaan": "Cerai dalam *Suci/Haid/Mengandung",
        "hal_hal_lain": "Kenyataan lain (page 2, all lines of text)",
        "tarikh_daftar_hijri": "page 2 bottom: Tarikh - Hijrah (registration date)",
        "tarikh_daftar": "page 2 bottom: Tarikh - Masihi (registration date)",
        "nama_pendaftar": "the registrar's name printed or stamped under the signature",
        "jawatan_pendaftar": "the registrar's position/title under their name",
    },
    "rujuk_legacy": {
        "bil": "1. Bilangan Daftar Rujuk",
        "tarikh_daftar": "2. Tarikh H/M",
        "nama_suami": "3. Nama Suami",
        "id_suami": "Nama Suami's No. Kad Pengenalan",
        "tarikh_lahir_suami": "Nama Suami's Tarikh Lahir",
        "pekerjaan_suami": "Nama Suami's Pekerjaan",
        "alamat_suami": "Nama Suami's Alamat Rumah (all lines)",
        "alamat_pejabat_suami": "Nama Suami's Alamat Pejabat",
        "nama_isteri": "4. Nama Isteri",
        "id_isteri": "Nama Isteri's No. Kad Pengenalan",
        "tarikh_lahir_isteri": "Nama Isteri's Tarikh Lahir",
        "pekerjaan_isteri": "Nama Isteri's Pekerjaan",
        "alamat_isteri": "Nama Isteri's Alamat Rumah (all lines)",
        "alamat_pejabat_isteri": "Nama Isteri's Alamat Pejabat",
        "nama_pendaftar": "Saya ... (Nama Pendaftar)",
        "bil_cerai": "Bilangan Daftar Cerai",
        "tarikh_cerai": "Tarikh Perceraian H/M",
        "tarikh_rujuk": "Tarikh Rujuk H/M",
        "hal_hal_lain": "page 2: any other typed statement (Hal-hal lain / Kenyataan lain)",
        "jumlah_bayaran": "page 2: Jumlah Bayaran",
    },
    "rujuk_modern": {
        "bil": "Bilangan Daftar rujuk",
        "tarikh_daftar_hijri": "Tarikh: Hijrah (next to Bilangan Daftar rujuk)",
        "tarikh_daftar": "Tarikh: Masihi (next to Bilangan Daftar rujuk)",
        "nama_suami": "1. Nama Suami",
        "id_suami": "1. No.Kad Pengenalan/Pasport (suami)",
        "tarikh_lahir_suami": "1. Tarikh Lahir (suami)",
        "bangsa_suami": "1. Bangsa (suami)",
        "warganegara_suami": "1. Warganegara (suami)",
        "alamat_suami": "1. Alamat (Rumah) (suami, all lines)",
        "tarikh_masuk_islam_suami": "1. Tarikh masuk Islam (Jika mualaf) (suami)",
        "no_kad_perakuan_islam_suami": "1. No.kad perakuan Islam (suami)",
        "nama_isteri": "2. Nama Isteri",
        "id_isteri": "2. No.Kad Pengenalan/Pasport (isteri)",
        "tarikh_lahir_isteri": "2. Tarikh Lahir (isteri)",
        "bangsa_isteri": "2. Bangsa (isteri)",
        "warganegara_isteri": "2. Warganegara (isteri)",
        "alamat_isteri": "2. Alamat (Rumah) (isteri, all lines)",
        "tarikh_masuk_islam_isteri": "2. Tarikh masuk Islam (Jika mualaf) (isteri)",
        "no_kad_perakuan_islam_isteri": "2. No.kad perakuan Islam (isteri)",
        "nama_pendaftar_saksi": "B. PERAKUAN RUJUK: Saya ... (Nama Pendaftar)",
        "tarikh_rujuk_hijri": "C. Tarikh Rujuk - Hijrah",
        "tarikh_rujuk": "C. Tarikh Rujuk - Masihi",
        "rujuk_kali": "Rujuk kali ke",
        "tarikh_nikah_hijri": "Tarikh Nikah - Hijrah",
        "tarikh_nikah": "Tarikh Nikah - Masihi",
        "bil_daftar_nikah": "Bil. Daftar Nikah",
        "tarikh_cerai_hijri": "Tarikh Cerai - Hijrah",
        "tarikh_cerai": "Tarikh Cerai - Masihi",
        "bil_cerai": "Bil. Daftar Cerai",
        "hal_hal_lain": "Kenyataan lain (all lines of text)",
        "nama_pendaftar": "the registrar's name printed or stamped under the signature",
        "jawatan_pendaftar": "the registrar's position/title under their name",
    },
}

# Fields whose answer is shown ONLY by which printed option is left not
# crossed out (nothing is typed). Gemini can't see the thin strike-through
# reliably -- on real cerai_modern samples it confidently picked a crossed-out
# option even with a zoomed crop -- so these are never taken from Gemini: the
# printed options are stored for a reviewer to resolve (as the Vision reader's
# whole-line text effectively was) and the record is always flagged for review.
STRUCK_CHOICE_FIELDS: dict[str, dict[str, str]] = {
    "cerai_modern": {
        "keadaan_talak": "*Raj'ie/Bain(Bain Sughra/Bain Kubra/Bain Muabbad) [check which options are crossed out]",
        "tempat_cerai": "*Dengan Kebenaran Mahkamah/Tanpa Kebenaran Mahkamah [check which option is crossed out]",
    },
}


def review_fields(template_name: str) -> tuple[str, ...]:
    return tuple(STRUCK_CHOICE_FIELDS.get(template_name, {}))


_PROMPT = """
These are the {pages} page(s) of ONE scanned, typewritten Malaysian Islamic family-law certificate
("{title}"). For each key below, copy EXACTLY the typed value filled in for the printed label shown
(page number in brackets).

Rules:
- Copy the value as typed: same spelling, digits, punctuation and date style. Do not reformat,
  translate, correct or complete it. Where a value continues onto the next line, include it all.
- Never include the printed label itself, its numbering, or its dotted/dashed leader line.
- Where the printed label offers alternatives marked with "*" or separated by "/" (for example
  "*Raj'ie/Bain", "*Dengan Kebenaran Mahkamah/Tanpa Kebenaran Mahkamah", "*Suci/Haid/Mengandung"),
  the options that are CROSSED OUT do not apply: the option left NOT crossed out IS the value --
  return it even when nothing else is typed on that line (never null in that case). If a value is
  also typed after the label, return that typed value instead.
- Use null if the value is blank, "-", or only dots.
- Never move a value into a different key, and never repeat one value under two keys.

{fields}

Return only JSON.
""".strip()


def template_for(config_template: str) -> str:
    if config_template not in FIELD_LABELS:
        raise KeyError(f"no Gemini field labels for typed template {config_template!r}")
    return config_template


def build_request(template_name: str, record_type: str, page_count: int) -> dict[str, Any]:
    """The prompt text and generation config for one certificate -- shared by
    the live call below and gemini_page_batch.py's Batch Mode requests, so
    both send exactly the same request."""
    regions = TEMPLATES[template_name]["regions"]
    labels = FIELD_LABELS[template_for(template_name)]
    fields = "\n".join(f'- {key}: "{labels.get(key, key.replace("_", " "))}" (page {page})' for key, (page, _) in regions.items())
    prompt = _PROMPT.format(pages=page_count, title=_TITLES.get(record_type.lower(), "certificate"), fields=fields)
    return {
        "prompt": prompt,
        "generation_config": {
            "temperature": 0.0,
            "max_output_tokens": _MAX_OUTPUT_TOKENS,
            "response_mime_type": "application/json",
            "response_schema": {
                "type": "OBJECT",
                "properties": {key: {"type": "STRING", "nullable": True} for key in regions},
            },
        },
    }


def render_pages(pdf: Path, dpi: int = RENDER_DPI) -> list[bytes]:
    import pymupdf

    with pymupdf.open(pdf) as document:
        return [page.get_pixmap(dpi=dpi).tobytes("jpeg") for page in document]


def raw_fields_from_payload(payload: Mapping[str, Any], template_name: str) -> dict[str, RawField]:
    """Gemini's {region_key: text} answer as the RawFields typed/extractor.py
    would have produced -- confidence 1.0 for a filled region, 0.0 for an
    empty one (validate_record's word-confidence check then behaves as for a
    clean OCR read)."""
    regions = TEMPLATES[template_name]["regions"]
    struck = STRUCK_CHOICE_FIELDS.get(template_name, {})
    fields: dict[str, RawField] = {}
    for key, (page, region) in regions.items():
        if key in struck:
            fields[key] = RawField(
                key=key,
                output_name=key.replace("_", " ").title(),
                page_number=page,
                region=region,
                raw_text=struck[key],
                confidence=0.0,
            )
            continue
        value = payload.get(key)
        text = "" if value is None else str(value).strip()
        fields[key] = RawField(
            key=key,
            output_name=key.replace("_", " ").title(),
            page_number=page,
            region=region,
            raw_text=text,
            confidence=1.0 if text else 0.0,
        )
    return fields


class GeminiTypedReader:
    """Live (synchronous) Gemini read of one certificate. Returns None on any
    failure so the caller falls back to the Vision reader for that PDF."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        client: Any = None,
        *,
        attempts: int = 4,
        initial_delay_seconds: float = 2.0,
        sleep: Any = None,
    ) -> None:
        self.model = model
        self._client = client
        self._attempts = attempts
        self._initial_delay = initial_delay_seconds
        import time

        self._sleep = sleep or time.sleep

    def _get_client(self) -> Any:
        if self._client is None:
            from google import genai

            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            self._client = genai.Client(api_key=api_key) if api_key else genai.Client()
        return self._client

    def read(self, pdf: Path, template_name: str, record_type: str) -> dict[str, RawField] | None:
        try:
            from google.genai import types

            images = render_pages(pdf)
            request = build_request(template_name, record_type, len(images))
            contents = [request["prompt"], *(types.Part.from_bytes(data=img, mime_type="image/jpeg") for img in images)]
            config = types.GenerateContentConfig(**request["generation_config"])
            delay = self._initial_delay
            for attempt in range(1, self._attempts + 1):
                try:
                    response = self._get_client().models.generate_content(
                        model=self.model, contents=contents, config=config
                    )
                    break
                except Exception as error:
                    # Google's "high demand" 503s (and 429/500s) are routine and
                    # short-lived -- worth a few retries before giving up the
                    # more accurate reader for this PDF.
                    code = getattr(error, "code", None)
                    if attempt == self._attempts or code not in _TRANSIENT_CODES:
                        raise
                    self._sleep(delay)
                    delay *= 2
            return raw_fields_from_payload(json.loads(response.text), template_name)
        except Exception:
            LOGGER.warning("Gemini typed read failed for %s; falling back to Vision", pdf.name, exc_info=True)
            return None
