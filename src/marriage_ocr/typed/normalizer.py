from __future__ import annotations

from decimal import Decimal, InvalidOperation
import re
from typing import Iterable

from marriage_ocr.models import ExtractedRecord
from marriage_ocr.typed.models import RawField
from marriage_ocr.refinement.text_corrections import generate_date_candidates


BIL_PATTERN = re.compile(r"\b\d+\s*/\s*\d{4}\b")
DATE_PATTERN = re.compile(r"\b(\d{1,2})\s*([./-])\s*(\d{1,2})\s*\2\s*(\d{4})\b")
MAS_KAHWIN_PATTERN = re.compile(r"\bRM\s*([0-9][0-9\s,]*(?:\.[0-9]{1,2})?)", re.IGNORECASE)
_IC_DIGITS = re.compile(r"\D+")
_LEADING_LABELS = {
    "nama_suami": re.compile(r"^\s*(nama\s*:?\s*)", re.IGNORECASE),
    "nama_isteri": re.compile(r"^\s*(nama\s*:?\s*)", re.IGNORECASE),
    "nama_pendaftar": re.compile(r"^\s*(nama\s*:?\s*)", re.IGNORECASE),
    "nama_wali": re.compile(r"^\s*(nama\s*:?\s*)", re.IGNORECASE),
    "hubungan_wali": re.compile(r"^\s*(hubungan\s*:?\s*)", re.IGNORECASE),
    "saksi_1": re.compile(r"^\s*(saksi\s*1\s*:?\s*)", re.IGNORECASE),
    "saksi_2": re.compile(r"^\s*(saksi\s*2\s*:?\s*)", re.IGNORECASE),
    "tarikh_nikah": re.compile(r"^\s*(tarikh\s*nikah\s*:?\s*)", re.IGNORECASE),
    "alamat_pendaftar": re.compile(r"^\s*(alamat\s*:?\s*)", re.IGNORECASE),
    "mas_kahwin": re.compile(r"^\s*(mas\s*kahwin\s*:?\s*)", re.IGNORECASE),
}
_TRAILING_NOISE_PATTERN = re.compile(
    r"\b(?:NO\.?\s*SIRI|NO\.?\s*SIN|UMUR|BANGSA|WARGANEGARA|ALAMAT|PENDAFTAR|HUBUNGAN|SAKSI\s+PERTAMA|SAKSI\s+KEDUA)\b.*$",
    re.IGNORECASE,
)
_LOCATION_NOISE = (
    "DAERAH",
    "SELANGOR",
    "SUNGAI",
    "BESAR",
    "KAMPUNG",
    "PEJABAT",
    "AGAMA",
    "ISLAM",
    "WARGANEGARA",
    "BANGSA",
    "UMUR",
    "NO",
    "SIRI",
    "SIN",
    "TARIKH",
    "HIJRAH",
    "MASIHI",
    "PENDAFTAR",
)
_NAME_HINTS = {
    "nama_suami": ("BIN", "BINTI", "HAJI", "HJ", "TUAN", "USTAZ"),
    "nama_isteri": ("BIN", "BINTI", "HAJI", "HJ", "PUAN", "CIK"),
    "nama_wali": ("BIN", "BINTI", "HAJI", "HJ", "TUAN", "USTAZ"),
    "nama_pendaftar": ("BIN", "BINTI", "HAJI", "HJ", "TUAN", "USTAZ"),
    "saksi_1": ("BIN", "BINTI", "HAJI", "HJ"),
    "saksi_2": ("BIN", "BINTI", "HAJI", "HJ"),
}


def normalize_bil(raw: str | None) -> str | None:
    if raw is None:
        return None
    match = BIL_PATTERN.search(str(raw))
    if not match:
        return None
    return re.sub(r"\s*/\s*", "/", match.group(0)).strip()


_OLD_IC_LETTER_PATTERN = re.compile(r"(?:[A-Z]{1,3}/[A-Z]{1,3}|[A-Z])[-\s./]*\d{5,7}")


def normalize_ic(raw: str | None) -> tuple[str | None, str | None]:
    if raw is None:
        return (None, None)
    text = str(raw).upper()
    new_ic_patterns = (
        re.compile(r"\b\d{6}[\s./-]*\d{2}[\s./-]*\d{4}\b"),
        re.compile(r"\b\d{12}\b"),
    )
    for pattern in new_ic_patterns:
        match = pattern.search(text)
        if match:
            digits = re.sub(r"\D", "", match.group(0))
            if len(digits) == 12:
                return (None, digits)
    # Old-format ICs are letter-prefixed (e.g. "A1192345", "R/F119395"), never a
    # bare digit run -- without this, e.g. a wali's or witness's old IC on a
    # typed form was silently dropped entirely (not even the digits survived,
    # since \b\d{7,8}\b never matches when the letter is directly attached
    # with no separator, which is the normal old-IC format).
    letter_match = _OLD_IC_LETTER_PATTERN.search(text)
    if letter_match:
        return (re.sub(r"[\s.]", "", letter_match.group(0)), None)
    old_match = re.search(r"\b\d{7,8}\b", text)
    if old_match:
        return (old_match.group(0), None)
    return (None, None)


def normalize_age(raw: str | None, *, min_age: int, max_age: int) -> int | None:
    if raw is None:
        return None
    lines = [line.strip() for line in str(raw).splitlines() if line.strip()]
    source = lines[0] if lines else ""
    match = re.search(r"\b(\d{1,3})\b", source)
    if not match:
        return None
    age = int(match.group(1))
    if min_age <= age <= max_age:
        return age
    return None


def normalize_date_preserving_style(raw: str | None) -> str | None:
    if raw is None:
        return None

    for line in (part.strip() for part in str(raw).splitlines() if part.strip()):
        candidates = generate_date_candidates(line, field_name="date")
        if candidates:
            return candidates[0].value

    candidates = generate_date_candidates(str(raw).strip(), field_name="date")
    if candidates:
        return candidates[0].value

    return None


def normalize_mas_kahwin(raw: str | None) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    match = MAS_KAHWIN_PATTERN.search(text)
    if not match:
        return None
    numeric = match.group(1).replace(" ", "").replace(",", "")
    try:
        amount = Decimal(numeric)
    except InvalidOperation:
        return None
    if "." in numeric:
        decimals = len(numeric.split(".", 1)[1])
    else:
        decimals = 0
    quantize_pattern = {0: "0", 1: "0.0", 2: "0.00"}[min(decimals, 2)]
    normalized = format(amount.quantize(Decimal(quantize_pattern)), "f")
    if "." in normalized:
        integer_part, decimal_part = normalized.split(".", 1)
        integer_part = f"{int(integer_part):,}"
        decimal_part = decimal_part[: min(decimals, 2)]
        if decimals == 0:
            return f"RM {integer_part}"
        return f"RM {integer_part}.{decimal_part}"
    return f"RM {int(normalized):,}"


def _strip_label(value: str, field_key: str) -> str:
    pattern = _LEADING_LABELS.get(field_key)
    result = value
    if pattern is not None:
        result = pattern.sub("", result, count=1)
    result = result.replace(":", " ")
    result = re.sub(r"[ \t]+", " ", result)
    return result.strip()


def _strip_trailing_noise(value: str) -> str:
    return _TRAILING_NOISE_PATTERN.sub("", value).strip()


def _score_line_for_field(line: str, field_key: str | None) -> tuple[int, int, int]:
    upper = line.upper()
    words = re.findall(r"[A-Z@']+", upper)
    score = len(words)
    if any(char.isdigit() for char in line):
        score -= 5
    if any(marker in upper for marker in _LOCATION_NOISE):
        score -= 3
    if field_key in _NAME_HINTS:
        score += sum(2 for hint in _NAME_HINTS[field_key] if hint in upper)
    if field_key == "hubungan_wali":
        if any(token in upper for token in ("BAPA", "KANDUNG", "WALI", "HAKIM")):
            score += 4
    if field_key == "alamat_pendaftar":
        if any(token in upper for token in ("PEJABAT", "ALAMAT", "TEMPAT")):
            score += 3
        if any(char.isdigit() for char in line):
            score += 2
    return (score, len(words), -len(line))


def _select_best_line(lines: list[str], field_key: str | None) -> str | None:
    candidates = [line for line in lines if line]
    if not candidates:
        return None
    if field_key == "alamat_pendaftar":
        address_candidates = []
        for line in candidates:
            upper = line.upper()
            if any(marker in upper for marker in ("TARIKH", "HIJRAH", "MASIHI")):
                continue
            if any(token in upper for token in ("PEJABAT", "AGAMA", "ISLAM", "DAERAH", "SELANGOR", "SUNGAI", "KAMPUNG", "JALAN", "BESAR", "TAWAR", "LEMAN")) or any(char.isdigit() for char in line):
                address_candidates.append(line)
        if address_candidates:
            candidates = address_candidates
    else:
        no_digit = [line for line in candidates if not any(char.isdigit() for char in line)]
        if no_digit:
            candidates = no_digit
    return max(candidates, key=lambda line: _score_line_for_field(line, field_key))


def normalize_plain_text(raw: str | None, *, field_key: str | None = None) -> str | None:
    if raw is None:
        return None
    lines = [line.strip() for line in str(raw).splitlines() if line.strip()]
    if not lines:
        return None
    cleaned_lines = []
    for line in lines:
        line = _strip_label(line, field_key)
        line = _strip_trailing_noise(line)
        line = re.sub(r"[ \t]+", " ", line).strip(" ,")
        if line:
            cleaned_lines.append(line)
    value = _select_best_line(cleaned_lines, field_key)
    if value is None:
        return None
    value = re.sub(r"\s+", " ", value).strip(" ,")
    return value or None


def normalize_name(raw: str | None, *, field_key: str | None = None) -> str | None:
    return normalize_plain_text(raw, field_key=field_key)


def _first_non_empty(values: Iterable[str | None]) -> str | None:
    for value in values:
        if value:
            return value
    return None


def build_extracted_record(raw_fields: dict[str, RawField]) -> ExtractedRecord:
    bil = normalize_bil(raw_fields.get("bil").raw_text if raw_fields.get("bil") else None)

    nama_suami = normalize_name(raw_fields.get("nama_suami").raw_text if raw_fields.get("nama_suami") else None, field_key="nama_suami")
    ic_lama_suami, ic_baru_suami = normalize_ic(raw_fields.get("id_suami").raw_text if raw_fields.get("id_suami") else None)
    umur_suami = normalize_age(raw_fields.get("umur_suami").raw_text if raw_fields.get("umur_suami") else None, min_age=16, max_age=120)

    nama_isteri = normalize_name(raw_fields.get("nama_isteri").raw_text if raw_fields.get("nama_isteri") else None, field_key="nama_isteri")
    ic_lama_isteri, ic_baru_isteri = normalize_ic(raw_fields.get("id_isteri").raw_text if raw_fields.get("id_isteri") else None)
    umur_isteri = normalize_age(raw_fields.get("umur_isteri").raw_text if raw_fields.get("umur_isteri") else None, min_age=16, max_age=120)

    mas_kahwin = normalize_mas_kahwin(raw_fields.get("mas_kahwin").raw_text if raw_fields.get("mas_kahwin") else None)
    nama_pendaftar = normalize_plain_text(raw_fields.get("nama_pendaftar").raw_text if raw_fields.get("nama_pendaftar") else None, field_key="nama_pendaftar")
    alamat_pendaftar = normalize_plain_text(raw_fields.get("alamat_pendaftar").raw_text if raw_fields.get("alamat_pendaftar") else None, field_key="alamat_pendaftar")
    nama_wali = normalize_name(raw_fields.get("nama_wali").raw_text if raw_fields.get("nama_wali") else None, field_key="nama_wali")
    hubungan_wali = normalize_plain_text(raw_fields.get("hubungan_wali").raw_text if raw_fields.get("hubungan_wali") else None, field_key="hubungan_wali")
    saksi_1 = normalize_plain_text(raw_fields.get("saksi_1").raw_text if raw_fields.get("saksi_1") else None, field_key="saksi_1")
    saksi_2 = normalize_plain_text(raw_fields.get("saksi_2").raw_text if raw_fields.get("saksi_2") else None, field_key="saksi_2")
    tarikh_nikah = normalize_date_preserving_style(raw_fields.get("tarikh_nikah").raw_text if raw_fields.get("tarikh_nikah") else None)

    record = ExtractedRecord(
        bil=bil,
        nama_suami=nama_suami,
        ic_lama_suami=ic_lama_suami,
        ic_baru_suami=ic_baru_suami,
        id_suami_raw=raw_fields.get("id_suami").raw_text if raw_fields.get("id_suami") else None,
        umur_suami=umur_suami,
        nama_isteri=nama_isteri,
        ic_lama_isteri=ic_lama_isteri,
        ic_baru_isteri=ic_baru_isteri,
        id_isteri_raw=raw_fields.get("id_isteri").raw_text if raw_fields.get("id_isteri") else None,
        umur_isteri=umur_isteri,
        mas_kahwin=mas_kahwin,
        mas_kahwin_raw=raw_fields.get("mas_kahwin").raw_text if raw_fields.get("mas_kahwin") else None,
        nama_pendaftar=nama_pendaftar,
        alamat_pendaftar=alamat_pendaftar,
        nama_wali=nama_wali,
        hubungan_wali=hubungan_wali,
        saksi_1=saksi_1,
        saksi_2=saksi_2,
        tarikh_nikah=tarikh_nikah,
        tarikh_nikah_raw=raw_fields.get("tarikh_nikah").raw_text if raw_fields.get("tarikh_nikah") else None,
        tarikh_keluar=None,
        tarikh_keluar_raw=None,
        raw_bil=raw_fields.get("bil").raw_text if raw_fields.get("bil") else None,
        raw_suami_isteri=_first_non_empty(
            [
                raw_fields.get("nama_suami").raw_text if raw_fields.get("nama_suami") else None,
                raw_fields.get("id_suami").raw_text if raw_fields.get("id_suami") else None,
                raw_fields.get("nama_isteri").raw_text if raw_fields.get("nama_isteri") else None,
                raw_fields.get("id_isteri").raw_text if raw_fields.get("id_isteri") else None,
            ]
        ),
        raw_pendaftar=_first_non_empty(
            [
                raw_fields.get("nama_pendaftar").raw_text if raw_fields.get("nama_pendaftar") else None,
                raw_fields.get("alamat_pendaftar").raw_text if raw_fields.get("alamat_pendaftar") else None,
            ]
        ),
        raw_wali=raw_fields.get("nama_wali").raw_text if raw_fields.get("nama_wali") else None,
        raw_hubungan_wali=raw_fields.get("hubungan_wali").raw_text if raw_fields.get("hubungan_wali") else None,
        raw_saksi=_first_non_empty(
            [
                raw_fields.get("saksi_1").raw_text if raw_fields.get("saksi_1") else None,
                raw_fields.get("saksi_2").raw_text if raw_fields.get("saksi_2") else None,
            ]
        ),
        raw_tarikh_nikah=raw_fields.get("tarikh_nikah").raw_text if raw_fields.get("tarikh_nikah") else None,
        raw_tarikh_keluar=None,
        raw_remarks=None,
    )
    return record
