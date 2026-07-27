from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
import re
from typing import Mapping, Sequence

from marriage_ocr.corrections import (
    clean_name,
    correct_relationship,
    normalize_ic_ocr,
    normalize_numeric_ocr,
    normalize_ocr_text,
)
from marriage_ocr.models import ExtractedRecord, OcrResult
from marriage_ocr.ocr import RecordOcrOutput


RELATIONSHIP_VALUES = [
    "BAPA KANDUNG",
    "SAUDARA LELAKI",
    "WALI HAKIM",
    "BAPA",
    "ABANG",
    "SAUDARA",
    "LELAKI",
    "DATUK",
]


NAME_STOPWORDS = {
    "TAHUN",
    "THN",
    "RM",
    "NO",
    "NIKAH",
    "KELUAR",
}


DATE_PATTERN = re.compile(r"\b(\d{1,2})\s*[-./:,;]\s*(\d{1,2})\s*[-./:,;]\s*(\d{2,4})\b")
# OCR sometimes collapses 27.8.94 into 27.894; keep this as a fallback only.
COMPACT_DATE_PATTERN = re.compile(r"\b(\d{1,2})\s*[-./:,;]\s*(\d)(\d{2})\b")
AGE_PATTERN = re.compile(r"(?:^|[\s/\-])([1-9]\d{1,2})\s*(?:TAHUN|TAHUH|TAHN|THN|THN\.|TAHUN\.)\b")
NEW_IC_PATTERN = re.compile(r"\b(\d{6})[-\s./]?(\d{2})[-\s./]?(\d{4})\b")
OLD_IC_PATTERN = re.compile(r"\b([A-Z])[-\s./]?(\d{5,8})\b")
LEGACY_NUMERIC_IC_PATTERN = re.compile(r"\b(\d{5,8})\b")
DETAIL_HINT_PATTERN = re.compile(r"(?:\b[ARWGS][-\s./]?\d{5,8}\b|\b\d{5,8}\b|\b\d{6}[-\s./]?\d{2}[-\s./]?\d{4}\b|\b\d{2,3}\s*(?:TAHUN|TAHUH|TAHN|THN)\b)")
NUMBERING_PREFIX_PATTERN = re.compile(r"^\s*(?:[\[(]?\d+[\])\-.]?|[①②③④⑤⑥⑦⑧⑨])\s*")
NUMBERING_SPLIT_PATTERN = re.compile(r"(?:^|\n|\s)(?:[\[(]?[12][\])\-.]|[①②])\s*")
BIL_WITH_YEAR_PATTERN = re.compile(r"\b(\d{3,5})\s*[/\\|IL]\s*(\d{2,4})\b")
BIL_NUMBER_PATTERN = re.compile(r"\b(\d{1,5})\b")
NON_NAME_PATTERN = re.compile(r"[^A-Z\s.'/()-]")


@dataclass(frozen=True)
class ParsedMoney:
    normalized: str | None
    raw: str | None
    needs_review: bool


@dataclass(frozen=True)
class ParsedDate:
    normalized: str | None
    raw: str | None
    needs_review: bool


@dataclass(frozen=True)
class ParsedIdentifiers:
    ic_lama: str | None
    ic_baru: str | None
    raw: str | None


@dataclass(frozen=True)
class ParsedPendaftar:
    nama: str | None
    alamat: str | None
    issues: list[str]


@dataclass(frozen=True)
class ParsedWali:
    nama: str | None
    hubungan: str | None
    issues: list[str]


@dataclass(frozen=True)
class ParsedSaksi:
    saksi_1: str | None
    saksi_2: str | None
    issues: list[str]


def parse_bil(text: str) -> str | None:
    """Parse the BIL field and keep the register year suffix when OCR sees it.

    Examples: 460/94, 460|94, 46O/94 -> 460/94.
    """
    normalized = _normalize_text(text)
    candidate = (
        normalized.replace("O", "0")
        .replace("Q", "0")
        .replace("S", "5")
        .replace("L", "1")
        .replace("I", "1")
    )

    match = BIL_WITH_YEAR_PATTERN.search(candidate)
    if match is not None:
        number = _clean_bil_number(match.group(1))
        year = match.group(2)[-2:]
        if number is not None:
            return f"{number}/{year}"

    match = BIL_NUMBER_PATTERN.search(candidate)
    if match is not None:
        return _clean_bil_number(match.group(1))

    return None


def _clean_bil_number(value: str) -> str | None:
    digits = re.sub(r"\D", "", value)
    if len(digits) < 1:
        return None
    # OCR sometimes glues a trailing year-like digit; keep the most plausible
    # 3-digit ledger number for 1990s registers such as 460/94.
    if len(digits) > 3 and digits.startswith("4"):
        digits = digits[:3]
    return str(int(digits))


def parse_money(text: str) -> ParsedMoney:
    lines = _meaningful_lines(text)
    for line in reversed(lines):
        normalized = _normalize_money_from_line(line)
        if normalized is not None:
            return ParsedMoney(
                normalized=normalized,
                raw=line,
                needs_review="RM" not in _normalize_text(line),
            )
    return ParsedMoney(normalized=None, raw=None, needs_review=False)


def parse_ages(text: str) -> list[int]:
    ages: list[int] = []
    for match in AGE_PATTERN.finditer(_normalize_for_numeric_ocr(text)):
        value = int(match.group(1))
        if 15 <= value <= 100:
            ages.append(value)
    return ages


def parse_identifiers(text: str) -> ParsedIdentifiers:
    normalized = _normalize_for_ic_ocr(text)
    old_ic_match = OLD_IC_PATTERN.search(normalized)
    new_ic_match = NEW_IC_PATTERN.search(normalized)

    old_ic = None
    legacy_numeric_match = None
    if old_ic_match is not None:
        old_ic = f"{old_ic_match.group(1)}.{old_ic_match.group(2)}"
    else:
        legacy_numeric_match = LEGACY_NUMERIC_IC_PATTERN.search(normalized)
        if legacy_numeric_match is not None and NEW_IC_PATTERN.search(normalized) is None:
            old_ic = legacy_numeric_match.group(1)

    new_ic = None
    if new_ic_match is not None:
        new_ic = f"{new_ic_match.group(1)}-{new_ic_match.group(2)}-{new_ic_match.group(3)}"

    raw_parts = []
    if old_ic_match is not None:
        raw_parts.append(old_ic_match.group(0).strip())
    elif legacy_numeric_match is not None:
        raw_parts.append(legacy_numeric_match.group(0).strip())
    if new_ic_match is not None:
        raw_parts.append(new_ic_match.group(0).strip())

    return ParsedIdentifiers(
        ic_lama=old_ic,
        ic_baru=new_ic,
        raw=" / ".join(raw_parts) or None,
    )


def parse_date(text: str) -> ParsedDate:
    normalized = _normalize_for_numeric_ocr(text)
    for pattern in (DATE_PATTERN, COMPACT_DATE_PATTERN):
        for match in pattern.finditer(normalized):
            raw_value = match.group(0)
            parsed = _normalize_date_parts(match.group(1), match.group(2), match.group(3))
            if parsed is not None:
                return ParsedDate(normalized=parsed, raw=raw_value, needs_review=False)
    meaningful = " ".join(_meaningful_lines(text))
    return ParsedDate(
        normalized=None,
        raw=meaningful or None,
        needs_review=bool(meaningful),
    )


def parse_spouse_cell(text: str) -> tuple[dict[str, str | int | None], list[str]]:
    lines = _meaningful_lines(text)
    if not lines:
        return {}, ["spouse_cell_empty"]

    money = parse_money(text)
    filtered_lines = [line for line in lines if line != money.raw]
    people = _extract_spouse_people(filtered_lines)
    issues: list[str] = []

    if len(people) < 2:
        issues.append("spouse_names_incomplete")

    husband = people[0] if len(people) >= 1 else _empty_person()
    wife = people[1] if len(people) >= 2 else _empty_person()

    parsed = {
        "nama_suami": husband["name"],
        "ic_lama_suami": husband["ids"].ic_lama,
        "ic_baru_suami": husband["ids"].ic_baru,
        "id_suami_raw": husband["ids"].raw,
        "umur_suami": husband["age"],
        "nama_isteri": wife["name"],
        "ic_lama_isteri": wife["ids"].ic_lama,
        "ic_baru_isteri": wife["ids"].ic_baru,
        "id_isteri_raw": wife["ids"].raw,
        "umur_isteri": wife["age"],
        "mas_kahwin": money.normalized,
        "mas_kahwin_raw": money.raw,
    }

    if money.normalized and money.needs_review:
        issues.append("mas_kahwin_missing_rm")
    if parsed["umur_suami"] is None:
        issues.append("umur_suami_missing")
    if parsed["umur_isteri"] is None:
        issues.append("umur_isteri_missing")
    if parsed["ic_lama_suami"] is None and parsed["ic_baru_suami"] is None:
        issues.append("id_suami_missing")
    if parsed["ic_lama_isteri"] is None and parsed["ic_baru_isteri"] is None:
        issues.append("id_isteri_missing")
    if parsed["nama_suami"] is not None and _looks_name_suspicious(str(parsed["nama_suami"])):
        issues.append("nama_suami_suspicious")
    if parsed["nama_isteri"] is not None and _looks_name_suspicious(str(parsed["nama_isteri"])):
        issues.append("nama_isteri_suspicious")

    return parsed, issues

def parse_pendaftar_cell(text: str) -> ParsedPendaftar:
    lines = _meaningful_lines(text)
    if not lines:
        return ParsedPendaftar(nama=None, alamat=None, issues=["pendaftar_empty"])

    corrected_lines = [_fix_common_malay_ocr(line) for line in lines]
    nama = clean_name(corrected_lines[0])
    alamat_lines = [_normalize_free_text(line) for line in corrected_lines[1:]]
    issues: list[str] = []
    if not alamat_lines:
        issues.append("pendaftar_address_missing")

    return ParsedPendaftar(
        nama=nama,
        alamat="\n".join(line for line in alamat_lines if line) or None,
        issues=issues,
    )

def parse_wali_cells(wali_text: str, hubungan_text: str) -> ParsedWali:
    wali_text = normalize_ocr_text(wali_text)
    hubungan_text = normalize_ocr_text(hubungan_text)
    wali_lines = _meaningful_lines(wali_text)
    hubungan_lines = _meaningful_lines(hubungan_text)
    issues: list[str] = []

    combined = " ".join(wali_lines + hubungan_lines)
    nama, hubungan = _split_wali_name_relationship(combined)
    hubungan = correct_relationship(hubungan)

    if hubungan is None:
        issues.append("wali_relationship_missing")
    if nama is None:
        issues.append("wali_name_missing")

    return ParsedWali(nama=clean_name(nama), hubungan=hubungan, issues=issues)


def parse_saksi_cell(text: str) -> ParsedSaksi:
    normalized_text = _normalize_number_markers(str(text))
    raw_lines = _meaningful_lines(normalized_text)
    issues: list[str] = []

    parts = _split_numbered_people(normalized_text)
    if len(parts) < 2:
        cleaned_lines = [_strip_numbering(line) for line in raw_lines]
        cleaned_lines = [line for line in cleaned_lines if line]
        midpoint = max(1, len(cleaned_lines) // 2)
        parts = [" ".join(cleaned_lines[:midpoint]), " ".join(cleaned_lines[midpoint:])] if len(cleaned_lines) > 2 else cleaned_lines

    names = [clean_name(part) for part in parts]
    names = [name for name in names if name]

    saksi_1 = names[0] if len(names) >= 1 else None
    saksi_2 = names[1] if len(names) >= 2 else None
    if saksi_1 is None:
        issues.append("saksi_missing")
    if saksi_1 is not None and saksi_2 is None:
        issues.append("saksi_2_missing")

    return ParsedSaksi(saksi_1=saksi_1, saksi_2=saksi_2, issues=issues)

def parse_record_ocr(
    cell_results: Mapping[str, OcrResult],
    *,
    source_record: str | None = None,
    crop_folder: str | None = None,
    raw_ocr_json: str | None = None,
) -> ExtractedRecord:
    spouse_text = _ocr_text(cell_results.get("suami_isteri"))
    pendaftar_text = _ocr_text(cell_results.get("pendaftar"))
    wali_text = _ocr_text(cell_results.get("wali"))
    hubungan_text = _ocr_text(cell_results.get("hubungan_wali"))
    saksi_text = _ocr_text(cell_results.get("saksi"))
    nikah_text = _ocr_text(cell_results.get("tarikh_nikah"))
    keluar_text = _ocr_text(cell_results.get("tarikh_keluar"))
    bil_text = _ocr_text(cell_results.get("bil"))
    remarks_text = _ocr_text(cell_results.get("remarks"))

    spouse_fields, spouse_issues = parse_spouse_cell(spouse_text)
    pendaftar = parse_pendaftar_cell(pendaftar_text)
    wali = parse_wali_cells(wali_text, hubungan_text)
    saksi = parse_saksi_cell(saksi_text)
    nikah = parse_date(nikah_text)
    keluar = parse_date(keluar_text)

    bil = parse_bil(bil_text)

    record = ExtractedRecord(
        bil=bil,
        source_record=source_record,
        crop_folder=crop_folder,
        raw_ocr_json=raw_ocr_json,
        raw_bil=bil_text or None,
        raw_suami_isteri=spouse_text or None,
        raw_pendaftar=pendaftar_text or None,
        raw_wali=wali_text or None,
        raw_hubungan_wali=hubungan_text or None,
        raw_saksi=saksi_text or None,
        raw_tarikh_nikah=nikah_text or None,
        raw_tarikh_keluar=keluar_text or None,
        raw_remarks=remarks_text or None,
        nama_pendaftar=pendaftar.nama,
        alamat_pendaftar=pendaftar.alamat,
        nama_wali=wali.nama,
        hubungan_wali=wali.hubungan,
        saksi_1=saksi.saksi_1,
        saksi_2=saksi.saksi_2,
        tarikh_nikah=nikah.normalized,
        tarikh_nikah_raw=nikah.raw,
        tarikh_keluar=keluar.normalized,
        tarikh_keluar_raw=keluar.raw,
        remarks=_normalize_free_text(remarks_text) or None,
        **spouse_fields,
    )

    issues = spouse_issues + pendaftar.issues + wali.issues + saksi.issues
    if nikah.needs_review:
        issues.append("tarikh_nikah_unparsed")
    if keluar.needs_review:
        issues.append("tarikh_keluar_unparsed")

    record.review_reason = issues
    record.status_review = "REVIEW" if issues else "OK"
    return record


def parse_record_ocr_output(
    record_output: RecordOcrOutput,
    *,
    include_crop_folder: bool = True,
) -> ExtractedRecord:
    raw_json_text = None
    if record_output.raw_json_path is not None and record_output.raw_json_path.exists():
        raw_json_text = record_output.raw_json_path.read_text(encoding="utf-8")

    return parse_record_ocr(
        record_output.cell_results,
        source_record=f"record_{record_output.record_index:03d}",
        crop_folder=str(record_output.record_dir) if include_crop_folder else None,
        raw_ocr_json=raw_json_text,
    )


def save_parsed_record(record: ExtractedRecord, path: str | Path) -> None:
    output_path = Path(path)
    output_path.write_text(json.dumps(record.to_dict(), indent=2, ensure_ascii=True), encoding="utf-8")


def _empty_person() -> dict[str, object | None]:
    return {"name": None, "ids": ParsedIdentifiers(None, None, None), "age": None, "detail": ""}


def _extract_spouse_people(lines: Sequence[str]) -> list[dict[str, object | None]]:
    lines = [line for line in lines if not _is_ocr_placeholder(line)]
    """Parse spouse cell in strict ledger order.

    Expected handwritten sequence is:
    husband name -> husband IC -> husband age -> wife name -> wife IC -> wife age -> mas kahwin.
    A name may span multiple OCR lines, so a second non-numeric line after the first name
    stays with the same spouse until an IC/age detail is seen.
    """
    people: list[dict[str, object | None]] = []
    index = 0
    while index < len(lines) and len(people) < 2:
        name_parts: list[str] = []
        detail_parts: list[str] = []

        while index < len(lines):
            line = lines[index]
            split = _split_name_detail_line(line)
            if split["name"] and not name_parts:
                name_parts.append(str(split["name"]))
                if split["detail"]:
                    detail_parts.append(str(split["detail"]))
                    index += 1
                    break
                index += 1
                continue
            if _is_detail_line(line):
                break
            name_parts.append(line)
            index += 1

        while index < len(lines):
            line = lines[index]
            if _is_detail_line(line):
                detail_parts.append(line)
                index += 1
                continue
            break

        name = _normalize_name(_fix_common_malay_ocr(" ".join(name_parts)))
        detail = "\n".join(detail_parts)
        ids = parse_identifiers(detail)
        ages = parse_ages(detail)
        people.append({"name": name, "ids": ids, "age": ages[0] if ages else None, "detail": detail})

        # If no progress was made because the OCR lines are very noisy, advance one line.
        if not name_parts and not detail_parts:
            index += 1

    return people


def _is_detail_line(line: str) -> bool:
    return DETAIL_HINT_PATTERN.search(_normalize_for_numeric_ocr(line)) is not None


def _is_ocr_placeholder(line: str) -> bool:
    normalized = _normalize_text(line)
    return "MOCK_OCR" in normalized or ("[" in normalized and "]" in normalized and ":" in normalized)


def _split_name_detail_line(line: str) -> dict[str, str | None]:
    normalized = _normalize_for_numeric_ocr(line)
    matches = [match for match in DETAIL_HINT_PATTERN.finditer(normalized)]
    if not matches:
        return {"name": None, "detail": None}
    marker = min(match.start() for match in matches)
    name = _normalize_free_text(normalized[:marker])
    detail = _normalize_free_text(normalized[marker:])
    return {"name": name or None, "detail": detail or None}


def _split_wali_name_relationship(text: str) -> tuple[str | None, str | None]:
    normalized = normalize_ocr_text(text)
    relation = None
    parenthetical = re.search(r"\(([^)]{2,})\)", normalized)
    if parenthetical is not None:
        relation = correct_relationship(parenthetical.group(1))

    name_text = re.sub(r"\([^)]*\)", " ", normalized)
    name_text = re.sub(r"\s+", " ", name_text).strip()

    words = name_text.split()
    relation_word_count = 0
    if relation is None:
        for size in range(min(4, len(words)), 0, -1):
            candidate = " ".join(words[-size:])
            candidate_relation = correct_relationship(candidate)
            if candidate_relation is not None:
                relation = candidate_relation
                relation_word_count = size
                break

    if relation_word_count:
        words = words[:-relation_word_count]

    name_text = " ".join(words)
    name_text = re.sub(r"\bC\s*(?=BAPA|LELAKI|SAUDARA)\b", " ", name_text)
    nama = clean_name(name_text)
    return nama, relation


def _normalize_number_markers(text: str) -> str:
    replacements = {
        "①": "\n1) ",
        "②": "\n2) ",
        "③": "\n3) ",
        "④": "\n4) ",
        "❶": "\n1) ",
        "❷": "\n2) ",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def _split_numbered_people(text: str) -> list[str]:
    normalized = _normalize_number_markers(text)
    matches = list(NUMBERING_SPLIT_PATTERN.finditer(normalized))
    if len(matches) < 2:
        return []
    parts: list[str] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(normalized)
        value = _normalize_free_text(normalized[start:end])
        if value:
            parts.append(value)
    return parts


def _normalize_for_numeric_ocr(text: str) -> str:
    return normalize_numeric_ocr(text)


def _normalize_for_ic_ocr(text: str) -> str:
    return normalize_ic_ocr(text)


def _fix_common_malay_ocr(text: str) -> str:
    normalized = _normalize_free_text(text)
    replacements = [
        (r"\b(?:ITJ|LTJ|IJT|H1|HI)\s*\.?", "HJ."),
        (r"\bBIR\b", "BIN"),
        (r"\bBJN\b", "BIN"),
        (r"\bB1N\b", "BIN"),
        (r"\bBINT!\b", "BINTI"),
        (r"\bPENDAFTAPAN\b", "PENDAFTARAN"),
        (r"\bPENOAFTARAN\b", "PENDAFTARAN"),
        (r"\bNIKAHH\b", "NIKAH"),
        (r"\bMASJIO\b", "MASJID"),
        (r"\bMASJ1D\b", "MASJID"),
        (r"\bJAMEK\b", "JAMEK"),
        (r"\bKAWASAH\b", "KAWASAN"),
        (r"\bKAWASAM\b", "KAWASAN"),
        (r"\bISLAN\b", "ISLAM"),
        (r"\bHAMDAN\b", "HAMIDAN"),
    ]
    for pattern, replacement in replacements:
        normalized = re.sub(pattern, replacement, normalized)
    return _normalize_free_text(normalized)


def _normalize_money_from_line(line: str) -> str | None:
    upper = _normalize_text(line)
    candidate = upper.replace("O", "0")
    candidate = re.sub(r"(?<=\d)[IL](?=\d)", "1", candidate)
    groups = re.findall(r"\d+", candidate)
    if not groups:
        return None

    if "RM" not in candidate and len(groups[-1]) < 2 and len(groups) == 1:
        return None

    if len(groups) >= 2 and len(groups[-1]) <= 2:
        integer_part = "".join(groups[:-1]) or "0"
        fractional_part = groups[-1].ljust(2, "0")[:2]
    elif len(groups) == 1 and len(groups[0]) >= 3:
        integer_part = groups[0][:-2]
        fractional_part = groups[0][-2:]
    elif len(groups) == 1:
        integer_part = groups[0]
        fractional_part = "00"
    else:
        return None

    normalized_integer = str(int(integer_part))
    return f"RM {normalized_integer}.{fractional_part}"


def _normalize_date_parts(day_text: str, month_text: str, year_text: str) -> str | None:
    day_value = int(day_text)
    month_value = int(month_text)
    year_value = int(year_text)

    if len(year_text) == 2:
        year_value = 2000 + year_value if year_value <= 30 else 1900 + year_value

    try:
        parsed = date(year_value, month_value, day_value)
    except ValueError:
        return None

    return parsed.strftime("%d-%m-%Y")


def _extract_person_segment(
    lines: Sequence[str],
    name_indices: Sequence[int],
    segment_index: int,
) -> tuple[str | None, str]:
    if segment_index >= len(name_indices):
        return None, ""

    name_index = name_indices[segment_index]
    next_index = name_indices[segment_index + 1] if segment_index + 1 < len(name_indices) else len(lines)
    name = _normalize_name(lines[name_index])
    detail_lines = lines[name_index + 1 : next_index]
    return name, "\n".join(detail_lines)


def _extract_relationship(text: str) -> str | None:
    normalized = _normalize_text(text)
    parenthetical = re.search(r"\(([^)]{2,})\)", normalized)
    if parenthetical is not None:
        relation = _normalize_free_text(parenthetical.group(1))
        relation = re.sub(r"^C\s*(?=BAPA|LELAKI|SAUDARA)", "", relation).strip()
        return correct_relationship(relation)

    for value in RELATIONSHIP_VALUES:
        if value in normalized:
            return value

    candidate = re.sub(r"[^A-Z ]", " ", normalized)
    candidate = re.sub(r"\s+", " ", candidate).strip()
    if candidate:
        return correct_relationship(candidate)
    return None


def _is_probable_name_line(line: str) -> bool:
    normalized = _normalize_text(line)
    if not normalized:
        return False
    if AGE_PATTERN.search(normalized) or NEW_IC_PATTERN.search(normalized) or OLD_IC_PATTERN.search(normalized):
        return False
    if "RM" in normalized:
        return False
    if sum(character.isdigit() for character in normalized) > 2:
        return False
    tokens = [token for token in normalized.split() if token]
    if not tokens:
        return False
    if len(tokens) == 1 and tokens[0] in NAME_STOPWORDS:
        return False
    return any(token not in NAME_STOPWORDS for token in tokens)


def _looks_name_suspicious(name: str) -> bool:
    normalized = _normalize_text(name)
    if not normalized:
        return True
    return bool(NON_NAME_PATTERN.search(normalized))


def _normalize_name(text: str) -> str | None:
    normalized = _normalize_free_text(text)
    return normalized or None


def _normalize_free_text(text: str) -> str:
    return " ".join(_normalize_text(text).split())


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).replace("\r", "\n").replace("\t", " ")).strip().upper()


def _meaningful_lines(text: str) -> list[str]:
    return [_normalize_free_text(line) for line in str(text).splitlines() if _normalize_free_text(line)]


def _ocr_text(result: OcrResult | None) -> str:
    return result.text if result is not None else ""


def _strip_numbering(text: str) -> str:
    return NUMBERING_PREFIX_PATTERN.sub("", text).strip()
