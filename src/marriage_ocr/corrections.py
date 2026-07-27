from __future__ import annotations

import re
from difflib import get_close_matches


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

COMMON_MALAY_TERMS = {
    "B1N": "BIN",
    "8IN": "BIN",
    "BJN": "BIN",
    "BINT1": "BINTI",
    "B1NTI": "BINTI",
    "8INTI": "BINTI",
    "BT.": "BT",
    "BTE.": "BTE",
    "HJ.": "HJ",
    "HJH.": "HJH",
    "MOHD.": "MOHD",
    "KG.": "KG",
}

DATE_CHAR_FIX = str.maketrans(
    {
        "O": "0",
        "o": "0",
        "I": "1",
        "l": "1",
        "|": "1",
        "S": "5",
        "B": "8",
    }
)

IC_BODY_CHAR_FIX = str.maketrans(
    {
        "O": "0",
        "o": "0",
        "Q": "0",
        "q": "0",
        "I": "1",
        "l": "1",
        "|": "1",
        "S": "5",
        "B": "8",
    }
)


def normalize_ocr_text(text: str | None) -> str:
    if not text:
        return ""

    value = str(text).upper()
    value = value.replace("\r", "\n")
    value = re.sub(r"[ \t]+", " ", value)

    for wrong, correct in COMMON_MALAY_TERMS.items():
        if wrong.endswith("."):
            pattern = rf"\b{re.escape(wrong[:-1])}\.(?=\W|$)"
        else:
            pattern = rf"\b{re.escape(wrong)}\b"
        value = re.sub(pattern, correct, value)

    return value.strip()


def normalize_numeric_ocr(text: str | None) -> str:
    value = normalize_ocr_text(text)
    return value.translate(DATE_CHAR_FIX)


def normalize_ic_ocr(text: str | None) -> str:
    value = normalize_ocr_text(text)
    if not value:
        return ""

    value = re.sub(r"[^A-Z0-9./\-\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()

    parts: list[str] = []
    for part in re.split(r"(\s+|[./-])", value):
        if not part or part.isspace() or part in {".", "/", "-"}:
            parts.append(part)
            continue

        if len(part) == 1 and part.isalpha():
            parts.append(part)
            continue

        if len(part) > 1 and part[0].isalpha() and any(character.isdigit() for character in part[1:]):
            parts.append(part[0] + part[1:].translate(IC_BODY_CHAR_FIX))
            continue

        parts.append(part.translate(IC_BODY_CHAR_FIX))

    return "".join(parts).strip()


def correct_relationship(text: str | None) -> str | None:
    value = normalize_ocr_text(text)
    if not value:
        return None

    value = re.sub(r"[^A-Z ]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()

    if value in RELATIONSHIP_VALUES:
        return value

    matches = get_close_matches(value, RELATIONSHIP_VALUES, n=1, cutoff=0.72)
    if matches:
        return matches[0]

    return None


def clean_name(text: str | None) -> str | None:
    value = normalize_ocr_text(text)
    if not value:
        return None

    value = re.sub(r"[^A-Z\s.'/()-]", " ", value)
    value = re.sub(r"\b(?:TAHUN|TAHN|TAHUH|THN|RM|NO|NIKAH|KELUAR)\b", " ", value)
    value = re.sub(r"\s+", " ", value).strip()

    return value or None


def clean_free_text(text: str | None) -> str | None:
    value = normalize_ocr_text(text)
    value = re.sub(r"\s+", " ", value).strip()
    return value or None
