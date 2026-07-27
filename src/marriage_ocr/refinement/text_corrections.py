from __future__ import annotations

from datetime import date
import re

from marriage_ocr.refinement.models import FieldCandidate


NAME_ALLOWED_CHARS_PATTERN = re.compile(r"[^A-Z0-9\s.'/()-]")
NAME_TOKEN_CORRECTIONS = {
    "B1N": ("BIN",),
    "BJN": ("BIN",),
    "BLN": ("BIN",),
    "BINT1": ("BINTI",),
    "B1NTI": ("BINTI",),
    "M0HD": ("MOHD",),
    "MOHO": ("MOHD",),
    "A8D": ("ABD",),
    "HI.": ("HJ.",),
}
NUMERIC_OCR_SUBSTITUTIONS = {
    "O": "0",
    "Q": "0",
    "I": "1",
    "L": "1",
    "|": "1",
    "Z": "2",
    "S": "5",
    "G": "6",
    "B": "8",
}
LEGACY_IC_PATTERN = re.compile(r"^(?P<prefix>[A-Z])?(?P<body>[A-Z0-9|]{5,8})(?P<suffix>[A-Z])?$")
MODERN_IC_ANALYSIS_PATTERN = re.compile(r"^[A-Z0-9|]{12}$")


def generate_name_candidates(raw: str | None, *, field_name: str) -> list[FieldCandidate]:
    original_value = raw
    prepared = _prepare_name(raw)
    if not prepared:
        return []

    candidates: list[FieldCandidate] = []
    _append_candidate(
        candidates,
        value=prepared,
        field_name=field_name,
        source="safe_normalisation",
        substitutions=0,
        original_value=original_value,
        requires_retry_ocr=False,
        requires_review=False,
    )

    corrected_tokens: list[str] = []
    substitutions = 0
    changed = False
    for token in prepared.split():
        corrected = _correct_name_token(token)
        corrected_tokens.append(corrected)
        if corrected != token:
            substitutions += 1
            changed = True

    if changed:
        corrected_value = " ".join(corrected_tokens)
        _append_candidate(
            candidates,
            value=corrected_value,
            field_name=field_name,
            source="typo_rule",
            substitutions=substitutions,
            original_value=original_value,
            requires_retry_ocr=True,
            requires_review=True,
        )

    return candidates


def generate_ic_candidates(raw: str | None, *, field_name: str) -> list[FieldCandidate]:
    original_value = raw
    prepared = _prepare_ic_for_analysis(raw)
    if not prepared:
        return []

    candidates: list[FieldCandidate] = []
    modern_candidate = _generate_modern_ic_candidate(prepared, field_name=field_name, original_value=original_value)
    if modern_candidate is not None:
        candidates.append(modern_candidate)

    legacy_candidate = _generate_legacy_ic_candidate(prepared, field_name=field_name, original_value=original_value)
    if legacy_candidate is not None:
        candidates.append(legacy_candidate)

    return sorted(
        candidates,
        key=lambda candidate: (-candidate.validity_score, candidate.substitutions, candidate.value),
    )


def generate_date_candidates(raw: str | None, *, field_name: str) -> list[FieldCandidate]:
    original_value = raw
    substitution_count = _count_date_substitutions(raw or "")
    prepared = _prepare_date_for_analysis(raw)
    if not prepared:
        return []

    candidates: list[FieldCandidate] = []
    seen: set[str] = set()
    for day_text, month_text, year_text in _date_parts_from_text(prepared):
        normalized = _normalize_date_parts(day_text, month_text, year_text)
        if normalized is None or normalized in seen:
            continue
        seen.add(normalized)
        _append_candidate(
            candidates,
            value=normalized,
            field_name=field_name,
            source="typo_rule" if substitution_count else "safe_normalisation",
            substitutions=substitution_count,
            original_value=original_value,
            requires_retry_ocr=substitution_count > 0,
            requires_review=False,
        )

    return sorted(
        candidates,
        key=lambda candidate: (-candidate.validity_score, candidate.substitutions, candidate.value),
    )


def is_valid_malaysian_ic(value: str | None) -> bool:
    if not value:
        return False

    compact = re.sub(r"[^A-Z0-9]", "", str(value).upper())
    if len(compact) == 12 and compact.isdigit():
        return _is_valid_modern_ic_digits(compact)

    if re.fullmatch(r"[A-Z]\.\d{5,8}[A-Z]?", str(value).upper()):
        return True

    if re.fullmatch(r"\d{5,8}", str(value).upper()):
        return True

    return False


def is_valid_date(value: str | None) -> bool:
    if not value:
        return False

    text = str(value).strip()
    if not text:
        return False

    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        year_text, month_text, day_text = text.split("-")
        return _normalize_date_parts(day_text, month_text, year_text) is not None

    if re.fullmatch(r"\d{2}-\d{2}-\d{4}", text):
        day_text, month_text, year_text = text.split("-")
        return _normalize_date_parts(day_text, month_text, year_text) is not None

    return bool(generate_date_candidates(text, field_name="date"))


def is_suspicious_name(value: str | None) -> bool:
    if not value:
        return True

    prepared = _prepare_name(value)
    if not prepared:
        return True

    if any(character.isdigit() for character in prepared):
        return True

    return bool(NAME_ALLOWED_CHARS_PATTERN.search(prepared))


def _append_candidate(
    candidates: list[FieldCandidate],
    *,
    value: str,
    field_name: str,
    source: str,
    substitutions: int,
    original_value: str | None,
    requires_retry_ocr: bool,
    requires_review: bool,
) -> None:
    if any(candidate.value == value for candidate in candidates):
        return

    candidates.append(
        FieldCandidate(
            value=value,
            source=source,
            validity_score=max(0.0, 1.0 - substitutions * 0.05),
            ocr_confidence=None,
            plausibility_score=max(0.0, 0.98 - substitutions * 0.04),
            similarity_score=max(0.0, 0.99 - substitutions * 0.05),
            substitutions=substitutions,
            metadata={
                "field_name": field_name,
                "original_value": original_value,
                "requires_retry_ocr": requires_retry_ocr,
                "requires_review": requires_review,
            },
        )
    )


def _prepare_name(raw: str | None) -> str:
    if raw is None:
        return ""

    value = str(raw).upper().replace("\r", "\n")
    value = re.sub(r"\s+", " ", value.replace("\n", " ")).strip()
    value = NAME_ALLOWED_CHARS_PATTERN.sub(" ", value)
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"^[\s'\",;:!?-]+|[\s'\",;:!?-]+$", "", value)
    return re.sub(r"\s+", " ", value).strip()


def _correct_name_token(token: str) -> str:
    token_key = token.upper()
    corrections = NAME_TOKEN_CORRECTIONS.get(token_key)
    if corrections:
        return corrections[0]
    return token


def _prepare_ic_for_analysis(raw: str | None) -> str:
    if raw is None:
        return ""
    value = str(raw).upper()
    value = re.sub(r"[^A-Z0-9|./\-\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _generate_modern_ic_candidate(
    prepared: str,
    *,
    field_name: str,
    original_value: str | None,
) -> FieldCandidate | None:
    compact = re.sub(r"[^A-Z0-9|]", "", prepared)
    if not MODERN_IC_ANALYSIS_PATTERN.fullmatch(compact):
        return None

    normalized_digits, substitutions = _replace_numeric_ambiguities(compact)
    if not normalized_digits.isdigit() or not _is_valid_modern_ic_digits(normalized_digits):
        return None

    return FieldCandidate(
        value=f"{normalized_digits[:6]}-{normalized_digits[6:8]}-{normalized_digits[8:]}",
        source="typo_rule" if substitutions else "safe_normalisation",
        validity_score=max(0.0, 1.0 - substitutions * 0.05),
        ocr_confidence=None,
        plausibility_score=max(0.0, 0.99 - substitutions * 0.03),
        similarity_score=max(0.0, 0.99 - substitutions * 0.05),
        substitutions=substitutions,
        metadata={
            "field_name": field_name,
            "original_value": original_value,
            "requires_retry_ocr": False,
            "requires_review": False,
        },
    )


def _generate_legacy_ic_candidate(
    prepared: str,
    *,
    field_name: str,
    original_value: str | None,
) -> FieldCandidate | None:
    compact = re.sub(r"[^A-Z0-9|]", "", prepared)
    if len(compact) == 12:
        return None

    match = LEGACY_IC_PATTERN.fullmatch(compact)
    if match is None:
        return None

    prefix = match.group("prefix") or ""
    body = match.group("body")
    suffix = match.group("suffix") or ""
    normalized_body, substitutions = _replace_numeric_ambiguities(body)
    if not normalized_body.isdigit():
        return None

    formatted = normalized_body
    if prefix:
        formatted = f"{prefix}.{formatted}"
    if suffix:
        formatted = f"{formatted}{suffix}"

    return FieldCandidate(
        value=formatted,
        source="typo_rule" if substitutions else "safe_normalisation",
        validity_score=max(0.0, 0.98 - substitutions * 0.05),
        ocr_confidence=None,
        plausibility_score=max(0.0, 0.97 - substitutions * 0.03),
        similarity_score=max(0.0, 0.99 - substitutions * 0.05),
        substitutions=substitutions,
        metadata={
            "field_name": field_name,
            "original_value": original_value,
            "requires_retry_ocr": False,
            "requires_review": False,
        },
    )


def _replace_numeric_ambiguities(value: str) -> tuple[str, int]:
    converted: list[str] = []
    substitutions = 0
    for character in value:
        replacement = NUMERIC_OCR_SUBSTITUTIONS.get(character, character)
        if replacement != character:
            substitutions += 1
        converted.append(replacement)
    return "".join(converted), substitutions


def _is_valid_modern_ic_digits(digits: str) -> bool:
    if not re.fullmatch(r"\d{12}", digits):
        return False
    return _valid_birth_date_from_yyMMdd(digits[:6])


def _valid_birth_date_from_yyMMdd(value: str) -> bool:
    year = int(value[:2])
    month = int(value[2:4])
    day = int(value[4:6])
    if not 1 <= month <= 12:
        return False

    for century in (1900, 2000):
        try:
            date(century + year, month, day)
            return True
        except ValueError:
            continue
    return False


def _prepare_date_for_analysis(raw: str | None) -> str:
    if raw is None:
        return ""
    value = str(raw).upper()
    value = value.translate(str.maketrans(NUMERIC_OCR_SUBSTITUTIONS))
    value = re.sub(r"[^0-9./\-\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _date_parts_from_text(prepared: str) -> list[tuple[str, str, str]]:
    matches: list[tuple[str, str, str]] = []
    groups = [group for group in re.split(r"[-./\s]+", prepared) if group]

    if len(groups) == 3:
        matches.append((groups[0], groups[1], groups[2]))
    elif len(groups) == 2 and len(groups[1]) == 3:
        matches.append((groups[0], groups[1][0], groups[1][1:]))

    return matches


def _count_date_substitutions(raw: str) -> int:
    count = 0
    for character in str(raw).upper():
        replacement = NUMERIC_OCR_SUBSTITUTIONS.get(character, character)
        if replacement != character:
            count += 1
    return count


def _normalize_date_parts(day_text: str, month_text: str, year_text: str) -> str | None:
    if not (day_text.isdigit() and month_text.isdigit() and year_text.isdigit()):
        return None

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
