from __future__ import annotations

from collections import Counter
from dataclasses import replace
import re
from typing import Iterable

from marriage_ocr.models import ExtractedRecord


_BIL_NUMBER_RE = re.compile(r"\b(\d{3,5})(?:\s*/\s*(\d{2,4}))?\b")
_FOUR_DIGIT_YEAR_RE = re.compile(r"\b(?:19|20)(\d{2})\b")
_BIL_YEAR_RE = re.compile(r"/\s*(\d{2,4})\b")


def correct_bil_sequence(
    records: Iterable[ExtractedRecord],
    *,
    enabled: bool = True,
    start_number: int | None = None,
    year: str | int | None = None,
) -> list[ExtractedRecord]:
    """Fill/repair BIL values using the fact that ledger rows are sequential.

    OCR frequently misses the small red BIL column or reads only part of it. This
    function infers one continuous sequence from the OCR-visible BIL candidates
    and then fills every row. Explicit config values win over inference.
    """

    output = list(records)
    if not enabled or not output:
        return output

    inferred_year = _normalize_year(year) or _infer_year(output)
    inferred_start = start_number if start_number is not None else _infer_start_number(output)
    if inferred_start is None:
        return output

    corrected: list[ExtractedRecord] = []
    for index, record in enumerate(output):
        bil_value = f"{inferred_start + index}/{inferred_year}" if inferred_year else str(inferred_start + index)
        corrected.append(replace(record, bil=bil_value))
    return corrected


def _infer_start_number(records: list[ExtractedRecord]) -> int | None:
    votes: Counter[int] = Counter()
    for index, record in enumerate(records):
        for candidate in _bil_number_candidates(record):
            if 100 <= candidate <= 9999:
                votes[candidate - index] += 1
    if not votes:
        return None
    # Prefer the sequence explaining the most rows; tie-breaker: lowest start.
    return sorted(votes.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _bil_number_candidates(record: ExtractedRecord) -> list[int]:
    text = "\n".join(value or "" for value in [record.bil, record.raw_bil])
    cleaned = (
        text.upper()
        .replace("O", "0")
        .replace("Q", "0")
        .replace("S", "5")
        .replace("I", "1")
        .replace("L", "1")
    )
    candidates: list[int] = []
    for match in _BIL_NUMBER_RE.finditer(cleaned):
        digits = re.sub(r"\D", "", match.group(1))
        if len(digits) >= 3:
            if len(digits) > 3 and digits.startswith("4"):
                digits = digits[:3]
            try:
                candidates.append(int(digits))
            except ValueError:
                pass
    return candidates


def _infer_year(records: list[ExtractedRecord]) -> str | None:
    votes: Counter[str] = Counter()
    for record in records:
        # BIL suffix is the strongest signal, e.g. 460/94.
        for value in [record.bil, record.raw_bil]:
            if not value:
                continue
            for match in _BIL_YEAR_RE.finditer(str(value)):
                year = match.group(1)[-2:]
                if year.isdigit():
                    votes[year] += 5

        # Parsed dates contribute the calendar year only when it is four digits.
        for value in [record.tarikh_nikah, record.tarikh_keluar, record.raw_tarikh_nikah, record.raw_tarikh_keluar]:
            if not value:
                continue
            for match in _FOUR_DIGIT_YEAR_RE.finditer(str(value)):
                year = match.group(1)[-2:]
                if year.isdigit():
                    votes[year] += 1
    if not votes:
        return None
    return votes.most_common(1)[0][0]


def _normalize_year(value: str | int | None) -> str | None:
    if value is None:
        return None
    digits = re.sub(r"\D", "", str(value))
    if len(digits) < 2:
        return None
    return digits[-2:]
