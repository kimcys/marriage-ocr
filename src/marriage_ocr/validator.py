# src/marriage_ocr/validator.py

import re
from datetime import datetime


def clean_text(value):
    if value is None:
        return ""

    return str(value).strip()


def is_valid_malaysian_ic(value: str) -> bool:
    value = clean_text(value).replace("-", "")

    if not re.fullmatch(r"\d{12}", value):
        return False

    mm = int(value[2:4])
    dd = int(value[4:6])

    if not 1 <= mm <= 12:
        return False

    if not 1 <= dd <= 31:
        return False

    return True


def looks_like_name(value: str) -> bool:
    value = clean_text(value)

    if len(value) < 3:
        return False

    if re.search(r"\d{4,}", value):
        return False

    return True


def is_valid_date(value: str) -> bool:
    value = clean_text(value)

    if not value:
        return False

    formats = [
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%Y-%m-%d",
    ]

    for fmt in formats:
        try:
            datetime.strptime(value, fmt)
            return True
        except ValueError:
            pass

    return False


def validate_record(record: dict) -> dict:
    errors = []

    if not clean_text(record.get("bil")):
        errors.append("Missing bil")

    if not looks_like_name(record.get("nama_suami")):
        errors.append("Suspicious nama_suami")

    if not looks_like_name(record.get("nama_isteri")):
        errors.append("Suspicious nama_isteri")

    ic_suami = clean_text(record.get("ic_baru_suami"))
    ic_isteri = clean_text(record.get("ic_baru_isteri"))

    if ic_suami and not is_valid_malaysian_ic(ic_suami):
        errors.append("Invalid ic_baru_suami")

    if ic_isteri and not is_valid_malaysian_ic(ic_isteri):
        errors.append("Invalid ic_baru_isteri")

    tarikh_nikah = clean_text(record.get("tarikh_nikah"))

    if tarikh_nikah and not is_valid_date(tarikh_nikah):
        errors.append("Invalid tarikh_nikah")

    if errors:
        record["status"] = "REVIEW"
        record["confidence"] = min(float(record.get("confidence", 0.5)), 0.5)
    else:
        record["status"] = "OK"
        record["confidence"] = max(float(record.get("confidence", 0.8)), 0.8)

    record["validation_errors"] = errors

    return record