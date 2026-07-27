from __future__ import annotations

from statistics import mean
from typing import Sequence

from marriage_ocr.typed.models import (
    PageOcrResult,
    PositionedWord,
    RawField,
    Region,
    TemplateTransform,
)
from marriage_ocr.typed.template import apply_transform, estimate_transform, get_region


FIELD_OUTPUT_NAMES = {
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


def words_in_region(
    words: Sequence[PositionedWord],
    region: Region,
    *,
    tolerance: float,
) -> tuple[PositionedWord, ...]:
    expanded = region.expand(tolerance)
    selected: list[PositionedWord] = []
    for word in words:
        centre_x, centre_y = word.centre
        if expanded.x1 <= centre_x <= expanded.x2 and expanded.y1 <= centre_y <= expanded.y2:
            selected.append(word)
    return tuple(selected)


def join_words_in_reading_order(
    words: Sequence[PositionedWord],
    *,
    line_tolerance: float = 0.008,
) -> str:
    ordered = sorted(words, key=lambda word: ((word.y1 + word.y2) / 2.0, word.x1))
    lines: list[list[PositionedWord]] = []
    line_centres: list[float] = []
    for word in ordered:
        centre_y = (word.y1 + word.y2) / 2.0
        if not lines or abs(centre_y - line_centres[-1]) > line_tolerance:
            lines.append([word])
            line_centres.append(centre_y)
        else:
            lines[-1].append(word)
            line_centres[-1] = sum(
                (item.y1 + item.y2) / 2.0 for item in lines[-1]
            ) / len(lines[-1])
    return "\n".join(
        " ".join(word.text for word in sorted(line, key=lambda item: item.x1)).strip()
        for line in lines
        if line
    )


def extract_raw_fields(
    page_results: Sequence[PageOcrResult],
    *,
    boundary_tolerance: float = 0.01,
) -> dict[str, RawField]:
    by_page = {result.page_number: result for result in page_results}
    transforms = {
        page_number: estimate_transform(result.words, page_number=page_number)
        for page_number, result in by_page.items()
    }
    extracted: dict[str, RawField] = {}
    for field_key, output_name in FIELD_OUTPUT_NAMES.items():
        page_number, base_region = get_region(field_key)
        page_result = by_page.get(page_number)
        transform = transforms.get(page_number, TemplateTransform())
        if transform.safe:
            region = apply_transform(base_region, transform)
        else:
            region = apply_transform(
                base_region,
                TemplateTransform(dx=transform.dx, dy=transform.dy, scale_x=1.0, scale_y=1.0),
            )
        selected = words_in_region(
            page_result.words if page_result else (),
            region,
            tolerance=boundary_tolerance,
        )
        confidence = mean(word.confidence for word in selected) if selected else 0.0
        extracted[field_key] = RawField(
            key=field_key,
            output_name=output_name,
            page_number=page_number,
            region=region,
            raw_text=join_words_in_reading_order(selected),
            confidence=confidence,
            words=selected,
        )
    return extracted
