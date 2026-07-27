from __future__ import annotations

import re
from statistics import median
from typing import Sequence

from marriage_ocr.typed.models import PositionedWord, Region, TemplateTransform


BORANG_4B_REGIONS: dict[str, tuple[int, Region]] = {
    "bil": (1, Region(0.205, 0.307, 0.600, 0.337)),
    "nama_suami": (1, Region(0.315, 0.468, 0.935, 0.493)),
    "id_suami": (1, Region(0.345, 0.492, 0.610, 0.519)),
    "umur_suami": (1, Region(0.680, 0.492, 0.825, 0.519)),
    "nama_isteri": (1, Region(0.315, 0.605, 0.935, 0.633)),
    "id_isteri": (1, Region(0.345, 0.630, 0.615, 0.658)),
    "umur_isteri": (1, Region(0.680, 0.630, 0.825, 0.658)),
    "nama_wali": (1, Region(0.315, 0.746, 0.935, 0.774)),
    "hubungan_wali": (1, Region(0.315, 0.786, 0.935, 0.814)),
    "saksi_1": (2, Region(0.325, 0.137, 0.940, 0.165)),
    "saksi_2": (2, Region(0.325, 0.229, 0.940, 0.257)),
    "tarikh_nikah": (2, Region(0.315, 0.355, 0.505, 0.374)),
    "alamat_pendaftar": (2, Region(0.170, 0.378, 0.940, 0.405)),
    "nama_pendaftar": (2, Region(0.350, 0.404, 0.940, 0.432)),
    "mas_kahwin": (2, Region(0.235, 0.448, 0.940, 0.476)),
}

BORANG_4B_ANCHORS: dict[int, dict[str, tuple[float, float]]] = {
    1: {
        "A MAKLUMAT PASANGAN": (0.105, 0.425),
        "SUAMI": (0.105, 0.450),
        "ISTERI": (0.105, 0.585),
        "B MAKLUMAT WALI": (0.105, 0.720),
    },
    2: {
        "C MAKLUMAT SAKSI": (0.105, 0.100),
        "D BUTIR BUTIR PERNIKAHAN": (0.105, 0.325),
    },
}

_TOKEN_RE = re.compile(r"[^A-Z0-9 ]+")
_WHITESPACE_RE = re.compile(r"\s+")


def get_region(field_key: str) -> tuple[int, Region]:
    return BORANG_4B_REGIONS[field_key]


def _clean_token(value: str) -> str:
    return _WHITESPACE_RE.sub(" ", _TOKEN_RE.sub(" ", value.upper())).strip()


def _tokenise_words(words: Sequence[PositionedWord]) -> list[str]:
    tokens: list[str] = []
    for word in sorted(words, key=lambda item: (item.page_number, item.y1, item.x1)):
        cleaned = _clean_token(word.text)
        if cleaned:
            tokens.extend(cleaned.split())
    return tokens


def _anchor_matches(words: Sequence[PositionedWord], phrase: str) -> list[tuple[float, float]]:
    tokens = _tokenise_words(words)
    phrase_tokens = _clean_token(phrase).split()
    if not phrase_tokens:
        return []

    words_by_page = sorted(words, key=lambda item: (item.y1, item.x1))
    cleaned_words = [_clean_token(word.text).split() for word in words_by_page]
    flattened: list[tuple[str, PositionedWord]] = []
    for word, tokens_for_word in zip(words_by_page, cleaned_words, strict=True):
        for token in tokens_for_word:
            flattened.append((token, word))

    matches: list[tuple[float, float]] = []
    for index in range(0, max(0, len(flattened) - len(phrase_tokens) + 1)):
        candidate_tokens = [token for token, _ in flattened[index : index + len(phrase_tokens)]]
        if candidate_tokens != phrase_tokens:
            continue
        anchor_word = flattened[index][1]
        matches.append((anchor_word.x1, anchor_word.y1))
    return matches


def estimate_transform(words: Sequence[PositionedWord], page_number: int) -> TemplateTransform:
    page_words = [word for word in words if word.page_number == page_number]
    if not page_words:
        return TemplateTransform(safe=False, diagnostics=("no words for page",))

    observed_dx: list[float] = []
    observed_dy: list[float] = []
    anchors_found: list[tuple[float, float, float, float]] = []

    for phrase, (expected_x, expected_y) in BORANG_4B_ANCHORS.get(page_number, {}).items():
        matches = _anchor_matches(page_words, phrase)
        if not matches:
            continue
        observed_x, observed_y = matches[0]
        observed_dx.append(observed_x - expected_x)
        observed_dy.append(observed_y - expected_y)
        anchors_found.append((expected_x, expected_y, observed_x, observed_y))

    if not observed_dx:
        return TemplateTransform(safe=False, diagnostics=("no anchor matches found",))

    dx = float(median(observed_dx))
    dy = float(median(observed_dy))
    scale_x = 1.0
    scale_y = 1.0
    diagnostics: list[str] = []

    if len(anchors_found) >= 2:
        expected_spans = []
        observed_spans = []
        for index, (expected_x, expected_y, observed_x, observed_y) in enumerate(anchors_found):
            for later_expected_x, later_expected_y, later_observed_x, later_observed_y in anchors_found[index + 1 :]:
                expected_span = abs(later_expected_y - expected_y)
                observed_span = abs(later_observed_y - observed_y)
                if expected_span > 0:
                    expected_spans.append(expected_span)
                    observed_spans.append(observed_span)
        if expected_spans:
            ratios = [observed / expected for observed, expected in zip(observed_spans, expected_spans, strict=True)]
            scale_y = float(median(ratios))
            if scale_y < 0.98 or scale_y > 1.02:
                diagnostics.append(f"scale_y clamped from {scale_y:.4f}")
                scale_y = 1.0

    safe = (
        abs(dx) <= 0.05
        and abs(dy) <= 0.05
        and 0.95 <= scale_x <= 1.05
        and 0.95 <= scale_y <= 1.05
    )
    if not safe:
        diagnostics.append(f"unsafe transform dx={dx:.4f} dy={dy:.4f} scale_x={scale_x:.4f} scale_y={scale_y:.4f}")
    return TemplateTransform(dx=dx, dy=dy, scale_x=scale_x, scale_y=scale_y, safe=safe, diagnostics=tuple(diagnostics))


def apply_transform(region: Region, transform: TemplateTransform) -> Region:
    return Region(
        region.x1 * transform.scale_x + transform.dx,
        region.y1 * transform.scale_y + transform.dy,
        region.x2 * transform.scale_x + transform.dx,
        region.y2 * transform.scale_y + transform.dy,
    )
