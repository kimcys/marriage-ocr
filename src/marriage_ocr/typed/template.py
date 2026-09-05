from __future__ import annotations

import re
from statistics import median
from typing import Any, Sequence

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


# ---------------------------------------------------------------------------
# Cerai / Rujuk typed certificates -- NOT single templates each. Real
# attached samples (input/cerai - typed/*.pdf, input/rujuk - typed/*.pdf)
# show the same legacy/modern split already found in the handwritten
# ledgers: a pre-2003 single-page "legacy" form (Borang 9/Form 9 for Cerai,
# Borang 5/Form 5 for Rujuk, both under Enakmen No. 4 Tahun 1984) and a
# post-2003 two-page "modern" form (Cerai printed as either "Borang 8" or
# "Borang 10" depending on print run; Rujuk as either "Borang 7B" or
# "Borang 8B" -- confirmed field-for-field identical between those label
# variants on the same enactment/subsection, so they're one template each).
# Coordinates below are first-pass measurements from one representative
# sample per template (grid-overlay method, not guessed) -- see the plan
# for calibration status; verify against `page_N_regions.png` debug output
# before trusting at volume, same as every other layout in this project.
# ---------------------------------------------------------------------------

CERAI_MODERN_REGIONS: dict[str, tuple[int, Region]] = {
    "bil": (1, Region(0.29, 0.310, 0.60, 0.328)),
    "bil_daftar_nikah": (1, Region(0.41, 0.330, 0.75, 0.348)),
    "bil_daftar_rujuk_asal": (1, Region(0.41, 0.355, 0.85, 0.373)),
    "bilangan_kes_mal": (1, Region(0.26, 0.378, 0.60, 0.397)),
    "tarikh_nikah_hijri": (1, Region(0.34, 0.404, 0.55, 0.421)),
    "tarikh_nikah": (1, Region(0.70, 0.404, 0.90, 0.421)),
    "tarikh_rujuk_hijri": (1, Region(0.34, 0.428, 0.55, 0.446)),
    "tarikh_rujuk": (1, Region(0.70, 0.428, 0.90, 0.446)),
    "tempat_nikah_daerah": (1, Region(0.34, 0.452, 0.52, 0.470)),
    "tempat_nikah_negeri": (1, Region(0.70, 0.452, 0.90, 0.470)),
    "nama_suami": (1, Region(0.23, 0.477, 0.90, 0.495)),
    "id_suami": (1, Region(0.35, 0.502, 0.56, 0.520)),
    "bangsa_suami": (1, Region(0.70, 0.502, 0.90, 0.520)),
    "tarikh_lahir_suami": (1, Region(0.30, 0.526, 0.48, 0.544)),
    "warganegara_suami": (1, Region(0.68, 0.526, 0.90, 0.544)),
    "alamat_suami": (1, Region(0.20, 0.548, 0.92, 0.572)),
    "pekerjaan_suami": (1, Region(0.21, 0.579, 0.90, 0.597)),
    "nama_isteri": (1, Region(0.22, 0.604, 0.90, 0.622)),
    "id_isteri": (1, Region(0.35, 0.628, 0.56, 0.646)),
    "bangsa_isteri": (1, Region(0.70, 0.628, 0.90, 0.646)),
    "tarikh_lahir_isteri": (1, Region(0.30, 0.652, 0.48, 0.670)),
    "warganegara_isteri": (1, Region(0.68, 0.652, 0.90, 0.670)),
    "alamat_isteri": (1, Region(0.20, 0.680, 0.92, 0.705)),
    "pekerjaan_isteri": (1, Region(0.21, 0.705, 0.90, 0.723)),
    "keadaan_talak": (1, Region(0.29, 0.729, 0.90, 0.747)),
    "talak_kali_ke": (1, Region(0.30, 0.749, 0.50, 0.760)),
    "jumlah_talak": (1, Region(0.70, 0.749, 0.90, 0.760)),
    "bayaran_tebus_talak": (1, Region(0.36, 0.769, 0.90, 0.787)),
    "tempat_cerai": (1, Region(0.29, 0.791, 0.90, 0.809)),
    "tempat_bercerai": (1, Region(0.45, 0.811, 0.90, 0.828)),
    "tarikh_cerai_hijri": (1, Region(0.33, 0.832, 0.60, 0.850)),
    "tarikh_cerai": (1, Region(0.70, 0.832, 0.90, 0.850)),
    "cerai_dalam_keadaan": (1, Region(0.42, 0.855, 0.90, 0.873)),
    "hal_hal_lain": (2, Region(0.10, 0.130, 0.92, 0.440)),
    "tarikh_daftar_hijri": (2, Region(0.25, 0.441, 0.50, 0.459)),
    "tarikh_daftar": (2, Region(0.28, 0.455, 0.50, 0.472)),
    "nama_pendaftar": (2, Region(0.45, 0.455, 0.92, 0.472)),
    "jawatan_pendaftar": (2, Region(0.45, 0.474, 0.92, 0.502)),
}

CERAI_MODERN_ANCHORS: dict[int, dict[str, tuple[float, float]]] = {
    1: {
        "BILANGAN DAFTAR CERAI": (0.124, 0.314),
        "NAMA SUAMI": (0.124, 0.482),
        "NAMA ISTERI": (0.124, 0.608),
        "JENIS PERCERAIAN": (0.124, 0.733),
    },
    2: {
        "LAIN LAIN KENYATAAN": (0.117, 0.136),
    },
}

CERAI_LEGACY_REGIONS: dict[str, tuple[int, Region]] = {
    "bil": (1, Region(0.34, 0.289, 0.50, 0.308)),
    "no_sijil_perakuan_nikah_rujuk": (1, Region(0.36, 0.322, 0.62, 0.336)),
    "no_permohonan_cerai": (1, Region(0.34, 0.336, 0.62, 0.357)),
    "nama_suami": (1, Region(0.31, 0.362, 0.70, 0.377)),
    "id_suami": (1, Region(0.31, 0.390, 0.46, 0.406)),
    "tarikh_lahir_suami": (1, Region(0.63, 0.388, 0.78, 0.407)),
    "pekerjaan_suami": (1, Region(0.31, 0.412, 0.90, 0.421)),
    "alamat_suami": (1, Region(0.31, 0.439, 0.90, 0.457)),
    "nama_isteri": (1, Region(0.31, 0.462, 0.70, 0.476)),
    "id_isteri": (1, Region(0.31, 0.481, 0.46, 0.499)),
    "tarikh_lahir_isteri": (1, Region(0.63, 0.482, 0.78, 0.499)),
    "pekerjaan_isteri": (1, Region(0.31, 0.505, 0.55, 0.519)),
    "alamat_isteri": (1, Region(0.31, 0.531, 0.90, 0.550)),
    "tarikh_nikah": (1, Region(0.31, 0.552, 0.62, 0.576)),
    "saksi_1": (1, Region(0.31, 0.571, 0.70, 0.592)),
    "saksi_2": (1, Region(0.31, 0.603, 0.60, 0.620)),
    "tempat_cerai": (1, Region(0.20, 0.626, 0.47, 0.641)),
    "keadaan_talak": (1, Region(0.595, 0.622, 0.97, 0.663)),
    "bayaran_tebus_talak": (1, Region(0.39, 0.663, 0.90, 0.673)),
    "tempat_bercerai": (1, Region(0.25, 0.676, 0.56, 0.696)),
    "tarikh_cerai": (1, Region(0.68, 0.676, 0.90, 0.704)),
    "cerai_dalam_keadaan": (1, Region(0.42, 0.706, 0.90, 0.718)),
    "hal_hal_lain": (1, Region(0.08, 0.723, 0.95, 0.865)),
    "jumlah_bayaran": (1, Region(0.20, 0.866, 0.40, 0.883)),
    "tarikh_daftar": (1, Region(0.17, 0.913, 0.42, 0.934)),
}

CERAI_LEGACY_ANCHORS: dict[int, dict[str, tuple[float, float]]] = {
    1: {
        "SURAT PERAKUAN CERAI": (0.348, 0.246),
        "NAMA SUAMI": (0.084, 0.375),
        "NAMA ISTERI": (0.083, 0.470),
    },
}

RUJUK_MODERN_REGIONS: dict[str, tuple[int, Region]] = {
    "bil": (1, Region(0.27, 0.330, 0.45, 0.350)),
    "tarikh_daftar_hijri": (1, Region(0.71, 0.330, 0.90, 0.350)),
    "tarikh_daftar": (1, Region(0.71, 0.354, 0.90, 0.369)),
    "nama_suami": (1, Region(0.25, 0.418, 0.90, 0.437)),
    "id_suami": (1, Region(0.36, 0.439, 0.55, 0.459)),
    "tarikh_lahir_suami": (1, Region(0.74, 0.439, 0.90, 0.459)),
    "bangsa_suami": (1, Region(0.21, 0.463, 0.40, 0.479)),
    "warganegara_suami": (1, Region(0.55, 0.463, 0.75, 0.479)),
    "alamat_suami": (1, Region(0.27, 0.487, 0.92, 0.511)),
    "tarikh_masuk_islam_suami": (1, Region(0.39, 0.533, 0.55, 0.554)),
    "no_kad_perakuan_islam_suami": (1, Region(0.73, 0.533, 0.92, 0.554)),
    "nama_isteri": (1, Region(0.25, 0.568, 0.90, 0.586)),
    "id_isteri": (1, Region(0.36, 0.588, 0.55, 0.608)),
    "tarikh_lahir_isteri": (1, Region(0.74, 0.588, 0.90, 0.608)),
    "bangsa_isteri": (1, Region(0.21, 0.613, 0.40, 0.628)),
    "warganegara_isteri": (1, Region(0.55, 0.613, 0.75, 0.628)),
    "alamat_isteri": (1, Region(0.27, 0.634, 0.92, 0.668)),
    "tarikh_masuk_islam_isteri": (1, Region(0.39, 0.668, 0.55, 0.689)),
    "no_kad_perakuan_islam_isteri": (1, Region(0.73, 0.668, 0.92, 0.689)),
    "nama_pendaftar_saksi": (1, Region(0.19, 0.737, 0.45, 0.757)),
    "tarikh_rujuk_hijri": (2, Region(0.27, 0.160, 0.50, 0.180)),
    "tarikh_rujuk": (2, Region(0.27, 0.181, 0.50, 0.198)),
    "rujuk_kali": (2, Region(0.62, 0.160, 0.90, 0.180)),
    "tarikh_nikah_hijri": (2, Region(0.27, 0.202, 0.50, 0.219)),
    "tarikh_nikah": (2, Region(0.27, 0.220, 0.50, 0.238)),
    "bil_daftar_nikah": (2, Region(0.65, 0.200, 0.90, 0.220)),
    "tarikh_cerai_hijri": (2, Region(0.27, 0.240, 0.50, 0.260)),
    "tarikh_cerai": (2, Region(0.27, 0.261, 0.50, 0.278)),
    "bil_cerai": (2, Region(0.65, 0.241, 0.90, 0.259)),
    "hal_hal_lain": (2, Region(0.23, 0.294, 0.92, 0.415)),
    "nama_pendaftar": (2, Region(0.55, 0.456, 0.92, 0.478)),
    "jawatan_pendaftar": (2, Region(0.55, 0.481, 0.92, 0.512)),
}

RUJUK_MODERN_ANCHORS: dict[int, dict[str, tuple[float, float]]] = {
    1: {
        "A MAKLUMAT PASANGAN": (0.115, 0.390),
        # "B." prefix required -- bare "PERAKUAN RUJUK" also matches the
        # page header ("SURAT PERAKUAN RUJUK"), and since that occurs
        # earlier on the page it was wrongly picked as the anchor's first
        # match, producing a huge bogus dy. Confirmed against a real sample.
        "B PERAKUAN RUJUK": (0.117, 0.709),
    },
    2: {
        "C MAKLUMAT AM": (0.117, 0.131),
    },
}

RUJUK_LEGACY_REGIONS: dict[str, tuple[int, Region]] = {
    "bil": (1, Region(0.34, 0.391, 0.62, 0.409)),
    "tarikh_daftar": (1, Region(0.28, 0.415, 0.50, 0.437)),
    "nama_suami": (1, Region(0.28, 0.447, 0.92, 0.466)),
    "id_suami": (1, Region(0.28, 0.470, 0.46, 0.492)),
    "tarikh_lahir_suami": (1, Region(0.60, 0.470, 0.80, 0.492)),
    "pekerjaan_suami": (1, Region(0.28, 0.497, 0.70, 0.517)),
    "alamat_suami": (1, Region(0.28, 0.520, 0.75, 0.540)),
    "alamat_pejabat_suami": (1, Region(0.28, 0.545, 0.90, 0.565)),
    "nama_isteri": (1, Region(0.28, 0.570, 0.92, 0.586)),
    "id_isteri": (1, Region(0.28, 0.590, 0.50, 0.611)),
    "tarikh_lahir_isteri": (1, Region(0.60, 0.590, 0.82, 0.611)),
    "pekerjaan_isteri": (1, Region(0.28, 0.618, 0.55, 0.636)),
    "alamat_isteri": (1, Region(0.28, 0.640, 0.75, 0.660)),
    "alamat_pejabat_isteri": (1, Region(0.28, 0.663, 0.90, 0.682)),
    "nama_pendaftar": (1, Region(0.24, 0.682, 0.60, 0.702)),
    "bil_cerai": (1, Region(0.27, 0.868, 0.45, 0.887)),
    "tarikh_cerai": (1, Region(0.27, 0.888, 0.50, 0.918)),
    "tarikh_rujuk": (1, Region(0.60, 0.888, 0.90, 0.918)),
    "hal_hal_lain": (2, Region(0.08, 0.100, 0.95, 0.170)),
    "jumlah_bayaran": (2, Region(0.20, 0.171, 0.40, 0.192)),
}

RUJUK_LEGACY_ANCHORS: dict[int, dict[str, tuple[float, float]]] = {
    1: {
        "SURAT PERAKUAN RUJUK": (0.344, 0.317),
        "NAMA SUAMI": (0.115, 0.456),
        "NAMA ISTERI": (0.118, 0.574),
    },
    2: {
        "LAIN LAIN KENYATAAN": (0.083, 0.105),
    },
}

TEMPLATES: dict[str, dict[str, Any]] = {
    "borang_4b": {"regions": BORANG_4B_REGIONS, "anchors": BORANG_4B_ANCHORS, "pages": 2},
    "cerai_modern": {"regions": CERAI_MODERN_REGIONS, "anchors": CERAI_MODERN_ANCHORS, "pages": 2},
    "cerai_legacy": {"regions": CERAI_LEGACY_REGIONS, "anchors": CERAI_LEGACY_ANCHORS, "pages": 1},
    "rujuk_modern": {"regions": RUJUK_MODERN_REGIONS, "anchors": RUJUK_MODERN_ANCHORS, "pages": 2},
    "rujuk_legacy": {"regions": RUJUK_LEGACY_REGIONS, "anchors": RUJUK_LEGACY_ANCHORS, "pages": 2},
}

_TOKEN_RE = re.compile(r"[^A-Z0-9 ]+")
_WHITESPACE_RE = re.compile(r"\s+")


def get_region(field_key: str, template_name: str = "borang_4b") -> tuple[int, Region]:
    return TEMPLATES[template_name]["regions"][field_key]


def _clean_token(value: str) -> str:
    return _WHITESPACE_RE.sub(" ", _TOKEN_RE.sub(" ", value.upper())).strip()


_ANCHOR_LINE_TOLERANCE = 0.008


def _words_in_reading_order(words: Sequence[PositionedWord]) -> list[PositionedWord]:
    """Group words into visual lines (by y-centre, within tolerance) before
    sorting by x -- a plain sort by (y1, x1) is fragile for a multi-word
    anchor phrase: OCR-reported y1 varies by a hair's width between words on
    the same printed line (different glyph ascender heights), which can flip
    two same-line words out of reading order and silently break the phrase
    match. Confirmed against a real Rujuk-legacy sample where "SURAT" had
    y1=0.317 but "PERAKUAN"/"RUJUK" on the same line had y1=0.316, sorting
    them before "SURAT" and making the whole anchor unmatchable."""
    ordered = sorted(words, key=lambda word: ((word.y1 + word.y2) / 2.0, word.x1))
    lines: list[list[PositionedWord]] = []
    line_centres: list[float] = []
    for word in ordered:
        centre_y = (word.y1 + word.y2) / 2.0
        if not lines or abs(centre_y - line_centres[-1]) > _ANCHOR_LINE_TOLERANCE:
            lines.append([word])
            line_centres.append(centre_y)
        else:
            lines[-1].append(word)
            line_centres[-1] = sum((item.y1 + item.y2) / 2.0 for item in lines[-1]) / len(lines[-1])
    result: list[PositionedWord] = []
    for line in lines:
        result.extend(sorted(line, key=lambda item: item.x1))
    return result


def _tokenise_words(words: Sequence[PositionedWord]) -> list[str]:
    tokens: list[str] = []
    for word in _words_in_reading_order(words):
        cleaned = _clean_token(word.text)
        if cleaned:
            tokens.extend(cleaned.split())
    return tokens


def _anchor_matches(words: Sequence[PositionedWord], phrase: str) -> list[tuple[float, float]]:
    tokens = _tokenise_words(words)
    phrase_tokens = _clean_token(phrase).split()
    if not phrase_tokens:
        return []

    words_by_page = _words_in_reading_order(words)
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


def estimate_transform(
    words: Sequence[PositionedWord], page_number: int, template_name: str = "borang_4b"
) -> TemplateTransform:
    page_words = [word for word in words if word.page_number == page_number]
    if not page_words:
        return TemplateTransform(safe=False, diagnostics=("no words for page",))

    observed_dx: list[float] = []
    observed_dy: list[float] = []
    anchors_found: list[tuple[float, float, float, float]] = []

    anchors: dict[str, tuple[float, float]] = TEMPLATES[template_name]["anchors"].get(page_number, {})
    for phrase, (expected_x, expected_y) in anchors.items():
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
