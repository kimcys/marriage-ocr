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
# Typed Nikah certificates -- NOT one template. Real attached samples
# (input/nikah - typed/*.pdf) split the same way Cerai/Rujuk already do: a
# pre-2003 legacy form under Enakmen No. 4 Tahun 1984 ("Borang 3A/Form 3A",
# Seksyen 26, titled "SURAT PERAKUAN NIKAH") and a post-2003 modern form
# under Enakmen No. 2 Tahun 2003 ("Borang 4B", Subseksyen 26(1) dan (2)).
# BORANG_4B_REGIONS above predates this split and was calibrated against
# neither real layout -- confirmed by processing all 5 real attached
# samples: 3 were legacy Borang 3A (1990, 2000, 2005) and 2 were modern
# Borang 4B (2009, 2020), and BORANG_4B_REGIONS produced garbled output on
# effectively all of them (its tarikh_nikah/alamat_pendaftar/nama_pendaftar
# regions bleed into each other -- see typed/validator.py's
# _CONTAMINATION_LABELS comment, which already documented this). Kept as-is
# for backward compatibility (marriage-be's TYPED_BORANG_4B document type
# still points at config/typed_borang4b.yaml, which still names template:
# borang_4b) -- new documents should route to nikah_legacy/nikah_modern
# below via triage.py instead.
#
# Coordinates below are ground-truth word positions read directly from a
# real Vision full_page_vision.json debug capture for one representative
# sample per layout (input/nikah - typed/014970379082009.pdf for modern,
# input/nikah - typed/01490612085274082005.pdf for legacy) -- NOT the PDF's
# own embedded text layer. That layer is badly garbled character-by-
# character on the legacy sample even though the page image itself is
# crisp, and its reported word positions don't reliably match what Vision
# actually detects on the rendered image -- an earlier pass at
# NIKAH_LEGACY_REGIONS built from that embedded layer produced a
# deterministic ~0.013 dy transform drift (confirmed by rerunning twice and
# getting the identical drift both times -- not OCR run-to-run variance),
# enough to shift every field into its neighbor's territory on this
# densely-packed form (~0.02 between rows). Every legacy coordinate below
# was re-measured from real per-word Vision output instead. Still
# first-pass in the sense that only one sample per layout was used -- verify
# against page_N_regions.png debug output before trusting either at volume.
#
# NIKAH_LEGACY (Borang 3A) is a real 2-page PDF, but every field this
# template extracts lives on page 1 -- page 2 of every legacy sample is a
# separate "SURAT PERAKUAN TALIQ" declaration (the husband's conditional-
# divorce pronouncement), not a continuation of the marriage certificate's
# own fields. It happens to restate the wife's name and the marriage date,
# but has no fields this template needs that aren't already on page 1, so
# there are deliberately no page-2 region entries below (TEMPLATES still
# registers this as "pages": 2, since render_typed_pdf hard-fails on an
# actual page-count mismatch).
# Widened against THREE real samples spanning 1990/2000/2005 (one calibration
# pass against a single 2005 sample wasn't enough -- confirmed empirically:
# a tight region calibrated on 2005 alone truncated longer names/addresses on
# the 2000 sample and missed several fields entirely on the 1990 sample).
# Root cause is NOT random OCR noise: estimate_transform's dx/dy (from the
# "SURAT PERAKUAN NIKAH"/"NAMA WALI" anchors) varies by real, consistent
# amounts across these three actual scans (dx from 0 to -0.087, dy from 0 to
# -0.034 on page 1), and scale_y correction gets clamped/dropped whenever the
# transform is flagged unsafe (dx or dy over the shared 0.05 threshold, which
# happens on both the 2000 and 1990 samples) -- see estimate_transform()'s
# clamp and extractor.py's fallback to scale=1.0. Rather than loosen that
# shared safety clamp (used by Cerai/Rujuk too, no evidence they need it
# loosened), every field's X range here is widened to comfortably span its
# whole printed row (left past the label, right past realistic content
# width) and Y is widened by a fixed +-0.02/-0.03 margin sized to the actual
# observed dy spread -- verified field-by-field against Vision word
# positions captured from all three samples' debug output. Text/name/address
# fields lean on normalize_plain_text's/_select_best_line's noise filtering
# (prefers the digit-free, name-hint-bearing line) to recover the right
# value even when a neighbouring row's label/text now falls inside the
# region too; ID/date/bil fields rely on their own regex search plus
# reading-order (top-to-bottom) precedence, since a field's own true value
# always appears before a neighbour's bleed-in text in the joined raw text.
NIKAH_LEGACY_REGIONS: dict[str, tuple[int, Region]] = {
    "no_siri": (1, Region(0.73, 0.216, 0.95, 0.303)),
    "bil": (1, Region(0.30, 0.292, 0.64, 0.343)),
    "tarikh_nikah": (1, Region(0.30, 0.337, 0.70, 0.401)),
    # y1 pulled up well above the label's own row -- confirmed on the 2000
    # sample that the Masihi date sometimes lands on the Hijri ("H ...") row
    # printed just above "4. Tarikh didaftar pada" instead of beside the
    # label itself, unlike 2005 where it's inline with the label.
    "tarikh_daftar": (1, Region(0.34, 0.385, 0.70, 0.436)),
    "nama_suami": (1, Region(0.18, 0.393, 0.75, 0.452)),
    "id_suami": (1, Region(0.20, 0.416, 0.58, 0.474)),
    "tarikh_lahir_suami": (1, Region(0.66, 0.417, 0.92, 0.475)),
    "alamat_suami": (1, Region(0.13, 0.439, 0.80, 0.498)),
    "nama_isteri": (1, Region(0.18, 0.458, 0.75, 0.517)),
    "id_isteri": (1, Region(0.20, 0.479, 0.58, 0.538)),
    "tarikh_lahir_isteri": (1, Region(0.66, 0.480, 0.92, 0.538)),
    "alamat_isteri": (1, Region(0.13, 0.500, 0.80, 0.559)),
    "nama_wali": (1, Region(0.18, 0.522, 0.75, 0.580)),
    "id_wali": (1, Region(0.20, 0.544, 0.60, 0.602)),
    "alamat_wali": (1, Region(0.13, 0.564, 0.75, 0.623)),
    "hubungan_wali": (1, Region(0.23, 0.584, 0.65, 0.643)),
    "saksi_1": (1, Region(0.13, 0.631, 0.75, 0.689)),
    "id_saksi_1": (1, Region(0.29, 0.654, 0.66, 0.711)),
    "saksi_2": (1, Region(0.13, 0.695, 0.75, 0.753)),
    "id_saksi_2": (1, Region(0.29, 0.716, 0.66, 0.774)),
    "mas_kahwin": (1, Region(0.28, 0.755, 0.63, 0.814)),
    "belanja_hantaran": (1, Region(0.32, 0.777, 0.67, 0.835)),
    # x2 kept short of ~0.69 -- a circular registrar stamp overlaps this row
    # further right on the real sample and its text ("...PERCERAIAN...")
    # would otherwise bleed in.
    "pemberian_lain": (1, Region(0.41, 0.842, 0.65, 0.900)),
    "jumlah_bayaran": (1, Region(0.20, 0.896, 0.65, 0.961)),
}

NIKAH_LEGACY_ANCHORS: dict[int, dict[str, tuple[float, float]]] = {
    1: {
        # Coordinates below are real Vision-detected word positions (from
        # a debug full_page_vision.json capture), NOT the PDF's own
        # embedded text layer -- that layer is badly garbled on this
        # template's samples (see the module comment above) and its
        # character-position data doesn't match what Vision actually
        # detects on the rendered image. Using it directly here (an
        # earlier mistake) produced a consistent ~0.013 downward dy drift
        # on every run -- deterministic, not OCR run-to-run variance, and
        # confirmed by rerunning twice and getting the identical drift both
        # times. On this densely-packed form (~0.02 between rows) that was
        # enough to shift every field into its neighbor's territory.
        "SURAT PERAKUAN NIKAH": (0.4065, 0.2634),
        "NAMA WALI": (0.2101, 0.5560),
    },
}

# Widened against TWO real samples (2009's own calibration scan plus a real
# 2020 scan) -- a single-sample calibration pass wasn't enough. Confirmed
# empirically that page-2 anchors alone underestimate a genuine ~14% larger
# inter-row spacing on the 2020 sample (estimate_transform's scale_y computed
# 1.1438 there, clamped to 1.0 since it's outside the shared [0.98,1.02]
# tolerance -- see the identical situation and rationale documented on
# NIKAH_LEGACY_REGIONS above), so only dy (not scale) corrects page 2, and
# error compounds with distance from the page's own anchor. Every row's Y
# range below is the union of both samples' real (Vision-detected) row
# position, each converted back to template space via that sample's own
# applied dy, plus a small fixed buffer -- not a guess. Rows that pack two
# sub-fields side by side (Nama Suami's ID+Umur row, Warganegara+Bangsa,
# Pernikahan Kali+Isteri Ke) keep a narrower per-field X split; solo-value
# rows (names, addresses, tempat/nama pendaftar) widen X to most of the row
# so a longer label on one print run doesn't truncate the value on either
# sample, the same reasoning as NIKAH_LEGACY_REGIONS.
NIKAH_MODERN_REGIONS: dict[str, tuple[int, Region]] = {
    # y1 negative -- no_siri sits well above both page-1 anchors, so the
    # anchor-derived dy (computed from content *below* it) overshoots
    # downward here; confirmed on a real 2020 sample where the serial's
    # word-centre landed just above y1=0 after that dy was applied.
    "no_siri": (1, Region(0.68, -0.02, 0.97, 0.063)),
    "bil": (1, Region(0.13, 0.277, 0.55, 0.321)),
    # tarikh_daftar keeps a narrower X (not widened as generously as other
    # fields) and stays out of the 0.68-0.86 band -- this form repeats its
    # own "No. Siri" serial as a faint stamp down the right margin (see
    # no_siri's own region above and its recurrence near alamat_wali below),
    # and normalize_date_preserving_style (unlike normalize_bil's forgiving
    # pattern search) rejects a date line with that trailing noise on it.
    "tarikh_daftar": (1, Region(0.55, 0.277, 0.90, 0.321)),
    "nama_suami": (1, Region(0.13, 0.422, 0.90, 0.451)),
    "id_suami": (1, Region(0.13, 0.440, 0.62, 0.471)),
    "umur_suami": (1, Region(0.58, 0.440, 0.90, 0.471)),
    "warganegara_suami": (1, Region(0.13, 0.456, 0.58, 0.491)),
    "bangsa_suami": (1, Region(0.53, 0.456, 0.90, 0.491)),
    "alamat_suami": (1, Region(0.13, 0.479, 0.92, 0.521)),
    "nama_isteri": (1, Region(0.13, 0.568, 0.90, 0.597)),
    "id_isteri": (1, Region(0.13, 0.590, 0.62, 0.621)),
    "umur_isteri": (1, Region(0.58, 0.590, 0.90, 0.621)),
    "warganegara_isteri": (1, Region(0.13, 0.607, 0.58, 0.641)),
    "bangsa_isteri": (1, Region(0.53, 0.607, 0.90, 0.641)),
    "alamat_isteri": (1, Region(0.13, 0.625, 0.92, 0.679)),
    "nama_wali": (1, Region(0.13, 0.714, 0.90, 0.748)),
    "id_wali": (1, Region(0.13, 0.730, 0.62, 0.771)),
    "umur_wali": (1, Region(0.58, 0.730, 0.90, 0.771)),
    "hubungan_wali": (1, Region(0.13, 0.746, 0.60, 0.793)),
    # x2 narrowed (this form's own "No. Siri" serial repeats as a faint
    # stamp down the right margin -- see tarikh_daftar's comment above; here
    # it bled in as a trailing "No. Siri : <serial>" on alamat_wali's own
    # second line).
    "alamat_wali": (1, Region(0.13, 0.765, 0.80, 0.828)),
    "saksi_1": (2, Region(0.13, 0.112, 0.90, 0.149)),
    "id_saksi_1": (2, Region(0.13, 0.134, 0.90, 0.168)),
    "saksi_2": (2, Region(0.13, 0.239, 0.90, 0.267)),
    "id_saksi_2": (2, Region(0.13, 0.258, 0.90, 0.290)),
    "tarikh_nikah_hijri": (2, Region(0.13, 0.363, 0.60, 0.399)),
    "tarikh_nikah": (2, Region(0.13, 0.379, 0.55, 0.409)),
    "hari_nikah": (2, Region(0.45, 0.363, 0.65, 0.399)),
    "masa_nikah": (2, Region(0.62, 0.363, 0.90, 0.399)),
    "tempat_nikah": (2, Region(0.13, 0.400, 0.90, 0.438)),
    "nama_pendaftar": (2, Region(0.13, 0.420, 0.90, 0.459)),
    "pernikahan_kali": (2, Region(0.13, 0.441, 0.50, 0.479)),
    "isteri_ke": (2, Region(0.45, 0.441, 0.75, 0.479)),
    "mas_kahwin": (2, Region(0.13, 0.462, 0.70, 0.502)),
    # x1 shifted right of "Belanja Hantaran:"/"Pemberian Lain (Jika Ada)"'s
    # own printed label (first-pass regions started inside the label
    # itself, capturing label text instead of the value beside it).
    "belanja_hantaran": (2, Region(0.13, 0.483, 0.60, 0.520)),
    "pemberian_lain": (2, Region(0.13, 0.504, 0.60, 0.541)),
}

NIKAH_MODERN_ANCHORS: dict[int, dict[str, tuple[float, float]]] = {
    1: {
        "A MAKLUMAT PASANGAN": (0.1562, 0.3854),
        "B MAKLUMAT WALI": (0.1566, 0.6965),
    },
    2: {
        "C MAKLUMAT SAKSI": (0.1574, 0.0871),
        "D BUTIR BUTIR PERNIKAHAN": (0.1570, 0.3364),
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
    # "pages": 2 even though every field this template extracts lives on
    # page 1 -- the PDF itself is genuinely 2 physical pages (page 2 is the
    # bundled "Surat Perakuan Taliq" declaration, see NIKAH_LEGACY_REGIONS's
    # own comment above), and render_typed_pdf hard-fails on a page-count
    # mismatch. Confirmed against all 3 real legacy samples: every one is a
    # 2-page PDF this way, not 1.
    "nikah_legacy": {"regions": NIKAH_LEGACY_REGIONS, "anchors": NIKAH_LEGACY_ANCHORS, "pages": 2},
    "nikah_modern": {"regions": NIKAH_MODERN_REGIONS, "anchors": NIKAH_MODERN_ANCHORS, "pages": 2},
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
