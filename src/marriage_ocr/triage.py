"""Auto-classify an arbitrary input file before routing it to a pipeline.

At small scale a human picks `--config config/handwritten_cerai_legacy.yaml`
per batch. At ~900k records arriving as unpredictable OneDrive dumps (mixed
typed/handwritten, mixed Nikah/Cerai/Rujuk, occasional Jawi-only pages), that
doesn't scale -- this module looks at one page of a file and decides:

- doc_type: "handwritten" | "typed" | "unknown"
- record_type: "nikah" | "cerai" | "rujuk" | None
- layout_variant: "legacy" | "modern" | None (None only for handwritten Nikah
  and unrouted/unknown pages -- typed Nikah splits legacy/modern the same
  way Cerai/Rujuk do, see NIKAH_LEGACY_REGIONS/NIKAH_MODERN_REGIONS in
  typed/template.py)
- is_jawi: whether the page is essentially entirely Jawi script and should
  be skipped before any Gemini call is ever made on it

Deliberately cheap-heuristics-first (per the user's stated preference):
keyword matching on OCR text that's already being paid for, no extra paid
model call. The one deliberate extra cost is a second, `ar`-hinted Vision
pass used only for the Jawi check (see _assess_jawi) -- worth it because a
`ms/en`-hinted pass over a genuinely Jawi page often returns garbled
Latin-looking junk rather than real Arabic characters, which would silently
under-count Jawi proportion on a single pass.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tempfile
from typing import Sequence

from marriage_ocr.document_loader import load_document_pages, write_image
from marriage_ocr.models import OcrResult
from marriage_ocr.ocr import GoogleVisionOcrEngine


# Arabic-script Unicode blocks Jawi text falls in.
_JAWI_UNICODE_RANGES: tuple[tuple[int, int], ...] = (
    (0x0600, 0x06FF),  # Arabic
    (0x0750, 0x077F),  # Arabic Supplement
    (0x08A0, 0x08FF),  # Arabic Extended-A
    (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
)

# Fraction of recognized alphabetic characters that must be Jawi/Arabic-range
# for a page to count as "entirely Jawi" and get skipped. NOT calibrated
# against a real Jawi sample -- none exists in this repo yet. A page with
# only an incidental Jawi label/translation should read far below this; a
# genuinely Jawi-written page should read far above it. Revisit once real
# Jawi samples are available, same as the layout-ratio configs were tuned.
DEFAULT_JAWI_PROPORTION_THRESHOLD = 0.85

_HANDWRITTEN_HEADER_KEYWORDS: dict[str, str] = {
    "PERKAHWINAN": "nikah",
    # "PERCERA" (not the full word) deliberately -- confirmed against two
    # real samples that OCR renders this word inconsistently: a two-page
    # spread's book-spine gap splits it ("DAFTAR PERCERAIA\nORANG ISLAMA",
    # input/cerai/image00001-4.jpg), and a separate sample misreads I->J
    # ("PERCERAJAN", input/cerai/image00002.jpg). "PERCERA" is the stem
    # both variants (and the correct spelling) share, and is not a substring
    # of "CERAI" alone, so it won't false-match Rujuk's own "Bil Cerai"
    # column header.
    "PERCERA": "cerai",
    "RUJUK": "rujuk",
}

# How many leading lines of the page's recognized text count as "the title
# region" for record_type/doc_type keyword matching. The title reliably
# appears as line 0 on every real sample checked; kept slightly generous
# (not just line 0) in case a title ever wraps across two lines, but tight
# enough to exclude row-level content -- confirmed necessary against a real
# sample where a Cerai record's own remarks read "RUJUK BUKU DAFTAR" (a
# cross-reference note, not the page's own type) at line 14.
_HEADER_REGION_LINES = 5

_MODERN_LAYOUT_KEYWORDS: tuple[str, ...] = ("CATATAN",)

# Confirmed against real samples for all three typed record types
# (input/nikah - typed, input/cerai - typed, input/rujuk - typed).
#
# ORDER MATTERS -- do not alphabetize or otherwise reshuffle this dict.
# Unlike the handwritten check above, this one is deliberately NOT limited to
# a header-region prefix (a typed certificate's own title can appear several
# lines down, after "No. Siri", "(KETUA PENDAFTAR)", enactment/subseksyen
# lines, etc. -- there is no fixed line count that reliably covers it on
# every real sample). That means it scans the WHOLE page text, and a real
# Cerai certificate's own body cross-references the original marriage (and,
# on some print runs, a prior reconciliation) via printed lines like
# "Bilangan Daftar Surat Perakuan Nikah" / "...Surat Perakuan Rujuk" --
# confirmed on input/cerai - typed/01740326145837082010.pdf and
# input/cerai - typed/02321004055031082020.pdf, both real Cerai certs whose
# own body text contains the exact substrings "SURAT PERAKUAN NIKAH" and
# "SURAT PERAKUAN RUJUK". Checking "cerai" first means a real Cerai page
# matches (and returns) on its own title before either cross-reference line
# is ever evaluated. Rujuk forms only cross-reference by bare "Bil Daftar
# Cerai"/"Bil Daftar Nikah" (no "Surat Perakuan" prefix, confirmed across all
# 3 real Rujuk samples), so they carry no equivalent risk against the Cerai
# or Nikah keywords below.
#
# Real print runs spell this both with and without the trailing K -- Borang
# 5 (1990 legacy sample) prints "SURAT PERAKUAN RUJUK", but Borang 8B (2010
# modern sample, input/rujuk - typed/03600124085069082010.pdf) prints "SURAT
# PERAKUAN RUJU'" (apostrophe, no K). Matching on the shared "RUJU" stem
# (short enough to still require the full "SURAT PERAKUAN" phrase before it,
# so it can't bare-word-match something unrelated) covers both.
_TYPED_HEADER_KEYWORDS: dict[str, str] = {
    "SURAT PERAKUAN CERAI": "cerai",
    "SURAT PERAKUAN RUJU": "rujuk",
    "SURAT PERAKUAN NIKAH": "nikah",
    "BORANG 4B": "nikah",
}

# Typed Cerai/Rujuk certificates print their governing enactment in the
# header ("Enakmen ... No. 4 Tahun 1984" for the legacy layout vs "... No. 2
# Tahun 2003" for the modern layout) -- confirmed across all 6 real attached
# typed samples. This is more reliable than keying off the printed "Borang N"
# label: the modern layout is printed under two different Borang numbers for
# the same enactment (8 or 10 for Cerai, 7B or 8B for Rujuk) across real
# samples that are otherwise field-for-field identical. Nikah's typed path
# (borang_4b) has no legacy/modern split, so this only applies when the
# matched record_type is cerai/rujuk.
_TYPED_LEGACY_ENACTMENT_KEYWORD = "TAHUN 1984"


@dataclass
class Classification:
    doc_type: str
    record_type: str | None
    layout_variant: str | None
    is_jawi: bool
    jawi_proportion: float
    notes: list[str] = field(default_factory=list)


def classify_file(
    file_path: Path,
    *,
    allowed_extensions: Sequence[str],
    pdf_dpi: int = 300,
    jawi_proportion_threshold: float = DEFAULT_JAWI_PROPORTION_THRESHOLD,
) -> Classification:
    """Classify a file by its first page. A scanned ledger book stays one
    record_type/layout throughout; a typed certificate PDF is inherently one
    record -- so the first page is representative of the whole file."""
    pages = load_document_pages(Path(file_path), list(allowed_extensions), pdf_dpi=pdf_dpi)
    if not pages:
        return Classification("unknown", None, None, False, 0.0, ["no pages found"])

    page = pages[0]
    with tempfile.TemporaryDirectory(prefix="marriage-ocr-triage-") as tmp_dir:
        page_path = Path(tmp_dir) / "page.jpg"
        write_image(page_path, page.image)

        primary_result = GoogleVisionOcrEngine({"language_hints": ["ms", "en"]}).read_image(page_path)
        jawi_result = GoogleVisionOcrEngine({"language_hints": ["ar"]}).read_image(page_path)

    is_jawi, jawi_proportion, jawi_notes = _assess_jawi(
        primary_result, jawi_result, jawi_proportion_threshold
    )
    doc_type, record_type, layout_variant, header_notes = _classify_headers(primary_result.text)

    return Classification(
        doc_type=doc_type,
        record_type=record_type,
        layout_variant=layout_variant,
        is_jawi=is_jawi,
        jawi_proportion=jawi_proportion,
        notes=[*header_notes, *jawi_notes],
    )


def _classify_headers(text: str) -> tuple[str, str | None, str | None, list[str]]:
    upper_text = text.upper()
    header_region = "\n".join(upper_text.splitlines()[:_HEADER_REGION_LINES])

    # Typed keywords first: they're specific multi-word phrases ("SURAT
    # PERAKUAN NIKAH"), whereas the handwritten keywords below are single
    # bare words loosened to survive the book-spine title split -- checking
    # handwritten first would let a typed certificate's body text (which
    # can legitimately mention "...kahwin, cerai dan rujuk...") false-match
    # on the bare word "RUJUK" before ever reaching the more specific check.
    for keyword, record_type in _TYPED_HEADER_KEYWORDS.items():
        if keyword in upper_text:
            layout_variant = None
            if record_type in ("cerai", "rujuk", "nikah"):
                layout_variant = (
                    "legacy" if _TYPED_LEGACY_ENACTMENT_KEYWORD in upper_text else "modern"
                )
            return "typed", record_type, layout_variant, []

    for keyword, record_type in _HANDWRITTEN_HEADER_KEYWORDS.items():
        if keyword in header_region:
            layout_variant = "modern" if any(k in upper_text for k in _MODERN_LAYOUT_KEYWORDS) else "legacy"
            return "handwritten", record_type, layout_variant, []

    return "unknown", None, None, ["no known header keyword matched"]


def _assess_jawi(
    primary_result: OcrResult,
    jawi_result: OcrResult,
    threshold: float,
) -> tuple[bool, float, list[str]]:
    primary_jawi, primary_alpha = _count_jawi_chars(primary_result.text)
    jawi_jawi, jawi_alpha = _count_jawi_chars(jawi_result.text)

    # Prefer whichever pass recognized more alphabetic content -- an ms/en
    # hinted pass over a genuinely Jawi page often returns little to no text
    # at all, while the ar-hinted pass recovers the real content.
    if jawi_alpha > primary_alpha:
        jawi_count, alpha_count, source = jawi_jawi, jawi_alpha, "ar-hinted pass"
    else:
        jawi_count, alpha_count, source = primary_jawi, primary_alpha, "ms/en-hinted pass"

    proportion = (jawi_count / alpha_count) if alpha_count else 0.0
    is_jawi = proportion >= threshold
    notes = [f"jawi_proportion={proportion:.2f} from {source} ({alpha_count} alphabetic chars recognized)"]
    return is_jawi, proportion, notes


def _count_jawi_chars(text: str) -> tuple[int, int]:
    jawi_count = 0
    alpha_count = 0
    for character in text:
        if not character.isalpha():
            continue
        alpha_count += 1
        if _is_jawi_character(character):
            jawi_count += 1
    return jawi_count, alpha_count


def _is_jawi_character(character: str) -> bool:
    code_point = ord(character)
    return any(start <= code_point <= end for start, end in _JAWI_UNICODE_RANGES)
