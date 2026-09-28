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
# for a page to count as Jawi and get skipped. Calibrated against real
# samples: 49 genuine Jawi register pages scored 0.61-0.90 (median 0.77),
# both classified one at a time and in stacks -- they are never "entirely"
# Jawi, since printed English/Rumi titles ("DIVORCE REGISTER"), Rumi notes and
# digits are mixed in -- while every real Rumi handwritten page scored 0.00
# (including when the ar-hinted pass was the one used). The old 0.85 guess
# missed 44 of those 49 Jawi pages, and one ("Sijil Rujuk" in a Rumi note)
# was routed to Gemini as a Rujuk register. 0.40 sits well clear of both.
DEFAULT_JAWI_PROPORTION_THRESHOLD = 0.40

# Small edit-distance tolerance for a handwritten keyword match -- 1 for a
# keyword of 6 letters or fewer, 2 for anything longer. Confirmed necessary
# against a real 154-photo handwritten Nikah ledger batch where "PERKAHWINAN"
# was misread as "PERKAHWAN"/"PERKAHNAN"/"PERKAHWNAN" (a dropped letter) or
# "PERKAH NAN" (a stray inserted space) on roughly 1 in 4 pages -- an exact
# substring check missed all of these even though the header was genuinely
# present and legible to a human. Scaled by length (not a flat tolerance) so
# a short keyword like "RUJUK" (5 letters) doesn't become so loose it starts
# matching unrelated short words by coincidence.
def _fuzzy_tolerance(keyword: str) -> int:
    return 1 if len(keyword) <= 6 else 2


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous_row = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current_row = [i]
        for j, char_b in enumerate(b, start=1):
            cost = 0 if char_a == char_b else 1
            current_row.append(
                min(
                    previous_row[j] + 1,  # deletion
                    current_row[j - 1] + 1,  # insertion
                    previous_row[j - 1] + cost,  # substitution
                )
            )
        previous_row = current_row
    return previous_row[-1]


def _fuzzy_keyword_in_region(keyword: str, region: str) -> bool:
    """True if `keyword` appears verbatim in `region`, or any single
    whitespace-separated word in `region` (or two adjacent words joined, to
    tolerate a stray inserted space splitting the keyword in two) is within
    _fuzzy_tolerance(keyword) edits of it."""
    if keyword in region:
        return True
    max_distance = _fuzzy_tolerance(keyword)
    words = region.split()
    candidates = [*words, *(a + b for a, b in zip(words, words[1:]))]
    return any(
        abs(len(candidate) - len(keyword)) <= max_distance
        and _levenshtein(keyword, candidate) <= max_distance
        for candidate in candidates
    )


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
    page_ocr_output: Path | None = None,
) -> Classification:
    """Classify a file by its first page. A scanned ledger book stays one
    record_type/layout throughout; a typed certificate PDF is inherently one
    record -- so the first page is representative of the whole file.

    With `page_ocr_output`, a PDF's page 1 is sent as the exact PNG
    process-typed would render and send itself, and -- if it classifies as a
    non-Jawi typed document -- its `ms/en` Vision result is saved there for
    `process-typed --page1-ocr` to reuse (see typed/page_ocr_cache.py)."""
    file_path = Path(file_path)
    pages = load_document_pages(file_path, list(allowed_extensions), pdf_dpi=pdf_dpi)
    if not pages:
        return Classification("unknown", None, None, False, 0.0, ["no pages found"])

    page = pages[0]
    save_page_ocr = page_ocr_output is not None and file_path.suffix.lower() == ".pdf"
    with tempfile.TemporaryDirectory(prefix="marriage-ocr-triage-") as tmp_dir:
        page_path = Path(tmp_dir) / ("page.png" if save_page_ocr else "page.jpg")
        write_image(page_path, page.image)

        primary_result, primary_annotation = GoogleVisionOcrEngine(
            {"language_hints": ["ms", "en"]}
        ).read_image_annotated(page_path)
        doc_type, record_type, layout_variant, header_notes = _classify_headers(primary_result.text)
        # A matched typed-form title means a printed Rumi form, never an
        # all-Jawi page -- skip the (billed) ar-hinted pass for it and judge
        # Jawi proportion from the ms/en pass alone. Handwritten/unknown
        # pages still get both passes (see module docstring for why).
        if doc_type == "typed":
            jawi_result = OcrResult()
        else:
            jawi_result = GoogleVisionOcrEngine({"language_hints": ["ar"]}).read_image(page_path)

    is_jawi, jawi_proportion, jawi_notes = _assess_jawi(
        primary_result, jawi_result, jawi_proportion_threshold
    )

    if save_page_ocr and page_ocr_output is not None and doc_type == "typed" and not is_jawi:
        from marriage_ocr.typed.page_ocr_cache import write_page_ocr_cache
        from marriage_ocr.typed.vision import annotation_to_page_result

        height, width = page.image.shape[:2]
        write_page_ocr_cache(
            page_ocr_output,
            annotation_to_page_result(primary_annotation, source_file=file_path.name, page_number=1),
            dpi=pdf_dpi,
            width=width,
            height=height,
        )

    return Classification(
        doc_type=doc_type,
        record_type=record_type,
        layout_variant=layout_variant,
        is_jawi=is_jawi,
        jawi_proportion=jawi_proportion,
        notes=[*header_notes, *jawi_notes],
    )


# Stacked classify (classify_files_stacked): up to this many single-image
# files per stacked image, separated by white bands so each page's words
# stay attributable to it. Vision rejects a request over 40MB (base64 adds a
# third), so a stack whose JPEG exceeds MAX_STACK_JPEG_BYTES is refused and
# the caller classifies those files one by one instead.
MAX_STACK_SIZE = 3
MAX_STACK_JPEG_BYTES = 20 * 1024 * 1024
_STACK_GAP_PX = 80
_STACK_JPEG_QUALITY = 95
_PDF_STACK_TOP_FRACTION = 0.40


class StackTooLargeError(ValueError):
    pass


def classify_files_stacked(
    file_paths: Sequence[Path],
    *,
    allowed_extensions: Sequence[str],
    jawi_proportion_threshold: float = DEFAULT_JAWI_PROPORTION_THRESHOLD,
) -> list[Classification]:
    """Classify up to MAX_STACK_SIZE single-page image files with the same
    two Vision passes classify_file makes for ONE file (ms/en + ar), by
    stacking them into one tall image and judging each page only from the
    words inside its own band -- same header keywords, same Jawi rule.
    Validated against per-file classify_file on real handwritten samples
    (27/27 identical, stacks of 2 and 3, mixed legacy/modern). PDFs contribute
    their first page; a stack of only typed forms skips the ar pass. Classify only
    reads large printed titles/column headers, which survive the taller
    image; don't reuse this for field extraction (a stacked typed-form test
    changed ~6% of extracted fields)."""
    import cv2
    import numpy as np

    paths = [Path(p) for p in file_paths]
    if not 1 <= len(paths) <= MAX_STACK_SIZE:
        raise ValueError(f"a stack holds 1-{MAX_STACK_SIZE} files, got {len(paths)}")

    images = []
    for path in paths:
        # A PDF contributes only the top of its first page: a typed form's
        # title and enactment line ("No. 4 TAHUN 1984" = legacy) are all
        # classify reads, and in a full 3-page-tall stack that small print
        # was lost -- every legacy form read as modern (46/82 correct). Top
        # of page only keeps it sharp. Images (handwritten) stay whole.
        pages = load_document_pages(path, list(allowed_extensions), pdf_dpi=300)
        if not pages:
            raise ValueError(f"no page found in {path.name}")
        image = pages[0].image
        if path.suffix.lower() == ".pdf":
            image = image[: max(1, int(image.shape[0] * _PDF_STACK_TOP_FRACTION))]
        images.append(image)

    width = max(image.shape[1] for image in images)
    rows: list[np.ndarray] = []
    spans: list[tuple[int, int]] = []
    y = 0
    for image in images:
        padded = np.full((image.shape[0], width, 3), 255, dtype=np.uint8)
        padded[:, : image.shape[1]] = image
        rows.extend([padded, np.full((_STACK_GAP_PX, width, 3), 255, dtype=np.uint8)])
        spans.append((y, y + image.shape[0]))
        y += image.shape[0] + _STACK_GAP_PX
    stacked = np.vstack(rows[:-1])
    height = stacked.shape[0]

    with tempfile.TemporaryDirectory(prefix="marriage-ocr-triage-stack-") as tmp_dir:
        stack_path = Path(tmp_dir) / "stack.jpg"
        if not cv2.imwrite(str(stack_path), stacked, [cv2.IMWRITE_JPEG_QUALITY, _STACK_JPEG_QUALITY]):
            raise OSError("failed to write stacked classify image")
        if stack_path.stat().st_size > MAX_STACK_JPEG_BYTES:
            raise StackTooLargeError(
                f"stacked image is {stack_path.stat().st_size} bytes (limit {MAX_STACK_JPEG_BYTES})"
            )

        def band_texts(language_hints: tuple[str, ...]) -> list[str]:
            from marriage_ocr.typed.extractor import join_words_in_reading_order
            from marriage_ocr.typed.vision import annotation_to_page_result

            _, annotation = GoogleVisionOcrEngine({"language_hints": list(language_hints)}).read_image_annotated(
                stack_path
            )

            def band_of(top_px: float, bottom_px: float) -> int | None:
                centre = (top_px + bottom_px) / 2
                return next((i for i, (top, bottom) in enumerate(spans) if top <= centre < bottom), None)

            # PDFs (typed forms): whole Vision text blocks in Vision's own
            # order, as classify_file's text is -- keeps a certificate title
            # on its own line, apart from a stamp printed beside it.
            blocks: list[list[str]] = [[] for _ in spans]
            for page in getattr(annotation, "pages", []) or []:
                for block in getattr(page, "blocks", []) or []:
                    ys = [float(getattr(v, "y", 0) or 0) for v in getattr(block.bounding_box, "vertices", []) or []]
                    band = band_of(min(ys), max(ys)) if ys else None
                    if band is None:
                        continue
                    for paragraph in getattr(block, "paragraphs", []) or []:
                        words = [
                            "".join(str(getattr(sym, "text", "")) for sym in getattr(word, "symbols", []) or [])
                            for word in getattr(paragraph, "words", []) or []
                        ]
                        line = " ".join(w for w in words if w)
                        if line:
                            blocks[band].append(line)

            # Images (handwritten registers): words in reading order -- the
            # form validated 27/27 against classify_file on real ledgers.
            words = annotation_to_page_result(annotation, source_file=paths[0].name, page_number=1).words
            texts = []
            for index, (path, (top, bottom)) in enumerate(zip(paths, spans)):
                if path.suffix.lower() == ".pdf":
                    texts.append("\n".join(blocks[index]))
                else:
                    inside = tuple(w for w in words if top <= ((w.y1 + w.y2) / 2) * height < bottom)
                    texts.append(join_words_in_reading_order(inside))
            return texts

        primary_texts = band_texts(("ms", "en"))
        # Same rule as classify_file: a matched typed-form title means a
        # printed Rumi form, never Jawi -- so a stack of nothing but typed
        # forms skips the (billed) ar-hinted pass: 1 Vision call for 3 PDFs.
        if all(_classify_headers(text)[0] == "typed" for text in primary_texts):
            jawi_texts = [""] * len(primary_texts)
        else:
            jawi_texts = band_texts(("ar",))

    classifications = []
    for primary_text, jawi_text in zip(primary_texts, jawi_texts, strict=True):
        is_jawi, jawi_proportion, jawi_notes = _assess_jawi(
            OcrResult(text=primary_text), OcrResult(text=jawi_text), jawi_proportion_threshold
        )
        doc_type, record_type, layout_variant, header_notes = _classify_headers(primary_text)
        classifications.append(
            Classification(
                doc_type=doc_type,
                record_type=record_type,
                layout_variant=layout_variant,
                is_jawi=is_jawi,
                jawi_proportion=jawi_proportion,
                notes=[*header_notes, *jawi_notes, f"stacked classify ({len(paths)} files)"],
            )
        )
    return classifications


def _is_standalone_title(line: str, keyword: str) -> bool:
    stripped = line.strip(" .:'\u2019*")
    return stripped.startswith(keyword) and len(stripped) <= len(keyword) + 4


def _classify_headers(text: str) -> tuple[str, str | None, str | None, list[str]]:
    upper_text = text.upper()
    header_region = "\n".join(upper_text.splitlines()[:_HEADER_REGION_LINES])

    # Typed keywords first: they're specific multi-word phrases ("SURAT
    # PERAKUAN NIKAH"), whereas the handwritten keywords below are single
    # bare words loosened to survive the book-spine title split -- checking
    # handwritten first would let a typed certificate's body text (which
    # can legitimately mention "...kahwin, cerai dan rujuk...") false-match
    # on the bare word "RUJUK" before ever reaching the more specific check.
    matched = [(keyword, record_type) for keyword, record_type in _TYPED_HEADER_KEYWORDS.items() if keyword in upper_text]
    if matched:
        # More than one certificate title can appear on a page: a Cerai form
        # has a "Bilangan Daftar Surat Perakuan Rujuk" field label, and a real
        # Rujuk form carried a registrar's stamp reading "No. Siri Surat
        # Perakuan Cerai / Ruju'". The actual title stands on its own line;
        # those mentions sit inside longer lines. Prefer a standalone match,
        # else keep the original order (Cerai, then Rujuk, then Nikah).
        standalone = [
            (keyword, record_type)
            for keyword, record_type in matched
            if any(_is_standalone_title(line, keyword) for line in upper_text.splitlines())
        ]
        keyword, record_type = (standalone or matched)[0]
        layout_variant = "legacy" if _TYPED_LEGACY_ENACTMENT_KEYWORD in upper_text else "modern"
        return "typed", record_type, layout_variant, []

    for keyword, record_type in _HANDWRITTEN_HEADER_KEYWORDS.items():
        if _fuzzy_keyword_in_region(keyword, header_region):
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
