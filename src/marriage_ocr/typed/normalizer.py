from __future__ import annotations

from decimal import Decimal, InvalidOperation
import re
from typing import Iterable

from marriage_ocr.models import ExtractedRecord
from marriage_ocr.typed.models import RawField
from marriage_ocr.refinement.text_corrections import generate_date_candidates


# \d{2,4}, not a fixed \d{4} -- confirmed on a real 1990 Borang 3A sample
# ("384/90") that the oldest legacy forms print a 2-digit year for Bilangan
# Daftar, not the 4-digit year every later form uses.
BIL_PATTERN = re.compile(r"\b\d+\s*/\s*\d{2,4}\b")
DATE_PATTERN = re.compile(r"\b(\d{1,2})\s*([./-])\s*(\d{1,2})\s*\2\s*(\d{4})\b")
# Legacy Cerai/Rujuk certs (Borang 5/9, 1984 enactment) print amounts with a
# "$" prefix (pre-Ringgit-renaming Malaysian dollar notation) rather than
# "RM" -- widened to accept either so the same pattern/normalizer covers
# mas_kahwin, bayaran_tebus_talak, and jumlah_bayaran alike.
MAS_KAHWIN_PATTERN = re.compile(r"(?:RM|\$)\s*([0-9][0-9\s,]*(?:\.[0-9]{1,2})?)", re.IGNORECASE)
_IC_DIGITS = re.compile(r"\D+")
_LEADING_LABELS = {
    "nama_suami": re.compile(r"^\s*(\d\s*\.\s*)?(nama\s*(suami)?\s*:?\s*)", re.IGNORECASE),
    "nama_isteri": re.compile(r"^\s*(\d\s*\.?\s*)?(nama\s*(isteri)?\s*:?\s*)", re.IGNORECASE),
    # The real printed label repeats "Pendaftar" twice ("Nama Pendaftar /
    # Pen . Pendaftar :") -- a pattern that only strips the first "Nama
    # Pendaftar" left "/ Pen . Pendaftar ..." in front of the actual name,
    # and since PENDAFTAR is also a generic _TRAILING_NOISE_PATTERN
    # keyword (for stripping bleed from OTHER fields), that leftover
    # "Pendaftar" then nuked the rest of the value as if it were noise.
    "nama_pendaftar": re.compile(r"^\s*(nama\s*pendaftar\s*(/\s*pen\s*\.?\s*)?(pendaftar)?\s*:?\s*)", re.IGNORECASE),
    "nama_wali": re.compile(r"^\s*(\d\s*\.\s*)?(nama\s*wali\s*:?\s*)", re.IGNORECASE),
    # (per)? -- the legacy Borang 3A form prints "Perhubungan", not "Hubungan"
    # (modern's wording); without it this label was never stripped at all on
    # legacy samples, since "Perhubungan" has no word boundary before
    # "hubungan" for a plain "hubungan" pattern to match.
    "hubungan_wali": re.compile(r"^\s*(per)?(hubungan\s*:?\s*)", re.IGNORECASE),
    # (saksi\s*1)? optional, not required -- confirmed on real modern-nikah
    # samples that the printed label for each witness's own row is a bare
    # "Nama :", with "SAKSI PERTAMA"/"SAKSI KEDUA" only appearing once, as a
    # separate section header above both witnesses' Nama/IC/Umur rows, never
    # repeated per-field. Requiring "saksi 1"/"saksi 2" literally here meant
    # this pattern never matched real "Nama : <name>" text at all, so the
    # label was never stripped -- and since bare NAMA is also a generic
    # _TRAILING_NOISE_PATTERN keyword (added to strip *bleed* of a
    # neighbouring Nama row into other fields), that keyword then wiped the
    # witness's own name out entirely on every single record.
    "saksi_1": re.compile(r"^\s*(\(\s*i\s*\)\s*)?(nama\s*(saksi\s*1)?\s*:?\s*)", re.IGNORECASE),
    "saksi_2": re.compile(r"^\s*(\(\s*ii\s*\)\s*)?(nama\s*(saksi\s*2)?\s*:?\s*)", re.IGNORECASE),
    "tarikh_nikah": re.compile(r"^\s*(tarikh\s*nikah\s*:?\s*)", re.IGNORECASE),
    # Both phrasings seen on real samples: "Tarikh Nikah Hijrah : ..." and
    # "Tarikh Nikah : Hijrah ...". Needs its own label (rather than falling
    # through with field_key=None) because "TARIKH NIKAH" is also a generic
    # _TRAILING_NOISE_PATTERN keyword -- without stripping it here first,
    # that keyword wiped this field's own correctly-labelled value to
    # nothing, the same self-erasure already fixed for no_siri/nama_pendaftar.
    "tarikh_nikah_hijri": re.compile(r"^\s*(tarikh\s*nikah\s*(hijrah)?\s*:?\s*(hijrah\s*:?\s*)?)", re.IGNORECASE),
    "alamat_pendaftar": re.compile(r"^\s*(alamat\s*:?\s*)", re.IGNORECASE),
    "mas_kahwin": re.compile(r"^\s*(mas\s*kahwin\s*:?\s*)", re.IGNORECASE),
    # Added for typed Nikah's legacy/modern split -- these fields are new
    # this pass and were falling through with no label stripped at all
    # (normalize_plain_text/normalize_address were being called without a
    # field_key at all in _build_nikah_typed_record, so none of these ever
    # took effect; fixed there too).
    "bangsa_suami": re.compile(r"^\s*(bangsa\s*:?\s*)", re.IGNORECASE),
    "bangsa_isteri": re.compile(r"^\s*(bangsa\s*:?\s*)", re.IGNORECASE),
    "warganegara_suami": re.compile(r"^\s*(warganegara\s*:?\s*)", re.IGNORECASE),
    "warganegara_isteri": re.compile(r"^\s*(warganegara\s*:?\s*)", re.IGNORECASE),
    # \(? -- a "(Ketua Pendaftar)" rubber stamp is printed overlapping the
    # Isteri/Wali address rows on real client samples, and Vision sometimes
    # reads its opening "(" as sitting BEFORE "Alamat" rather than after it
    # ("( Alamat : ENDAFTAR ) LUBUK BAKAK ..."); without tolerating that
    # leading "(" here, the label pattern never matched at all, so "Alamat"
    # itself survived as unstripped text and then matched _TRAILING_NOISE_
    # PATTERN's own bare ALAMAT keyword -- wiping the field's entire real
    # value as if it were a bled-in neighbouring row. (alamat\s*){1,2} --
    # Vision also sometimes doubles the label itself ("Alamat Alamat :"),
    # which a single-occurrence match left a leftover "Alamat :" prefix
    # that triggered that same self-erasure.
    "alamat_suami": re.compile(r"^\s*\(?\s*(alamat\s*){1,2}(rumah)?\s*:?\s*", re.IGNORECASE),
    "alamat_isteri": re.compile(r"^\s*\(?\s*(alamat\s*){1,2}(rumah)?\s*:?\s*", re.IGNORECASE),
    # Typed Cerai/Rujuk's printed label is "Alamat (Rumah):" -- confirmed on
    # real client samples, the field's own region boundary sometimes clips
    # everything except the label's closing ") :" tail, leaving that bare
    # fragment glued onto the front of the real address with no "alamat"
    # text present at all for the pattern above to match against (e.g.
    # ") : NO 11 , JLN SPEKTRUM U16 / 30 , ..."). See
    # _ADDRESS_RUMAH_LABEL_TAIL_BLEED below, applied in _strip_label.
    "alamat_wali": re.compile(r"^\s*\(?\s*(alamat\s*){1,2}(pejabat)?\s*:?\s*", re.IGNORECASE),
    "tempat_nikah": re.compile(r"^\s*(tempat\s*:?\s*)", re.IGNORECASE),
    "pernikahan_kali": re.compile(r"^\s*(pernikahan\s*kali\s*:?\s*)", re.IGNORECASE),
    # (isteri\s*)? -- "Pernikahan Kali : ... Isteri ke : ..." is one printed
    # row; when Vision splits it across two joined lines, the wrap can start
    # mid-label at bare "ke :" with "Isteri" left on the line above, so the
    # full-phrase-only pattern never matched at all.
    "isteri_ke": re.compile(r"^\s*((isteri\s*)?ke\s*:?\s*)", re.IGNORECASE),
    # No literal "hari"/"masa" text precedes these on a bled-in continuation
    # line (e.g. "1441 Hari : JUMAAT") -- non-greedy so it also strips a
    # neighbouring field's tail sitting in front of the real label.
    "hari_nikah": re.compile(r"^\s*(.*?\bhari\b\s*:?\s*)", re.IGNORECASE),
    "masa_nikah": re.compile(r"^\s*(.*?\bmasa\b\s*:?\s*)", re.IGNORECASE),
    # Vision's own printed label repeats the field name twice on typed
    # Nikah's "No. Siri : ######" stamp/header; without stripping it here,
    # the leftover "No. Siri" text then matched _TRAILING_NOISE_PATTERN's
    # own "NO. SIRI" keyword (added there to strip *bleed* of this label
    # into OTHER fields) and erased the real serial along with it.
    "no_siri": re.compile(r"^\s*(no\.?\s*sir[in]?\s*:?\s*)", re.IGNORECASE),
    # (belanja\s*)? -- Vision OCR's line-grouping sometimes splits "Belanja"
    # onto a different line than "Hantaran ..." (confirmed on a real legacy
    # sample), which used to defeat the whole match since it previously
    # required both words together.
    "belanja_hantaran": re.compile(r"^\s*((belanja\s*)?hantaran\s*:?\s*)", re.IGNORECASE),
    "pemberian_lain": re.compile(r"^\s*(pemberian\s*lain\s*(\(\s*jika\s*ada\s*\))?\s*:?\s*)", re.IGNORECASE),
}
_TRAILING_NOISE_PATTERN = re.compile(
    r"\b(?:"
    r"NO\.?\s*SIRI|NO\.?\s*SIN|NO\.?\s*KAD\s*PENGENALAN|KAD\s*PENGENALAN|PASPORT|"
    # NAMA bare (not just "NAMA SAKSI"/"NAMA WALI") -- Nikah's widened
    # regions (see NIKAH_LEGACY_REGIONS/NIKAH_MODERN_REGIONS) routinely
    # bleed in a whole neighbouring "N. Nama Isteri/Suami/Wali ..." row
    # ahead of or after a field's real value; that entire bled-in line
    # needs to disappear (not just have its own label stripped), and a
    # line that becomes empty after this is dropped by normalize_wrapped_
    # field/normalize_remarks's non-empty-line filter.
    r"UMUR|BANGSA|WARGANEGARA|ALAMAT|PENDAFTAR|(?:PER)?HUBUNGAN|NAMA|"
    r"SAKSI\s+PERTAMA|SAKSI\s+KEDUA|"
    # MASIHI/HIJRAH bare -- typed Nikah modern's "Tarikh Nikah : Hijrah ...
    # Masihi <date>" row sits directly above tempat_nikah's widened region
    # (see NIKAH_MODERN_REGIONS) and bleeds in as its own whole line; these
    # calendar-system labels never appear inside real address/name content.
    r"MASIHI|HIJRAH|"
    # HARI bare -- tarikh_nikah_hijri and hari_nikah share one printed row
    # ("... Hijrah ____ Hari ____ Masa : ..."), so a blank Hijri value (real
    # on some samples) leaves just the next section's bare "Hari" label
    # behind as if it were content; hari_nikah's own leading-label pattern
    # already consumes "hari" from the FRONT of its own line before this
    # runs, so this only ever strips a bled-in trailing occurrence.
    # TEMPAT bare -- confirmed on a real client sample: tarikh_nikah_hijri's
    # region bled in tempat_nikah's whole "Tempat : PEJABAT AGAMA ISLAM
    # DAERAH SABAK" row from directly below when the Hijri date itself was
    # blank, the same "next section's row survives when this one is blank"
    # pattern already fixed for HARI below. (?!\s*\d) -- confirmed on a
    # real Thai/Indonesian client address ("NO . 15 TEMPAT 2 , MUKIM
    # TUJUNG") that "Tempat" is also a genuine Malay/Indonesian address
    # word (a lot/block number), always followed there by a bare digit;
    # the real bled-in "Tempat Nikah"/"Tempat :" label is never followed
    # by one, so this excludes only that address usage.
    r"TARIKH\s*LAHIR|TARIKH\s*MASUK\s*ISLAM|TARIKH\s*NIKAH|TARIKH\s*DAFTAR|HARI|TEMPAT(?!\s*\d)|"
    # ISTERI bare, not just "ISTERI KE" -- pernikahan_kali's own row can wrap
    # so only "Isteri" (no "ke") lands at its tail (confirmed on a real
    # sample); by the time isteri_ke's own leading-label pattern runs on its
    # own line, "Isteri" is already stripped from the front there, so
    # broadening this to bare ISTERI doesn't touch that field's own value.
    # KAHWIN bare, not just "MAS KAHWIN" -- same line-split issue as ISTERI
    # above ("Mas" and "Kahwin ... RM80.00" landing on separate joined
    # lines on a real legacy sample), which let mas_kahwin's own value
    # survive unstripped and get mistaken for belanja_hantaran's.
    # MASA bare -- hari_nikah and masa_nikah were widened to share the full
    # row width (see NIKAH_MODERN_REGIONS), so hari_nikah's own capture now
    # routinely includes the "Masa : <time>" tail sitting right after the
    # day name on the same printed line.
    # (?!\s*KE\b) -- confirmed on a real client sample that isteri_ke's own
    # value sometimes reads as the combined phrase "Pernikahan Kali Ke 3"
    # rather than a bare ordinal word; without this exclusion, this keyword
    # (meant to strip a bled-in "Pernikahan Kali" row from OTHER fields)
    # wiped that entire line -- including the "Ke 3" that made it isteri_
    # ke's own real content -- before the ordinal-word/digit selection
    # logic in _select_best_line ever got a candidate to choose.
    r"PERNIKAHAN\s*KALI(?!\s*KE\b)|ISTERI(?:\s*KE)?|(?:MAS\s*)?KAHWIN|MASA|BELANJA\s*HANTARAN|PEMBERIAN\s*LAIN|"
    r"JUMLAH\s*BAYARAN|TANDATANGAN"
    r")\b.*$",
    re.IGNORECASE,
)
# Vision occasionally reads a printed tick mark or a dotted fill-in-the-blank
# line (both purely visual, not text) as trailing characters -- a lone
# checkmark/cross is never legitimate content, and a *run* of 2+ dots/dashes
# is a form's underline rather than punctuation (unlike a single trailing
# abbreviation dot, e.g. "P.P.T.", which this leaves alone).
_TRAILING_SYMBOL_NOISE = re.compile(r"(?:[✓✗]+|[.\-–—_]{2,})\s*$")
# A bare, dangling "No." (or "No", or "No. :" with the label's own trailing
# colon and nothing after it) at the very end of a line -- typed Nikah's own
# "No. Siri : ######" stamp repeats down the page's right margin (see the
# module-level comments on NIKAH_MODERN_REGIONS' no_siri/tarikh_daftar/
# alamat_wali entries), and a field's own right boundary sometimes lands
# between "No." and its serial, leaving just this fragment behind --
# confirmed on a real client sample where this bled into nama_isteri as a
# trailing "No. :". Anchored at end-of-line so it never touches a real
# in-progress house number like "NO. 19 , JLN ..." (there is always more
# text after "No." there).
_TRAILING_BARE_NO_PATTERN = re.compile(r"\(?\s*\bno\.?\s*:?\s*$", re.IGNORECASE)
# The same "No. Siri : ######" stamp bleed, but with the serial's digits
# still attached (so the line doesn't end right after "No." -- the pattern
# above only matches when nothing follows). Confirmed on several real
# client samples, each a different OCR misreading of "No. Siri": "Nc Sid
# 016213", "Na Siri 018945", "NC SH 016180", and a bare "No. 018964" with
# "Siri" dropped entirely. Requires the trailing digit run to be
# serial-length (4-7 digits) specifically so this can't fire on a real
# in-progress house number ("NO. 19 , JLN ...", 1-3 digits) or swallow a
# genuine name that happens to start with "Na"/"No" (e.g. "NASIR", "NOOR")
# -- those never have a 4+ digit run immediately after, so the required
# \d{4,7} simply won't be there to complete the match. Not anchored to
# end-of-line like the bare pattern above, since this bleed has shown up
# mid-line too (a name followed by this fragment followed by more text).
# \s*:?\s* (not just \s+) -- confirmed on a real client sample where this
# bled into nama_wali as "KAMSANI BIN SAMADI No. Sirt : 018973"; the stray
# colon between the label and its serial broke the old whitespace-only
# gap, leaving the trailing digits (and their -5 scoring penalty in
# _score_line_for_field) in place to make the real name lose to unrelated
# noise on another candidate line.
_STRAY_NO_SIRI_PATTERN = re.compile(r"\b(?:n[oac]\.?\s*s[a-z]{1,4}|no\.?)\s*:?\s*\d{4,7}\b", re.IGNORECASE)
# The "( KETUA PENDAFTAR )" registrar-stamp is a *bounded* parenthetical
# (a title, "KETUA" plus exactly one more word) that can land mid-line,
# not just at the end -- confirmed on a real modern sample where Vision
# merged it onto the same reading-order line as alamat_wali's own real
# second address line ("( KETUA PE JENIANG , KEDAH"). Unlike
# _TRAILING_NOISE_PATTERN's keywords (real bleed of a whole neighbouring
# field, safe to strip to end-of-line), stripping to end-of-line here would
# also eat that real trailing content, so this only removes the bounded
# "(KETUA <one word>)" span itself, truncated or not, wherever it sits.
_KETUA_STAMP_PATTERN = re.compile(r"\(?\s*ketua\s+[a-z]+\.?\s*\)?\s*", re.IGNORECASE)
# The same stamp, but with nothing legible after "KETUA" (its own following
# word/paren got cut off entirely, not just abbreviated) -- confirmed on
# three real client samples: "... SELANGOR CKETUA" (the stamp's leading "("
# misread as a stray "C" glued directly to the word), "... SELANGOR (
# KETUA" (the "(" read correctly, still nothing legible after "KETUA"), and
# a bare "... JALAN RIZAB KETUA" (no paren survived at all). _KETUA_STAMP_
# PATTERN above requires a word after "ketua" and matches none of these.
# Three alternatives, not one permissive pattern -- an earlier, looser
# version ("optional single letter, optional whitespace, then ketua") also
# matched into the *end of a real preceding word* ("JALAN RIZAB KETUA" ->
# "JALAN RIZA", eating the "B") since nothing stopped the stray-letter slot
# from grabbing it. Each alternative below is anchored to a genuinely
# distinct, non-overlapping shape: a lone letter with no word character
# before it (negative lookbehind) directly glued to "ketua"; an explicit
# "(" (with or without a following space); or "ketua" as its own word
# (\b), which requires a non-letter immediately before "k" and so cannot
# reach into "RIZAB".
_TRAILING_BARE_KETUA_PATTERN = re.compile(r"(?:(?<![a-z])[a-z]ketua|\(\s*ketua|\bketua)\s*$", re.IGNORECASE)
# The same "(Ketua Pendaftar)" stamp, but as a LEADING fragment on
# alamat_isteri/alamat_suami/alamat_wali specifically -- confirmed across
# roughly half of a real 20-document client batch, this stamp physically
# overlaps the Isteri/Wali address rows (never Suami's), so it lands right
# after (or, when it blocks the label match entirely, right before) the
# printed "Alamat :" label, or as a leading fragment on a wrapped
# continuation line. _KETUA_STAMP_PATTERN above requires the literal word
# "ketua" and only strips a *bounded* one-word span, which real samples
# defeated two ways: (1) the generic bare PENDAFTAR keyword in
# _TRAILING_NOISE_PATTERN ran first and wiped everything from "PENDAFTAR"
# to end-of-line -- including the real street address sitting right after
# it on the same line ("Alamat KETUA PENDAFTAR PARIT 13 SUNGAI PANJANG ,"
# -> only "45300 SUNGAI BESAR , SELANGOR" survived, the street silently
# dropped); (2) the stamp's own OCR misreading rarely spells "ketua"
# correctly at all (ETUA/TUA/TOA/NUA/CTUA/KETER/ETDA, missing the leading
# K or whole syllables), which the literal-only pattern above doesn't
# match, and the misspelled fragment then sat in front of the real address
# untouched. This is the same underlying self-erasure bug already fixed
# for no_siri/nama_pendaftar/saksi_1/saksi_2 (a keyword meant to strip a
# whole bled-in NEIGHBOURING row instead nukes real content when the
# field's OWN captured text happens to contain that keyword) -- applied
# here in _strip_label, before _strip_trailing_noise's generic keyword
# scan ever runs, so the real street/town text after the stamp fragment
# is never in the blast radius. Scoped to only the three alamat fields
# (never applied elsewhere) since "tua" is a real Malay word (as in
# "Kampung Tua") that would be unsafe to strip generically -- restricting
# the match to the leading position of an alamat field's own line, right
# where this stamp is confirmed to sit, keeps that safe.
_ADDRESS_STAMP_PREFIX = re.compile(
    r"^[(:\s]*(?:"
    # KETUA-variant token(s) followed by a recognised PENDAFTAR-variant --
    # both parts confirmed present, safe to consume regardless of what
    # follows (the real address text starts right after).
    r"(?:(?:ketua|cktua|cketua|ctua|etua|etda|tua|toa|nua|keter|ketga|ket)\.?\s*){1,2}"
    r"(?:pendaftar[a-z]*|pendaptari|pendaft|pendan|endaftar[a-z]*|p\s*aftary|dastari|datt)"
    r"|"
    # KETUA-variant token(s) with nothing else recognisable following --
    # either nothing at all (a lone "NUA"/"TUA" stamp fragment bled in as
    # its own whole line) or a postcode's digit run bled onto the same
    # line ("KETUA 68100 BATU CAVES ..."), both unambiguous since a real
    # trailing word would be letters, not end-of-line or a digit. Anything
    # else (a real trailing WORD this doesn't recognise, e.g. "KETUA PE
    # JENIANG , KEDAH", where "PE" is an abbreviation _KETUA_STAMP_PATTERN
    # below already handles correctly) is deliberately left alone here --
    # consuming just "KETUA " and stopping would strand "PE" unstripped,
    # since that removes _KETUA_STAMP_PATTERN's own "ketua" anchor.
    r"(?:(?:ketua|cktua|cketua|ctua|etua|etda|tua|toa|nua|keter|ketga|ket)\.?\s*){1,2}(?=\d|$)"
    r"|"
    # A PENDAFTAR-variant alone, with no KETUA-part in front -- Vision
    # merged the two into one bled-in line elsewhere on some samples, so
    # only the second half survives on THIS field's own line/continuation.
    r"(?:pendaftar[a-z]*|pendaptari|pendaft|pendan|endaftar[a-z]*|p\s*aftary|dastari|datt)"
    r")\s*\)?\s*",
    re.IGNORECASE,
)
# Typed Cerai/Rujuk's printed "Alamat (Rumah):" label -- confirmed on real
# client samples, the field's own region boundary sometimes clips everything
# except the label's closing ") :" tail, leaving that bare fragment glued
# onto the front of the real address with no "alamat" text present at all
# (e.g. ") : NO 11 , JLN SPEKTRUM U16 / 30 , ..."), so the _LEADING_LABELS
# pattern above (which requires the literal word "alamat") never matches.
# Scoped to only alamat_isteri/alamat_suami in _strip_label (never applied
# elsewhere) since a bare leading ") :" is this label's own specific tail,
# not a generally-safe-to-strip shape on other fields.
_ADDRESS_RUMAH_LABEL_TAIL_BLEED = re.compile(r"^\s*\)\s*:\s*")
# "Tarikh masuk Islam (Jika mualaf):" and "No.kad perakuan Islam:" are the
# next two printed labels immediately below alamat_isteri/alamat_suami on
# the real Rujuk form -- confirmed on real client samples, the "Tarikh
# masuk" lead-in gets clipped by this field's own region boundary, leaving
# "Islam ( Jika mualaf ) : No.kad perakuan Islam :" bled onto the tail of
# the real address. Scoped to only alamat_isteri/alamat_suami in
# _strip_trailing_noise (never added to the generic _TRAILING_NOISE_PATTERN
# keyword list) since a bare ISLAM keyword IS real content elsewhere, e.g.
# alamat_pendaftar's "PEJABAT AGAMA ISLAM DAERAH ...".
_ISLAM_LABEL_TAIL_BLEED = re.compile(r"\bislam\s*\(\s*jika\s*mualaf\s*\)\s*:?.*$", re.IGNORECASE)
_LOCATION_NOISE = (
    "DAERAH",
    "SELANGOR",
    "SUNGAI",
    "BESAR",
    "KAMPUNG",
    "PEJABAT",
    "AGAMA",
    "ISLAM",
    "WARGANEGARA",
    "BANGSA",
    "UMUR",
    "NO",
    "SIRI",
    "SIN",
    "TARIKH",
    "HIJRAH",
    "MASIHI",
    "PENDAFTAR",
    # Compass-direction words -- confirmed on a real client sample that a
    # street name's own direction suffix ("PARIT 7 1/2 BARAT") bled into
    # bangsa_isteri as its own stray line ("BARAT ,"), which then TIED the
    # real "MELAYU" answer on word count (both one word, no digits) and won
    # the tie-break on being the shorter string. These never appear as a
    # real Bangsa/Warganegara value, only as address bleed.
    "BARAT",
    "TIMUR",
    "UTARA",
    "SELATAN",
)
# Closed set of real Bangsa/Warganegara values seen on real client
# samples plus other common Malaysian ethnicities/nearby nationalities --
# used only as a positive scoring bonus (see _score_line_for_field), so an
# unlisted-but-genuine value simply doesn't get the bonus rather than being
# rejected outright.
_BANGSA_WARGANEGARA_VALUES = (
    "MELAYU",
    "CINA",
    "INDIA",
    "IBAN",
    "KADAZAN",
    "BAJAU",
    "MELANAU",
    "BIDAYUH",
    "MURUT",
    "JAWA",
    "BUGIS",
    "SIAM",
    "THAI",
    "ARAB",
    "PAKISTAN",
    "MYANMAR",
    "FILIPINO",
    "BANGLADESH",
    "MALAYSIA",
    "THAILAND",
    "INDONESIA",
    "SINGAPURA",
)
_NAME_HINTS = {
    "nama_suami": ("BIN", "BINTI", "HAJI", "HJ", "TUAN", "USTAZ"),
    "nama_isteri": ("BIN", "BINTI", "HAJI", "HJ", "PUAN", "CIK"),
    "nama_wali": ("BIN", "BINTI", "HAJI", "HJ", "TUAN", "USTAZ"),
    "nama_pendaftar": ("BIN", "BINTI", "HAJI", "HJ", "TUAN", "USTAZ"),
    "saksi_1": ("BIN", "BINTI", "HAJI", "HJ"),
    "saksi_2": ("BIN", "BINTI", "HAJI", "HJ"),
}


def normalize_bil(raw: str | None) -> str | None:
    if raw is None:
        return None
    match = BIL_PATTERN.search(str(raw))
    if not match:
        return None
    return re.sub(r"\s*/\s*", "/", match.group(0)).strip()


_OLD_IC_LETTER_PATTERN = re.compile(r"(?:[A-Z]{1,3}/[A-Z]{1,3}|[A-Z])[-\s./]*\d{5,7}")


def normalize_ic(raw: str | None) -> tuple[str | None, str | None]:
    if raw is None:
        return (None, None)
    text = str(raw).upper()
    new_ic_patterns = (
        re.compile(r"\b\d{6}[\s./-]*\d{2}[\s./-]*\d{4}\b"),
        re.compile(r"\b\d{12}\b"),
    )
    for pattern in new_ic_patterns:
        match = pattern.search(text)
        if match:
            digits = re.sub(r"\D", "", match.group(0))
            if len(digits) == 12:
                return (None, digits)
    # Old-format ICs are letter-prefixed (e.g. "A1192345", "R/F119395"), never a
    # bare digit run -- without this, e.g. a wali's or witness's old IC on a
    # typed form was silently dropped entirely (not even the digits survived,
    # since \b\d{7,8}\b never matches when the letter is directly attached
    # with no separator, which is the normal old-IC format).
    letter_match = _OLD_IC_LETTER_PATTERN.search(text)
    if letter_match:
        return (re.sub(r"[\s.]", "", letter_match.group(0)), None)
    old_match = re.search(r"\b\d{7,8}\b", text)
    if old_match:
        return (old_match.group(0), None)
    return (None, None)


def normalize_age(raw: str | None, *, min_age: int, max_age: int) -> int | None:
    if raw is None:
        return None
    lines = [line.strip() for line in str(raw).splitlines() if line.strip()]
    source = lines[0] if lines else ""
    match = re.search(r"\b(\d{1,3})\b", source)
    if not match:
        return None
    age = int(match.group(1))
    if min_age <= age <= max_age:
        return age
    return None


_DATE_CALENDAR_LABEL_PREFIX = re.compile(r"^\s*(masihi|hijrah)\s*:?\s*", re.IGNORECASE)


def _strip_calendar_label(line: str) -> str:
    # generate_date_candidates's own OCR-digit-confusion substitution (S->5,
    # I->1, applied to the whole string before any date parsing) turns the
    # printed label "Masihi" into garbage like "51 1" -- confirmed on a real
    # client sample where this left "Masihi 28.04.2009" with 5 groups after
    # splitting instead of 3, so no date candidate was ever generated at all
    # despite a perfectly clean date being right there. This label routinely
    # sits directly in front of the date it introduces (both Nikah and
    # Cerai/Rujuk templates print "Masihi"/"Hijrah" before their respective
    # calendar's date on the same line), so stripping it first is broadly
    # useful, not just for this one field.
    return _DATE_CALENDAR_LABEL_PREFIX.sub("", line)


def normalize_date_preserving_style(raw: str | None) -> str | None:
    if raw is None:
        return None

    for line in (part.strip() for part in str(raw).splitlines() if part.strip()):
        candidates = generate_date_candidates(_strip_calendar_label(line), field_name="date")
        if candidates:
            return candidates[0].value

    candidates = generate_date_candidates(_strip_calendar_label(str(raw).strip()), field_name="date")
    if candidates:
        return candidates[0].value

    return None


_YEAR_ONLY_PATTERN = re.compile(r"^(1[89]\d{2}|20\d{2})$")


def normalize_birth_year_or_date(raw: str | None) -> str | None:
    """Legacy Cerai/Rujuk certs (Borang 5/9) record only the birth *year* for
    Tarikh Lahir -- confirmed against real samples -- unlike every other
    date field on these forms, which is a full day.month.year. A bare year
    doesn't match generate_date_candidates's day/month/year parser, so fall
    back to accepting it verbatim rather than silently losing the data."""
    date_value = normalize_date_preserving_style(raw)
    if date_value is not None:
        return date_value
    if raw is None:
        return None
    for line in (part.strip() for part in str(raw).splitlines() if part.strip()):
        if _YEAR_ONLY_PATTERN.match(line):
            return line
    return None


def normalize_mas_kahwin(raw: str | None) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    match = MAS_KAHWIN_PATTERN.search(text)
    if not match:
        return None
    numeric = match.group(1).replace(" ", "").replace(",", "")
    try:
        amount = Decimal(numeric)
    except InvalidOperation:
        return None
    if "." in numeric:
        decimals = len(numeric.split(".", 1)[1])
    else:
        decimals = 0
    quantize_pattern = {0: "0", 1: "0.0", 2: "0.00"}[min(decimals, 2)]
    normalized = format(amount.quantize(Decimal(quantize_pattern)), "f")
    if "." in normalized:
        integer_part, decimal_part = normalized.split(".", 1)
        integer_part = f"{int(integer_part):,}"
        decimal_part = decimal_part[: min(decimals, 2)]
        if decimals == 0:
            return f"RM {integer_part}"
        return f"RM {integer_part}.{decimal_part}"
    return f"RM {int(normalized):,}"


# Generic alias for reuse on Cerai's bayaran_tebus_talak and the legacy
# forms' jumlah_bayaran -- same "RM or $" amount shape as mas_kahwin,
# normalized uniformly to "RM ..." regardless of which prefix the source
# form printed (Malaysia's own pre-Ringgit "$" notation on the 1984-
# enactment forms means the same currency, just an older symbol).
normalize_money = normalize_mas_kahwin


_LEADING_SECTION_NUMBER = re.compile(r"^\s*\(?\s*\d{1,2}\s*[.)]\s*")
# Typed Nikah modern's own page-section headers ("A. Maklumat Pasangan",
# "B. Maklumat Wali", "C. Maklumat Saksi", "D. Butir - Butir Pernikahan" --
# see NIKAH_MODERN_ANCHORS) -- confirmed on several real client samples
# that widening tempat_nikah's region far enough to fix its own drift bug
# (see NIKAH_MODERN_REGIONS) also reaches this anchor line, and unlike
# normalize_plain_text's best-line selection, normalize_wrapped_field JOINS
# every non-empty line rather than picking one, so this header survived
# glued onto the front of the real address ("D. BUTIR - BUTIR PERNIKAHAN
# PEJABAT AGAMA ISLAM DAERAH SABAK BERNAM , ..."). Never real content for
# any field, so stripped unconditionally the same way _LEADING_SECTION_
# NUMBER is above.
_LEADING_PAGE_SECTION_HEADER = re.compile(
    r"^\s*[a-d]\s*\.?\s*-?\s*(maklumat\s+pasangan|maklumat\s+wali|maklumat\s+saksi|butir\s*-?\s*butir\s+pernikahan)\s*",
    re.IGNORECASE,
)


def _strip_label(value: str, field_key: str) -> str:
    # Applied before the field-specific pattern, not just after -- Nikah's
    # typed regions are wide enough now (to tolerate cross-sample drift,
    # see NIKAH_LEGACY_REGIONS/NIKAH_MODERN_REGIONS) that a neighbouring
    # row's own leading "6." / "7." section number routinely bleeds into a
    # field's captured text; that prefix is never real content for any
    # field, so it comes off unconditionally rather than per-field.
    result = _LEADING_SECTION_NUMBER.sub("", value)
    result = _LEADING_PAGE_SECTION_HEADER.sub("", result)
    # A printed fill-in blank ("......... Bangsa : MELAYU") can sit BEFORE
    # the field's own label, not just after it -- e.g. isteri's row prints
    # a checkbox-style "*" placeholder ahead of "Umur", and once that's
    # stripped by _strip_trailing_noise the next line starts with a run of
    # dots before "Bangsa :". Every _LEADING_LABELS pattern anchors on
    # `^\s*`, so leading dots/dashes/underscores block the label pattern
    # from matching at all unless they're removed first.
    result = re.sub(r"^[\s.\-_]+", "", result)
    pattern = _LEADING_LABELS.get(field_key)
    if pattern is not None:
        result = pattern.sub("", result, count=1)
    if field_key in ("alamat_isteri", "alamat_suami", "alamat_wali"):
        # See _ADDRESS_STAMP_PREFIX -- strips a leading "(Ketua Pendaftar)"
        # stamp fragment (in whichever misspelling Vision produced) before
        # _strip_trailing_noise's generic PENDAFTAR/ALAMAT keywords get a
        # chance to run and wipe the real street text that follows it on
        # the same line.
        result = _ADDRESS_STAMP_PREFIX.sub("", result, count=1)
    if field_key in ("alamat_isteri", "alamat_suami"):
        # See _ADDRESS_RUMAH_LABEL_TAIL_BLEED -- strips a bare leading
        # ") :" tail of typed Cerai/Rujuk's own "Alamat (Rumah):" label.
        result = _ADDRESS_RUMAH_LABEL_TAIL_BLEED.sub("", result, count=1)
    # A leading colon only -- not every colon in the string. masa_nikah's
    # value can itself contain one ("9:30 PM"); a blanket replace used to
    # turn that into "9 30 PM" whenever the label pattern above left a
    # stray colon in front of it.
    result = re.sub(r"^\s*:\s*", "", result)
    # A label's own printed fill-in blank ("Nama Suami ....." / "Alamat ...")
    # is dots/dashes/underscores immediately after the label -- strip
    # whatever the field-specific pattern above left in place, the same way
    # _strip_trailing_noise already strips that same filler at the far end.
    result = re.sub(r"^[\s.\-_]+", "", result)
    result = re.sub(r"[ \t]+", " ", result)
    return result.strip()


def _strip_trailing_noise(value: str, field_key: str | None = None) -> str:
    value = _TRAILING_NOISE_PATTERN.sub("", value).strip()
    if field_key in ("alamat_isteri", "alamat_suami"):
        # See _ISLAM_LABEL_TAIL_BLEED -- strips the next two fields' own
        # bled-in labels ("Islam ( Jika mualaf ) : No.kad perakuan Islam :")
        # from the tail of the real address.
        value = _ISLAM_LABEL_TAIL_BLEED.sub("", value).strip()
    value = _TRAILING_SYMBOL_NOISE.sub("", value).strip()
    value = _TRAILING_BARE_NO_PATTERN.sub("", value).strip()
    # Skipped for no_siri itself -- "No <digits>" (or "No. Siri <digits>")
    # is that field's own legitimate content, not bleed to strip away, and
    # this runs before _select_best_line's own no_siri handling ever gets a
    # chance to choose between candidate lines.
    if field_key != "no_siri":
        value = _STRAY_NO_SIRI_PATTERN.sub("", value).strip()
    value = _KETUA_STAMP_PATTERN.sub("", value).strip()
    return _TRAILING_BARE_KETUA_PATTERN.sub("", value).strip()


def _score_line_for_field(line: str, field_key: str | None) -> tuple[int, int, int]:
    upper = line.upper()
    words = re.findall(r"[A-Z@']+", upper)
    score = len(words)
    if any(char.isdigit() for char in line):
        score -= 5
    if any(marker in upper for marker in _LOCATION_NOISE):
        score -= 3
    if any(marker in upper for marker in ("KAD", "PENGENALAN", "PASPORT")):
        # A No. Kad Pengenalan/Pasport label fragment is never a legitimate
        # value for any plain-text field -- confirmed on a real client
        # sample where Vision doubled this label's own words ("No. No. Kad
        # Kad Peng Pengenalan /"), which broke the exact-phrase
        # _TRAILING_NOISE_PATTERN match this would normally get fully
        # stripped by, leaving a 6-word garbage line that outscored the
        # real one-word "MALAYSIA" answer for warganegara_isteri on pure
        # word count. A strong fixed penalty (not proportional to how many
        # of the three words matched) since the doubling's exact shape is
        # unpredictable -- this only needs to reliably lose to a real
        # short answer, not model the noise precisely.
        score -= 8
    if field_key in _NAME_HINTS:
        score += sum(2 for hint in _NAME_HINTS[field_key] if hint in upper)
    if field_key == "hubungan_wali":
        # WALI deliberately excluded -- confirmed on a real client sample
        # that this field's own region also bled in a "Wali : <name>"
        # label fragment (a mis-split "Nama Wali :" row) sitting ahead of
        # the real "Hubungan : BAPA KANDUNG" line; rewarding bare "WALI"
        # let that noise line (short, but plus this bonus) outscore the
        # real one. "WALI HAKIM" (a real relationship value) still scores
        # fine on HAKIM alone, so this doesn't need WALI to be recognised.
        if any(token in upper for token in ("BAPA", "KANDUNG", "HAKIM")):
            score += 4
    if field_key in ("pernikahan_kali", "isteri_ke"):
        # These fields' real value is always a short ordinal word -- without
        # this, best-line selection's plain word-count scoring prefers a
        # longer neighbouring field's bleed (confirmed on a real sample: a
        # registrar's name outscored the correct "PERTAMA" on word count
        # alone once Nikah's widened regions started letting that bleed in).
        if any(token in upper for token in ("PERTAMA", "KEDUA", "KETIGA", "KEEMPAT", "KELIMA")):
            score += 10
        # A person's name is never a valid isteri_ke/pernikahan_kali value --
        # confirmed on a real client sample where isteri_ke came back as
        # nama_pendaftar's own full name ("MOHD YUSOF BIN MOHD TAHIR") with
        # no ordinal word anywhere in the candidates to outscore it.
        if any(hint in upper for hint in ("BIN", "BINTI", "HAJI", "HJ", "USTAZ", "TUAN")):
            score -= 6
    if field_key == "alamat_pendaftar":
        if any(token in upper for token in ("PEJABAT", "ALAMAT", "TEMPAT")):
            score += 3
        if any(char.isdigit() for char in line):
            score += 2
    if field_key in ("bangsa_isteri", "bangsa_suami", "warganegara_isteri", "warganegara_suami"):
        # A person's name is never a valid Bangsa/Warganegara value -- same
        # bug already fixed for pernikahan_kali/isteri_ke above. Confirmed
        # on two real client samples where a bled-in name fragment from the
        # row above ("BIN ABDUL LATIF", "MOHD HATA") outscored the real
        # one-word "MELAYU" answer on plain word count alone.
        if any(hint in upper for hint in ("BIN", "BINTI", "HAJI", "HJ", "USTAZ", "TUAN")):
            score -= 6
        # A bare given-name fragment with none of the markers above (e.g.
        # "MOHD HATA") still outscores a real one-word answer on plain word
        # count -- confirmed on a real client sample. Rather than guess at
        # every possible name shape, reward the actual closed set of
        # Bangsa/Warganegara values seen on real samples from this form
        # (both fields share this scorer, and "INDONESIA"/"THAI" are used
        # as both a Warganegara AND a Bangsa answer on real samples here).
        if any(token == upper for token in _BANGSA_WARGANEGARA_VALUES):
            score += 6
    return (score, len(words), -len(line))


def _select_best_line(lines: list[str], field_key: str | None) -> str | None:
    candidates = [line for line in lines if line]
    if not candidates:
        return None
    if field_key == "alamat_pendaftar":
        address_candidates = []
        for line in candidates:
            upper = line.upper()
            if any(marker in upper for marker in ("TARIKH", "HIJRAH", "MASIHI")):
                continue
            if any(token in upper for token in ("PEJABAT", "AGAMA", "ISLAM", "DAERAH", "SELANGOR", "SUNGAI", "KAMPUNG", "JALAN", "BESAR", "TAWAR", "LEMAN")) or any(char.isdigit() for char in line):
                address_candidates.append(line)
        if address_candidates:
            candidates = address_candidates
    elif field_key in ("no_siri", "tarikh_nikah_hijri"):
        # no_siri is a bare serial number -- the opposite of every other
        # plain_text field here, which is text and wants the no-digit
        # branch below. Without this, a widened region that picks up a
        # neighbouring watermark/stamp misread (confirmed on a real legacy
        # sample: raw text "AARAT\nNo 092113") loses the real value, since
        # the default branch actively prefers the digit-free garbage line.
        # tarikh_nikah_hijri's real value ("03 J'AWAL 1430" style) also has
        # digits -- confirmed on a real client sample where a bled-in,
        # digit-free "D. BUTIR - BUTIR PERNIKAHAN" section header outscored
        # the field's own correct (digit-containing) value this same way.
        digit_candidates = [line for line in candidates if any(char.isdigit() for char in line)]
        if digit_candidates:
            candidates = digit_candidates
    elif field_key in ("pernikahan_kali", "isteri_ke"):
        # This field's real value is always one of a closed set of ordinal
        # words -- prefer an ordinal-word candidate over everything else
        # FIRST, rather than filtering by digit presence the way no_siri/
        # tarikh_nikah_hijri do above (tried previously: confirmed on
        # several real client samples that once these two fields' regions
        # were widened enough to also pick up an address's own bled-in
        # postcode, the digit-preferring filter picked that digit-bearing
        # address line over the real, digit-free "PERTAMA" line outright --
        # excluding it before the ordinal-word scoring bonus ever got a
        # chance to run).
        ordinal_candidates = [
            line
            for line in candidates
            if any(token in line.upper() for token in ("PERTAMA", "KEDUA", "KETIGA", "KEEMPAT", "KELIMA"))
        ]
        if ordinal_candidates:
            candidates = ordinal_candidates
        else:
            # No spelled-out ordinal word anywhere -- fall back to
            # preferring a digit-bearing candidate the same way no_siri/
            # tarikh_nikah_hijri do (a real value can still be a bare
            # parenthetical digit once OCR drops the word itself, e.g.
            # "( 1 )", which the default no-digit branch below would
            # otherwise throw away in favour of an unrelated neighbour).
            digit_candidates = [line for line in candidates if any(char.isdigit() for char in line)]
            if digit_candidates:
                candidates = digit_candidates
        # A person's name is never a valid value here -- the -6 scoring
        # penalty above only helps when there's a *better* candidate to
        # prefer instead, but on a real client sample this field's only
        # candidate at all was a bled-in "MOHD YUSOF BIN MOHD TAHIR" (the
        # true value was genuinely blank on that document), so scoring
        # alone still picked it: max() over one candidate returns that
        # candidate regardless of its score. Drop any name-shaped,
        # non-ordinal candidate outright, unless doing so would leave
        # nothing at all to select from.
        name_free = [
            line
            for line in candidates
            if not any(hint in line.upper() for hint in ("BIN", "BINTI", "HAJI", "HJ", "USTAZ", "TUAN"))
        ]
        if name_free:
            candidates = name_free
    else:
        no_digit = [line for line in candidates if not any(char.isdigit() for char in line)]
        if no_digit:
            candidates = no_digit
    return max(candidates, key=lambda line: _score_line_for_field(line, field_key))


def normalize_plain_text(raw: str | None, *, field_key: str | None = None) -> str | None:
    if raw is None:
        return None
    lines = [line.strip() for line in str(raw).splitlines() if line.strip()]
    if not lines:
        return None
    cleaned_lines = []
    for line in lines:
        line = _strip_label(line, field_key)
        line = _strip_trailing_noise(line, field_key)
        line = re.sub(r"[ \t]+", " ", line).strip(" ,")
        # A line can reduce to a bare placeholder mark once its own real
        # content (e.g. "Umur : 64 Tahun") is stripped as bled-in noise from
        # a neighbouring field -- confirmed on a real sample where a
        # checkbox-style "*" preceded "Umur" on bangsa_isteri's own region,
        # leaving a lone "*" that would otherwise outrank the real answer on
        # the next line ("Bangsa : MELAYU") since _select_best_line has no
        # other candidate to prefer over a short non-empty string.
        if line and _PLACEHOLDER_LINE_PATTERN.match(line):
            continue
        if line:
            cleaned_lines.append(line)
    value = _select_best_line(cleaned_lines, field_key)
    if value is None:
        return None
    value = re.sub(r"\s+", " ", value).strip(" ,")
    if field_key in ("pernikahan_kali", "isteri_ke") and value:
        # Last-resort reject, after selection: on a real client sample this
        # field's only candidate at all was a bled-in registrar's name (the
        # true value was genuinely blank on that document), so there was
        # nothing for _select_best_line's filtering/scoring above to prefer
        # instead -- a wrong confident name is worse than admitting the
        # field is unreadable here.
        upper = value.upper()
        has_name_hint = any(hint in upper for hint in ("BIN", "BINTI", "HAJI", "HJ", "USTAZ", "TUAN"))
        has_ordinal = any(token in upper for token in ("PERTAMA", "KEDUA", "KETIGA", "KEEMPAT", "KELIMA"))
        if has_name_hint and not has_ordinal:
            return None
    return value or None


def normalize_name(raw: str | None, *, field_key: str | None = None) -> str | None:
    return normalize_plain_text(raw, field_key=field_key)


# Includes "*" -- typed Nikah modern's printed form uses a bare asterisk as
# a checkbox/reference mark ahead of some rows (e.g. before "Umur"); once
# that row's real content is stripped as bled-in noise, the lone "*" left
# behind is placeholder junk, not a candidate value.
_PLACEHOLDER_LINE_PATTERN = re.compile(r"^[.\-_:\s*]*$")


def normalize_remarks(raw: str | None) -> str | None:
    """For genuinely multi-line free-text remarks (e.g. typed Cerai/Rujuk's
    "Lain-lain Kenyataan" / "Kenyataan lain" blocks) -- unlike
    normalize_plain_text, which picks a single "best" line and is right for
    single-value fields (names), this joins every non-placeholder line so a
    remark spanning several sentences isn't silently collapsed to just one
    of them. The region for this field is deliberately generous (covers
    several blank dotted lines below any real remark) so most lines read
    back are pure placeholder underscores/dots -- those are dropped.
    """
    if raw is None:
        return None
    lines = [line.strip() for line in str(raw).splitlines() if line.strip()]
    cleaned_lines = [
        re.sub(r"\s+", " ", line).strip(" ,")
        for line in lines
        if not _PLACEHOLDER_LINE_PATTERN.match(line)
    ]
    cleaned_lines = [line for line in cleaned_lines if line]
    if not cleaned_lines:
        return None
    return " ".join(cleaned_lines)


# Cerai/Rujuk typed-certificate addresses can wrap across two printed lines
# (confirmed against a real Cerai modern sample) -- unlike borang_4b's
# single-line alamat_pendaftar, so join every real line the same way remarks
# do instead of normalize_plain_text's "pick one best line" behaviour, which
# would silently drop half the address.
normalize_address = normalize_remarks


def normalize_wrapped_field(raw: str | None, *, field_key: str | None = None) -> str | None:
    """Like normalize_remarks/normalize_address (joins every real line,
    rather than normalize_plain_text's "pick one best line" -- needed
    because typed Nikah's own addresses sometimes wrap a trailing state
    name onto its own line, confirmed on a real modern sample), but ALSO
    strips each line's own leading label and trailing next-field noise
    first via the same _strip_label/_strip_trailing_noise normalize_plain_text
    uses. Deliberately separate from normalize_address/normalize_remarks
    (used as-is by Cerai/Rujuk) rather than changing their behaviour --
    those fields' regions were calibrated tightly enough that no label ever
    lands in the captured text, so adding stripping there is unnecessary
    risk for no benefit; typed Nikah's regions are deliberately wider (to
    tolerate the layout variance between samples), so any given line here
    is more likely to carry a label or a neighbouring field's bleed that
    needs stripping before joining.
    """
    if raw is None:
        return None
    lines = [line.strip() for line in str(raw).splitlines() if line.strip()]
    cleaned_lines = []
    for line in lines:
        if _PLACEHOLDER_LINE_PATTERN.match(line):
            continue
        line = _strip_label(line, field_key)
        line = _strip_trailing_noise(line, field_key)
        line = re.sub(r"\s+", " ", line).strip(" ,")
        if line:
            cleaned_lines.append(line)
    if not cleaned_lines:
        return None
    return " ".join(cleaned_lines)

# The registrar's printed job title/office (e.g. "Pendaftar Perkahwinan,
# Perceraian Dan Rujuk Orang Islam" / "Pejabat Agama Islam Daerah Petaling")
# wraps across two or three printed lines on both Cerai and Rujuk modern
# certs (confirmed against real samples) -- same multi-line join as
# addresses, for the same reason.
_JAWATAN_PENDAFTAR_TITLE_START = re.compile(r"\bpendaftar\b", re.IGNORECASE)


def normalize_jawatan(raw: str | None) -> str | None:
    """The jawatan_pendaftar region on real Cerai/Rujuk samples sometimes
    also captures the registrar's own name sitting directly above or beside
    the job-title text (confirmed on several real client samples, e.g.
    "MOHD AZWAN BIN SELAMAT $ 3 Pendaftar Perkahwinan , Perceraian Dan Rujuk
    Orang Islam") -- the real job title always starts with the word
    "Pendaftar" ("Registrar"), so anything before its first occurrence in
    the joined text is the bled-in name, never part of the real title.
    """
    joined = normalize_remarks(raw)
    if joined is None:
        return None
    match = _JAWATAN_PENDAFTAR_TITLE_START.search(joined)
    if match:
        return joined[match.start():].strip(" ,") or None
    return joined


def _first_non_empty(values: Iterable[str | None]) -> str | None:
    for value in values:
        if value:
            return value
    return None


def _raw(raw_fields: dict[str, RawField], key: str) -> str | None:
    field = raw_fields.get(key)
    return field.raw_text if field else None


def build_extracted_record(raw_fields: dict[str, RawField], *, template_name: str = "borang_4b") -> ExtractedRecord:
    if template_name == "nikah_legacy" or template_name == "nikah_modern":
        return _build_nikah_typed_record(raw_fields, template_name=template_name)
    if template_name == "cerai_modern" or template_name == "cerai_legacy":
        return _build_cerai_record(raw_fields, template_name=template_name)
    if template_name == "rujuk_modern" or template_name == "rujuk_legacy":
        return _build_rujuk_record(raw_fields, template_name=template_name)
    return _build_nikah_record(raw_fields)


def _build_nikah_record(raw_fields: dict[str, RawField]) -> ExtractedRecord:
    bil = normalize_bil(raw_fields.get("bil").raw_text if raw_fields.get("bil") else None)

    nama_suami = normalize_name(raw_fields.get("nama_suami").raw_text if raw_fields.get("nama_suami") else None, field_key="nama_suami")
    ic_lama_suami, ic_baru_suami = normalize_ic(raw_fields.get("id_suami").raw_text if raw_fields.get("id_suami") else None)
    umur_suami = normalize_age(raw_fields.get("umur_suami").raw_text if raw_fields.get("umur_suami") else None, min_age=16, max_age=120)

    nama_isteri = normalize_name(raw_fields.get("nama_isteri").raw_text if raw_fields.get("nama_isteri") else None, field_key="nama_isteri")
    ic_lama_isteri, ic_baru_isteri = normalize_ic(raw_fields.get("id_isteri").raw_text if raw_fields.get("id_isteri") else None)
    umur_isteri = normalize_age(raw_fields.get("umur_isteri").raw_text if raw_fields.get("umur_isteri") else None, min_age=16, max_age=120)

    mas_kahwin = normalize_mas_kahwin(raw_fields.get("mas_kahwin").raw_text if raw_fields.get("mas_kahwin") else None)
    nama_pendaftar = normalize_plain_text(raw_fields.get("nama_pendaftar").raw_text if raw_fields.get("nama_pendaftar") else None, field_key="nama_pendaftar")
    alamat_pendaftar = normalize_plain_text(raw_fields.get("alamat_pendaftar").raw_text if raw_fields.get("alamat_pendaftar") else None, field_key="alamat_pendaftar")
    nama_wali = normalize_name(raw_fields.get("nama_wali").raw_text if raw_fields.get("nama_wali") else None, field_key="nama_wali")
    hubungan_wali = normalize_plain_text(raw_fields.get("hubungan_wali").raw_text if raw_fields.get("hubungan_wali") else None, field_key="hubungan_wali")
    saksi_1 = normalize_plain_text(raw_fields.get("saksi_1").raw_text if raw_fields.get("saksi_1") else None, field_key="saksi_1")
    saksi_2 = normalize_plain_text(raw_fields.get("saksi_2").raw_text if raw_fields.get("saksi_2") else None, field_key="saksi_2")
    tarikh_nikah = normalize_date_preserving_style(raw_fields.get("tarikh_nikah").raw_text if raw_fields.get("tarikh_nikah") else None)

    record = ExtractedRecord(
        bil=bil,
        nama_suami=nama_suami,
        ic_lama_suami=ic_lama_suami,
        ic_baru_suami=ic_baru_suami,
        id_suami_raw=raw_fields.get("id_suami").raw_text if raw_fields.get("id_suami") else None,
        umur_suami=umur_suami,
        nama_isteri=nama_isteri,
        ic_lama_isteri=ic_lama_isteri,
        ic_baru_isteri=ic_baru_isteri,
        id_isteri_raw=raw_fields.get("id_isteri").raw_text if raw_fields.get("id_isteri") else None,
        umur_isteri=umur_isteri,
        mas_kahwin=mas_kahwin,
        mas_kahwin_raw=raw_fields.get("mas_kahwin").raw_text if raw_fields.get("mas_kahwin") else None,
        nama_pendaftar=nama_pendaftar,
        alamat_pendaftar=alamat_pendaftar,
        nama_wali=nama_wali,
        hubungan_wali=hubungan_wali,
        saksi_1=saksi_1,
        saksi_2=saksi_2,
        tarikh_nikah=tarikh_nikah,
        tarikh_nikah_raw=raw_fields.get("tarikh_nikah").raw_text if raw_fields.get("tarikh_nikah") else None,
        tarikh_keluar=None,
        tarikh_keluar_raw=None,
        raw_bil=raw_fields.get("bil").raw_text if raw_fields.get("bil") else None,
        raw_suami_isteri=_first_non_empty(
            [
                raw_fields.get("nama_suami").raw_text if raw_fields.get("nama_suami") else None,
                raw_fields.get("id_suami").raw_text if raw_fields.get("id_suami") else None,
                raw_fields.get("nama_isteri").raw_text if raw_fields.get("nama_isteri") else None,
                raw_fields.get("id_isteri").raw_text if raw_fields.get("id_isteri") else None,
            ]
        ),
        raw_pendaftar=_first_non_empty(
            [
                raw_fields.get("nama_pendaftar").raw_text if raw_fields.get("nama_pendaftar") else None,
                raw_fields.get("alamat_pendaftar").raw_text if raw_fields.get("alamat_pendaftar") else None,
            ]
        ),
        raw_wali=raw_fields.get("nama_wali").raw_text if raw_fields.get("nama_wali") else None,
        raw_hubungan_wali=raw_fields.get("hubungan_wali").raw_text if raw_fields.get("hubungan_wali") else None,
        raw_saksi=_first_non_empty(
            [
                raw_fields.get("saksi_1").raw_text if raw_fields.get("saksi_1") else None,
                raw_fields.get("saksi_2").raw_text if raw_fields.get("saksi_2") else None,
            ]
        ),
        raw_tarikh_nikah=raw_fields.get("tarikh_nikah").raw_text if raw_fields.get("tarikh_nikah") else None,
        raw_tarikh_keluar=None,
        raw_remarks=None,
    )
    return record


def _build_nikah_typed_record(raw_fields: dict[str, RawField], *, template_name: str) -> ExtractedRecord:
    """Shared by nikah_legacy (Borang 3A) and nikah_modern (Borang 4B) -- same
    pattern as _build_cerai_record/_build_rujuk_record below: one field set
    covering both eras' superset, _raw() no-ops for whichever fields the
    other era's REGIONS dict doesn't define. Deliberately separate from the
    original _build_nikah_record above (kept untouched for borang_4b
    backward compatibility) rather than merged into it, since that one's
    field set and region key names (e.g. "id_suami" feeding split ic_lama/
    ic_baru only, no bangsa/warganegara/alamat/wali-IC/saksi-IC at all)
    predate and don't match either real layout.

    id_wali/id_saksi_1/id_saksi_2 go through normalize_ic() and take
    ic_baru-or-ic_lama the same way ic_suami/ic_isteri do below in
    _build_cerai_record, even though ExtractedRecord stores them as single
    fields (ic_wali/ic_saksi_1/ic_saksi_2, no lama/baru split) -- that
    normalizer already handles both the old short-form and modern 12-digit
    IC formats robustly, which a plain text passthrough would not.

    Legacy's tarikh_lahir_suami/tarikh_lahir_isteri (the form prints a
    birthdate, not an age) reuses the same shared field Cerai/Rujuk's typed
    forms already populate for the identical reason -- umur_suami/
    umur_isteri stay null for legacy, exactly as they already do for typed
    Cerai/Rujuk legacy samples that show a birthdate instead of an age.
    """
    ic_lama_suami, ic_baru_suami = normalize_ic(_raw(raw_fields, "id_suami"))
    ic_lama_isteri, ic_baru_isteri = normalize_ic(_raw(raw_fields, "id_isteri"))
    ic_lama_wali, ic_baru_wali = normalize_ic(_raw(raw_fields, "id_wali"))
    ic_wali = ic_baru_wali or ic_lama_wali
    ic_lama_saksi_1, ic_baru_saksi_1 = normalize_ic(_raw(raw_fields, "id_saksi_1"))
    ic_saksi_1 = ic_baru_saksi_1 or ic_lama_saksi_1
    ic_lama_saksi_2, ic_baru_saksi_2 = normalize_ic(_raw(raw_fields, "id_saksi_2"))
    ic_saksi_2 = ic_baru_saksi_2 or ic_lama_saksi_2

    record = ExtractedRecord(
        record_type="NIKAH",
        bil=normalize_bil(_raw(raw_fields, "bil")),
        no_siri=normalize_plain_text(_raw(raw_fields, "no_siri"), field_key="no_siri"),
        tarikh_daftar=normalize_date_preserving_style(_raw(raw_fields, "tarikh_daftar")),
        nama_suami=normalize_name(_raw(raw_fields, "nama_suami"), field_key="nama_suami"),
        ic_lama_suami=ic_lama_suami,
        ic_baru_suami=ic_baru_suami,
        id_suami_raw=_raw(raw_fields, "id_suami"),
        umur_suami=normalize_age(_raw(raw_fields, "umur_suami"), min_age=16, max_age=120),
        tarikh_lahir_suami=normalize_birth_year_or_date(_raw(raw_fields, "tarikh_lahir_suami")),
        warganegara_suami=normalize_plain_text(_raw(raw_fields, "warganegara_suami"), field_key="warganegara_suami"),
        bangsa_suami=normalize_plain_text(_raw(raw_fields, "bangsa_suami"), field_key="bangsa_suami"),
        alamat_suami=normalize_wrapped_field(_raw(raw_fields, "alamat_suami"), field_key="alamat_suami"),
        nama_isteri=normalize_name(_raw(raw_fields, "nama_isteri"), field_key="nama_isteri"),
        ic_lama_isteri=ic_lama_isteri,
        ic_baru_isteri=ic_baru_isteri,
        id_isteri_raw=_raw(raw_fields, "id_isteri"),
        umur_isteri=normalize_age(_raw(raw_fields, "umur_isteri"), min_age=16, max_age=120),
        tarikh_lahir_isteri=normalize_birth_year_or_date(_raw(raw_fields, "tarikh_lahir_isteri")),
        warganegara_isteri=normalize_plain_text(_raw(raw_fields, "warganegara_isteri"), field_key="warganegara_isteri"),
        bangsa_isteri=normalize_plain_text(_raw(raw_fields, "bangsa_isteri"), field_key="bangsa_isteri"),
        alamat_isteri=normalize_wrapped_field(_raw(raw_fields, "alamat_isteri"), field_key="alamat_isteri"),
        nama_wali=normalize_name(_raw(raw_fields, "nama_wali"), field_key="nama_wali"),
        ic_wali=ic_wali,
        umur_wali=normalize_age(_raw(raw_fields, "umur_wali"), min_age=16, max_age=120),
        hubungan_wali=normalize_plain_text(_raw(raw_fields, "hubungan_wali"), field_key="hubungan_wali"),
        alamat_wali=normalize_wrapped_field(_raw(raw_fields, "alamat_wali"), field_key="alamat_wali"),
        saksi_1=normalize_plain_text(_raw(raw_fields, "saksi_1"), field_key="saksi_1"),
        ic_saksi_1=ic_saksi_1,
        saksi_2=normalize_plain_text(_raw(raw_fields, "saksi_2"), field_key="saksi_2"),
        ic_saksi_2=ic_saksi_2,
        tarikh_nikah=normalize_date_preserving_style(_raw(raw_fields, "tarikh_nikah")),
        tarikh_nikah_raw=_raw(raw_fields, "tarikh_nikah"),
        tarikh_nikah_hijri=normalize_plain_text(_raw(raw_fields, "tarikh_nikah_hijri"), field_key="tarikh_nikah_hijri"),
        hari_nikah=normalize_plain_text(_raw(raw_fields, "hari_nikah"), field_key="hari_nikah"),
        masa_nikah=normalize_plain_text(_raw(raw_fields, "masa_nikah"), field_key="masa_nikah"),
        tempat_nikah=normalize_wrapped_field(_raw(raw_fields, "tempat_nikah"), field_key="tempat_nikah"),
        nama_pendaftar=normalize_plain_text(_raw(raw_fields, "nama_pendaftar"), field_key="nama_pendaftar"),
        pernikahan_kali=normalize_plain_text(_raw(raw_fields, "pernikahan_kali"), field_key="pernikahan_kali"),
        isteri_ke=normalize_plain_text(_raw(raw_fields, "isteri_ke"), field_key="isteri_ke"),
        mas_kahwin=normalize_mas_kahwin(_raw(raw_fields, "mas_kahwin")),
        mas_kahwin_raw=_raw(raw_fields, "mas_kahwin"),
        belanja_hantaran=normalize_plain_text(_raw(raw_fields, "belanja_hantaran"), field_key="belanja_hantaran"),
        pemberian_lain=normalize_plain_text(_raw(raw_fields, "pemberian_lain"), field_key="pemberian_lain"),
        jumlah_bayaran=normalize_money(_raw(raw_fields, "jumlah_bayaran")),
        raw_bil=_raw(raw_fields, "bil"),
        raw_suami_isteri=_first_non_empty([_raw(raw_fields, "nama_suami"), _raw(raw_fields, "id_suami")]),
        raw_pendaftar=_raw(raw_fields, "nama_pendaftar"),
        raw_wali=_raw(raw_fields, "nama_wali"),
        raw_hubungan_wali=_raw(raw_fields, "hubungan_wali"),
        raw_saksi=_first_non_empty([_raw(raw_fields, "saksi_1"), _raw(raw_fields, "saksi_2")]),
        raw_tarikh_nikah=_raw(raw_fields, "tarikh_nikah"),
    )
    return record


def _build_cerai_record(raw_fields: dict[str, RawField], *, template_name: str) -> ExtractedRecord:
    ic_lama_suami, ic_baru_suami = normalize_ic(_raw(raw_fields, "id_suami"))
    ic_lama_isteri, ic_baru_isteri = normalize_ic(_raw(raw_fields, "id_isteri"))

    record = ExtractedRecord(
        record_type="CERAI",
        bil=normalize_bil(_raw(raw_fields, "bil")),
        nama_suami=normalize_name(_raw(raw_fields, "nama_suami"), field_key="nama_suami"),
        ic_suami=ic_baru_suami or ic_lama_suami,
        ic_suami_raw=_raw(raw_fields, "id_suami"),
        bangsa_suami=normalize_plain_text(_raw(raw_fields, "bangsa_suami")),
        tarikh_lahir_suami=normalize_birth_year_or_date(_raw(raw_fields, "tarikh_lahir_suami")),
        warganegara_suami=normalize_plain_text(_raw(raw_fields, "warganegara_suami")),
        alamat_suami=normalize_address(_raw(raw_fields, "alamat_suami")),
        pekerjaan_suami=normalize_plain_text(_raw(raw_fields, "pekerjaan_suami")),
        nama_isteri=normalize_name(_raw(raw_fields, "nama_isteri"), field_key="nama_isteri"),
        ic_isteri=ic_baru_isteri or ic_lama_isteri,
        ic_isteri_raw=_raw(raw_fields, "id_isteri"),
        bangsa_isteri=normalize_plain_text(_raw(raw_fields, "bangsa_isteri")),
        tarikh_lahir_isteri=normalize_birth_year_or_date(_raw(raw_fields, "tarikh_lahir_isteri")),
        warganegara_isteri=normalize_plain_text(_raw(raw_fields, "warganegara_isteri")),
        alamat_isteri=normalize_address(_raw(raw_fields, "alamat_isteri")),
        pekerjaan_isteri=normalize_plain_text(_raw(raw_fields, "pekerjaan_isteri")),
        bil_daftar_nikah=normalize_plain_text(_raw(raw_fields, "bil_daftar_nikah")),
        bil_daftar_rujuk_asal=normalize_plain_text(_raw(raw_fields, "bil_daftar_rujuk_asal")),
        bilangan_kes_mal=normalize_plain_text(_raw(raw_fields, "bilangan_kes_mal")),
        no_sijil_perakuan_nikah_rujuk=normalize_plain_text(_raw(raw_fields, "no_sijil_perakuan_nikah_rujuk")),
        no_permohonan_cerai=normalize_plain_text(_raw(raw_fields, "no_permohonan_cerai")),
        tempat_nikah_daerah=normalize_plain_text(_raw(raw_fields, "tempat_nikah_daerah")),
        tempat_nikah_negeri=normalize_plain_text(_raw(raw_fields, "tempat_nikah_negeri")),
        tarikh_nikah=normalize_date_preserving_style(_raw(raw_fields, "tarikh_nikah")),
        tarikh_nikah_raw=_raw(raw_fields, "tarikh_nikah"),
        tarikh_nikah_hijri=normalize_plain_text(_raw(raw_fields, "tarikh_nikah_hijri")),
        tarikh_rujuk=normalize_date_preserving_style(_raw(raw_fields, "tarikh_rujuk")),
        tarikh_rujuk_raw=_raw(raw_fields, "tarikh_rujuk"),
        tarikh_rujuk_hijri=normalize_plain_text(_raw(raw_fields, "tarikh_rujuk_hijri")),
        keadaan_talak=normalize_plain_text(_raw(raw_fields, "keadaan_talak")),
        talak_kali_ke=normalize_plain_text(_raw(raw_fields, "talak_kali_ke")),
        jumlah_talak=normalize_plain_text(_raw(raw_fields, "jumlah_talak")),
        bayaran_tebus_talak=normalize_money(_raw(raw_fields, "bayaran_tebus_talak")),
        tempat_cerai=normalize_plain_text(_raw(raw_fields, "tempat_cerai")),
        tempat_bercerai=normalize_plain_text(_raw(raw_fields, "tempat_bercerai")),
        tarikh_cerai=normalize_date_preserving_style(_raw(raw_fields, "tarikh_cerai")),
        tarikh_cerai_raw=_raw(raw_fields, "tarikh_cerai"),
        tarikh_cerai_hijri=normalize_plain_text(_raw(raw_fields, "tarikh_cerai_hijri")),
        cerai_dalam_keadaan=normalize_plain_text(_raw(raw_fields, "cerai_dalam_keadaan")),
        saksi_1=normalize_plain_text(_raw(raw_fields, "saksi_1"), field_key="saksi_1"),
        saksi_2=normalize_plain_text(_raw(raw_fields, "saksi_2"), field_key="saksi_2"),
        jumlah_bayaran=normalize_money(_raw(raw_fields, "jumlah_bayaran")),
        hal_hal_lain=normalize_remarks(_raw(raw_fields, "hal_hal_lain")),
        nama_pendaftar=normalize_plain_text(_raw(raw_fields, "nama_pendaftar"), field_key="nama_pendaftar"),
        jawatan_pendaftar=normalize_jawatan(_raw(raw_fields, "jawatan_pendaftar")),
        tarikh_daftar=normalize_date_preserving_style(_raw(raw_fields, "tarikh_daftar")),
        tarikh_daftar_hijri=normalize_plain_text(_raw(raw_fields, "tarikh_daftar_hijri")),
        raw_bil=_raw(raw_fields, "bil"),
        raw_suami_isteri=_first_non_empty([_raw(raw_fields, "nama_suami"), _raw(raw_fields, "id_suami")]),
        raw_pendaftar=_raw(raw_fields, "nama_pendaftar"),
        raw_remarks=_raw(raw_fields, "hal_hal_lain"),
    )
    return record


def _build_rujuk_record(raw_fields: dict[str, RawField], *, template_name: str) -> ExtractedRecord:
    ic_lama_suami, ic_baru_suami = normalize_ic(_raw(raw_fields, "id_suami"))
    ic_lama_isteri, ic_baru_isteri = normalize_ic(_raw(raw_fields, "id_isteri"))

    record = ExtractedRecord(
        record_type="RUJUK",
        bil=normalize_bil(_raw(raw_fields, "bil")),
        nama_suami=normalize_name(_raw(raw_fields, "nama_suami"), field_key="nama_suami"),
        ic_suami=ic_baru_suami or ic_lama_suami,
        ic_suami_raw=_raw(raw_fields, "id_suami"),
        bangsa_suami=normalize_plain_text(_raw(raw_fields, "bangsa_suami")),
        tarikh_lahir_suami=normalize_birth_year_or_date(_raw(raw_fields, "tarikh_lahir_suami")),
        warganegara_suami=normalize_plain_text(_raw(raw_fields, "warganegara_suami")),
        alamat_suami=normalize_wrapped_field(_raw(raw_fields, "alamat_suami"), field_key="alamat_suami"),
        alamat_pejabat_suami=normalize_address(_raw(raw_fields, "alamat_pejabat_suami")),
        pekerjaan_suami=normalize_plain_text(_raw(raw_fields, "pekerjaan_suami")),
        tarikh_masuk_islam_suami=normalize_date_preserving_style(_raw(raw_fields, "tarikh_masuk_islam_suami")),
        no_kad_perakuan_islam_suami=normalize_plain_text(_raw(raw_fields, "no_kad_perakuan_islam_suami")),
        nama_isteri=normalize_name(_raw(raw_fields, "nama_isteri"), field_key="nama_isteri"),
        ic_isteri=ic_baru_isteri or ic_lama_isteri,
        ic_isteri_raw=_raw(raw_fields, "id_isteri"),
        bangsa_isteri=normalize_plain_text(_raw(raw_fields, "bangsa_isteri")),
        tarikh_lahir_isteri=normalize_birth_year_or_date(_raw(raw_fields, "tarikh_lahir_isteri")),
        warganegara_isteri=normalize_plain_text(_raw(raw_fields, "warganegara_isteri")),
        alamat_isteri=normalize_wrapped_field(_raw(raw_fields, "alamat_isteri"), field_key="alamat_isteri"),
        alamat_pejabat_isteri=normalize_address(_raw(raw_fields, "alamat_pejabat_isteri")),
        pekerjaan_isteri=normalize_plain_text(_raw(raw_fields, "pekerjaan_isteri")),
        tarikh_masuk_islam_isteri=normalize_date_preserving_style(_raw(raw_fields, "tarikh_masuk_islam_isteri")),
        no_kad_perakuan_islam_isteri=normalize_plain_text(_raw(raw_fields, "no_kad_perakuan_islam_isteri")),
        tempat_rujuk=normalize_plain_text(_raw(raw_fields, "tempat_rujuk")),
        rujuk_kali=normalize_plain_text(_raw(raw_fields, "rujuk_kali")),
        tarikh_rujuk=normalize_date_preserving_style(_raw(raw_fields, "tarikh_rujuk")),
        tarikh_rujuk_raw=_raw(raw_fields, "tarikh_rujuk"),
        tarikh_rujuk_hijri=normalize_plain_text(_raw(raw_fields, "tarikh_rujuk_hijri")),
        bil_daftar_nikah=normalize_plain_text(_raw(raw_fields, "bil_daftar_nikah")),
        tarikh_nikah=normalize_date_preserving_style(_raw(raw_fields, "tarikh_nikah")),
        tarikh_nikah_raw=_raw(raw_fields, "tarikh_nikah"),
        tarikh_nikah_hijri=normalize_plain_text(_raw(raw_fields, "tarikh_nikah_hijri")),
        bil_cerai=normalize_bil(_raw(raw_fields, "bil_cerai")),
        tarikh_cerai=normalize_date_preserving_style(_raw(raw_fields, "tarikh_cerai")),
        tarikh_cerai_raw=_raw(raw_fields, "tarikh_cerai"),
        tarikh_cerai_hijri=normalize_plain_text(_raw(raw_fields, "tarikh_cerai_hijri")),
        jumlah_bayaran=normalize_money(_raw(raw_fields, "jumlah_bayaran")),
        hal_hal_lain=normalize_remarks(_raw(raw_fields, "hal_hal_lain")),
        nama_pendaftar=normalize_plain_text(_raw(raw_fields, "nama_pendaftar"), field_key="nama_pendaftar"),
        jawatan_pendaftar=normalize_jawatan(_raw(raw_fields, "jawatan_pendaftar")),
        tarikh_daftar=normalize_date_preserving_style(_raw(raw_fields, "tarikh_daftar")),
        tarikh_daftar_hijri=normalize_plain_text(_raw(raw_fields, "tarikh_daftar_hijri")),
        raw_bil=_raw(raw_fields, "bil"),
        raw_suami_isteri=_first_non_empty([_raw(raw_fields, "nama_suami"), _raw(raw_fields, "id_suami")]),
        raw_pendaftar=_raw(raw_fields, "nama_pendaftar"),
        raw_remarks=_raw(raw_fields, "hal_hal_lain"),
    )
    return record
