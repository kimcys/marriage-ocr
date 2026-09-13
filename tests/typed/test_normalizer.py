import pytest

from marriage_ocr.typed.models import RawField, Region
from marriage_ocr.typed.normalizer import (
    build_extracted_record,
    normalize_age,
    normalize_bil,
    normalize_date_preserving_style,
    normalize_ic,
    normalize_mas_kahwin,
    normalize_plain_text,
    normalize_wrapped_field,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("04/2009 (SALINAN 1: 468/1984 S. BERNAM)", "04/2009"),
        ("330/2009 DAERAH SABAK BERNAM", "330/2009"),
        ("Bilangan Daftar: 32/2009", "32/2009"),
        # Regression: the oldest legacy Nikah forms (pre-2000, confirmed on a
        # real 1990 Borang 3A sample) print a 2-digit year, not the 4-digit
        # year every later form uses -- BIL_PATTERN used to require exactly
        # \d{4}, silently dropping this format entirely.
        ("1. Bilangan Daftar Nikah 384/90", "384/90"),
    ],
)
def test_normalize_bil_keeps_primary_number(raw: str, expected: str) -> None:
    assert normalize_bil(raw) == expected


def test_normalize_ic_classifies_old_and_new_numbers() -> None:
    assert normalize_ic("571018-10-5919") == (None, "571018105919")
    assert normalize_ic("6057990") == ("6057990", None)
    assert normalize_ic("12-34") == (None, None)


def test_normalize_ic_keeps_old_ic_letter_prefix() -> None:
    # Regression: old-format ICs are always letter-prefixed (e.g. "A1192345"),
    # never a bare digit run. Before this, normalize_ic only matched
    # \b\d{7,8}\b for the old-IC fallback, which never matches when a letter is
    # directly attached with no separator (the normal old-IC format) -- so a
    # wali's or witness's old IC on a typed form was dropped entirely, not just
    # missing its prefix.
    assert normalize_ic("A1192345") == ("A1192345", None)
    assert normalize_ic("No. Kad Pengenalan: A. 1192345") == ("A1192345", None)
    assert normalize_ic("R/F 119395") == ("R/F119395", None)


def test_normalize_date_normalizes_to_day_month_year() -> None:
    assert normalize_date_preserving_style("21.09.1984") == "21-09-1984"
    assert normalize_date_preserving_style(" 28 / 04 / 2009 ") == "28-04-2009"
    assert normalize_date_preserving_style("31.02.2009") is None
    assert normalize_date_preserving_style("25 ZULHIJJAH 1404\n21.09.1984") == "21-09-1984"


def test_normalize_mas_kahwin_preserves_rm_full_text() -> None:
    assert normalize_mas_kahwin("Mas Kahwin RM80.00") == "RM 80.00"
    assert normalize_mas_kahwin("RM1 500.00") == "RM 1,500.00"


def test_normalize_age_enforces_16_to_120() -> None:
    assert normalize_age("52 Tahun", min_age=16, max_age=120) == 52
    assert normalize_age("9 Tahun", min_age=16, max_age=120) is None
    assert normalize_age("Umur: 52 Tahun\nBangsa: MELAYU", min_age=16, max_age=120) == 52


def _raw(key: str, output_name: str, text: str) -> RawField:
    return RawField(key, output_name, 1, Region(0, 0, 1, 1), text, 0.95)


def test_build_extracted_record_maps_typed_fields() -> None:
    record = build_extracted_record(
        {
            "bil": _raw("bil", "Bil", "04/2009 (SALINAN 1: 468/1984)"),
            "nama_suami": _raw("nama_suami", "Nama Suami", "Nama: HENDON BIN MARIMIN"),
            "id_suami": _raw("id_suami", "IC Suami", "571018-10-5919"),
            "umur_suami": _raw("umur_suami", "Umur Suami", "52 Tahun"),
        }
    )

    assert record.bil == "04/2009"
    assert record.nama_suami == "HENDON BIN MARIMIN"
    assert record.ic_baru_suami == "571018105919"
    assert record.ic_lama_suami is None
    assert record.umur_suami == 52


def test_normalize_plain_text_strips_trailing_checkmark() -> None:
    # Regression: a printed tick mark on the form was being read by Vision
    # as a trailing "✓" character with no cleanup anywhere to remove it.
    assert normalize_plain_text("ABANG KANDUNG ✓", field_key="hubungan_wali") == "ABANG KANDUNG"


def test_normalize_plain_text_strips_trailing_dot_leader() -> None:
    # Regression: a dotted fill-in-the-blank line on the form was being
    # read as a run of trailing periods.
    assert normalize_plain_text("BAPA KANDUNG ..............", field_key="hubungan_wali") == "BAPA KANDUNG"


def test_normalize_plain_text_keeps_a_single_trailing_abbreviation_dot() -> None:
    # A lone trailing dot is real punctuation (an abbreviation), not a
    # form's dotted underline -- only *runs* of 2+ get stripped.
    assert normalize_plain_text("TUAN HAJI IDRIS BIN HAJI RAMLI , P.P.T.", field_key="nama_wali") == (
        "TUAN HAJI IDRIS BIN HAJI RAMLI , P.P.T."
    )


def test_normalize_name_strips_following_noise_lines() -> None:
    assert (
        build_extracted_record(
            {
                "nama_isteri": _raw(
                    "nama_isteri",
                    "Nama Isteri",
                    "ABIDAH BINTI HALIDI @ HAJI HALIDI No Sin 016173\n: 6057990 Umur : 49 Tahun",
                )
            }
        ).nama_isteri
        == "ABIDAH BINTI HALIDI @ HAJI HALIDI"
    )


# --- Typed Nikah legacy/modern region-widening regressions -----------------
#
# Every case below is a REAL raw_text capture from a real sample, seen only
# after NIKAH_LEGACY_REGIONS/NIKAH_MODERN_REGIONS were widened to tolerate
# cross-sample drift (see template.py's module comments) -- widening
# regularly lets a neighbouring row's label/value bleed into a field's own
# captured text, and each of these confirms the normalizer still recovers
# the correct value despite that bleed. Previously verified only via ad-hoc
# real-OCR-call testing in-session, not as committed, repeatable tests.


def test_no_siri_survives_its_own_label_being_captured() -> None:
    # Regression: "No. Siri" is also a generic _TRAILING_NOISE_PATTERN
    # keyword (for stripping *bleed* of this label into other fields) --
    # without no_siri's own leading-label pattern stripping it first, that
    # keyword self-erased the serial along with the label.
    assert normalize_plain_text("No. Siri : 018336", field_key="no_siri") == "018336"


def test_no_siri_prefers_the_digit_line_over_bled_in_garbage() -> None:
    # Regression: a widened region picked up a neighbouring watermark/stamp
    # misread on the same real sample ("AARAT"); best-line selection's
    # default digit-avoidance actively preferred that digit-free garbage
    # over the real serial. ("No 092113" survives whole here, unlike the
    # self-erasure case above, because this line has no "Siri"/"Sid" word
    # for no_siri's own leading-label pattern to strip.)
    assert normalize_plain_text("AARAT\nNo 092113", field_key="no_siri") == "No 092113"


def test_nama_pendaftar_survives_its_own_repeated_label_word() -> None:
    # Regression: the real printed label repeats "Pendaftar" twice ("Nama
    # Pendaftar / Pen . Pendaftar :"); a pattern stripping only the first
    # occurrence left "/ Pen . Pendaftar ..." in front of the name, and
    # PENDAFTAR being a generic trailing-noise keyword then erased the rest.
    assert (
        normalize_plain_text(
            "Nama Pendaftar / Pen . Pendaftar NIK MOHAMAD BIN NIK ABDULLAH",
            field_key="nama_pendaftar",
        )
        == "NIK MOHAMAD BIN NIK ABDULLAH"
    )
    assert (
        normalize_plain_text(
            "Nama Pendaftar / Pen . Pendaftar : TN HJ MOHD KAMAL BIN HJ ABDUL WAHAB",
            field_key="nama_pendaftar",
        )
        == "TN HJ MOHD KAMAL BIN HJ ABDUL WAHAB"
    )


def test_hubungan_wali_strips_legacy_perhubungan_label() -> None:
    # Regression: the legacy Borang 3A form prints "Perhubungan", not
    # "Hubungan" (modern's wording) -- a plain "hubungan" pattern has no
    # word boundary before it inside "Perhubungan" and never matched at all.
    assert normalize_plain_text("Perhubungan ... Wali Hakim", field_key="hubungan_wali") == "Wali Hakim"


def test_hubungan_wali_survives_a_bled_in_section_number() -> None:
    # Regression: widened regions routinely bleed in a neighbouring row's
    # own leading "6."/"7." section number; that prefix is never real
    # content for any field.
    assert normalize_plain_text("7. BAPA", field_key="hubungan_wali") == "BAPA"


def test_belanja_hantaran_survives_belanja_hantaran_split_across_lines() -> None:
    # Regression: Vision's line-grouping sometimes splits "Belanja" onto a
    # different line than "Hantaran ...", which used to defeat the whole
    # label match since it previously required both words together.
    assert normalize_plain_text("Hantaran ..... TIADA", field_key="belanja_hantaran") == "TIADA"


def test_belanja_hantaran_does_not_pick_up_mas_kahwins_bled_in_value() -> None:
    # Regression: "Mas" and "Kahwin ... RM80.00" landed on separate joined
    # lines on a real legacy sample; mas_kahwin's own value used to survive
    # unstripped (bare "Kahwin", not "Mas Kahwin") and get mistaken for
    # belanja_hantaran's, once its own genuinely-blank line vanished.
    assert normalize_plain_text("Kahwin ......... RM80.00\nBelanja Hantaran .....", field_key="belanja_hantaran") is None


def test_isteri_ke_handles_a_bare_ke_continuation_line() -> None:
    # Regression: "Pernikahan Kali : ... Isteri ke : ..." is one printed
    # row; when Vision splits it across two joined lines, the wrap can
    # start mid-label at bare "ke :" with "Isteri" left on the prior line,
    # so a full-phrase-only pattern never matched.
    assert normalize_plain_text("ke : PERTAMA ( 1 )", field_key="isteri_ke") == "PERTAMA ( 1 )"


def test_isteri_ke_is_not_outscored_by_a_bled_in_name() -> None:
    # Regression: isteri_ke's real value ("PERTAMA ( 1 )") contains a
    # digit, which best-line selection's default digit-avoidance used to
    # throw away in favour of a longer, digit-free neighbouring field's
    # bleed (here nama_pendaftar's tail).
    assert (
        normalize_plain_text("HJ MOHD KAMAL BIN HJ ABDUL WAHAB\nke : PERTAMA", field_key="isteri_ke") == "PERTAMA"
    )


def test_pernikahan_kali_ignores_bled_in_isteri_and_mas_kahwin_rows() -> None:
    # Regression: on a real sample, pernikahan_kali's own row wrapped with
    # a bare trailing "Isteri" (its "ke :" continuation captured
    # separately), and a second bled-in row ("Mas Kahwin : RM 80.00")
    # outscored the correct short ordinal value before the ordinal-word
    # bonus and bare-ISTERI/KAHWIN noise stripping existed.
    assert (
        normalize_plain_text(
            "Pernikahan Kali : PERTAMA ( 1 ) Isteri\nMas Kahwin : RM 80.00",
            field_key="pernikahan_kali",
        )
        == "PERTAMA ( 1 )"
    )


def test_hari_nikah_strips_a_bled_in_hijri_year_prefix() -> None:
    # Regression: tarikh_nikah_hijri and hari_nikah share one printed row
    # ("... Hijrah ____ Hari ____ Masa : ..."); a non-greedy leading-label
    # pattern is needed to strip the Hijri year sitting in front of "Hari"
    # on a bled-in continuation line.
    assert normalize_plain_text("1441 Hari : JUMAAT", field_key="hari_nikah") == "JUMAAT"


def test_masa_nikah_preserves_the_colon_inside_a_time_value() -> None:
    # Regression: _strip_label used to blanket-replace every colon with a
    # space, which mangled a genuine time value's own colon ("9:30 PM" ->
    # "9 30 PM") once the leading "Masa :" label was stripped in front of it.
    assert normalize_plain_text("Masa : 9:30 PM", field_key="masa_nikah") == "9:30 PM"


def test_tarikh_nikah_hijri_survives_its_own_label_being_captured() -> None:
    # Regression: "Tarikh Nikah" (this field's own label) is also a generic
    # trailing-noise keyword for stripping bleed of that phrase into other
    # fields -- without a dedicated leading-label pattern stripping it
    # first, the field erased its own value.
    assert (
        normalize_plain_text(
            "Tarikh Nikah Hijrah : 06 JUMADAL AKHIRAH 1441 Hari : JUMAAT",
            field_key="tarikh_nikah_hijri",
        )
        == "06 JUMADAL AKHIRAH 1441"
    )


def test_tempat_nikah_drops_a_bled_in_masihi_date_line() -> None:
    # Regression: "Tarikh Nikah : Hijrah ... Masihi <date>" sits directly
    # above tempat_nikah's widened region and bled in as its own whole
    # line; MASIHI/HIJRAH are calendar-system labels that never appear
    # inside real address content.
    assert (
        normalize_wrapped_field(
            "Masihi 31-01-2020\nTempat : MASJID ASY - SYAKIRIN , TAMAN PUCHONG UTAMA",
            field_key="tempat_nikah",
        )
        == "MASJID ASY - SYAKIRIN , TAMAN PUCHONG UTAMA"
    )


def test_alamat_wali_strips_a_truncated_ketua_stamp_without_losing_real_content() -> None:
    # Regression: the "( KETUA PENDAFTAR )" registrar stamp (truncated by
    # Vision here to "( KETUA PE") sits close enough vertically to
    # alamat_wali's own real second address line that Vision merges them
    # onto one reading-order line together. A naive "strip from KETUA to
    # end of line" fix would also have erased "JENIANG , KEDAH".
    assert (
        normalize_wrapped_field(
            "Alamat : KAMPUNG SUNGAI PAU\n( KETUA PE JENIANG , KEDAH\nNo.",
            field_key="alamat_wali",
        )
        == "KAMPUNG SUNGAI PAU JENIANG , KEDAH"
    )


# --- Regressions from a real client OneDrive batch --------------------------
#
# Found by actually running process-typed against a real client-supplied
# link (20 real modern Nikah certificates), not guessed -- each case below
# is the literal raw_text this pipeline captured on one of those documents.


def test_tarikh_nikah_hijri_drops_a_bled_in_tempat_row() -> None:
    # Regression: on a real client sample, tarikh_nikah_hijri's Hijri date
    # was genuinely blank, and tempat_nikah's whole row bled in from directly
    # below instead ("Tempat : PEJABAT AGAMA ISLAM DAERAH SABAK"). TEMPAT
    # wasn't a recognised trailing-noise keyword, so it survived as if it
    # were tarikh_nikah_hijri's own value.
    assert normalize_plain_text("Tempat : PEJABAT AGAMA ISLAM DAERAH SABAK", field_key="tarikh_nikah_hijri") is None


def test_nama_isteri_drops_a_bled_in_no_colon_fragment() -> None:
    # Regression: a real client sample's nama_isteri came back as "MARIAM
    # BIN DEROM No. :" -- a neighbouring field's "No. <label> :" bled in
    # with a trailing colon _TRAILING_BARE_NO_PATTERN didn't yet cover
    # (it only matched a bare "No."/"No", not one with ": " after it).
    assert normalize_plain_text("MARIAM BIN DEROM No. :", field_key="nama_isteri") == "MARIAM BIN DEROM"


def test_alamat_wali_drops_a_garbled_ketua_stamp_with_no_following_word() -> None:
    # Regression: a real client sample's alamat_wali ended in "... SELANGOR
    # CKETUA" -- the same registrar stamp as the KETUA_STAMP_PATTERN case
    # above, but here Vision misread the stamp's leading "(" as a "C" *and*
    # dropped everything after "KETUA", so neither existing KETUA pattern
    # (which both expect a following word) matched.
    assert (
        normalize_wrapped_field("PARIT 7 BARAT 45300 SUNGAI BESAR , SELANGOR CKETUA", field_key="alamat_wali")
        == "PARIT 7 BARAT 45300 SUNGAI BESAR , SELANGOR"
    )


def test_alamat_wali_drops_a_bare_parenthesised_ketua_with_no_following_word() -> None:
    # Regression: a second real client sample ended in "... SELANGOR (
    # KETUA" -- the stamp's own leading "(" read correctly this time, but
    # still nothing legible after "KETUA".
    assert (
        normalize_wrapped_field(
            "NO . 1 SURAU PARIT SERONG SIMPANG LIMA 45300 SUNGAI BESAR , SELANGOR ( KETUA",
            field_key="alamat_wali",
        )
        == "NO . 1 SURAU PARIT SERONG SIMPANG LIMA 45300 SUNGAI BESAR , SELANGOR"
    )


def test_alamat_wali_drops_a_bare_trailing_ketua_word() -> None:
    # Regression: a third real client sample ended in "... RIZAB KETUA" --
    # the stamp's parenthesis missing entirely this time, just the bare word.
    assert (
        normalize_wrapped_field("PARIT 9 TIMUR JALAN RIZAB KETUA", field_key="alamat_wali")
        == "PARIT 9 TIMUR JALAN RIZAB"
    )


def test_isteri_ke_rejects_a_bled_in_registrars_name_outright() -> None:
    # Regression: a real client sample's isteri_ke raw text was just
    # "MOHD YUSOF BIN MOHD TAHIR" -- nama_pendaftar's own name, with no
    # ordinal-word candidate anywhere to prefer instead (the true value was
    # genuinely blank on that document). A wrong confident name is worse
    # than admitting the field is unreadable.
    assert normalize_plain_text("MOHD YUSOF BIN MOHD TAHIR", field_key="isteri_ke") is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Each a different OCR misreading of "No. Siri <serial>" bleeding
        # into nama_isteri on a real client sample -- _TRAILING_BARE_NO_
        # PATTERN only covered a bare "No." with nothing after it; none of
        # these have "Siri" spelled correctly, or have the serial's digits
        # still directly attached.
        ("AINAH BINTI TARMAN Nc Sid 016213", "AINAH BINTI TARMAN"),
        ("SITI APSIAH BINTI DARWI Na Siri 018945", "SITI APSIAH BINTI DARWI"),
        ("MARIAM BIN DEROM No. 018964", "MARIAM BIN DEROM"),
        ("SITI AISHAH BINTI SAHAMIN NC SH 016180", "SITI AISHAH BINTI SAHAMIN"),
    ],
)
def test_nama_isteri_drops_stray_no_siri_ocr_misreadings(raw: str, expected: str) -> None:
    assert normalize_plain_text(raw, field_key="nama_isteri") == expected


def test_stray_no_siri_pattern_does_not_eat_real_names_or_house_numbers() -> None:
    # A name starting with "Na"/"No" is never followed by a 4+ digit run
    # immediately after in real content, and a genuine in-progress house
    # number is 1-3 digits -- both must survive untouched.
    assert normalize_plain_text("AHMAD NASIR BIN ALI", field_key="nama_suami") == "AHMAD NASIR BIN ALI"
    assert normalize_plain_text("MOHD NOOR BIN ISMAIL", field_key="nama_suami") == "MOHD NOOR BIN ISMAIL"


def test_bangsa_isteri_drops_a_leading_checkbox_mark_and_dotted_fill_in() -> None:
    # Regression: a real client sample's bangsa_isteri region captured "*
    # Umur : 64 Tahun" (a checkbox mark ahead of the bled-in Umur row) as
    # its first line and "......... Bangsa : MELAYU" as its second -- the
    # bare "*" that survived Umur's own strip used to win selection outright
    # since normalize_plain_text never filtered placeholder-only lines, and
    # separately the leading dots blocked bangsa_isteri's own label pattern
    # from matching "Bangsa :" underneath them.
    assert (
        normalize_plain_text("* Umur : 64 Tahun\n......... Bangsa : MELAYU", field_key="bangsa_isteri")
        == "MELAYU"
    )


@pytest.mark.parametrize(
    ("raw", "field_key", "expected"),
    [
        # A directional address suffix bled in as bangsa_isteri's own
        # second candidate line and tied "MELAYU" on word count, winning
        # the tie-break by being the shorter string.
        ("410512-10-5116 Umur : 68\nBangsa : MELAYU\nBARAT ,", "bangsa_isteri", "MELAYU"),
        # A bled-in name fragment with a BIN/BINTI/HAJI-style marker.
        ("BIN ABDUL LATIF\nUmur : 60 Tahun\nBangsa : ............. MELAYU", "bangsa_suami", "MELAYU"),
        # A bled-in given-name fragment with NO recognisable marker at all
        # -- only a closed-set allow-list bonus for real Bangsa/Warganegara
        # values can outscore this, since word count alone favours it.
        ("MOHD HATA\nUmur : 27 Tahun\nBangsa : MELAYU", "bangsa_suami", "MELAYU"),
    ],
)
def test_bangsa_is_not_outscored_by_a_bled_in_name_fragment(raw: str, field_key: str, expected: str) -> None:
    assert normalize_plain_text(raw, field_key=field_key) == expected


def test_hubungan_wali_is_not_outscored_by_a_bled_in_wali_name_label() -> None:
    # Regression: a real client sample's hubungan_wali region captured a
    # mis-split "Nama Wali :" row ("Wali : KAMSANI BIN SAMADI") ahead of the
    # real "Hubungan : BAPA KANDUNG" line -- the bare "WALI" scoring bonus
    # (meant to help recognise "WALI HAKIM") rewarded that noise line too,
    # letting it outscore the real answer.
    raw = (
        "Wali : KAMSANI BIN SAMADI\n"
        "No. Kad Pengenalan / Pasport : TIADA MAKLUMAT\n"
        "Hubungan : BAPA KANDUNG\n"
        "( Alamat : ENDAFTAR ) LUBUK BAKAK D5 PADANG CERMIN"
    )
    assert normalize_plain_text(raw, field_key="hubungan_wali") == "BAPA KANDUNG"


def test_nama_wali_survives_a_no_siri_bleed_with_a_colon_before_the_serial() -> None:
    # Regression: on a real client sample "No. Sirt : 018973" (a colon
    # between the label and its serial) bled onto the same line as the real
    # name; _STRAY_NO_SIRI_PATTERN required whitespace only there, so the
    # trailing digits survived and their -5 digit penalty let an unrelated
    # bled-in noise line ("B. MAKLUMAT WALI") win selection instead.
    raw = "B. MAKLUMAT WALI\nNama Wali : KAMSANI BIN SAMADI No. Sirt : 018973"
    assert normalize_plain_text(raw, field_key="nama_wali") == "KAMSANI BIN SAMADI"


def test_isteri_ke_survives_its_own_combined_pernikahan_kali_phrasing() -> None:
    # Regression: on two real client samples, isteri_ke's own printed value
    # read as the combined phrase "Pernikahan Kali Ke 3"/"Ke 2" rather than
    # a bare ordinal word. The generic PERNIKAHAN KALI bleed-strip keyword
    # (meant to remove a neighbouring field's whole row) matched this
    # field's own value and wiped it to nothing.
    raw = "SUNGAI BESAR , SELANGOR\nMOHD YUSOF BIN MOHD TAHIR\nIsteri ke : PERNIKAHAN KALI KE 3"
    assert normalize_plain_text(raw, field_key="isteri_ke") == "PERNIKAHAN KALI KE 3"


def test_isteri_ke_still_prefers_a_real_ordinal_word_over_bled_in_address_digits() -> None:
    # Regression: widening isteri_ke's region to fix a vertical drift bug
    # also let an address's own postcode digits into its candidate pool;
    # the old digit-preferring branch (shared with no_siri/tarikh_nikah_
    # hijri) then excluded the real, digit-free "PERTAMA" line outright.
    raw = ": Masa : 5.00 PTG\nDAERAH SABAK BERNAM , 45300 SUNGAI BESAR , SELANGOR\nHAJI IDRIS BIN HAJI RAMLI , P.P.T.\nIsteri ke : PERTAMA"
    assert normalize_plain_text(raw, field_key="isteri_ke") == "PERTAMA"


def test_tempat_nikah_drops_a_bled_in_page_section_header() -> None:
    # Regression: widening tempat_nikah's region to fix a vertical drift bug
    # also reached this page's own "D. Butir - Butir Pernikahan" section
    # anchor line; unlike normalize_plain_text's best-line selection,
    # normalize_wrapped_field joins every non-empty line, so this header
    # survived glued onto the front of the real address.
    raw = (
        "D. BUTIR - BUTIR PERNIKAHAN\n"
        "Tarikh Nikah : Hijrah : 03 J ' AWAL 1430 Hari : SABTU Masa : 5.00 PTG\n"
        "Masihi 28.04.2009\n"
        "Tempat : PEJABAT AGAMA ISLAM DAERAH SABAK BERNAM , 45300 SUNGAI BESAR , SELANGOR"
    )
    assert (
        normalize_wrapped_field(raw, field_key="tempat_nikah")
        == "PEJABAT AGAMA ISLAM DAERAH SABAK BERNAM , 45300 SUNGAI BESAR , SELANGOR"
    )


def test_alamat_isteri_recovers_a_street_hidden_behind_a_ketua_pendaftar_stamp() -> None:
    # Regression: a "(Ketua Pendaftar)" rubber stamp physically overlaps the
    # Isteri/Wali address rows on real client samples. The generic bare
    # PENDAFTAR bleed-strip keyword matched the stamp fragment and wiped
    # everything after it on the same line -- including the real street
    # text sitting right after the stamp, not just the stamp itself.
    assert (
        normalize_wrapped_field(
            "Alamat KETUA PENDAFTAR PARIT 13 SUNGAI PANJANG ,\n45300 SUNGAI BESAR , SELANGOR",
            field_key="alamat_isteri",
        )
        == "PARIT 13 SUNGAI PANJANG 45300 SUNGAI BESAR , SELANGOR"
    )


def test_alamat_isteri_recovers_a_street_behind_a_doubled_ketua_stamp_misread() -> None:
    # Regression: on a real sample the stamp's own OCR misreading doubled
    # ("Alamat Alamat : KETUA CTUA PENDAFTAR ..."), which a single-pass
    # label strip and a single KETUA-variant match both failed to fully
    # consume, leaving the field's own generic ALAMAT/PENDAFTAR keywords to
    # wipe the whole line as if it were entirely bled-in noise.
    assert (
        normalize_wrapped_field(
            "Alamat Alamat : KETUA CTUA PENDAFTAR PARIT 7 BARAT ,\n45300 SUNGAI BESAR , SELANGOR",
            field_key="alamat_isteri",
        )
        == "PARIT 7 BARAT 45300 SUNGAI BESAR , SELANGOR"
    )


def test_alamat_wali_recovers_a_street_when_the_stamp_blocks_its_own_label() -> None:
    # Regression: the stamp's leading "(" sometimes lands ahead of "Alamat"
    # itself ("( Alamat : ENDAFTAR ) LUBUK BAKAK ..."), which used to block
    # alamat_wali's own leading-label pattern from matching at all -- the
    # unstripped "Alamat" then matched the generic bare ALAMAT bleed-strip
    # keyword and wiped the field's entire real value.
    assert (
        normalize_wrapped_field(
            "( Alamat : ENDAFTAR ) LUBUK BAKAK D5 PADANG CERMIN ,\nLAMPUNG SELATAN , BANDAR LAMPUNG , INDONESIA .",
            field_key="alamat_wali",
        )
        == "LUBUK BAKAK D5 PADANG CERMIN LAMPUNG SELATAN , BANDAR LAMPUNG , INDONESIA ."
    )
