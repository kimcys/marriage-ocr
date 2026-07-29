from marriage_ocr.refinement.text_corrections import (
    generate_date_candidates,
    generate_ic_candidates,
    generate_name_candidates,
)


def test_generate_name_candidates_normalizes_whitespace_and_multiline_without_retry() -> None:
    candidates = generate_name_candidates("  siti\nbinti   ali  ", field_name="nama_isteri")

    assert [candidate.value for candidate in candidates] == ["SITI BINTI ALI"]
    assert candidates[0].metadata["requires_retry_ocr"] is False
    assert candidates[0].metadata["original_value"] == "  siti\nbinti   ali  "


def test_generate_name_candidates_adds_connector_fix_with_retry() -> None:
    candidates = generate_name_candidates("AHMAD B1N ALI", field_name="nama_suami")

    assert [candidate.value for candidate in candidates] == ["AHMAD B1N ALI", "AHMAD BIN ALI"]
    assert candidates[1].metadata["requires_retry_ocr"] is True
    assert candidates[1].metadata["requires_review"] is True


def test_generate_name_candidates_preserves_valid_apostrophes_and_hyphens() -> None:
    candidates = generate_name_candidates("  O'Neil  Abdul-Rahman ", field_name="nama_suami")

    assert [candidate.value for candidate in candidates] == ["O'NEIL ABDUL-RAHMAN"]


def test_generate_name_candidates_preserves_abbreviation_periods() -> None:
    candidates = generate_name_candidates("HI.", field_name="nama_wali")

    assert [candidate.value for candidate in candidates] == ["HI.", "HJ."]
    assert candidates[1].metadata["requires_retry_ocr"] is True


def test_generate_name_candidates_keeps_uncommon_name_without_dictionary_replacement() -> None:
    candidates = generate_name_candidates("Zulqarnain", field_name="nama_suami")

    assert [candidate.value for candidate in candidates] == ["ZULQARNAIN"]


def test_generate_name_candidates_does_not_broadly_replace_non_connector_tokens() -> None:
    candidates = generate_name_candidates("MOH0D YUSOF", field_name="nama_suami")

    assert [candidate.value for candidate in candidates] == ["MOH0D YUSOF"]


def test_generate_name_candidates_applies_targeted_wife_name_corrections() -> None:
    candidates = generate_name_candidates("salwa bing ahmed samie", field_name="nama_isteri")

    assert [candidate.value for candidate in candidates] == ["SALWA BING AHMED SAMIE", "SALWA BINTI AHMED SAMION"]


def test_generate_name_candidates_strips_trailing_name_periods() -> None:
    candidates = generate_name_candidates("saadiah binti haji osman.", field_name="nama_isteri")

    assert [candidate.value for candidate in candidates] == ["SAADIAH BINTI HAJI OSMAN"]


def test_generate_ic_candidates_keeps_valid_modern_ic_without_retry() -> None:
    candidates = generate_ic_candidates("900101-10-1234", field_name="ic_baru")

    assert [candidate.value for candidate in candidates] == ["900101101234"]
    assert candidates[0].metadata["requires_retry_ocr"] is False


def test_generate_ic_candidates_repairs_numeric_ocr_confusions_when_valid() -> None:
    candidates = generate_ic_candidates("9O0101-1O-1234", field_name="ic_baru")

    assert [candidate.value for candidate in candidates] == ["900101101234"]


def test_generate_ic_candidates_rejects_invalid_birth_date() -> None:
    assert generate_ic_candidates("991332-10-1234", field_name="ic_baru") == []


def test_generate_ic_candidates_accepts_valid_leap_day_and_rejects_invalid_one() -> None:
    assert [candidate.value for candidate in generate_ic_candidates("000229-10-1234", field_name="ic_baru")] == [
        "000229101234"
    ]
    assert generate_ic_candidates("010229-10-1234", field_name="ic_baru") == []


def test_generate_ic_candidates_preserves_legacy_prefix() -> None:
    candidates = generate_ic_candidates("A 1192345", field_name="ic_lama")

    assert [candidate.value for candidate in candidates] == ["A1192345"]


def test_generate_ic_candidates_supports_slash_prefixed_legacy_id() -> None:
    candidates = generate_ic_candidates("R/F 119395", field_name="ic_lama")

    assert [candidate.value for candidate in candidates] == ["R/F119395"]


def test_generate_date_candidates_normalize_supported_inputs() -> None:
    assert [candidate.value for candidate in generate_date_candidates("27.8.94", field_name="tarikh_nikah")] == [
        "27-08-1994"
    ]
    assert [candidate.value for candidate in generate_date_candidates("1-10-94", field_name="tarikh_nikah")] == [
        "01-10-1994"
    ]


def test_generate_date_candidates_recovers_collapsed_and_ocr_confused_dates() -> None:
    assert [candidate.value for candidate in generate_date_candidates("27.894", field_name="tarikh_nikah")] == [
        "27-08-1994"
    ]
    candidates = generate_date_candidates("3O.O1.94", field_name="tarikh_nikah")
    assert [candidate.value for candidate in candidates] == ["30-01-1994"]
    assert candidates[0].metadata["requires_retry_ocr"] is True


def test_generate_date_candidates_reject_impossible_dates_and_validate_leap_years() -> None:
    assert generate_date_candidates("31/02/94", field_name="tarikh_nikah") == []
    assert [candidate.value for candidate in generate_date_candidates("29/02/96", field_name="tarikh_nikah")] == [
        "29-02-1996"
    ]
    assert generate_date_candidates("29/02/95", field_name="tarikh_nikah") == []
