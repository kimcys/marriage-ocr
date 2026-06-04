# Latest OCR Parser Enhancements

## Main changes

- Reworked spouse parsing to follow the ledger sequence strictly:
  `nama suami -> IC suami -> umur suami -> nama isteri -> IC isteri -> umur isteri -> mas kahwin`.
- Added support for names that span multiple OCR lines, for example `NOR AZNAN BIN ABDUL LATIFF` + `AHMAD YUTA.` stays as one husband name until an IC/age detail is found.
- Added IC parsing for old IC values separated by `/`, `-`, `.`, or whitespace, including numeric-only legacy IC values like `1595530`.
- Added age parsing for values after IC separators such as `A/1192345 - 25 TAHUN` and common OCR variants of `TAHUN`.
- Improved pendaftar normalization for common OCR errors such as `ITJ` -> `HJ.`, `BIR` -> `BIN`, `PENDAFTAPAN` -> `PENDAFTARAN`, `MASJIO` -> `MASJID`, and `HAMDAN` -> `HAMIDAN`.
- Improved wali parsing so the wali name can span both `Nama Wali` and `Hubungan` cells. Example: `SHAIKH` + `MOHD. BIN MOHD. JAINI (BAPA)` becomes name `SHAIKH MOHD. BIN MOHD. JAINI` and relationship `BAPA`.
- Improved saksi parsing for circled numbering (`①`, `②`) and multi-line witness names. Number markers are removed and lines between markers are joined.
- Improved date parsing for `tarikh nikah` and `tarikh keluar`, including normal dates (`27.8.94`, `1-10-94`) and collapsed OCR dates like `27.894`.
- Added placeholder filtering so mock OCR strings are not treated as real names.

## Files changed

- `src/marriage_ocr/parser/__init__.py`
- `tests/test_date_parser.py`

## Verification

- Full test suite passed: `38 passed`.
