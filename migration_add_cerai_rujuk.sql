-- SUPERSEDED: db_postgres.py's SCHEMA constant now creates these columns
-- (plus record_type/layout_variant) directly, since there was no deployed
-- database to migrate (confirmed empty/local-only). This file is kept only
-- as historical reference for the column design rationale below; a fresh
-- `init_db()` no longer needs it run separately. If a real deployed DB ever
-- needs upgrading from an older schema, reconcile this file against the
-- current SCHEMA in db_postgres.py first, since the two have since diverged
-- (e.g. this file's ic_baru_*/ic_lama_* framing predates the ic_suami/
-- ic_isteri single-field columns SCHEMA now defines for Cerai/Rujuk).
--
-- Extends the existing `records` table (db_postgres.py) to cover Nikah,
-- Cerai, and Rujuk in one wide table with nullable columns -- matching how
-- OCR_Data_Column.xlsx itself represents the three types (one sheet, one
-- Kategori column, columns populated per type).
--
-- Design note on IC columns: the existing table/ExtractedRecord already
-- splits ic_lama_suami (old-format IC) vs ic_baru_suami (modern 12-digit)
-- for Nikah. OCR_Data_Column.xlsx instead wants a single "IC/Passport
-- Suami" column. This migration keeps the existing ic_baru_suami /
-- ic_baru_isteri columns as-is (don't break what's already indexed) and
-- adds the extraction-time detail as raw columns; when you build the
-- Excel-matching export, coalesce ic_baru_* first, falling back to
-- ic_lama_* from raw_record for older documents where only the short-form
-- IC exists.

BEGIN;

ALTER TABLE records
    ADD COLUMN IF NOT EXISTS record_type TEXT NOT NULL DEFAULT 'NIKAH'
        CHECK (record_type IN ('NIKAH', 'CERAI', 'RUJUK'));

-- Fields already marked for Nikah in OCR_Data_Column.xlsx that aren't yet
-- flat columns on this table (previously only living in raw_record JSONB).
ALTER TABLE records
    ADD COLUMN IF NOT EXISTS no_rujukan TEXT,
    ADD COLUMN IF NOT EXISTS umur_suami INTEGER,
    ADD COLUMN IF NOT EXISTS umur_isteri INTEGER,
    ADD COLUMN IF NOT EXISTS nama_pendaftar TEXT,
    ADD COLUMN IF NOT EXISTS alamat_pendaftar TEXT,
    ADD COLUMN IF NOT EXISTS tarikh_nikah_hijri TEXT,
    ADD COLUMN IF NOT EXISTS tarikh_daftar TEXT,
    ADD COLUMN IF NOT EXISTS tarikh_keluar TEXT,
    ADD COLUMN IF NOT EXISTS hal_hal_lain TEXT;

-- Added on request: Wali (guardian), and witnesses split into two
-- columns instead of one combined field. Note this table already has a
-- single generic `wali` TEXT column from before -- that one is left in
-- place as a legacy/simplified field; nama_wali/hubungan_wali below are
-- the richer pair your extraction pipeline already produces internally
-- (see ExtractedRecord in gemini_extractor.py) and should be treated as
-- the source of truth for the Excel-matching export going forward.
ALTER TABLE records
    ADD COLUMN IF NOT EXISTS nama_wali TEXT,
    ADD COLUMN IF NOT EXISTS hubungan_wali TEXT,
    ADD COLUMN IF NOT EXISTS saksi_1 TEXT,
    ADD COLUMN IF NOT EXISTS saksi_2 TEXT;

-- Cerai-specific fields.
ALTER TABLE records
    ADD COLUMN IF NOT EXISTS keadaan_talak TEXT,
    ADD COLUMN IF NOT EXISTS tarikh_cerai TEXT,
    ADD COLUMN IF NOT EXISTS tempat_cerai TEXT,
    -- Placeholder for the duplicate "Tempat Cerai" header in the source
    -- sheet -- rename once its intended meaning is confirmed.
    ADD COLUMN IF NOT EXISTS tempat_cerai_2 TEXT;

-- Rujuk-specific fields.
ALTER TABLE records
    ADD COLUMN IF NOT EXISTS tarikh_rujuk TEXT,
    ADD COLUMN IF NOT EXISTS tempat_rujuk TEXT;

CREATE INDEX IF NOT EXISTS idx_records_record_type
ON records(record_type);

COMMIT;

-- Verification query -- should show all three types with row counts once
-- Cerai/Rujuk extraction is live:
--   SELECT record_type, COUNT(*) FROM records GROUP BY record_type;
