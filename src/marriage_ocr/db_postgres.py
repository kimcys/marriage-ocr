# src/marriage_ocr/db_postgres.py

import os
import re
from datetime import datetime, timezone
from urllib.parse import parse_qsl, unquote
from typing import Any

import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv


load_dotenv()


SCHEMA = """
CREATE TABLE IF NOT EXISTS batches (
    id BIGSERIAL PRIMARY KEY,
    batch_name TEXT UNIQUE NOT NULL,
    input_path TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'RUNNING',

    total_files INTEGER DEFAULT 0,
    processed_files INTEGER DEFAULT 0,
    ok_records INTEGER DEFAULT 0,
    review_records INTEGER DEFAULT 0,
    failed_records INTEGER DEFAULT 0,

    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS processed_files (
    id BIGSERIAL PRIMARY KEY,
    batch_id BIGINT REFERENCES batches(id) ON DELETE CASCADE,
    file_path TEXT UNIQUE NOT NULL,
    file_hash TEXT,
    status TEXT NOT NULL,
    processed_at TIMESTAMPTZ,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS records (
    id BIGSERIAL PRIMARY KEY,

    batch_id BIGINT REFERENCES batches(id) ON DELETE CASCADE,
    source_file TEXT NOT NULL,
    source_page INTEGER DEFAULT 1,
    source_record INTEGER DEFAULT 1,

    bil TEXT,
    nama_suami TEXT,
    ic_baru_suami TEXT,
    nama_isteri TEXT,
    ic_baru_isteri TEXT,
    tarikh_nikah TEXT,
    mas_kahwin TEXT,
    wali TEXT,

    raw_record JSONB,
    raw_ocr JSONB,

    status TEXT NOT NULL DEFAULT 'REVIEW',
    confidence DOUBLE PRECISION DEFAULT 0,
    validation_errors JSONB DEFAULT '[]'::jsonb,

    reviewed_by TEXT,
    reviewed_at TIMESTAMPTZ,

    created_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ,

    UNIQUE(source_file, source_page, source_record)
);

CREATE INDEX IF NOT EXISTS idx_records_status
ON records(status);

CREATE INDEX IF NOT EXISTS idx_records_batch_id
ON records(batch_id);

CREATE INDEX IF NOT EXISTS idx_records_ic_suami
ON records(ic_baru_suami);

CREATE INDEX IF NOT EXISTS idx_records_ic_isteri
ON records(ic_baru_isteri);

CREATE INDEX IF NOT EXISTS idx_processed_files_status
ON processed_files(status);
"""


def utcnow():
    return datetime.now(timezone.utc)


def get_connection():
    conninfo = _build_conninfo()
    return psycopg.connect(row_factory=dict_row, **conninfo)


def init_db():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA)
        conn.commit()


def count_records() -> int:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS count FROM records")
            row = cur.fetchone()

    if row is None:
        return 0

    if isinstance(row, dict):
        value = row.get("count")
        if value is None:
            value = next(iter(row.values()), 0)
    else:
        value = row[0]

    return int(value)


def _build_conninfo() -> dict[str, Any]:
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        return _conninfo_from_database_url(database_url)

    dbname = os.getenv("DATABASE_NAME") or os.getenv("POSTGRES_DB") or "marriagedb"
    user = os.getenv("DATABASE_USER") or os.getenv("POSTGRES_USER") or "postgres"
    password = os.getenv("DATABASE_PASSWORD") or os.getenv("POSTGRES_PASSWORD")
    host = os.getenv("DATABASE_HOST") or os.getenv("POSTGRES_HOST") or "localhost"
    port = os.getenv("DATABASE_PORT") or os.getenv("POSTGRES_PORT") or "5432"

    conninfo: dict[str, Any] = {
        "dbname": dbname,
        "user": user,
        "host": host,
        "port": port,
    }
    if password is not None:
        conninfo["password"] = password
    return conninfo


def _conninfo_from_database_url(database_url: str) -> dict[str, Any]:
    # Normalize raw env values so reserved characters in passwords do not get
    # misread as URI separators by libpq.
    if "://" not in database_url:
        return {"conninfo": database_url}

    scheme, rest = database_url.split("://", 1)
    if not scheme:
        return {"conninfo": database_url}

    authority = rest
    path_and_more = ""
    slash_index = rest.find("/")
    if slash_index != -1:
        authority = rest[:slash_index]
        path_and_more = rest[slash_index + 1 :]

    userinfo = ""
    hostport = authority
    at_index = authority.rfind("@")
    if at_index != -1:
        userinfo = authority[:at_index]
        hostport = authority[at_index + 1 :]

    conninfo: dict[str, Any] = {}
    if userinfo:
        username, password = _split_userinfo(userinfo)
        if username:
            conninfo["user"] = unquote(username)
        if password is not None:
            conninfo["password"] = unquote(password)

    if hostport:
        host, port = _split_hostport(hostport)
        if host:
            conninfo["host"] = unquote(host)
        if port:
            conninfo["port"] = port

    dbname, query = _split_path_and_query(path_and_more)
    if dbname:
        conninfo["dbname"] = unquote(dbname)

    if query:
        for key, value in parse_qsl(query, keep_blank_values=True):
            conninfo[key] = value

    if not conninfo:
        return {"conninfo": database_url}

    return conninfo


def _split_userinfo(userinfo: str) -> tuple[str, str | None]:
    if ":" not in userinfo:
        return userinfo, None
    return userinfo.split(":", 1)


def _split_hostport(hostport: str) -> tuple[str, str | None]:
    if hostport.startswith("["):
        closing_bracket = hostport.find("]")
        if closing_bracket == -1:
            return hostport, None
        host = hostport[1:closing_bracket]
        remainder = hostport[closing_bracket + 1 :]
        if remainder.startswith(":") and remainder[1:]:
            return host, remainder[1:]
        return host, None

    if hostport.count(":") == 1:
        host, port = hostport.rsplit(":", 1)
        return host, port or None

    return hostport, None


def _split_path_and_query(path_and_more: str) -> tuple[str, str]:
    if not path_and_more:
        return "", ""

    dbname = path_and_more
    query = ""

    if "?" in dbname:
        dbname, query = dbname.split("?", 1)
    if "#" in dbname:
        dbname, _fragment = dbname.split("#", 1)

    return dbname.lstrip("/"), query


def create_batch(batch_name: str, input_path: str, total_files: int) -> int:
    now = utcnow()

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO batches (
                    batch_name,
                    input_path,
                    status,
                    total_files,
                    started_at
                )
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (batch_name)
                DO UPDATE SET
                    input_path = EXCLUDED.input_path,
                    total_files = EXCLUDED.total_files
                RETURNING id
                """,
                (batch_name, input_path, "RUNNING", total_files, now),
            )

            row = cur.fetchone()
            conn.commit()
            return row["id"]


def is_file_done(file_path: str) -> bool:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT status
                FROM processed_files
                WHERE file_path = %s
                """,
                (file_path,),
            )

            row = cur.fetchone()

    return bool(row and row["status"] == "DONE")


def mark_file_done(batch_id: int, file_path: str, file_hash: str):
    now = utcnow()

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO processed_files (
                    batch_id,
                    file_path,
                    file_hash,
                    status,
                    processed_at
                )
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (file_path)
                DO UPDATE SET
                    batch_id = EXCLUDED.batch_id,
                    file_hash = EXCLUDED.file_hash,
                    status = EXCLUDED.status,
                    processed_at = EXCLUDED.processed_at,
                    error_message = NULL
                """,
                (batch_id, file_path, file_hash, "DONE", now),
            )

            cur.execute(
                """
                UPDATE batches
                SET processed_files = processed_files + 1
                WHERE id = %s
                """,
                (batch_id,),
            )

        conn.commit()


def mark_file_failed(batch_id: int, file_path: str, error_message: str):
    now = utcnow()

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO processed_files (
                    batch_id,
                    file_path,
                    status,
                    processed_at,
                    error_message
                )
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (file_path)
                DO UPDATE SET
                    batch_id = EXCLUDED.batch_id,
                    status = EXCLUDED.status,
                    processed_at = EXCLUDED.processed_at,
                    error_message = EXCLUDED.error_message
                """,
                (batch_id, file_path, "FAILED", now, error_message),
            )

            cur.execute(
                """
                UPDATE batches
                SET failed_records = failed_records + 1
                WHERE id = %s
                """,
                (batch_id,),
            )

        conn.commit()


def insert_record(
    batch_id: int,
    source_file: str,
    source_page: Any,
    source_record: Any,
    record: dict[str, Any],
):
    now = utcnow()

    source_page = _coerce_record_number(source_page, default=1)
    source_record = _coerce_record_number(source_record, default=1)
    status = record.get("status") or record.get("status_review") or "REVIEW"
    validation_errors = record.get("validation_errors")
    if validation_errors is None:
        validation_errors = record.get("review_reason", [])
    raw_ocr = record.get("raw_ocr")
    if raw_ocr is None:
        raw_ocr = record.get("raw_ocr_json")

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO records (
                    batch_id,
                    source_file,
                    source_page,
                    source_record,

                    bil,
                    nama_suami,
                    ic_baru_suami,
                    nama_isteri,
                    ic_baru_isteri,
                    tarikh_nikah,
                    mas_kahwin,
                    wali,

                    raw_record,
                    raw_ocr,

                    status,
                    confidence,
                    validation_errors,

                    created_at,
                    updated_at
                )
                VALUES (
                    %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s,
                    %s, %s, %s,
                    %s, %s
                )
                ON CONFLICT (source_file, source_page, source_record)
                DO UPDATE SET
                    bil = EXCLUDED.bil,
                    nama_suami = EXCLUDED.nama_suami,
                    ic_baru_suami = EXCLUDED.ic_baru_suami,
                    nama_isteri = EXCLUDED.nama_isteri,
                    ic_baru_isteri = EXCLUDED.ic_baru_isteri,
                    tarikh_nikah = EXCLUDED.tarikh_nikah,
                    mas_kahwin = EXCLUDED.mas_kahwin,
                    wali = EXCLUDED.wali,
                    raw_record = EXCLUDED.raw_record,
                    raw_ocr = EXCLUDED.raw_ocr,
                    status = EXCLUDED.status,
                    confidence = EXCLUDED.confidence,
                    validation_errors = EXCLUDED.validation_errors,
                    updated_at = EXCLUDED.updated_at
                """,
                (
                    batch_id,
                    source_file,
                    source_page,
                    source_record,

                    record.get("bil"),
                    record.get("nama_suami"),
                    record.get("ic_baru_suami"),
                    record.get("nama_isteri"),
                    record.get("ic_baru_isteri"),
                    record.get("tarikh_nikah"),
                    record.get("mas_kahwin"),
                    record.get("wali"),

                    psycopg.types.json.Jsonb(record),
                    psycopg.types.json.Jsonb(raw_ocr),

                    status,
                    record.get("confidence", 0.0),
                    psycopg.types.json.Jsonb(validation_errors),

                    now,
                    now,
                ),
            )

            if status == "OK":
                cur.execute(
                    """
                    UPDATE batches
                    SET ok_records = ok_records + 1
                    WHERE id = %s
                    """,
                    (batch_id,),
                )
            else:
                cur.execute(
                    """
                    UPDATE batches
                    SET review_records = review_records + 1
                    WHERE id = %s
                    """,
                    (batch_id,),
                )

        conn.commit()


def _coerce_record_number(value: Any, *, default: int) -> int:
    if value is None:
        return default

    if isinstance(value, bool):
        return default

    if isinstance(value, int):
        return value

    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit() or (stripped.startswith("-") and stripped[1:].isdigit()):
            return int(stripped)
        matches = re.findall(r"\d+", stripped)
        if matches:
            return int(matches[-1])

    try:
        return int(value)
    except (TypeError, ValueError):
        return default
