from marriage_ocr import db_postgres
from marriage_ocr.db_postgres import _build_conninfo, _conninfo_from_database_url


def test_conninfo_from_database_url_handles_reserved_password_characters():
    url = "postgresql://postgres:Aim@nHakim#712@localhost:5432/marriagedb"

    conninfo = _conninfo_from_database_url(url)

    assert conninfo["user"] == "postgres"
    assert conninfo["password"] == "Aim@nHakim#712"
    assert conninfo["host"] == "localhost"
    assert conninfo["port"] == "5432"
    assert conninfo["dbname"] == "marriagedb"


def test_build_conninfo_uses_discrete_database_env_vars(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("DATABASE_HOST", "127.0.0.1")
    monkeypatch.setenv("DATABASE_PORT", "5433")
    monkeypatch.setenv("DATABASE_NAME", "marriagedb")
    monkeypatch.setenv("DATABASE_USER", "postgres")
    monkeypatch.setenv("DATABASE_PASSWORD", "secret")

    conninfo = _build_conninfo()

    assert conninfo == {
        "dbname": "marriagedb",
        "user": "postgres",
        "host": "127.0.0.1",
        "port": "5433",
        "password": "secret",
    }


def test_count_records_returns_integer(monkeypatch):
    class DummyCursor:
        def execute(self, query, params=None):
            self.query = query

        def fetchone(self):
            return {"count": "7"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class DummyConnection:
        def cursor(self):
            return DummyCursor()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(db_postgres, "get_connection", lambda: DummyConnection())

    assert db_postgres.count_records() == 7


def test_insert_record_normalizes_status_and_source_numbers(monkeypatch):
    captured_params = []

    class DummyCursor:
        def execute(self, query, params=None):
            captured_params.append(params)

        def fetchone(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class DummyConnection:
        def __init__(self):
            self.cursor_obj = DummyCursor()
            self.committed = False

        def cursor(self):
            return self.cursor_obj

        def commit(self):
            self.committed = True

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(db_postgres, "get_connection", lambda: DummyConnection())

    db_postgres.insert_record(
        batch_id=7,
        source_file="sample.jpg",
        source_page="2",
        source_record="record_007",
        record={
            "bil": "12",
            "nama_suami": "MOHAMAD BIN YASMIN",
            "status_review": "OK",
            "review_reason": ["missing wali"],
            "raw_ocr_json": "{\"ok\": true}",
            "confidence": 0.91,
        },
    )

    # captured_params[0] is the pre-upsert SELECT for the previous status;
    # captured_params[1] is the actual INSERT ... ON CONFLICT DO UPDATE.
    assert captured_params[0] == ("sample.jpg", 2, 7)
    assert captured_params[1][2] == 2
    assert captured_params[1][3] == 7
    assert captured_params[1][14] == "OK"
    assert captured_params[1][16].obj == ["missing wali"]
    assert captured_params[1][13].obj == "{\"ok\": true}"


def test_get_connection_reuses_a_single_pool_across_calls(monkeypatch):
    """Regression: get_connection() used to call psycopg.connect(...)
    directly on every invocation -- one fresh TCP connection and auth
    handshake per file, per record, and per status update. At 1M-record
    scale (many parallel batch_runner instances, or one long-running
    process working through a huge backlog) that reconnect churn adds real
    latency and risks exhausting the database's max_connections. A shared
    ConnectionPool must be built once and reused across every call.
    """
    created = []

    class FakePool:
        def __init__(self, dsn, *, kwargs, min_size, max_size, open):
            created.append(
                {"dsn": dsn, "kwargs": kwargs, "min_size": min_size, "max_size": max_size, "open": open}
            )

        def connection(self):
            return "connection-cm"

    monkeypatch.setattr(db_postgres, "ConnectionPool", FakePool)
    monkeypatch.setattr(db_postgres, "_pool", None)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("DATABASE_HOST", "127.0.0.1")
    monkeypatch.setenv("DATABASE_NAME", "marriagedb")
    monkeypatch.setenv("DATABASE_USER", "postgres")

    first = db_postgres.get_connection()
    second = db_postgres.get_connection()

    assert first == "connection-cm"
    assert second == "connection-cm"
    assert len(created) == 1


class _FakeRecordsDB:
    """Minimal in-memory stand-in for the records/batches tables, enough to
    exercise insert_record's SELECT-then-upsert-then-bucket-adjust sequence
    across multiple calls (unlike the single-call Dummy* fixtures above)."""

    def __init__(self):
        self.records: dict[tuple[str, int, int], str] = {}
        self.batches = {"ok_records": 0, "review_records": 0}


class _FakeCursor:
    def __init__(self, db: "_FakeRecordsDB") -> None:
        self.db = db
        self._last_result = None

    def execute(self, query, params=None):
        normalized = " ".join(query.split())
        if normalized.startswith("SELECT status"):
            source_file, source_page, source_record = params
            key = (source_file, source_page, source_record)
            status = self.db.records.get(key)
            self._last_result = {"status": status} if status is not None else None
        elif normalized.startswith("INSERT INTO records"):
            source_file, source_page, source_record = params[1], params[2], params[3]
            status = params[14]
            self.db.records[(source_file, source_page, source_record)] = status
            self._last_result = None
        elif "ok_records" in normalized:
            self.db.batches["ok_records"] += params[0]
            self._last_result = None
        elif "review_records" in normalized:
            self.db.batches["review_records"] += params[0]
            self._last_result = None
        else:  # pragma: no cover - defensive
            raise AssertionError(f"Unexpected query: {normalized}")

    def fetchone(self):
        return self._last_result

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeConnection:
    def __init__(self, db: "_FakeRecordsDB") -> None:
        self.db = db

    def cursor(self):
        return _FakeCursor(self.db)

    def commit(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_insert_record_does_not_double_count_batch_status_on_retry(monkeypatch):
    """Regression: is_file_done()/mark_file_failed() exist specifically so a
    file that failed partway gets reprocessed on the next batch_runner run.
    Reprocessing re-inserts the same (source_file, source_page,
    source_record) records, hitting the ON CONFLICT DO UPDATE path --  which
    used to unconditionally re-increment ok_records/review_records as if
    each were a brand-new record. At 1M-record scale, where retries after a
    transient OCR/API failure are routine, batches.ok_records/review_records
    (surfaced directly by reports_postgres.print_report()) would drift
    upward without bound on every retry.
    """
    db = _FakeRecordsDB()
    monkeypatch.setattr(db_postgres, "get_connection", lambda: _FakeConnection(db))

    record = {"bil": "1", "status_review": "OK", "confidence": 0.95}

    db_postgres.insert_record(batch_id=1, source_file="a.jpg", source_page=1, source_record=1, record=record)
    assert db.batches == {"ok_records": 1, "review_records": 0}

    # Retry of the identical record (e.g. the file failed on a later page
    # and got reprocessed): must not double-count.
    db_postgres.insert_record(batch_id=1, source_file="a.jpg", source_page=1, source_record=1, record=record)
    assert db.batches == {"ok_records": 1, "review_records": 0}

    # Retry where the status changed (e.g. Gemini succeeded this time):
    # must move the record from one bucket to the other, not double-count.
    changed_record = {**record, "status_review": "REVIEW"}
    db_postgres.insert_record(batch_id=1, source_file="a.jpg", source_page=1, source_record=1, record=changed_record)
    assert db.batches == {"ok_records": 0, "review_records": 1}


def test_fetch_records_for_batch_returns_extracted_records(monkeypatch):
    class DummyCursor:
        def execute(self, query, params=None):
            self.query = query
            self.params = params

        def fetchall(self):
            return [
                {"raw_record": {"bil": "2", "nama_suami": "SECOND", "source_record": "record_002"}},
                {"raw_record": {"bil": "1", "nama_suami": "FIRST", "source_record": "record_001"}},
            ]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class DummyConnection:
        def cursor(self):
            return DummyCursor()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(db_postgres, "get_connection", lambda: DummyConnection())

    records = db_postgres.fetch_records_for_batch(7)

    assert [record.bil for record in records] == ["2", "1"]
    assert [record.nama_suami for record in records] == ["SECOND", "FIRST"]
