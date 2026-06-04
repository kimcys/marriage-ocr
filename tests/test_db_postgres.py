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

    assert captured_params[0][2] == 2
    assert captured_params[0][3] == 7
    assert captured_params[0][14] == "OK"
    assert captured_params[0][16].obj == ["missing wali"]
    assert captured_params[0][13].obj == "{\"ok\": true}"
