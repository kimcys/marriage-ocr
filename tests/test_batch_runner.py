import threading
import time
from pathlib import Path
from types import SimpleNamespace

from marriage_ocr import batch_runner
from marriage_ocr.batch_runner import normalize_record
from marriage_ocr.models import ExtractedRecord
from marriage_ocr.triage import Classification


def test_normalize_record_handles_extracted_record_dataclass():
    record = ExtractedRecord(
        bil="12",
        nama_suami="MOHAMAD BIN YASMIN",
        status_review="OK",
        review_reason=[],
    )

    normalized = normalize_record(record)

    assert normalized["bil"] == "12"
    assert normalized["nama_suami"] == "MOHAMAD BIN YASMIN"
    assert normalized["status_review"] == "OK"


def test_run_batch_exports_merged_xlsx(monkeypatch, tmp_path: Path):
    input_file = tmp_path / "input" / "sample.JPG"
    input_file.parent.mkdir(parents=True)
    input_file.write_bytes(b"fake-image")

    exported: dict[str, object] = {}

    monkeypatch.setattr(batch_runner, "list_input_files", lambda input_dir: [input_file])
    monkeypatch.setattr(batch_runner, "create_batch", lambda batch_name, input_path, total_files: 7)
    monkeypatch.setattr(batch_runner, "is_file_done", lambda file_path: False)
    monkeypatch.setattr(batch_runner, "file_sha256", lambda file_path: "hash")
    monkeypatch.setattr(batch_runner, "find_done_file_by_hash", lambda file_hash: None)
    monkeypatch.setattr(batch_runner, "find_duplicate_record", lambda **kwargs: None)
    monkeypatch.setattr(batch_runner, "insert_record", lambda **kwargs: None)
    monkeypatch.setattr(batch_runner, "mark_file_done", lambda *args, **kwargs: None)
    monkeypatch.setattr(batch_runner, "mark_file_failed", lambda **kwargs: None)
    monkeypatch.setattr(
        batch_runner,
        "process_input",
        lambda **kwargs: SimpleNamespace(records=[ExtractedRecord(bil="1", source_record="record_001")]),
    )
    monkeypatch.setattr(
        batch_runner,
        "fetch_records_for_batch",
        lambda batch_id: [ExtractedRecord(bil="1", source_record="record_001")],
    )

    def fake_export_records_to_xlsx(records, output_path, export_config, *, reset_output, skip_existing):
        exported["records"] = list(records)
        exported["output_path"] = output_path
        exported["export_config"] = dict(export_config)
        exported["reset_output"] = reset_output
        exported["skip_existing"] = skip_existing
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("fake-xlsx", encoding="utf-8")
        return SimpleNamespace(
            written_count=len(exported["records"]),
            skipped_duplicates=0,
            total_rows=len(exported["records"]),
            output_path=output_path,
        )

    monkeypatch.setattr(batch_runner, "export_records_to_xlsx", fake_export_records_to_xlsx)

    batch_runner.run_batch(
        input_dir=str(input_file.parent),
        batch_name="run_001",
        output_dir=str(tmp_path / "batch_output"),
        config_path="config/production.yaml",
    )

    assert exported["output_path"] == tmp_path / "batch_output" / "exports" / "run_001_merged.xlsx"
    assert exported["reset_output"] is True
    assert exported["skip_existing"] is False
    assert Path(exported["output_path"]).exists()


def _stub_common(monkeypatch, tmp_path: Path, input_file: Path):
    monkeypatch.setattr(batch_runner, "list_input_files", lambda input_dir: [input_file])
    monkeypatch.setattr(batch_runner, "create_batch", lambda batch_name, input_path, total_files: 7)
    monkeypatch.setattr(batch_runner, "is_file_done", lambda file_path: False)
    monkeypatch.setattr(batch_runner, "file_sha256", lambda file_path: "hash")
    monkeypatch.setattr(batch_runner, "find_done_file_by_hash", lambda file_hash: None)
    monkeypatch.setattr(batch_runner, "fetch_records_for_batch", lambda batch_id: [])
    # Safety net: if a test's own stubs raise unexpectedly, run_batch's
    # except-block calls mark_file_failed with real args -- fail loudly via
    # AssertionError instead of silently trying (and hanging on) a real DB
    # connection.
    monkeypatch.setattr(
        batch_runner,
        "mark_file_failed",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError(f"unexpected failure path: {kwargs}")),
    )
    monkeypatch.setattr(
        batch_runner,
        "export_records_to_xlsx",
        lambda records, output_path, export_config, **kwargs: SimpleNamespace(
            written_count=0, skipped_duplicates=0, total_rows=0, output_path=output_path
        ),
    )


def test_run_batch_auto_routes_handwritten_nikah_to_matching_config(monkeypatch, tmp_path: Path):
    input_file = tmp_path / "input" / "sample.jpg"
    input_file.parent.mkdir(parents=True)
    input_file.write_bytes(b"fake-image")

    _stub_common(monkeypatch, tmp_path, input_file)
    monkeypatch.setattr(batch_runner, "mark_file_done", lambda *args, **kwargs: None)
    monkeypatch.setattr(batch_runner, "insert_record", lambda **kwargs: None)
    monkeypatch.setattr(
        batch_runner.triage,
        "classify_file",
        lambda path, **kwargs: Classification(
            doc_type="handwritten", record_type="nikah", layout_variant="legacy",
            is_jawi=False, jawi_proportion=0.0,
        ),
    )

    seen_configs = []

    def fake_process_input(*, input_path, output_path, debug_path, config_path):
        seen_configs.append(config_path)
        return SimpleNamespace(records=[])

    monkeypatch.setattr(batch_runner, "process_input", fake_process_input)

    batch_runner.run_batch(
        input_dir=str(input_file.parent),
        batch_name="auto_route_nikah",
        output_dir=str(tmp_path / "batch_output"),
    )

    assert seen_configs == [Path("config/handwritten.yaml")]


def test_run_batch_skips_jawi_pages_without_calling_process_input(monkeypatch, tmp_path: Path):
    input_file = tmp_path / "input" / "jawi_page.jpg"
    input_file.parent.mkdir(parents=True)
    input_file.write_bytes(b"fake-image")

    _stub_common(monkeypatch, tmp_path, input_file)
    monkeypatch.setattr(batch_runner, "mark_file_done", lambda *args, **kwargs: None)
    monkeypatch.setattr(batch_runner, "process_input", lambda **kwargs: (_ for _ in ()).throw(
        AssertionError("process_input must not be called for a Jawi-skipped file")
    ))

    skip_calls = []
    monkeypatch.setattr(
        batch_runner,
        "mark_file_skipped",
        lambda batch_id, file_path, status, **kwargs: skip_calls.append((file_path, status, kwargs)),
    )
    monkeypatch.setattr(
        batch_runner.triage,
        "classify_file",
        lambda path, **kwargs: Classification(
            doc_type="handwritten", record_type="rujuk", layout_variant="legacy",
            is_jawi=True, jawi_proportion=0.97, notes=["mostly Jawi"],
        ),
    )

    batch_runner.run_batch(
        input_dir=str(input_file.parent),
        batch_name="auto_route_jawi",
        output_dir=str(tmp_path / "batch_output"),
    )

    assert len(skip_calls) == 1
    file_path, status, kwargs = skip_calls[0]
    assert status == "SKIPPED_JAWI"
    assert kwargs["is_jawi"] is True
    assert kwargs["jawi_proportion"] == 0.97


def test_run_batch_skips_files_with_a_matching_content_hash(monkeypatch, tmp_path: Path):
    input_file = tmp_path / "input" / "reupload.jpg"
    input_file.parent.mkdir(parents=True)
    input_file.write_bytes(b"fake-image")

    _stub_common(monkeypatch, tmp_path, input_file)
    monkeypatch.setattr(batch_runner, "find_done_file_by_hash", lambda file_hash: "original/scan.jpg")
    monkeypatch.setattr(
        batch_runner.triage,
        "classify_file",
        lambda path, **kwargs: (_ for _ in ()).throw(
            AssertionError("classify_file must not be called for a hash-duplicate file")
        ),
    )
    monkeypatch.setattr(batch_runner, "process_input", lambda **kwargs: (_ for _ in ()).throw(
        AssertionError("process_input must not be called for a hash-duplicate file")
    ))

    skip_calls = []
    monkeypatch.setattr(
        batch_runner,
        "mark_file_skipped",
        lambda batch_id, file_path, status, **kwargs: skip_calls.append((file_path, status, kwargs)),
    )

    batch_runner.run_batch(
        input_dir=str(input_file.parent),
        batch_name="auto_route_duplicate_file",
        output_dir=str(tmp_path / "batch_output"),
    )

    assert len(skip_calls) == 1
    file_path, status, kwargs = skip_calls[0]
    assert file_path == str(input_file)
    assert status == "DUPLICATE_FILE"
    assert "original/scan.jpg" in kwargs["notes"][0]


def test_run_batch_flags_content_duplicate_records_before_insert(monkeypatch, tmp_path: Path):
    input_file = tmp_path / "input" / "sample.jpg"
    input_file.parent.mkdir(parents=True)
    input_file.write_bytes(b"fake-image")

    _stub_common(monkeypatch, tmp_path, input_file)
    monkeypatch.setattr(batch_runner, "mark_file_done", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        batch_runner.triage,
        "classify_file",
        lambda path, **kwargs: Classification(
            doc_type="handwritten", record_type="nikah", layout_variant="legacy",
            is_jawi=False, jawi_proportion=0.0,
        ),
    )
    monkeypatch.setattr(
        batch_runner,
        "process_input",
        lambda **kwargs: SimpleNamespace(
            records=[ExtractedRecord(bil="1/2010", ic_baru_suami="740326145837", nama_suami="X")]
        ),
    )
    monkeypatch.setattr(batch_runner, "find_duplicate_record", lambda **kwargs: 99)

    insert_calls = []
    monkeypatch.setattr(
        batch_runner,
        "insert_record",
        lambda **kwargs: insert_calls.append(kwargs),
    )

    batch_runner.run_batch(
        input_dir=str(input_file.parent),
        batch_name="content_duplicate",
        output_dir=str(tmp_path / "batch_output"),
    )

    assert len(insert_calls) == 1
    inserted_record = insert_calls[0]["record"]
    assert inserted_record["is_duplicate"] is True
    assert inserted_record["duplicate_of_record_id"] == 99


def test_run_batch_routes_typed_doc_type_to_typed_pipeline(monkeypatch, tmp_path: Path):
    input_file = tmp_path / "input" / "certificate.pdf"
    input_file.parent.mkdir(parents=True)
    input_file.write_bytes(b"fake-pdf")

    _stub_common(monkeypatch, tmp_path, input_file)
    monkeypatch.setattr(batch_runner, "mark_file_done", lambda *args, **kwargs: None)
    monkeypatch.setattr(batch_runner, "process_input", lambda **kwargs: (_ for _ in ()).throw(
        AssertionError("process_input must not be called for a typed doc_type")
    ))
    monkeypatch.setattr(
        batch_runner.triage,
        "classify_file",
        lambda path, **kwargs: Classification(
            doc_type="typed", record_type="nikah", layout_variant=None,
            is_jawi=False, jawi_proportion=0.0,
        ),
    )

    typed_calls = []

    def fake_process_typed_documents_no_csv(**kwargs):
        typed_calls.append(kwargs)
        return []

    monkeypatch.setattr(
        batch_runner, "process_typed_documents_no_csv", fake_process_typed_documents_no_csv
    )

    batch_runner.run_batch(
        input_dir=str(input_file.parent),
        batch_name="auto_route_typed",
        output_dir=str(tmp_path / "batch_output"),
    )

    assert len(typed_calls) == 1
    assert typed_calls[0]["config_path"] == Path("config/typed_borang4b.yaml")


def test_run_batch_blocks_typed_record_type_with_no_template(monkeypatch, tmp_path: Path):
    # layout_variant=None for a typed cerai/rujuk classification has no
    # ROUTING_TABLE entry (typed cerai/rujuk are only routed for "legacy"/
    # "modern", matching triage's enactment-year detection) -- this only
    # happens if triage ever fails to detect the enactment year, so it must
    # still fall through to BLOCKED_NO_TEMPLATE rather than guessing a config.
    input_file = tmp_path / "input" / "certificate.pdf"
    input_file.parent.mkdir(parents=True)
    input_file.write_bytes(b"fake-pdf")

    _stub_common(monkeypatch, tmp_path, input_file)
    monkeypatch.setattr(batch_runner, "process_input", lambda **kwargs: (_ for _ in ()).throw(
        AssertionError("process_input must not be called for a blocked file")
    ))
    monkeypatch.setattr(
        batch_runner, "process_typed_documents_no_csv", lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("process_typed_documents_no_csv must not be called with no matching template")
        )
    )
    monkeypatch.setattr(
        batch_runner.triage,
        "classify_file",
        lambda path, **kwargs: Classification(
            doc_type="typed", record_type="cerai", layout_variant=None,
            is_jawi=False, jawi_proportion=0.0,
        ),
    )

    skip_calls = []
    monkeypatch.setattr(
        batch_runner,
        "mark_file_skipped",
        lambda batch_id, file_path, status, **kwargs: skip_calls.append((file_path, status, kwargs)),
    )

    batch_runner.run_batch(
        input_dir=str(input_file.parent),
        batch_name="auto_route_blocked",
        output_dir=str(tmp_path / "batch_output"),
    )

    assert len(skip_calls) == 1
    assert skip_calls[0][1] == "BLOCKED_NO_TEMPLATE"


def test_run_batch_routes_typed_cerai_modern_to_its_own_config(monkeypatch, tmp_path: Path):
    input_file = tmp_path / "input" / "certificate.pdf"
    input_file.parent.mkdir(parents=True)
    input_file.write_bytes(b"fake-pdf")

    _stub_common(monkeypatch, tmp_path, input_file)
    monkeypatch.setattr(batch_runner, "mark_file_done", lambda *args, **kwargs: None)
    monkeypatch.setattr(batch_runner, "process_input", lambda **kwargs: (_ for _ in ()).throw(
        AssertionError("process_input must not be called for a typed doc_type")
    ))
    monkeypatch.setattr(
        batch_runner.triage,
        "classify_file",
        lambda path, **kwargs: Classification(
            doc_type="typed", record_type="cerai", layout_variant="modern",
            is_jawi=False, jawi_proportion=0.0,
        ),
    )

    typed_calls = []

    def fake_process_typed_documents_no_csv(**kwargs):
        typed_calls.append(kwargs)
        return []

    monkeypatch.setattr(
        batch_runner, "process_typed_documents_no_csv", fake_process_typed_documents_no_csv
    )

    batch_runner.run_batch(
        input_dir=str(input_file.parent),
        batch_name="auto_route_typed_cerai",
        output_dir=str(tmp_path / "batch_output"),
    )

    assert len(typed_calls) == 1
    assert typed_calls[0]["config_path"] == Path("config/typed_cerai_modern.yaml")


def test_run_batch_processes_multiple_files_concurrently_with_workers(monkeypatch, tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True)
    input_files = []
    for i in range(5):
        f = input_dir / f"sample_{i}.jpg"
        f.write_bytes(b"fake-image")
        input_files.append(f)

    _stub_common(monkeypatch, tmp_path, input_files[0])
    monkeypatch.setattr(batch_runner, "list_input_files", lambda input_dir: list(input_files))
    monkeypatch.setattr(batch_runner, "mark_file_done", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        batch_runner.triage,
        "classify_file",
        lambda path, **kwargs: Classification(
            doc_type="handwritten", record_type="nikah", layout_variant="legacy",
            is_jawi=False, jawi_proportion=0.0,
        ),
    )
    monkeypatch.setattr(batch_runner, "find_duplicate_record", lambda **kwargs: None)

    def fake_process_input(*, input_path, output_path, debug_path, config_path):
        time.sleep(0.02)
        return SimpleNamespace(records=[ExtractedRecord(bil=input_path.stem, source_record="record_001")])

    monkeypatch.setattr(batch_runner, "process_input", fake_process_input)

    insert_calls = []
    lock = threading.Lock()

    def fake_insert_record(**kwargs):
        with lock:
            insert_calls.append(kwargs)

    monkeypatch.setattr(batch_runner, "insert_record", fake_insert_record)

    batch_runner.run_batch(
        input_dir=str(input_dir),
        batch_name="concurrent_run",
        output_dir=str(tmp_path / "batch_output"),
        workers=3,
    )

    assert len(insert_calls) == 5
    assert {call["source_file"] for call in insert_calls} == {str(f) for f in input_files}


def test_run_batch_isolates_one_failing_file_from_the_rest(monkeypatch, tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True)
    good_files = [input_dir / "good_1.jpg", input_dir / "good_2.jpg"]
    bad_file = input_dir / "bad.jpg"
    for f in good_files + [bad_file]:
        f.write_bytes(b"fake-image")

    _stub_common(monkeypatch, tmp_path, good_files[0])
    monkeypatch.setattr(batch_runner, "list_input_files", lambda input_dir: [*good_files, bad_file])
    monkeypatch.setattr(batch_runner, "mark_file_done", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        batch_runner.triage,
        "classify_file",
        lambda path, **kwargs: Classification(
            doc_type="handwritten", record_type="nikah", layout_variant="legacy",
            is_jawi=False, jawi_proportion=0.0,
        ),
    )
    monkeypatch.setattr(batch_runner, "find_duplicate_record", lambda **kwargs: None)
    monkeypatch.setattr(batch_runner, "insert_record", lambda **kwargs: None)

    def fake_process_input(*, input_path, output_path, debug_path, config_path):
        if input_path == bad_file:
            raise RuntimeError("simulated OCR crash")
        return SimpleNamespace(records=[ExtractedRecord(bil="1", source_record="record_001")])

    monkeypatch.setattr(batch_runner, "process_input", fake_process_input)

    failed_calls = []
    done_calls = []
    monkeypatch.setattr(
        batch_runner, "mark_file_failed",
        lambda **kwargs: failed_calls.append(kwargs),
    )
    monkeypatch.setattr(
        batch_runner, "mark_file_done",
        lambda *args, **kwargs: done_calls.append(args),
    )

    batch_runner.run_batch(
        input_dir=str(input_dir),
        batch_name="isolated_failure",
        output_dir=str(tmp_path / "batch_output"),
        workers=3,
    )

    assert len(failed_calls) == 1
    assert failed_calls[0]["file_path"] == str(bad_file)
    assert {args[1] for args in done_calls} == {str(f) for f in good_files}


def test_run_batch_processes_typed_files_concurrently(monkeypatch, tmp_path: Path):
    # Typed files used to serialize against each other behind a CSV-writer
    # lock (see git history). Now that typed results go straight into
    # Postgres like the general path, they should run fully concurrently --
    # this proves at least two files' process_typed_documents_no_csv calls
    # are in flight at the same time, not one-at-a-time.
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True)
    input_files = []
    for i in range(4):
        f = input_dir / f"cert_{i}.pdf"
        f.write_bytes(b"fake-pdf")
        input_files.append(f)

    _stub_common(monkeypatch, tmp_path, input_files[0])
    monkeypatch.setattr(batch_runner, "list_input_files", lambda input_dir: list(input_files))
    monkeypatch.setattr(batch_runner, "mark_file_done", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        batch_runner.triage,
        "classify_file",
        lambda path, **kwargs: Classification(
            doc_type="typed", record_type="nikah", layout_variant=None,
            is_jawi=False, jawi_proportion=0.0,
        ),
    )

    state_lock = threading.Lock()
    state = {"current": 0, "max_seen": 0}

    def fake_process_typed_documents_no_csv(**kwargs):
        with state_lock:
            state["current"] += 1
            state["max_seen"] = max(state["max_seen"], state["current"])
        time.sleep(0.05)
        with state_lock:
            state["current"] -= 1
        return []

    monkeypatch.setattr(
        batch_runner, "process_typed_documents_no_csv", fake_process_typed_documents_no_csv
    )

    batch_runner.run_batch(
        input_dir=str(input_dir),
        batch_name="typed_concurrency",
        output_dir=str(tmp_path / "batch_output"),
        workers=4,
    )

    assert state["max_seen"] > 1


def test_run_batch_inserts_typed_results_with_mapped_status(monkeypatch, tmp_path: Path):
    input_file = tmp_path / "input" / "certificate.pdf"
    input_file.parent.mkdir(parents=True)
    input_file.write_bytes(b"fake-pdf")

    _stub_common(monkeypatch, tmp_path, input_file)
    monkeypatch.setattr(batch_runner, "mark_file_done", lambda *args, **kwargs: None)
    monkeypatch.setattr(batch_runner, "find_duplicate_record", lambda **kwargs: None)
    monkeypatch.setattr(
        batch_runner.triage,
        "classify_file",
        lambda path, **kwargs: Classification(
            doc_type="typed", record_type="nikah", layout_variant=None,
            is_jawi=False, jawi_proportion=0.0,
        ),
    )

    typed_record = ExtractedRecord(
        record_type="NIKAH",
        bil="04/2009",
        nama_suami="HENDON BIN MARIMIN",
        ic_baru_suami="571018105919",
        source_file="certificate.pdf",
        source_page=1,
        source_record="01000122140838082020",  # a real typed source_stem: far too big for an INTEGER column
    )
    typed_result = SimpleNamespace(
        record=typed_record,
        processing_status=batch_runner.ProcessingStatus.SUCCESS_WITH_RETRY,
        failed_fields=("umur_suami",),
        error_message="",
    )
    monkeypatch.setattr(
        batch_runner, "process_typed_documents_no_csv", lambda **kwargs: [typed_result]
    )

    insert_calls = []
    monkeypatch.setattr(batch_runner, "insert_record", lambda **kwargs: insert_calls.append(kwargs))

    batch_runner.run_batch(
        input_dir=str(input_file.parent),
        batch_name="typed_db_insert",
        output_dir=str(tmp_path / "batch_output"),
    )

    assert len(insert_calls) == 1
    call = insert_calls[0]
    # source_record is forced to a small constant rather than the raw
    # source_stem, which would overflow the records table's INTEGER column.
    assert call["source_page"] == 1
    assert call["source_record"] == 1
    assert call["record"]["status_review"] == "OK"
    assert call["record"]["bil"] == "04/2009"
    assert "typed field failed: umur_suami" in call["record"]["review_reason"]


def test_sync_gemini_batch_results_updates_matching_rows_and_preserves_dedup_flags(monkeypatch):
    record = ExtractedRecord(
        bil="1",
        nama_suami="AHMAD BIN ALI",
        source_file="a.jpg",
        source_page=1,
        source_record=2,
        status_review="OK",
        confidence=0.9,
    )

    monkeypatch.setattr(
        batch_runner,
        "get_existing_record_identity",
        lambda source_file, source_page, source_record: {
            "batch_id": 7,
            "is_duplicate": True,
            "duplicate_of_record_id": 42,
        },
    )
    insert_calls = []
    monkeypatch.setattr(
        batch_runner,
        "insert_record",
        lambda **kwargs: insert_calls.append(kwargs),
    )

    summary = batch_runner.sync_gemini_batch_results([record])

    assert summary == {"synced": 1, "skipped_no_match": []}
    assert len(insert_calls) == 1
    call = insert_calls[0]
    assert call["batch_id"] == 7
    assert call["source_file"] == "a.jpg"
    assert call["source_page"] == 1
    assert call["source_record"] == 2
    assert call["record"]["is_duplicate"] is True
    assert call["record"]["duplicate_of_record_id"] == 42


def test_sync_gemini_batch_results_skips_records_with_no_matching_db_row(monkeypatch):
    record = ExtractedRecord(
        bil="1",
        source_file="a.jpg",
        source_page=1,
        source_record=1,
    )

    monkeypatch.setattr(
        batch_runner, "get_existing_record_identity", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        batch_runner,
        "insert_record",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("insert_record must not be called when there's no matching row")
        ),
    )

    summary = batch_runner.sync_gemini_batch_results([record])

    assert summary["synced"] == 0
    assert summary["skipped_no_match"] == ["a.jpg#1.1"]
