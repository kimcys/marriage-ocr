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
    monkeypatch.setattr(
        batch_runner,
        "process_typed_input",
        lambda **kwargs: typed_calls.append(kwargs),
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
        batch_runner, "process_typed_input", lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("process_typed_input must not be called with no matching template")
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
    monkeypatch.setattr(
        batch_runner,
        "process_typed_input",
        lambda **kwargs: typed_calls.append(kwargs),
    )

    batch_runner.run_batch(
        input_dir=str(input_file.parent),
        batch_name="auto_route_typed_cerai",
        output_dir=str(tmp_path / "batch_output"),
    )

    assert len(typed_calls) == 1
    assert typed_calls[0]["config_path"] == Path("config/typed_cerai_modern.yaml")
