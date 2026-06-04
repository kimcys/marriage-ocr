from pathlib import Path
from types import SimpleNamespace

from marriage_ocr import batch_runner
from marriage_ocr.batch_runner import normalize_record
from marriage_ocr.models import ExtractedRecord


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
    monkeypatch.setattr(batch_runner, "mark_file_done", lambda **kwargs: None)
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
