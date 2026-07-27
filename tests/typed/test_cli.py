from pathlib import Path

from typer.testing import CliRunner

from marriage_ocr.cli import app
from marriage_ocr.typed.models import TypedBatchResult


runner = CliRunner()


def test_process_typed_command_delegates_paths(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def fake_process_typed_input(**kwargs):
        captured.update(kwargs)
        return TypedBatchResult(records=(), discovered_pdfs=0, written_rows=0)

    monkeypatch.setattr("marriage_ocr.typed.pipeline.process_typed_input", fake_process_typed_input)
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    result = runner.invoke(
        app,
        [
            "process-typed",
            "--input",
            str(input_dir),
            "--output",
            str(tmp_path / "typed_records.csv"),
            "--debug",
            str(tmp_path / "debug"),
            "--config",
            "config/typed_borang4b.yaml",
            "--skip-existing",
        ],
    )

    assert result.exit_code == 0
    assert captured["skip_existing"] is True
    assert captured["output_path"].name == "typed_records.csv"

