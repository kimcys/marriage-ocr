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


def test_process_typed_command_requires_config_explicitly(tmp_path: Path) -> None:
    # Regression: --config used to default to config/typed_borang4b.yaml --
    # a template documented in typed/template.py as never recalibrated
    # against real samples and known to bleed tarikh_nikah/alamat_pendaftar/
    # nama_pendaftar into each other. A real client's typed Nikah batch was
    # silently processed through it (instead of the actually-recalibrated
    # nikah_legacy/nikah_modern) simply because --config was omitted from
    # the command. No safe default exists across Nikah/Cerai/Rujuk x
    # legacy/modern, so this must be a hard requirement, not a guess.
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
        ],
    )
    assert result.exit_code != 0
    assert "--config" in result.output

