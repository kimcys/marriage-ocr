from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from marriage_ocr import triage
from marriage_ocr.cli import app
from marriage_ocr.triage import Classification

runner = CliRunner()


def _existing_file(tmp_path: Path) -> Path:
    path = tmp_path / "sample.jpg"
    path.write_bytes(b"fake-image")
    return path


def test_classify_reports_routable_result_with_config_path(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        triage,
        "classify_file",
        lambda path, **kwargs: Classification(
            doc_type="handwritten", record_type="cerai", layout_variant="legacy",
            is_jawi=False, jawi_proportion=0.0, notes=["some note"],
        ),
    )

    result = runner.invoke(app, ["classify", "--input", str(_existing_file(tmp_path))])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload == {
        "doc_type": "handwritten",
        "record_type": "cerai",
        "layout_variant": "legacy",
        "is_jawi": False,
        "jawi_proportion": 0.0,
        "notes": ["some note"],
        "status": "ROUTABLE",
        "config_path": "config/handwritten_cerai_legacy.yaml",
    }


def test_classify_reports_skipped_jawi_with_no_config_path(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        triage,
        "classify_file",
        lambda path, **kwargs: Classification(
            doc_type="handwritten", record_type="nikah", layout_variant="legacy",
            is_jawi=True, jawi_proportion=0.97, notes=["mostly Jawi"],
        ),
    )

    result = runner.invoke(app, ["classify", "--input", str(_existing_file(tmp_path))])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "SKIPPED_JAWI"
    assert payload["config_path"] is None


def test_classify_reports_blocked_no_template_for_unroutable_typed_combo(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        triage,
        "classify_file",
        lambda path, **kwargs: Classification(
            doc_type="typed", record_type="cerai", layout_variant=None,
            is_jawi=False, jawi_proportion=0.0, notes=[],
        ),
    )

    result = runner.invoke(app, ["classify", "--input", str(_existing_file(tmp_path))])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "BLOCKED_NO_TEMPLATE"
    assert payload["config_path"] is None


def test_classify_reports_needs_manual_classification_for_unroutable_handwritten_combo(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        triage,
        "classify_file",
        lambda path, **kwargs: Classification(
            doc_type="handwritten", record_type="cerai", layout_variant="unknown",
            is_jawi=False, jawi_proportion=0.0, notes=[],
        ),
    )

    result = runner.invoke(app, ["classify", "--input", str(_existing_file(tmp_path))])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "NEEDS_MANUAL_CLASSIFICATION"
    assert payload["config_path"] is None


def test_classify_fails_fast_for_missing_input(tmp_path: Path) -> None:
    result = runner.invoke(app, ["classify", "--input", str(tmp_path / "does_not_exist.jpg")])

    assert result.exit_code != 0
