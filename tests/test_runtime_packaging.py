from pathlib import Path
import json

from marriage_ocr.config import load_runtime_config
from marriage_ocr.error_reporting import write_error_report
from marriage_ocr.logging_config import get_logger, setup_logging


def test_load_runtime_config_applies_env_file_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "ocr:",
                "  engine: mock",
                "logging:",
                "  level: INFO",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    env_path = tmp_path / ".env.production"
    env_path.write_text(
        "\n".join(
            [
                "MARRIAGE_OCR_LOG_LEVEL=DEBUG",
                "MARRIAGE_OCR__OCR__ENGINE=paddle",
                "MARRIAGE_OCR_LOG_DIR=runtime-logs",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = load_runtime_config(
        config_path,
        env={
            "MARRIAGE_OCR_ENV_FILE": str(env_path),
        },
    )

    assert loaded.env_file == env_path
    assert loaded.data["ocr"]["engine"] == "paddle"
    assert loaded.data["logging"]["level"] == "DEBUG"
    assert loaded.data["logging"]["directory"] == "runtime-logs"


def test_handwritten_config_uses_cell_crops_and_public_export() -> None:
    loaded = load_runtime_config(Path("config/handwritten.yaml"))

    assert loaded.data["ocr"]["mode"] == "cell_crops"
    assert loaded.data["ocr"]["field_refinement"]["enabled"] is True
    assert loaded.data["llm"]["enabled"] is True
    assert loaded.data["llm"]["provider"] == "gemini"
    assert loaded.data["ocr"]["field_refinement"]["max_variants_per_field"] == 4
    assert loaded.data["ocr"]["field_refinement"]["minimum_candidate_score"] == 0.65
    assert loaded.data["ocr"]["field_refinement"]["minimum_score_improvement"] == 0.05
    assert loaded.data["llm"]["prompt_mode"] == "handwritten_aggressive"
    assert loaded.data["llm"]["prefer_gemini_threshold"] == 0.55
    assert loaded.data["llm"]["review_below_field_confidence"] == 0.65
    assert loaded.data["export"]["include_raw_columns"] is False
    assert loaded.data["export"]["columns"] == [
        "Bil",
        "Nama Suami",
        "IC Lama Suami",
        "IC Baru Suami",
        "Umur Suami",
        "Nama Isteri",
        "IC Lama Isteri",
        "IC Baru Isteri",
        "Umur Isteri",
        "Mas Kahwin",
        "Nama Pendaftar",
        "Alamat Pendaftar",
        "Nama Wali",
        "Hubungan Wali",
        "Saksi 1",
        "Saksi 2",
        "Tarikh Nikah",
        "Tarikh Keluar",
        "Remarks",
        "Confidence",
        "Status",
    ]


def test_error_report_uses_logging_runtime_paths(tmp_path: Path) -> None:
    runtime = setup_logging(
        "process",
        {
            "directory": str(tmp_path / "logs"),
            "error_report_dir": str(tmp_path / "logs" / "error_reports"),
            "console": False,
            "file": True,
        },
    )
    logger = get_logger("marriage_ocr.test")
    logger.info("runtime started")

    report = write_error_report(
        ValueError("boom"),
        command_name="process",
        runtime=runtime,
        config_path=Path("config/production.yaml"),
        extra_context={"input": "input/sample.jpg"},
        argv=["python", "-m", "marriage_ocr.cli", "process"],
        env={"MARRIAGE_OCR_LOG_LEVEL": "INFO"},
    )

    payload = json.loads(report.report_path.read_text(encoding="utf-8"))

    assert runtime.log_path.exists()
    assert report.report_path.exists()
    assert payload["command_name"] == "process"
    assert payload["config_path"] == "config/production.yaml"
    assert payload["exception"]["type"] == "ValueError"
    assert payload["context"]["input"] == "input/sample.jpg"
    assert payload["log_path"] == str(runtime.log_path)
