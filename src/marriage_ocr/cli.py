from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import typer
from rich.console import Console

from marriage_ocr.config import LoadedConfig, load_runtime_config
from marriage_ocr.error_reporting import write_error_report
from marriage_ocr.logging_config import LoggingRuntime, get_logger, setup_logging
from marriage_ocr.pipeline import ProcessProgress, process_input


app = typer.Typer(help="Marriage register OCR pipeline")
console = Console()


def load_config(path: Path) -> dict[str, Any]:
    return load_runtime_config(path).data


def _load_command_runtime(command_name: str, config_path: Path) -> tuple[dict[str, Any], LoadedConfig, LoggingRuntime]:
    loaded = load_runtime_config(config_path)
    runtime = setup_logging(command_name, loaded.data.get("logging", {}))
    logger = get_logger(f"marriage_ocr.{command_name}")
    logger.info("Loaded configuration from %s", config_path)
    if loaded.env_file is not None:
        logger.info("Loaded environment file %s", loaded.env_file)
    logger.info("Logging to %s", runtime.log_path)
    return loaded.data, loaded, runtime


def _handle_command_error(
    error: Exception,
    *,
    command_name: str,
    config_path: Path,
    runtime: LoggingRuntime | None = None,
    extra_context: dict[str, Any] | None = None,
) -> None:
    report = write_error_report(
        error,
        command_name=command_name,
        runtime=runtime,
        config_path=config_path,
        extra_context=extra_context,
    )
    logger = get_logger(f"marriage_ocr.{command_name}")
    logger.exception("%s failed; error report written to %s", command_name, report.report_path)
    console.print(f"[bold red]{command_name} failed[/bold red]")
    console.print(f"Error report: {report.report_path}")
    if runtime is not None:
        console.print(f"Log file: {runtime.log_path}")
    raise typer.Exit(code=1) from error


@app.callback()
def main() -> None:
    """Marriage register OCR pipeline."""


@app.command()
def process(
    input: Path = typer.Option(..., "--input", "-i", help="Input image, PDF, or folder"),
    output: Path = typer.Option(..., "--output", "-o", help="Output XLSX path"),
    debug: Path = typer.Option(Path("debug"), "--debug", help="Debug output folder"),
    config: Path = typer.Option(Path("config/default.yaml"), "--config", help="Config file"),
    reset_output: bool = typer.Option(False, "--reset-output", help="Delete old XLSX before processing"),
    layout_only: bool = typer.Option(False, "--layout-only", help="Only detect layout/crops"),
    skip_existing: bool = typer.Option(False, "--skip-existing", help="Skip duplicate records"),
) -> None:
    runtime: LoggingRuntime | None = None

    try:
        cfg, _, runtime = _load_command_runtime("process", config)
        logger = get_logger("marriage_ocr.process")

        console.print("[bold green]Marriage OCR process started[/bold green]")
        console.print(f"Input: {input}")
        console.print(f"Output: {output}")
        console.print(f"Debug: {debug}")
        console.print(f"Config: {config}")
        console.print(f"Log file: {runtime.log_path}")
        console.print(f"OCR engine: {cfg.get('ocr', {}).get('engine')}")
        console.print(f"Reset output: {reset_output}")
        console.print(f"Layout only: {layout_only}")
        console.print(f"Skip existing: {skip_existing}")

        logger.info(
            "Process started input=%s output=%s debug=%s reset_output=%s layout_only=%s skip_existing=%s",
            input,
            output,
            debug,
            reset_output,
            layout_only,
            skip_existing,
        )

        if not input.exists():
            raise typer.BadParameter(f"Input path does not exist: {input}")

        def _print_progress(progress: ProcessProgress) -> None:
            console.print(progress.message)

        process_input(
            input_path=input,
            output_path=output,
            debug_path=debug,
            config_path=config,
            reset_output=reset_output,
            layout_only=layout_only,
            skip_existing=skip_existing,
            progress_callback=_print_progress,
        )
    except typer.BadParameter:
        raise
    except Exception as error:
        _handle_command_error(
            error,
            command_name="process",
            config_path=config,
            runtime=runtime,
            extra_context={
                "input": str(input),
                "output": str(output),
                "debug": str(debug),
                "reset_output": reset_output,
                "layout_only": layout_only,
                "skip_existing": skip_existing,
            },
        )


@app.command()
def review(
    debug: Path = typer.Option(Path("debug"), "--debug", help="Debug output folder with record crops"),
    export: Path = typer.Option(
        Path("data/reviewed_exports/daftar_perkahwinan_reviewed.xlsx"),
        "--export",
        help="Corrected XLSX output path",
    ),
    config: Path = typer.Option(Path("config/default.yaml"), "--config", help="Config file"),
    reviewer: str = typer.Option("", "--reviewer", help="Reviewer name saved with corrections"),
    port: int = typer.Option(8501, "--port", min=1, max=65535, help="Streamlit port"),
) -> None:
    runtime: LoggingRuntime | None = None

    try:
        cfg, _, runtime = _load_command_runtime("review", config)
        logger = get_logger("marriage_ocr.review")
        review_cfg = cfg.get("review", {})
        training_cfg = cfg.get("training_export", {})

        if export == Path("data/reviewed_exports/daftar_perkahwinan_reviewed.xlsx") and review_cfg.get("export_path"):
            export = Path(str(review_cfg.get("export_path")))
        training_output_dir = Path(str(training_cfg.get("output_dir", "data/ground_truth")))

        if not debug.exists():
            raise typer.BadParameter(f"Debug path does not exist: {debug}")

        export.parent.mkdir(parents=True, exist_ok=True)
        training_output_dir.parent.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["MARRIAGE_OCR_DEBUG_ROOT"] = str(debug)
        env["MARRIAGE_OCR_REVIEW_EXPORT_PATH"] = str(export)
        env["MARRIAGE_OCR_TRAINING_OUTPUT_DIR"] = str(training_output_dir)
        env["MARRIAGE_OCR_TRAINING_VERIFIED_ONLY"] = str(training_cfg.get("verified_only", True)).lower()
        env["MARRIAGE_OCR_TRAINING_VALIDATION_RATIO"] = str(training_cfg.get("validation_ratio", 0.20))
        env["MARRIAGE_OCR_REVIEWER_NAME"] = reviewer or str(review_cfg.get("reviewer_name", ""))

        app_path = Path(__file__).with_name("review_app.py")
        command = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--server.headless",
            "true",
            "--server.address",
            "127.0.0.1",
            "--server.port",
            str(port),
        ]

        console.print("[bold green]Marriage OCR review UI starting[/bold green]")
        console.print(f"Debug: {debug}")
        console.print(f"Export: {export}")
        console.print(f"Training Output: {training_output_dir}")
        console.print(f"Config: {config}")
        console.print(f"Log file: {runtime.log_path}")
        console.print(f"Reviewer: {env['MARRIAGE_OCR_REVIEWER_NAME'] or '(none)'}")
        console.print(f"URL: http://127.0.0.1:{port}")

        logger.info(
            "Review UI starting debug=%s export=%s training_output=%s reviewer=%s port=%s",
            debug,
            export,
            training_output_dir,
            env["MARRIAGE_OCR_REVIEWER_NAME"] or "(none)",
            port,
        )
        subprocess.run(command, check=True, env=env)
    except typer.BadParameter:
        raise
    except Exception as error:
        _handle_command_error(
            error,
            command_name="review",
            config_path=config,
            runtime=runtime,
            extra_context={
                "debug": str(debug),
                "export": str(export),
                "reviewer": reviewer,
                "port": port,
            },
        )


@app.command("web")
def web(
    port: int = typer.Option(8502, "--port", min=1, max=65535, help="Streamlit port"),
) -> None:
    app_path = Path(__file__).with_name("web_app.py")
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.headless",
        "true",
        "--server.address",
        "127.0.0.1",
        "--server.port",
        str(port),
    ]
    subprocess.run(command, check=True, env=os.environ.copy())


@app.command("export-training")
def export_training(
    debug: Path = typer.Option(Path("debug"), "--debug", help="Debug output folder with reviewed records"),
    output_dir: Path = typer.Option(Path("data/ground_truth"), "--output-dir", help="Training data output folder"),
    config: Path = typer.Option(Path("config/default.yaml"), "--config", help="Config file"),
    verified_only: bool = typer.Option(True, "--verified-only/--include-unverified", help="Only export verified labels"),
    reset_output: bool = typer.Option(True, "--reset-output/--append-output", help="Reset previous training export"),
) -> None:
    runtime: LoggingRuntime | None = None

    try:
        cfg, _, runtime = _load_command_runtime("export-training", config)
        logger = get_logger("marriage_ocr.export_training")
        training_cfg = cfg.get("training_export", {})

        if output_dir == Path("data/ground_truth") and training_cfg.get("output_dir"):
            output_dir = Path(str(training_cfg.get("output_dir")))

        if not debug.exists():
            raise typer.BadParameter(f"Debug path does not exist: {debug}")

        output_dir.mkdir(parents=True, exist_ok=True)

        from marriage_ocr.training_export import export_training_dataset

        summary = export_training_dataset(
            debug_root=debug,
            output_dir=output_dir,
            export_config=training_cfg,
            verified_only=verified_only,
            reset_output=reset_output,
        )

        console.print("[bold green]Training data export complete[/bold green]")
        console.print(f"Debug: {debug}")
        console.print(f"Output Dir: {summary.output_dir}")
        console.print(f"Log file: {runtime.log_path}")
        console.print(f"Labels: {summary.labels_path}")
        console.print(f"Train Split: {summary.train_path}")
        console.print(f"Validation Split: {summary.validation_path}")
        console.print(f"Manifest: {summary.manifest_path}")
        console.print(
            f"Examples: {summary.total_examples} total "
            f"({summary.train_examples} train / {summary.validation_examples} val); "
            f"skipped {summary.skipped_unverified_records} unverified record(s) and "
            f"{summary.skipped_empty_labels} empty label(s)"
        )

        logger.info(
            "Training export completed output_dir=%s examples=%s train=%s val=%s skipped_unverified=%s skipped_empty=%s",
            summary.output_dir,
            summary.total_examples,
            summary.train_examples,
            summary.validation_examples,
            summary.skipped_unverified_records,
            summary.skipped_empty_labels,
        )
    except typer.BadParameter:
        raise
    except Exception as error:
        _handle_command_error(
            error,
            command_name="export-training",
            config_path=config,
            runtime=runtime,
            extra_context={
                "debug": str(debug),
                "output_dir": str(output_dir),
                "verified_only": verified_only,
                "reset_output": reset_output,
            },
        )


if __name__ == "__main__":
    app()
