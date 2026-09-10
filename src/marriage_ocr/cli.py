from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import typer
from dotenv import load_dotenv
from rich.console import Console

from marriage_ocr.config import LoadedConfig, load_runtime_config
from marriage_ocr.error_reporting import write_error_report
from marriage_ocr.logging_config import LoggingRuntime, get_logger, setup_logging
from marriage_ocr.pipeline import ProcessProgress, process_input


app = typer.Typer(help="Marriage register OCR pipeline")
console = Console()

onedrive_app = typer.Typer(help="Pull an arbitrary OneDrive share link to local disk")
app.add_typer(onedrive_app, name="onedrive")

gemini_batch_app = typer.Typer(help="Re-run Gemini semantic extraction via Batch Mode (~50% cheaper, async)")
app.add_typer(gemini_batch_app, name="gemini-batch")

DEFAULT_ONEDRIVE_TOKEN_CACHE = Path(".onedrive_token_cache.json")


def load_config(path: Path) -> dict[str, Any]:
    return load_runtime_config(path).data


def _load_command_runtime(command_name: str, config_path: Path) -> tuple[dict[str, Any], LoadedConfig, LoggingRuntime]:
    loaded = load_runtime_config(config_path)
    # load_runtime_config's own .env parsing only ever feeds the
    # MARRIAGE_OCR_*-prefixed override mechanism into `loaded.data` -- it
    # never touches the real process environment. Anything read directly via
    # os.getenv() elsewhere (GEMINI_API_KEY in gemini_extractor.py,
    # GOOGLE_APPLICATION_CREDENTIALS for the Vision client) would otherwise
    # silently stay unset for every CLI command here, even with a correctly
    # filled-in .env file. db_postgres.py already gets this for free via its
    # own load_dotenv() call, but only when something imports it (e.g.
    # batch_runner) -- `process`/`process-typed` do not. override=False
    # keeps real environment variables (e.g. from a parent shell) winning
    # over .env, matching load_runtime_config's own MARRIAGE_OCR_* precedence.
    if loaded.env_file is not None:
        load_dotenv(loaded.env_file, override=False)
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


@app.command("classify")
def classify(
    input: Path = typer.Option(..., "--input", "-i", help="Input image or PDF to classify"),
) -> None:
    """Classify one file's doc_type/record_type/layout_variant and print the
    result -- plus which config it would route to, if any -- as one JSON
    object on stdout.

    For an external caller (e.g. marriage-be) that receives files of unknown
    type (a OneDrive dump can mix handwritten/typed, Nikah/Cerai/Rujuk) and
    needs to know which config to run `process`/`process-typed` with before
    it can do so. Reuses triage.classify_file and batch_runner.ROUTING_TABLE
    directly rather than duplicating that routing decision here -- those
    stay the single source of truth for what routes where.
    """
    from marriage_ocr import triage
    from marriage_ocr.batch_runner import ROUTING_TABLE, SUPPORTED_EXTENSIONS

    if not input.exists():
        raise typer.BadParameter(f"Input path does not exist: {input}")

    classification = triage.classify_file(input, allowed_extensions=sorted(SUPPORTED_EXTENSIONS))

    result: dict[str, Any] = {
        "doc_type": classification.doc_type,
        "record_type": classification.record_type,
        "layout_variant": classification.layout_variant,
        "is_jawi": classification.is_jawi,
        "jawi_proportion": classification.jawi_proportion,
        "notes": classification.notes,
    }

    if classification.is_jawi:
        result["status"] = "SKIPPED_JAWI"
        result["config_path"] = None
    else:
        route_key = (classification.doc_type, classification.record_type, classification.layout_variant)
        routed_config = ROUTING_TABLE.get(route_key)
        if routed_config is None:
            result["status"] = (
                "BLOCKED_NO_TEMPLATE" if classification.doc_type == "typed" else "NEEDS_MANUAL_CLASSIFICATION"
            )
            result["config_path"] = None
        else:
            result["status"] = "ROUTABLE"
            result["config_path"] = routed_config

    print(json.dumps(result, ensure_ascii=False))


@app.command()
def process(
    input: Path = typer.Option(..., "--input", "-i", help="Input image, PDF, or folder"),
    output: Path = typer.Option(..., "--output", "-o", help="Output CSV or XLSX path"),
    debug: Path = typer.Option(Path("debug"), "--debug", help="Debug output folder when artifacts are retained"),
    config: Path = typer.Option(Path("config/default.yaml"), "--config", help="Config file"),
    reset_output: bool = typer.Option(False, "--reset-output", help="Delete old output before processing"),
    layout_only: bool = typer.Option(False, "--layout-only", help="Only detect layout/crops"),
    skip_existing: bool = typer.Option(False, "--skip-existing", help="Skip duplicate records"),
) -> None:
    runtime: LoggingRuntime | None = None

    try:
        cfg, _, runtime = _load_command_runtime("process", config)
        logger = get_logger("marriage_ocr.process")
        retain_debug_artifacts = bool(cfg.get("debug", {}).get("retain_artifacts", False))

        console.print("[bold green]Marriage OCR process started[/bold green]")
        console.print(f"Input: {input}")
        console.print(f"Output: {output}")
        console.print(f"Debug artifacts: {'retained at ' + str(debug) if retain_debug_artifacts else 'disabled'}")
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

        result = process_input(
            input_path=input,
            output_path=output,
            debug_path=debug,
            config_path=config,
            retain_debug_artifacts=retain_debug_artifacts,
            reset_output=reset_output,
            layout_only=layout_only,
            skip_existing=skip_existing,
            progress_callback=_print_progress,
        )
        console.print(f"Refinement retry OCR calls: {result.refinement_ocr_calls}")
        if result.failed_pages:
            # A page failing no longer crashes the whole run (see process_input's
            # per-page error handling), but a caller driving this CLI as a
            # subprocess -- e.g. marriage-be's job executor -- decides success or
            # failure purely from the exit code. Exiting non-zero here, even
            # though good pages' records were already exported, is what makes a
            # partial failure show up as a FAILED (and therefore retryable) job
            # instead of silently reporting success with missing data.
            console.print(
                f"[bold red]{len(result.failed_pages)} page(s) failed and were skipped: "
                f"{result.failed_pages}[/bold red]"
            )
            raise typer.Exit(code=1)
    except typer.BadParameter:
        raise
    except typer.Exit:
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


@app.command("process-typed")
def process_typed(
    input_path: Path = typer.Option(..., "--input", help="Input PDF folder or file"),
    output_path: Path = typer.Option(Path("output/typed_records.csv"), "--output", help="Typed CSV output path"),
    debug_path: Path = typer.Option(Path("debug/typed"), "--debug", help="Typed debug output folder"),
    config_path: Path = typer.Option(Path("config/typed_borang4b.yaml"), "--config", help="Typed config file"),
    reset_output: bool = typer.Option(False, "--reset-output", help="Delete old typed CSV before processing"),
    skip_existing: bool = typer.Option(False, "--skip-existing", help="Skip typed rows already processed successfully"),
) -> None:
    runtime: LoggingRuntime | None = None

    try:
        cfg, _, runtime = _load_command_runtime("process-typed", config_path)
        logger = get_logger("marriage_ocr.process_typed")
        retain_debug_artifacts = bool(cfg.get("debug", {}).get("retain_artifacts", False))
        console.print("[bold green]Marriage OCR typed process started[/bold green]")
        console.print(f"Input: {input_path}")
        console.print(f"Output: {output_path}")
        console.print(
            f"Debug artifacts: {'retained at ' + str(debug_path) if retain_debug_artifacts else 'disabled'}"
        )
        console.print(f"Config: {config_path}")
        console.print(f"Log file: {runtime.log_path}")
        console.print(f"Reset output: {reset_output}")
        console.print(f"Skip existing: {skip_existing}")
        logger.info(
            "Typed process started input=%s output=%s debug=%s reset_output=%s skip_existing=%s",
            input_path,
            output_path,
            debug_path,
            reset_output,
            skip_existing,
        )

        if not input_path.exists():
            raise typer.BadParameter(f"Input path does not exist: {input_path}")
        if not config_path.exists():
            raise typer.BadParameter(f"Config path does not exist: {config_path}")

        from marriage_ocr.typed.pipeline import process_typed_input

        result = process_typed_input(
            input_path=input_path,
            output_path=output_path,
            debug_path=debug_path,
            config_path=config_path,
            reset_output=reset_output,
            skip_existing=skip_existing,
            retain_debug_artifacts=retain_debug_artifacts,
        )
        console.print(
            f"Typed OCR complete: discovered={result.discovered_pdfs} written={result.written_rows} skipped={len(result.skipped_files)}"
        )
    except typer.BadParameter:
        raise
    except Exception as error:
        _handle_command_error(
            error,
            command_name="process-typed",
            config_path=config_path,
            runtime=runtime,
            extra_context={
                "input": str(input_path),
                "output": str(output_path),
                "debug": str(debug_path),
                "reset_output": reset_output,
                "skip_existing": skip_existing,
            },
        )


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


@onedrive_app.command("fetch-public")
def onedrive_fetch_public(
    url: str = typer.Option(..., "--url", help="OneDrive/SharePoint sharing link"),
    dest: Path = typer.Option(..., "--dest", help="Local folder to save the downloaded file(s) into"),
) -> None:
    """Download a link shared as "Anyone with the link" with a plain HTTP
    GET -- no Microsoft sign-in, no Entra ID app registration, no
    --client-id needed. Use this when you can't set up `onedrive login`
    (e.g. no Entra ID access) and the client is able to share the link as
    "Anyone". Fails with a clear error if the link actually requires
    sign-in; fall back to `onedrive login` + `onedrive fetch` in that case."""
    from marriage_ocr.onedrive_ingest import download_anonymous_share

    console.print("[bold green]OneDrive anonymous fetch[/bold green]")
    console.print(f"URL: {url}")
    console.print(f"Dest: {dest}")

    downloaded = download_anonymous_share(url, dest)

    console.print(f"[bold green]Downloaded {len(downloaded)} file(s) to {dest}[/bold green]")


@onedrive_app.command("login")
def onedrive_login(
    client_id: str = typer.Option(
        ..., "--client-id", envvar="MARRIAGE_OCR_ONEDRIVE_CLIENT_ID",
        help="Azure AD app (client) ID from your app registration",
    ),
    token_cache: Path = typer.Option(
        DEFAULT_ONEDRIVE_TOKEN_CACHE, "--token-cache",
        help="Local file to cache the login token in (gitignored)",
    ),
) -> None:
    """One-time interactive device-code login. Run this once; `onedrive
    fetch` reuses and silently refreshes the cached token afterwards."""
    from marriage_ocr.onedrive_ingest import OneDriveClient

    console.print("[bold green]OneDrive login[/bold green]")
    client = OneDriveClient(client_id=client_id, token_cache_path=token_cache)
    client.login()
    console.print(f"Login cached at {token_cache}")


@onedrive_app.command("fetch")
def onedrive_fetch(
    url: str = typer.Option(..., "--url", help="OneDrive sharing link"),
    dest: Path = typer.Option(..., "--dest", help="Local folder to mirror the share into"),
    client_id: str = typer.Option(
        ..., "--client-id", envvar="MARRIAGE_OCR_ONEDRIVE_CLIENT_ID",
        help="Azure AD app (client) ID from your app registration",
    ),
    token_cache: Path = typer.Option(
        DEFAULT_ONEDRIVE_TOKEN_CACHE, "--token-cache",
        help="Local file the login token is cached in (see `onedrive login`)",
    ),
) -> None:
    """Resolve a OneDrive share link and mirror its full contents under
    --dest, preserving folder structure. Point --input at --dest afterwards
    for `process` / `process-typed` / the batch orchestrator."""
    from marriage_ocr.onedrive_ingest import OneDriveClient

    console.print("[bold green]OneDrive fetch[/bold green]")
    console.print(f"URL: {url}")
    console.print(f"Dest: {dest}")

    client = OneDriveClient(client_id=client_id, token_cache_path=token_cache)
    downloaded = client.download_share(url, dest)

    console.print(f"[bold green]Downloaded {len(downloaded)} file(s) to {dest}[/bold green]")


@gemini_batch_app.command("run")
def gemini_batch_run(
    debug_root: Path = typer.Option(
        ..., "--debug-root",
        help="Debug folder from a prior run (needs debug.retain_artifacts: true) to re-extract",
    ),
    config: Path = typer.Option(
        Path("config/default.yaml"), "--config",
        help="Config file to read llm/validation/record_type settings from",
    ),
    display_name: str | None = typer.Option(
        None, "--display-name", help="Optional label for the batch job in Google's console",
    ),
    poll_seconds: float = typer.Option(
        30.0, "--poll-seconds", help="Seconds between job-status checks while waiting",
    ),
    timeout_seconds: float | None = typer.Option(
        None, "--timeout-seconds", help="Give up waiting after this long (default: wait indefinitely)",
    ),
    sync_db: bool = typer.Option(
        False, "--sync-db",
        help=(
            "Write merged results back into the production Postgres `records` table "
            "(matched by source_file/source_page/source_record), so a batch_runner.run_batch "
            "run done with llm.enabled: false actually ends up with Gemini-refined values in "
            "the DB instead of just in validated_record.json on disk."
        ),
    ),
) -> None:
    """Submit every record crop under --debug-root as one Gemini Batch Mode
    job (~50% cheaper than the synchronous path), wait for it, and overwrite
    each record's validated_record.json with the merged result -- same
    format the Streamlit review UI already reads.

    Only useful against a run that had `llm.enabled: false` (so Gemini was
    never called synchronously) and `debug.retain_artifacts: true` (so
    full_record.jpg / raw_ocr.json / parsed_record.json exist per record).
    This does not touch `process`/`process-typed`/`batch_runner` at all --
    Batch Mode's async, wait-for-the-whole-job shape can't fit their
    real-time per-record flow.
    """
    from llm import run_batch_extraction

    runtime: LoggingRuntime | None = None

    try:
        cfg, _, runtime = _load_command_runtime("gemini-batch-run", config)
        llm_cfg = dict(cfg.get("llm", {}))
        ocr_cfg = cfg.get("ocr", {})
        validation_input = {
            **cfg.get("validation", {}),
            "min_average_confidence": ocr_cfg.get("min_average_confidence", 0.50),
        }
        record_type = str(cfg.get("record_type", "nikah")).strip().lower()
        layout_variant = str(cfg.get("layout_variant", "legacy")).strip().lower()
        llm_cfg.setdefault("record_type", record_type)
        llm_cfg.setdefault("layout_variant", layout_variant)
        llm_cfg["save_raw_json"] = False

        console.print("[bold green]Gemini batch run started[/bold green]")
        console.print(f"Debug root: {debug_root}")
        console.print(f"Config: {config}")
        console.print(f"Log file: {runtime.log_path}")

        summary = run_batch_extraction(
            debug_root,
            llm_config=llm_cfg,
            validation_config=validation_input,
            record_type=record_type,
            prefer_gemini_threshold=float(llm_cfg.get("prefer_gemini_threshold", 0.70)),
            review_below_field_confidence=float(llm_cfg.get("review_below_field_confidence", 0.80)),
            display_name=display_name,
            poll_seconds=poll_seconds,
            timeout_seconds=timeout_seconds,
            skip_gemini_when_parser_ok=bool(llm_cfg.get("skip_gemini_when_parser_ok", False)),
            skip_gemini_min_confidence=float(llm_cfg.get("skip_gemini_min_confidence", 0.90)),
        )

        console.print(f"Submitted: {summary.submitted}")
        console.print(f"Merged: {summary.merged}")
        if summary.skipped_parser_confident:
            console.print(
                f"Skipped (parser already confident, no Gemini needed): {len(summary.skipped_parser_confident)}"
            )
        if summary.skipped_no_parsed_record:
            console.print(
                f"[yellow]Skipped (no parsed_record.json): {len(summary.skipped_no_parsed_record)}[/yellow]"
            )
        if summary.failed:
            console.print(f"[bold red]Failed: {len(summary.failed)}[/bold red]")
            for key, error_text in summary.failed.items():
                console.print(f"  {key}: {error_text}")

        if sync_db:
            from marriage_ocr.batch_runner import sync_gemini_batch_results

            sync_summary = sync_gemini_batch_results(summary.validated_records)
            console.print(f"Synced to DB: {sync_summary['synced']}")
            if sync_summary["skipped_no_match"]:
                console.print(
                    f"[yellow]No matching DB row (skipped): {len(sync_summary['skipped_no_match'])}[/yellow]"
                )
    except typer.BadParameter:
        raise
    except typer.Exit:
        raise
    except Exception as error:
        _handle_command_error(
            error,
            command_name="gemini-batch-run",
            config_path=config,
            runtime=runtime,
            extra_context={"debug_root": str(debug_root)},
        )


if __name__ == "__main__":
    app()
