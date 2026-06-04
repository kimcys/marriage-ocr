from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import sys
import traceback
from typing import Any, Mapping
from uuid import uuid4

from marriage_ocr.logging_config import LoggingRuntime


@dataclass(frozen=True)
class ErrorReport:
    error_id: str
    report_path: Path


def write_error_report(
    error: BaseException,
    *,
    command_name: str,
    runtime: LoggingRuntime | None = None,
    config_path: str | Path | None = None,
    extra_context: Mapping[str, Any] | None = None,
    argv: list[str] | None = None,
    env: Mapping[str, str] | None = None,
) -> ErrorReport:
    timestamp = datetime.now()
    error_id = uuid4().hex[:12]
    report_dir = runtime.error_report_dir if runtime is not None else Path("logs/error_reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    report_path = report_dir / f"error-{command_name}-{timestamp.strftime('%Y%m%d-%H%M%S')}-{error_id}.json"
    environment = dict(env or os.environ)
    payload = {
        "error_id": error_id,
        "timestamp": timestamp.isoformat(timespec="seconds"),
        "command_name": command_name,
        "argv": list(argv or sys.argv),
        "cwd": str(Path.cwd()),
        "config_path": str(config_path) if config_path is not None else None,
        "log_path": str(runtime.log_path) if runtime is not None else None,
        "python_version": sys.version,
        "platform": platform.platform(),
        "exception": {
            "type": error.__class__.__name__,
            "message": str(error),
            "traceback": traceback.format_exception(type(error), error, error.__traceback__),
        },
        "context": dict(extra_context or {}),
        "environment": {
            key: value
            for key, value in sorted(environment.items())
            if key.startswith("MARRIAGE_OCR")
        },
    }
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return ErrorReport(error_id=error_id, report_path=report_path)
