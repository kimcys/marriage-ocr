from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Mapping


DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


@dataclass(frozen=True)
class LoggingRuntime:
    command_name: str
    log_dir: Path
    log_path: Path
    error_report_dir: Path
    level_name: str


def setup_logging(command_name: str, config: Mapping[str, Any] | None = None) -> LoggingRuntime:
    logging_config = dict(config or {})
    level_name = str(logging_config.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)

    log_dir = Path(str(logging_config.get("directory", "logs")))
    error_report_dir = Path(str(logging_config.get("error_report_dir", log_dir / "error_reports")))
    filename_prefix = str(logging_config.get("filename_prefix", "marriage-ocr"))
    max_bytes = int(logging_config.get("max_bytes", 5 * 1024 * 1024))
    backup_count = int(logging_config.get("backup_count", 5))
    enable_console = bool(logging_config.get("console", True))
    enable_file = bool(logging_config.get("file", True))

    log_dir.mkdir(parents=True, exist_ok=True)
    error_report_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"{filename_prefix}-{command_name}-{timestamp}.log"

    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()

    root_logger.setLevel(level)
    formatter = logging.Formatter(DEFAULT_FORMAT, datefmt=DEFAULT_DATE_FORMAT)

    if enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    if enable_file:
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    else:
        log_path.touch()

    logging.captureWarnings(True)
    return LoggingRuntime(
        command_name=command_name,
        log_dir=log_dir,
        log_path=log_path,
        error_report_dir=error_report_dir,
        level_name=level_name,
    )


def get_logger(name: str = "marriage_ocr") -> logging.Logger:
    return logging.getLogger(name)
