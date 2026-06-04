from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import yaml


ENV_FILE_VARIABLE = "MARRIAGE_OCR_ENV_FILE"
ENV_NESTED_PREFIX = "MARRIAGE_OCR__"
DEFAULT_ENV_FILE = ".env"

ENV_ALIAS_PATHS: dict[str, tuple[str, ...]] = {
    "MARRIAGE_OCR_OCR_ENGINE": ("ocr", "engine"),
    "MARRIAGE_OCR_OCR_MIN_AVERAGE_CONFIDENCE": ("ocr", "min_average_confidence"),
    "MARRIAGE_OCR_PDF_DPI": ("input", "pdf_dpi"),
    "MARRIAGE_OCR_LOG_LEVEL": ("logging", "level"),
    "MARRIAGE_OCR_LOG_DIR": ("logging", "directory"),
    "MARRIAGE_OCR_ERROR_DIR": ("logging", "error_report_dir"),
    "MARRIAGE_OCR_REVIEW_EXPORT_PATH": ("review", "export_path"),
    "MARRIAGE_OCR_REVIEWER_NAME": ("review", "reviewer_name"),
    "MARRIAGE_OCR_TRAINING_OUTPUT_DIR": ("training_export", "output_dir"),
    "MARRIAGE_OCR_VALIDATION_RATIO": ("training_export", "validation_ratio"),
}


@dataclass(frozen=True)
class LoadedConfig:
    data: dict[str, Any]
    env_file: Path | None
    effective_env: dict[str, str]


def load_runtime_config(
    config_path: str | Path,
    *,
    env: Mapping[str, str] | None = None,
) -> LoadedConfig:
    config_file = Path(config_path)
    effective_env = _effective_environment(config_file, env=env)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with config_file.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    env_overrides = _build_env_overrides(effective_env)
    merged = _deep_merge(data, env_overrides)
    env_file = _resolve_env_file(config_file, effective_env)

    return LoadedConfig(
        data=merged,
        env_file=env_file if env_file.exists() else None,
        effective_env=effective_env,
    )


def _effective_environment(
    config_path: Path,
    *,
    env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    base_env = dict(env or os.environ)
    env_file = _resolve_env_file(config_path, base_env)
    env_values = load_env_file(env_file) if env_file.exists() else {}
    env_values.update(base_env)
    return env_values


def load_env_file(path: str | Path) -> dict[str, str]:
    env_path = Path(path)
    if not env_path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue

        key, raw_value = line.split("=", 1)
        key = key.strip()
        value = _strip_wrapping_quotes(raw_value.strip())
        values[key] = value
    return values


def _resolve_env_file(config_path: Path, env: Mapping[str, str]) -> Path:
    raw_path = env.get(ENV_FILE_VARIABLE, DEFAULT_ENV_FILE)
    env_path = Path(raw_path)
    if env_path.is_absolute():
        return env_path
    return config_path.parent.parent / env_path if not env_path.exists() else env_path


def _build_env_overrides(env: Mapping[str, str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}

    for env_name, config_path in ENV_ALIAS_PATHS.items():
        if env_name not in env:
            continue
        _set_nested_value(overrides, config_path, _parse_env_scalar(env[env_name]))

    for env_name, raw_value in env.items():
        if not env_name.startswith(ENV_NESTED_PREFIX):
            continue
        remainder = env_name[len(ENV_NESTED_PREFIX) :]
        if not remainder:
            continue
        path = tuple(part.strip().lower() for part in remainder.split("__") if part.strip())
        if not path:
            continue
        _set_nested_value(overrides, path, _parse_env_scalar(raw_value))

    return overrides


def _parse_env_scalar(raw_value: str) -> Any:
    if raw_value == "":
        return ""

    parsed = yaml.safe_load(raw_value)
    if parsed is None and raw_value.strip().upper() != "NULL":
        return raw_value
    return parsed


def _set_nested_value(target: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    cursor = target
    for part in path[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            child = {}
            cursor[part] = child
        cursor = child
    cursor[path[-1]] = value


def _deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for key, value in override.items():
            if key in merged:
                merged[key] = _deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged
    return override


def _strip_wrapping_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value
