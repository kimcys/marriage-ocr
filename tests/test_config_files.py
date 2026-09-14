from pathlib import Path

import pytest

from marriage_ocr.config import load_runtime_config
from marriage_ocr.exporter import EXPORT_COLUMN_TO_FIELD

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
CONFIG_FILES = sorted(CONFIG_DIR.glob("*.yaml"))


@pytest.mark.parametrize("config_path", CONFIG_FILES, ids=lambda path: path.name)
def test_export_columns_are_all_real_export_columns(config_path: Path) -> None:
    # Regression: a shipped config's own export.columns list can silently
    # drift out of sync with exporter.py's EXPORT_COLUMN_TO_FIELD whenever a
    # column is renamed or dropped there (confirmed twice this session --
    # once for Nikah's own Pemberian Lain/Kad Pengenalan-Passport Wali-
    # Saksi columns, once for Cerai/Rujuk's No Rujukan/No Telefon/Tempat
    # Rujuk/Bil Daftar Rujukan/Catatan Raw) -- _normalize_configured_columns
    # only discovers this the moment a real handwritten batch of that record
    # type is processed, failing every single document in it with
    # "Unsupported export column: <name>". This runs the same validation at
    # test time instead, against every shipped config, so a future column
    # rename/removal can't silently take an entire record type's handwritten
    # pipeline down again.
    loaded = load_runtime_config(config_path, env={})
    columns = (loaded.data.get("export") or {}).get("columns")
    if not columns:
        return
    unsupported = [column for column in columns if column not in EXPORT_COLUMN_TO_FIELD]
    assert not unsupported, f"{config_path.name} references unsupported export column(s): {unsupported}"
