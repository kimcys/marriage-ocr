from __future__ import annotations

import hashlib
from pathlib import Path


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    file_path = Path(path)

    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            hasher.update(chunk)

    return hasher.hexdigest()
