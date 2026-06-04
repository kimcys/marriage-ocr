from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import cv2
import numpy as np


_SANITIZE_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class DocumentPage:
    source_path: Path
    relative_source: Path
    page_index: int
    total_pages: int
    image: np.ndarray
    debug_name: str

    @property
    def source_page(self) -> int:
        return self.page_index + 1


def load_document_pages(
    input_path: Path,
    allowed_extensions: Iterable[str],
    *,
    pdf_dpi: int = 200,
) -> list[DocumentPage]:
    normalized_extensions = {extension.lower() for extension in allowed_extensions}
    input_files = _collect_input_files(input_path, normalized_extensions)
    base_dir = input_path if input_path.is_dir() else input_path.parent

    pages: list[DocumentPage] = []
    for document_path in input_files:
        relative_source = document_path.relative_to(base_dir)
        suffix = document_path.suffix.lower()
        if suffix == ".pdf":
            pages.extend(_render_pdf_pages(document_path, relative_source, pdf_dpi))
            continue

        pages.append(
            DocumentPage(
                source_path=document_path,
                relative_source=relative_source,
                page_index=0,
                total_pages=1,
                image=_load_raster_image(document_path),
                debug_name=_build_debug_name(relative_source, 0, 1),
            )
        )

    return pages


def write_image(path: Path, image: np.ndarray, *, format_suffix: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = format_suffix or path.suffix or ".jpg"
    success, encoded = cv2.imencode(suffix, image)
    if not success:
        raise ValueError(f"Failed to encode image for {path}")

    path.write_bytes(encoded.tobytes())


def _collect_input_files(input_path: Path, allowed_extensions: set[str]) -> list[Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() not in allowed_extensions:
            raise ValueError(
                f"Unsupported input file type: {input_path.suffix}. "
                f"Allowed: {sorted(allowed_extensions)}"
            )
        return [input_path]

    files = [
        path
        for path in sorted(input_path.rglob("*"))
        if path.is_file() and path.suffix.lower() in allowed_extensions
    ]
    if files:
        return files

    raise FileNotFoundError(
        f"No supported input files found under {input_path}. "
        f"Allowed: {sorted(allowed_extensions)}"
    )


def _load_raster_image(path: Path) -> np.ndarray:
    image_bytes = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to load image: {path}")
    return image


def _render_pdf_pages(path: Path, relative_source: Path, pdf_dpi: int) -> list[DocumentPage]:
    if pdf_dpi <= 0:
        raise ValueError(f"pdf_dpi must be positive, got {pdf_dpi}")

    try:
        import fitz
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PDF input requires PyMuPDF (`fitz`). Install project dependencies "
            "or remove PDF files from the input batch."
        ) from exc

    pages: list[DocumentPage] = []
    zoom = pdf_dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    with fitz.open(path) as document:
        total_pages = document.page_count
        for page_index in range(total_pages):
            page = document.load_page(page_index)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)

            image = np.frombuffer(pixmap.samples, dtype=np.uint8)
            if pixmap.n == 1:
                grayscale = image.reshape(pixmap.height, pixmap.width)
                page_image = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)
            else:
                rgb = image.reshape(pixmap.height, pixmap.width, pixmap.n)
                page_image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            pages.append(
                DocumentPage(
                    source_path=path,
                    relative_source=relative_source,
                    page_index=page_index,
                    total_pages=total_pages,
                    image=page_image.copy(),
                    debug_name=_build_debug_name(relative_source, page_index, total_pages),
                )
            )

    return pages


def _build_debug_name(relative_source: Path, page_index: int, total_pages: int) -> str:
    stem_parts = [_sanitize_path_part(part) for part in relative_source.with_suffix("").parts]
    stem = "__".join(part for part in stem_parts if part) or "page"
    if relative_source.suffix.lower() == ".pdf" or total_pages > 1:
        return f"{stem}_page_{page_index + 1:03d}"
    return stem


def _sanitize_path_part(value: str) -> str:
    sanitized = _SANITIZE_PATTERN.sub("_", value).strip("._")
    return sanitized or "page"
