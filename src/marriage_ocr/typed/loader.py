from __future__ import annotations

from pathlib import Path

import cv2
import fitz
import numpy as np

from marriage_ocr.typed.models import RenderedPage


def discover_typed_pdfs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            raise ValueError(f"Typed input must be a PDF or directory: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Typed input does not exist: {input_path}")
    return sorted(
        (
            path
            for path in input_path.iterdir()
            if path.is_file() and path.suffix.lower() == ".pdf"
        ),
        key=lambda path: path.name.casefold(),
    )


def render_typed_pdf(
    pdf_path: Path,
    debug_dir: Path,
    *,
    dpi: int = 300,
) -> tuple[RenderedPage, RenderedPage]:
    document = fitz.open(pdf_path)
    try:
        if document.page_count != 2:
            raise ValueError(
                f"Typed Borang 4B expected exactly 2 pages, found {document.page_count}: {pdf_path.name}"
            )
        debug_dir.mkdir(parents=True, exist_ok=True)
        scale = dpi / 72.0
        matrix = fitz.Matrix(scale, scale)
        rendered: list[RenderedPage] = []
        for page_index in range(document.page_count):
            pixmap = document.load_page(page_index).get_pixmap(matrix=matrix, alpha=False)
            if pixmap.n == 1:
                array = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                    pixmap.height, pixmap.width
                )
                image = cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
            else:
                array = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                    pixmap.height, pixmap.width, pixmap.n
                )
                image = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)

            image_path = debug_dir / f"page_{page_index + 1}.png"
            if not cv2.imwrite(str(image_path), image):
                raise OSError(f"Failed to write rendered page: {image_path}")

            rendered.append(
                RenderedPage(
                    source_pdf=pdf_path,
                    source_file=pdf_path.name,
                    page_number=page_index + 1,
                    image_path=image_path,
                    width=pixmap.width,
                    height=pixmap.height,
                )
            )
        return (rendered[0], rendered[1])
    finally:
        document.close()

