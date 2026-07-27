from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from marriage_ocr.models import ExtractedRecord


class ProcessingStatus(str, Enum):
    SUCCESS = "SUCCESS"
    SUCCESS_WITH_RETRY = "SUCCESS_WITH_RETRY"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class Region:
    x1: float
    y1: float
    x2: float
    y2: float

    def expand(self, ratio: float) -> "Region":
        return Region(
            max(0.0, self.x1 - ratio),
            max(0.0, self.y1 - ratio),
            min(1.0, self.x2 + ratio),
            min(1.0, self.y2 + ratio),
        )


@dataclass(frozen=True)
class TemplateTransform:
    dx: float = 0.0
    dy: float = 0.0
    scale_x: float = 1.0
    scale_y: float = 1.0
    safe: bool = True
    diagnostics: tuple[str, ...] = ()


@dataclass(frozen=True)
class PositionedWord:
    text: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    page_number: int

    @property
    def centre(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


@dataclass(frozen=True)
class RenderedPage:
    source_pdf: Path
    source_file: str
    page_number: int
    image_path: Path
    width: int
    height: int


@dataclass(frozen=True)
class PageOcrResult:
    source_file: str
    page_number: int
    words: tuple[PositionedWord, ...]
    full_text: str
    raw_response: dict[str, Any]
    error_message: str = ""


@dataclass(frozen=True)
class RawField:
    key: str
    output_name: str
    page_number: int
    region: Region
    raw_text: str
    confidence: float
    words: tuple[PositionedWord, ...] = ()


@dataclass(frozen=True)
class FieldDiagnostic:
    key: str
    output_name: str
    valid: bool
    confidence: float
    issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class ValidationSummary:
    diagnostics: dict[str, FieldDiagnostic]
    retry_fields: tuple[str, ...]
    failed_fields: tuple[str, ...]
    meaningful_field_count: int


@dataclass(frozen=True)
class RetryCrop:
    source_file: str
    field_key: str
    page_number: int
    crop_path: Path
    region: Region


@dataclass(frozen=True)
class TypedDocumentResult:
    record: ExtractedRecord
    source_file: str
    processing_status: ProcessingStatus
    failed_fields: tuple[str, ...] = ()
    retry_count: int = 0
    error_message: str = ""

    @property
    def review_required(self) -> bool:
        return self.processing_status in {
            ProcessingStatus.REVIEW_REQUIRED,
            ProcessingStatus.FAILED,
        }

    @property
    def failed_fields_text(self) -> str:
        return ";".join(self.failed_fields)


@dataclass(frozen=True)
class TypedBatchResult:
    records: tuple[TypedDocumentResult, ...]
    discovered_pdfs: int
    written_rows: int
    skipped_files: tuple[str, ...] = ()
    status_counts: dict[str, int] = field(default_factory=dict)

