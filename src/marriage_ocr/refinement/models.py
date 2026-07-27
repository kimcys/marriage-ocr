from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FieldCandidate:
    value: str
    source: str
    validity_score: float
    ocr_confidence: float | None
    plausibility_score: float
    similarity_score: float
    substitutions: int
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class FieldRefinementDecision:
    field_name: str
    original_value: str | None
    selected_value: str | None
    candidates: tuple[FieldCandidate, ...] = ()
    selected_candidate: FieldCandidate | None = None
    requires_review: bool = True
    reason: str = ""


@dataclass(frozen=True)
class FieldRefinementSettings:
    enabled: bool = True
    max_variants_per_field: int = 3
    minimum_candidate_score: float = 0.75

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> FieldRefinementSettings:
        ocr = config.get("ocr", {})
        section = ocr.get("field_refinement", {}) if isinstance(ocr, dict) else {}
        if not isinstance(section, dict):
            section = {}
        return cls(
            enabled=bool(section.get("enabled", cls.enabled)),
            max_variants_per_field=int(
                section.get("max_variants_per_field", cls.max_variants_per_field)
            ),
            minimum_candidate_score=float(
                section.get("minimum_candidate_score", cls.minimum_candidate_score)
            ),
        )


@dataclass(frozen=True)
class FieldRefinementAuditRow:
    source_file: str
    page_number: int
    record_index: int
    field_name: str
    original_value: str | None
    selected_value: str | None
    original_score: float
    selected_score: float
    correction_type: str
    candidate_source: str
    reason: str
    requires_review: bool
    crop_path: str | None
    retry_count: int
