from __future__ import annotations

from dataclasses import replace
from difflib import SequenceMatcher
from pathlib import Path
import tempfile
from typing import Callable

from marriage_ocr.models import OcrResult
from marriage_ocr.ocr import OcrEngine, read_ocr_images
from marriage_ocr.refinement.models import (
    FieldCandidate,
    FieldRefinementDecision,
    FieldRefinementSettings,
)
from marriage_ocr.refinement.preprocess import RetryVariant, build_retry_variants, cleanup_retry_variants
from marriage_ocr.refinement.text_corrections import (
    generate_date_candidates,
    generate_ic_candidates,
    generate_name_candidates,
    is_suspicious_name,
    is_valid_date,
    is_valid_malaysian_ic,
)

CandidateGenerator = Callable[[str | None], list[FieldCandidate]]
SuspicionCheck = Callable[[str | None], bool]


def refine_field(
    field_name: str,
    original_value: str | None,
    parsed_value: str | None = None,
    *,
    crop_path: Path | None,
    engine: OcrEngine | None = None,
    ocr_engine: OcrEngine | None = None,
    settings: FieldRefinementSettings | None = None,
    save_retry_dir: Path | None = None,
) -> FieldRefinementDecision:
    active_engine = ocr_engine or engine
    if active_engine is None:
        raise ValueError("An OCR engine must be provided")

    active_settings = settings or FieldRefinementSettings()
    candidate_generator, is_suspicious = _field_strategy(field_name)
    base_value = parsed_value if parsed_value is not None else original_value

    initial_candidates = _merge_candidates(
        [
            _build_original_candidate(field_name=field_name, original_value=original_value, base_value=base_value),
            *_decorate_candidates(
                candidate_generator(base_value),
                field_name=field_name,
                original_value=original_value,
                parsed_value=parsed_value,
                source_prefix=None,
                ocr_result=None,
            ),
        ]
    )
    selected_initial = _select_best_candidate(initial_candidates, reference_value=base_value or original_value)

    if not active_settings.enabled:
        return _decision_from_candidates(
            field_name=field_name,
            original_value=original_value,
            selected_candidate=selected_initial,
            candidates=initial_candidates,
            reason="refinement_disabled",
            requires_review_fallback=False,
        )

    if not _should_retry(
        field_name=field_name,
        value=base_value,
        selected_candidate=selected_initial,
        is_suspicious=is_suspicious,
        settings=active_settings,
    ):
        return _decision_from_candidates(
            field_name=field_name,
            original_value=original_value,
            selected_candidate=selected_initial,
            candidates=initial_candidates,
            reason="accepted_without_retry",
            requires_review_fallback=False,
        )

    if crop_path is None or not crop_path.exists():
        return _fallback_decision(field_name=field_name, original_value=original_value, reason="missing_crop")

    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    retry_dir = save_retry_dir
    if retry_dir is None and not active_settings.save_retry_images:
        temp_dir = tempfile.TemporaryDirectory(prefix="marriage-ocr-retry-")
        retry_dir = Path(temp_dir.name)
    elif retry_dir is None:
        retry_dir = crop_path.parent / f"{crop_path.stem}.retry"

    variants: list[RetryVariant] = []
    retry_candidates: list[FieldCandidate] = []
    retry_errors = 0

    try:
        variants = build_retry_variants(crop_path, save_dir=retry_dir)
        max_variants = max(1, int(active_settings.max_variants_per_field))
        for attempt_index, variant in enumerate(variants[:max_variants], start=1):
            try:
                retry_result = active_engine.read_image(variant.image_path)
            except Exception:
                retry_errors += 1
                continue
            retry_candidates.extend(
                _decorate_candidates(
                    candidate_generator(retry_result.text),
                    field_name=field_name,
                    original_value=original_value,
                    parsed_value=parsed_value,
                    source_prefix=variant.source,
                    ocr_result=retry_result,
                    variant_name=variant.name,
                    retry_attempt=attempt_index,
                )
            )
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()
        elif save_retry_dir is None and not active_settings.save_retry_images and variants:
            cleanup_retry_variants(variants)

    combined_candidates = _merge_candidates([*initial_candidates, *retry_candidates])
    selected_candidate = _select_best_candidate(combined_candidates, reference_value=base_value or original_value)
    original_candidate = _find_original_candidate(initial_candidates)

    if selected_candidate is None:
        return _fallback_decision(field_name=field_name, original_value=original_value, reason="no_valid_candidates")

    if original_candidate is None and _candidate_score(selected_candidate, reference_value=base_value or original_value) >= active_settings.minimum_candidate_score:
        return _decision_from_candidates(
            field_name=field_name,
            original_value=original_value,
            selected_candidate=selected_candidate,
            candidates=combined_candidates,
            reason="accepted_after_retry",
            requires_review_fallback=False,
        )

    original_score = _candidate_score(original_candidate, reference_value=base_value or original_value) if original_candidate else 0.0
    selected_score = _candidate_score(selected_candidate, reference_value=base_value or original_value)
    score_improvement = selected_score - original_score

    if (
        retry_candidates
        and selected_candidate.source.startswith("retry_")
        and selected_score >= active_settings.minimum_candidate_score
        and score_improvement >= active_settings.minimum_score_improvement
    ):
        return _decision_from_candidates(
            field_name=field_name,
            original_value=original_value,
            selected_candidate=selected_candidate,
            candidates=combined_candidates,
            reason="accepted_after_retry",
            requires_review_fallback=False,
        )

    if retry_candidates or retry_errors:
        return _decision_from_candidates(
            field_name=field_name,
            original_value=original_value,
            selected_candidate=original_candidate or selected_candidate,
            candidates=combined_candidates,
            reason="retry_candidates_below_threshold" if retry_candidates else "retry_ocr_failed",
            requires_review_fallback=True,
        )

    return _decision_from_candidates(
        field_name=field_name,
        original_value=original_value,
        selected_candidate=selected_candidate,
        candidates=combined_candidates,
        reason="accepted_without_retry",
        requires_review_fallback=True,
    )


def _field_strategy(field_name: str) -> tuple[CandidateGenerator, SuspicionCheck]:
    if field_name.startswith("tarikh_"):
        return (
            lambda value: generate_date_candidates(value, field_name=field_name),
            lambda value: not is_valid_date(value),
        )
    if field_name.startswith("ic_") or field_name.startswith("id_"):
        return (
            lambda value: generate_ic_candidates(value, field_name=field_name),
            lambda value: not is_valid_malaysian_ic(value),
        )
    return (
        lambda value: generate_name_candidates(value, field_name=field_name),
        is_suspicious_name,
    )


def _decorate_candidates(
    candidates: list[FieldCandidate],
    *,
    field_name: str,
    original_value: str | None,
    parsed_value: str | None,
    source_prefix: str | None,
    ocr_result: OcrResult | None,
    variant_name: str | None = None,
    retry_attempt: int | None = None,
) -> list[FieldCandidate]:
    decorated: list[FieldCandidate] = []
    for candidate in candidates:
        metadata = dict(candidate.metadata)
        metadata["field_name"] = field_name
        metadata["original_value"] = original_value
        metadata["parsed_value"] = parsed_value
        if variant_name is not None:
            metadata["retry_variant"] = variant_name
        if retry_attempt is not None:
            metadata["retry_attempt"] = retry_attempt
        if ocr_result is not None:
            metadata["ocr_text"] = ocr_result.text
            metadata["ocr_confidence"] = ocr_result.average_confidence
        source = candidate.source
        if source_prefix is not None:
            source = source_prefix
        decorated.append(
            replace(
                candidate,
                source=source,
                validity_score=_field_validity_score(field_name, candidate.value),
                plausibility_score=_field_plausibility_score(field_name, candidate.value),
                ocr_confidence=ocr_result.average_confidence if ocr_result is not None else candidate.ocr_confidence,
                metadata=metadata,
            )
        )
    return decorated


def _build_original_candidate(
    *,
    field_name: str,
    original_value: str | None,
    base_value: str | None,
) -> FieldCandidate | None:
    if base_value is None:
        return None
    normalized = str(base_value).strip()
    if not normalized:
        return None
    return FieldCandidate(
        value=normalized,
        source="original_ocr",
        validity_score=_field_validity_score(field_name, normalized),
        ocr_confidence=None,
        plausibility_score=_field_plausibility_score(field_name, normalized),
        similarity_score=1.0,
        substitutions=0,
        metadata={
            "field_name": field_name,
            "original_value": original_value,
            "parsed_value": base_value,
            "requires_retry_ocr": False,
            "requires_review": False,
        },
    )


def _should_retry(
    *,
    field_name: str,
    value: str | None,
    selected_candidate: FieldCandidate | None,
    is_suspicious: SuspicionCheck,
    settings: FieldRefinementSettings,
) -> bool:
    if value is None or not str(value).strip():
        return True

    if field_name.startswith("tarikh_") and not settings.retry_dates:
        return False
    if (field_name.startswith("ic_") or field_name.startswith("id_")) and not settings.retry_ic_numbers:
        return False
    if not (field_name.startswith("tarikh_") or field_name.startswith("ic_") or field_name.startswith("id_")) and not settings.retry_names:
        return False

    if is_suspicious(value):
        return True

    if selected_candidate is not None and selected_candidate.source == "typo_rule":
        return True

    return False


def _merge_candidates(candidates: list[FieldCandidate]) -> list[FieldCandidate]:
    merged: dict[str, FieldCandidate] = {}
    for candidate in candidates:
        if candidate is None:
            continue
        current = merged.get(candidate.value)
        if current is None or _candidate_sort_key(candidate) < _candidate_sort_key(current):
            merged[candidate.value] = candidate
    return sorted(merged.values(), key=_candidate_sort_key)


def _select_best_candidate(candidates: list[FieldCandidate], *, reference_value: str | None) -> FieldCandidate | None:
    if not candidates:
        return None
    return sorted(candidates, key=lambda candidate: _candidate_sort_key(candidate, reference_value=reference_value))[0]


def _find_original_candidate(candidates: list[FieldCandidate]) -> FieldCandidate | None:
    for candidate in candidates:
        if candidate.source == "original_ocr":
            return candidate
    return candidates[0] if candidates else None


def _candidate_sort_key(
    candidate: FieldCandidate,
    *,
    reference_value: str | None = None,
) -> tuple[float, bool, int, float, str, str]:
    return (
        -_candidate_score(candidate, reference_value=reference_value),
        bool(candidate.metadata.get("requires_review", False)),
        candidate.substitutions,
        -candidate.validity_score,
        candidate.source,
        candidate.value,
    )


def _candidate_score(candidate: FieldCandidate | None, *, reference_value: str | None) -> float:
    if candidate is None:
        return 0.0

    if candidate.ocr_confidence is None:
        validity_weight = 0.475
        plausibility_weight = 0.375
        similarity_weight = 0.15
        ocr_weight = 0.0
    else:
        validity_weight = 0.35
        plausibility_weight = 0.25
        similarity_weight = 0.15
        ocr_weight = 0.25

    similarity = candidate.similarity_score
    if reference_value is not None:
        similarity = max(
            similarity,
            SequenceMatcher(None, str(reference_value).strip(), candidate.value.strip()).ratio(),
        )

    source_bonus = 0.0
    if candidate.source.startswith("retry_"):
        source_bonus = 0.10
    elif candidate.source == "typo_rule":
        source_bonus = -0.05
    elif candidate.source == "safe_normalisation":
        source_bonus = -0.02

    return (
        candidate.validity_score * validity_weight
        + candidate.plausibility_score * plausibility_weight
        + similarity * similarity_weight
        + (candidate.ocr_confidence or 0.0) * ocr_weight
        + source_bonus
    )


def _field_validity_score(field_name: str, value: str) -> float:
    if field_name.startswith("tarikh_"):
        return 1.0 if is_valid_date(value) else 0.0
    if field_name.startswith("ic_") or field_name.startswith("id_"):
        return 1.0 if is_valid_malaysian_ic(value) else 0.0
    return 1.0 if not is_suspicious_name(value) else 0.25


def _field_plausibility_score(field_name: str, value: str) -> float:
    if field_name.startswith("tarikh_"):
        return 1.0 if is_valid_date(value) else 0.0
    if field_name.startswith("ic_") or field_name.startswith("id_"):
        return 1.0 if is_valid_malaysian_ic(value) else 0.0
    return 1.0 if not is_suspicious_name(value) else 0.35


def _decision_from_candidates(
    *,
    field_name: str,
    original_value: str | None,
    selected_candidate: FieldCandidate | None,
    candidates: list[FieldCandidate],
    reason: str,
    requires_review_fallback: bool,
) -> FieldRefinementDecision:
    if selected_candidate is None:
        return _fallback_decision(field_name=field_name, original_value=original_value, reason=reason)
    return FieldRefinementDecision(
        field_name=field_name,
        original_value=original_value,
        selected_value=selected_candidate.value,
        candidates=tuple(candidates),
        selected_candidate=selected_candidate,
        requires_review=bool(selected_candidate.metadata.get("requires_review", False) or requires_review_fallback),
        reason=reason,
    )


def _fallback_decision(
    *,
    field_name: str,
    original_value: str | None,
    reason: str,
) -> FieldRefinementDecision:
    return FieldRefinementDecision(
        field_name=field_name,
        original_value=original_value,
        selected_value=original_value,
        candidates=(),
        selected_candidate=None,
        requires_review=True,
        reason=reason,
    )
