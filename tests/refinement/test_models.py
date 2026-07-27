from marriage_ocr.config import load_runtime_config
from marriage_ocr.refinement.models import (
    FieldCandidate,
    FieldRefinementAuditRow,
    FieldRefinementDecision,
    FieldRefinementSettings,
)


def test_field_candidate_preserves_original_metadata() -> None:
    candidate = FieldCandidate(
        value="AHMAD BIN ALI",
        source="typo_rule",
        validity_score=0.88,
        ocr_confidence=0.79,
        plausibility_score=0.91,
        similarity_score=0.86,
        substitutions=1,
        metadata={"rule": "B1N->BIN"},
    )

    assert candidate.source == "typo_rule"
    assert candidate.metadata["rule"] == "B1N->BIN"


def test_refinement_settings_default_when_section_missing(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("ocr:\n  engine: mock\n", encoding="utf-8")

    loaded = load_runtime_config(config_path)
    settings = FieldRefinementSettings.from_config(loaded.data)

    assert settings.enabled is True
    assert settings.max_variants_per_field == 3
    assert settings.minimum_candidate_score == 0.75


def test_decision_and_audit_row_keep_review_and_original_value() -> None:
    candidate = FieldCandidate("AHMAD BIN ALI", "typo_rule", 0.88, None, 0.91, 0.86, 1, {})
    decision = FieldRefinementDecision(
        field_name="nama_suami",
        original_value="AHMAD B1N ALI",
        selected_value=candidate.value,
        candidates=(candidate,),
        selected_candidate=candidate,
        requires_review=False,
        reason="connector typo",
    )
    audit = FieldRefinementAuditRow(
        source_file="sample.pdf",
        page_number=1,
        record_index=1,
        field_name=decision.field_name,
        original_value=decision.original_value,
        selected_value=decision.selected_value,
        original_score=0.61,
        selected_score=0.88,
        correction_type="connector_typo",
        candidate_source=candidate.source,
        reason=decision.reason,
        requires_review=decision.requires_review,
        crop_path=None,
        retry_count=0,
    )

    assert decision.selected_value == "AHMAD BIN ALI"
    assert decision.requires_review is False
    assert audit.original_value == "AHMAD B1N ALI"
    assert audit.requires_review is False
