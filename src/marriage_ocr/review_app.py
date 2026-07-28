from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import streamlit as st

from marriage_ocr.models import ExtractedRecord
from marriage_ocr.review_store import (
    ALLOWED_REVIEW_STATUSES,
    ReviewBundle,
    export_reviewed_records,
    load_review_bundles,
    save_corrected_record,
)
from marriage_ocr.training_export import export_training_dataset


DEFAULT_EXPORT_CONFIG = {
    "append": False,
    "dedupe": False,
    "sheet_name": "Records",
}

EDITABLE_FIELDS = [
    ("bil", "Bil"),
    ("nama_suami", "Nama Suami"),
    ("ic_lama_suami", "IC Lama Suami"),
    ("ic_baru_suami", "IC Baru Suami"),
    ("id_suami_raw", "ID Suami Raw"),
    ("umur_suami", "Umur Suami"),
    ("nama_isteri", "Nama Isteri"),
    ("ic_lama_isteri", "IC Lama Isteri"),
    ("ic_baru_isteri", "IC Baru Isteri"),
    ("id_isteri_raw", "ID Isteri Raw"),
    ("umur_isteri", "Umur Isteri"),
    ("mas_kahwin", "Mas Kahwin"),
    ("mas_kahwin_raw", "Mas Kahwin Raw"),
    ("nama_pendaftar", "Nama Pendaftar"),
    ("alamat_pendaftar", "Alamat Pendaftar"),
    ("nama_wali", "Nama Wali"),
    ("hubungan_wali", "Hubungan Wali"),
    ("saksi_1", "Saksi 1"),
    ("saksi_2", "Saksi 2"),
    ("tarikh_nikah", "Tarikh Nikah"),
    ("tarikh_nikah_raw", "Tarikh Nikah Raw"),
    ("tarikh_keluar", "Tarikh Keluar"),
    ("tarikh_keluar_raw", "Tarikh Keluar Raw"),
    ("remarks", "Remarks"),
]

TEXTAREA_FIELDS = {
    "alamat_pendaftar",
    "remarks",
}

INT_FIELDS = {
    "umur_suami",
    "umur_isteri",
}


st.set_page_config(page_title="Marriage OCR Review", layout="wide")


def main() -> None:
    debug_root = Path(os.environ.get("MARRIAGE_OCR_DEBUG_ROOT", "debug"))
    export_path = Path(os.environ.get("MARRIAGE_OCR_REVIEW_EXPORT_PATH", "data/reviewed_exports/daftar_perkahwinan_reviewed.xlsx"))
    training_output_dir = Path(os.environ.get("MARRIAGE_OCR_TRAINING_OUTPUT_DIR", "data/ground_truth"))
    training_verified_only = _parse_bool_env("MARRIAGE_OCR_TRAINING_VERIFIED_ONLY", True)
    training_validation_ratio = _parse_float_env("MARRIAGE_OCR_TRAINING_VALIDATION_RATIO", 0.20)
    default_reviewer = os.environ.get("MARRIAGE_OCR_REVIEWER_NAME", "").strip()

    bundles = load_review_bundles(debug_root)

    st.title("Marriage OCR Human Verification")
    st.caption(f"Debug root: {debug_root}")

    if not bundles:
        st.warning("No reviewable records found under the selected debug folder.")
        return

    filtered_bundles = _render_sidebar(
        bundles,
        export_path,
        training_output_dir,
        training_verified_only=training_verified_only,
        training_validation_ratio=training_validation_ratio,
    )
    if not filtered_bundles:
        st.warning("No records match the current filters.")
        return

    selected_index = _resolve_selected_index(filtered_bundles)
    selected_bundle = filtered_bundles[selected_index]

    _render_header_metrics(filtered_bundles)
    _render_record_summary(selected_bundle)
    _render_images(selected_bundle)
    _render_raw_ocr(selected_bundle)
    _render_refinement_audit(selected_bundle)
    _render_edit_form(selected_bundle, default_reviewer=default_reviewer)


def _render_sidebar(
    bundles: list[ReviewBundle],
    export_path: Path,
    training_output_dir: Path,
    *,
    training_verified_only: bool,
    training_validation_ratio: float,
) -> list[ReviewBundle]:
    st.sidebar.header("Review Queue")

    statuses = sorted({bundle.active_record.status_review for bundle in bundles})
    selected_statuses = st.sidebar.multiselect("Status Filter", options=statuses, default=statuses)
    verified_filter = st.sidebar.selectbox("Verified Filter", options=["All", "Verified", "Unverified"], index=0)
    search_text = st.sidebar.text_input("Search", value="").strip().lower()

    filtered_bundles = []
    for bundle in bundles:
        if selected_statuses and bundle.active_record.status_review not in selected_statuses:
            continue
        if verified_filter == "Verified" and not bundle.verified:
            continue
        if verified_filter == "Unverified" and bundle.verified:
            continue
        searchable = " ".join(
            [
                bundle.display_name,
                bundle.active_record.nama_suami or "",
                bundle.active_record.nama_isteri or "",
                bundle.active_record.bil or "",
            ]
        ).lower()
        if search_text and search_text not in searchable:
            continue
        filtered_bundles.append(bundle)

    selected_label = st.sidebar.selectbox(
        "Record",
        options=[bundle.display_name for bundle in filtered_bundles],
        index=0,
        key="selected_record_label",
    )
    st.session_state["selected_record_dir"] = str(
        next(bundle.record_dir for bundle in filtered_bundles if bundle.display_name == selected_label)
    )

    st.sidebar.divider()
    st.sidebar.subheader("Export")
    export_verified_only = st.sidebar.checkbox("Verified Only", value=False)
    if st.sidebar.button("Export Corrected XLSX", use_container_width=True):
        export_path.parent.mkdir(parents=True, exist_ok=True)
        summary = export_reviewed_records(
            debug_root=Path(os.environ.get("MARRIAGE_OCR_DEBUG_ROOT", "debug")),
            output_path=export_path,
            export_config=DEFAULT_EXPORT_CONFIG,
            verified_only=export_verified_only,
            reset_output=True,
        )
        st.sidebar.success(
            f"Wrote {summary.written_count} row(s) to {summary.output_path}"
            + (f"; skipped {summary.skipped_duplicates} duplicate(s)" if summary.skipped_duplicates else "")
        )
    st.sidebar.caption(f"Export path: {export_path}")

    st.sidebar.divider()
    st.sidebar.subheader("Training Data")
    training_verified_only = st.sidebar.checkbox("Verified Labels Only", value=training_verified_only)
    if st.sidebar.button("Export Training Dataset", use_container_width=True):
        training_output_dir.mkdir(parents=True, exist_ok=True)
        summary = export_training_dataset(
            debug_root=Path(os.environ.get("MARRIAGE_OCR_DEBUG_ROOT", "debug")),
            output_dir=training_output_dir,
            export_config={
                "verified_only": training_verified_only,
                "validation_ratio": training_validation_ratio,
            },
            verified_only=training_verified_only,
            reset_output=True,
        )
        st.sidebar.success(
            f"Exported {summary.total_examples} crop(s) to {summary.output_dir}"
            f" [{summary.train_examples} train / {summary.validation_examples} val]"
        )
    st.sidebar.caption(f"Training output: {training_output_dir}")

    return filtered_bundles


def _resolve_selected_index(filtered_bundles: list[ReviewBundle]) -> int:
    selected_record_dir = st.session_state.get("selected_record_dir")
    if not selected_record_dir:
        return 0

    for index, bundle in enumerate(filtered_bundles):
        if str(bundle.record_dir) == selected_record_dir:
            return index
    return 0


def _render_header_metrics(bundles: list[ReviewBundle]) -> None:
    total_count = len(bundles)
    review_count = sum(1 for bundle in bundles if bundle.active_record.status_review == "REVIEW")
    verified_count = sum(1 for bundle in bundles if bundle.verified)
    corrected_count = sum(1 for bundle in bundles if bundle.corrected_record is not None)

    metric_columns = st.columns(4)
    metric_columns[0].metric("Visible Records", total_count)
    metric_columns[1].metric("Needs Review", review_count)
    metric_columns[2].metric("Corrected", corrected_count)
    metric_columns[3].metric("Verified", verified_count)


def _render_record_summary(bundle: ReviewBundle) -> None:
    record = bundle.active_record
    st.subheader(bundle.display_name)

    summary_columns = st.columns(4)
    summary_columns[0].metric("Status", record.status_review)
    summary_columns[1].metric("Confidence", f"{record.confidence:.2f}")
    summary_columns[2].metric("Verified", "Yes" if bundle.verified else "No")
    summary_columns[3].metric("Corrected", "Yes" if bundle.corrected_record is not None else "No")

    if bundle.reviewed_at or bundle.reviewed_by:
        st.caption(
            "Last review: "
            + " | ".join(
                part
                for part in [
                    bundle.reviewed_at,
                    bundle.reviewed_by,
                ]
                if part
            )
        )

    if record.review_reason:
        st.warning("Review reason: " + "; ".join(record.review_reason))


def _render_images(bundle: ReviewBundle) -> None:
    left_column, right_column = st.columns([1.2, 1.0])
    with left_column:
        st.markdown("#### Full Record")
        if bundle.full_record_path is not None:
            st.image(str(bundle.full_record_path), use_container_width=True)
        else:
            st.info("`full_record.jpg` is not available for this record.")

    with right_column:
        st.markdown("#### Cell Crops")
        cell_names = list(bundle.cell_paths)
        if not cell_names:
            st.info("No cell crops found.")
            return

        crop_columns = st.columns(2)
        for index, cell_name in enumerate(cell_names):
            with crop_columns[index % 2]:
                st.image(str(bundle.cell_paths[cell_name]), caption=cell_name, use_container_width=True)


def _render_raw_ocr(bundle: ReviewBundle) -> None:
    with st.expander("Raw OCR", expanded=False):
        cells = bundle.raw_ocr.get("cells", {})
        if not cells:
            st.info("No raw OCR JSON available.")
            return

        for cell_name in sorted(cells):
            cell_payload = cells.get(cell_name, {})
            st.markdown(f"**{cell_name}**")
            st.caption(f"Average confidence: {float(cell_payload.get('average_confidence', 0.0)):.3f}")
            st.code(cell_payload.get("text", ""), language="text")


def _render_refinement_audit(bundle: ReviewBundle) -> None:
    with st.expander("Refinement Audit", expanded=False):
        if not bundle.refinement_audit_rows:
            st.info("No refinement audit metadata available.")
            return

        for row in bundle.refinement_audit_rows:
            st.markdown(f"**{row.field_name}**")
            left_column, right_column = st.columns(2)
            with left_column:
                st.caption("Original")
                st.code(row.original_value or "", language="text")
            with right_column:
                st.caption("Selected")
                st.code(row.selected_value or "", language="text")
            st.caption(
                "Source: "
                f"{row.candidate_source} | "
                f"Original score: {row.original_score:.2f} | "
                f"Selected score: {row.selected_score:.2f} | "
                f"Retry count: {row.retry_count} | "
                f"Requires review: {'Yes' if row.requires_review else 'No'}"
            )
            if row.reason:
                st.caption(f"Reason: {row.reason}")


def _render_edit_form(bundle: ReviewBundle, *, default_reviewer: str) -> None:
    record = bundle.active_record
    st.markdown("#### Editable Fields")

    with st.form(key=f"edit_form_{bundle.record_dir.name}"):
        form_values: dict[str, Any] = {}
        form_columns = st.columns(2)
        for index, (field_name, label) in enumerate(EDITABLE_FIELDS):
            target_column = form_columns[index % 2]
            value = getattr(record, field_name)
            widget_value = "" if value is None else str(value)
            with target_column:
                if field_name in TEXTAREA_FIELDS:
                    form_values[field_name] = st.text_area(label, value=widget_value, height=96)
                else:
                    form_values[field_name] = st.text_input(label, value=widget_value)

        st.markdown("#### Cell Labels For Training")
        cell_label_values: dict[str, str] = {}
        cell_label_columns = st.columns(2)
        for index, cell_name in enumerate(sorted(bundle.cell_paths)):
            target_column = cell_label_columns[index % 2]
            with target_column:
                cell_label_values[cell_name] = st.text_area(
                    f"{cell_name} label",
                    value=bundle.active_cell_labels.get(cell_name, ""),
                    height=120,
                )

        status_index = ALLOWED_REVIEW_STATUSES.index(record.status_review) if record.status_review in ALLOWED_REVIEW_STATUSES else 1
        status_review = st.selectbox("Status Review", options=ALLOWED_REVIEW_STATUSES, index=status_index)
        review_reason_text = st.text_area("Review Reason", value="\n".join(record.review_reason), height=120)
        verified = st.checkbox("Mark as verified", value=bundle.verified)
        reviewed_by = st.text_input("Reviewer", value=bundle.reviewed_by or default_reviewer)
        review_notes = st.text_area("Reviewer Notes", value=bundle.review_notes or "", height=96)

        submitted = st.form_submit_button("Save Corrections", use_container_width=True)
        if submitted:
            try:
                updated_record = _build_updated_record(
                    record,
                    form_values,
                    status_review=status_review,
                    review_reason_text=review_reason_text,
                )
            except ValueError as exc:
                st.error(str(exc))
            else:
                save_corrected_record(
                    bundle.record_dir,
                    updated_record,
                    verified=verified,
                    reviewed_by=reviewed_by,
                    review_notes=review_notes,
                    corrected_cells=cell_label_values,
                )
                st.session_state["selected_record_dir"] = str(bundle.record_dir)
                st.success(f"Saved corrections to {bundle.record_dir / 'corrected_record.json'}")
                st.rerun()


def _build_updated_record(
    record: ExtractedRecord,
    form_values: dict[str, Any],
    *,
    status_review: str,
    review_reason_text: str,
) -> ExtractedRecord:
    data = record.to_dict()

    for field_name, value in form_values.items():
        if field_name in INT_FIELDS:
            data[field_name] = _parse_optional_int(value)
            continue
        data[field_name] = _parse_optional_text(value)

    data["status_review"] = status_review
    data["review_reason"] = [
        line.strip()
        for line in review_reason_text.splitlines()
        if line.strip()
    ]
    return ExtractedRecord.from_dict(data)


def _parse_optional_text(value: Any) -> str | None:
    text = str(value).strip()
    return text or None


def _parse_optional_int(value: Any) -> int | None:
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f"Invalid integer value: {text}") from exc


def _parse_bool_env(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_float_env(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


if __name__ == "__main__":
    main()
