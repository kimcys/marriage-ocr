from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from marriage_ocr.review_store import ReviewBundle, load_review_bundles


NAME_FIELDS = frozenset(
    {
        "nama_suami",
        "nama_isteri",
        "nama_pendaftar",
        "nama_wali",
        "saksi_1",
        "saksi_2",
    }
)


@dataclass(frozen=True)
class RefinementBaseline:
    record_count: int
    name_exact_match_count: int
    ic_exact_match_count: int
    date_exact_match_count: int

    @classmethod
    def from_bundles(cls, bundles: list[ReviewBundle]) -> RefinementBaseline:
        name_exact_match_count = 0
        ic_exact_match_count = 0
        date_exact_match_count = 0

        for bundle in bundles:
            for row in bundle.refinement_audit_rows:
                reviewed_value = getattr(bundle.active_record, row.field_name, None)
                if not _is_exact_match(row.selected_value, reviewed_value):
                    continue
                if row.field_name in NAME_FIELDS:
                    name_exact_match_count += 1
                elif _is_ic_field(row.field_name):
                    ic_exact_match_count += 1
                elif _is_date_field(row.field_name):
                    date_exact_match_count += 1

        return cls(
            record_count=len(bundles),
            name_exact_match_count=name_exact_match_count,
            ic_exact_match_count=ic_exact_match_count,
            date_exact_match_count=date_exact_match_count,
        )


def build_refinement_baseline(review_root: Path, *, limit: int = 25) -> RefinementBaseline:
    reviewed = load_review_bundles(review_root)
    selected = [bundle for bundle in reviewed if bundle.verified][:limit]
    return RefinementBaseline.from_bundles(selected)


def _is_exact_match(left: str | None, right: object) -> bool:
    if left is None or right is None:
        return False
    return str(left).strip() == str(right).strip()


def _is_ic_field(field_name: str) -> bool:
    return field_name.startswith("ic_") or field_name.startswith("id_")


def _is_date_field(field_name: str) -> bool:
    return field_name.startswith("tarikh_")
