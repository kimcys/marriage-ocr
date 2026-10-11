from marriage_ocr.typed.models import RawField
from marriage_ocr.typed.pipeline import raw_text_snapshot


def _raw(key: str, text: str) -> RawField:
    return RawField(key=key, output_name=key, page_number=1, region=None, raw_text=text, confidence=1.0)  # type: ignore[arg-type]


def test_raw_text_snapshot_keeps_identity_fields_only_and_skips_blanks() -> None:
    fields = {
        "id_suami": _raw("id_suami", " A0785338 "),
        "nama_suami": _raw("nama_suami", ""),
        "alamat_suami": _raw("alamat_suami", "No. 8 Jalan 4/7C"),
        "mas_kahwin": _raw("mas_kahwin", "$.80.00"),
    }
    assert raw_text_snapshot(fields) == {"id_suami": "A0785338"}
