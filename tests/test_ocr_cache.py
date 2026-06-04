from pathlib import Path

from marriage_ocr.ocr_cache import file_sha256


def test_file_sha256_matches_known_digest(tmp_path: Path) -> None:
    data_path = tmp_path / "sample.txt"
    data_path.write_text("marriage-ocr\n", encoding="utf-8")

    assert file_sha256(data_path) == "15f98cefcf4e25a573fa8494a31b4653ab09bb58fb0854402a60a7d22447ff5e"
