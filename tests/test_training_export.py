from pathlib import Path
import json

from marriage_ocr.models import ExtractedRecord
from marriage_ocr.review_store import save_corrected_record
from marriage_ocr.training_export import export_training_dataset


EXPORT_CONFIG = {
    "crops_dir": "training_crops",
    "labels_file": "labels.tsv",
    "train_file": "train.tsv",
    "validation_file": "val.tsv",
    "manifest_file": "manifest.jsonl",
    "stats_file": "stats.json",
}


def test_training_export_writes_crops_labels_and_manifest(tmp_path: Path) -> None:
    record_dir = _create_record_dir(tmp_path, source_record="record_001")
    save_corrected_record(
        record_dir,
        _make_record(source_record="record_001"),
        verified=True,
        reviewed_by="QA User",
        corrected_cells={
            "bil": "1",
            "suami_isteri": "MOHAMAD BIN YASMIN\nSITI BINTI ALI",
        },
    )

    output_dir = tmp_path / "ground_truth"
    summary = export_training_dataset(
        tmp_path,
        output_dir,
        {**EXPORT_CONFIG, "verified_only": True, "validation_ratio": 1.0},
        reset_output=True,
    )

    labels = (output_dir / "labels.tsv").read_text(encoding="utf-8").splitlines()
    manifest = [
        json.loads(line)
        for line in (output_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert summary.total_examples == 2
    assert summary.train_examples == 0
    assert summary.validation_examples == 2
    assert (output_dir / "training_crops" / "bil" / "000001.jpg").exists()
    assert (output_dir / "training_crops" / "suami_isteri" / "000002.jpg").exists()
    assert labels == [
        "training_crops/bil/000001.jpg\t1",
        "training_crops/suami_isteri/000002.jpg\tMOHAMAD BIN YASMIN\\nSITI BINTI ALI",
    ]
    assert manifest[0]["verified"] is True
    assert manifest[0]["used_corrected_label"] is True
    assert manifest[0]["dataset_split"] == "val"
    assert manifest[1]["label_text"] == "MOHAMAD BIN YASMIN\nSITI BINTI ALI"


def test_training_export_skips_unverified_records_by_default(tmp_path: Path) -> None:
    verified_dir = _create_record_dir(tmp_path, source_record="record_001")
    unverified_dir = _create_record_dir(tmp_path, source_record="record_002")

    save_corrected_record(
        verified_dir,
        _make_record(source_record="record_001"),
        verified=True,
        corrected_cells={"bil": "1", "suami_isteri": "VERIFIED"},
    )
    save_corrected_record(
        unverified_dir,
        _make_record(source_record="record_002"),
        verified=False,
        corrected_cells={"bil": "2", "suami_isteri": "UNVERIFIED"},
    )

    output_dir = tmp_path / "ground_truth"
    summary = export_training_dataset(
        tmp_path,
        output_dir,
        {**EXPORT_CONFIG, "verified_only": True, "validation_ratio": 0.0},
        reset_output=True,
    )

    labels = (output_dir / "labels.tsv").read_text(encoding="utf-8").splitlines()

    assert summary.total_examples == 2
    assert summary.skipped_unverified_records == 1
    assert labels == [
        "training_crops/bil/000001.jpg\t1",
        "training_crops/suami_isteri/000002.jpg\tVERIFIED",
    ]


def test_training_export_keeps_record_examples_in_same_split(tmp_path: Path) -> None:
    record_dir = _create_record_dir(tmp_path, source_record="record_001")
    save_corrected_record(
        record_dir,
        _make_record(source_record="record_001"),
        verified=True,
        corrected_cells={"bil": "1", "suami_isteri": "SAME RECORD"},
    )

    output_dir = tmp_path / "ground_truth"
    export_training_dataset(
        tmp_path,
        output_dir,
        {**EXPORT_CONFIG, "verified_only": True, "validation_ratio": 0.5},
        reset_output=True,
    )

    manifest = [
        json.loads(line)
        for line in (output_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert {item["dataset_split"] for item in manifest} in ({"train"}, {"val"})


def _create_record_dir(root: Path, *, source_record: str) -> Path:
    record_dir = root / "page_a" / "records" / source_record
    record_dir.mkdir(parents=True)

    validated_record = _make_record(source_record=source_record)
    (record_dir / "validated_record.json").write_text(
        json.dumps(validated_record.to_dict(), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    (record_dir / "parsed_record.json").write_text(
        json.dumps(validated_record.to_dict(), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    (record_dir / "raw_ocr.json").write_text(
        json.dumps(
            {
                "cells": {
                    "bil": {"text": "1"},
                    "suami_isteri": {"text": "RAW OCR"},
                }
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    (record_dir / "full_record.jpg").write_bytes(b"full")
    (record_dir / "bil.jpg").write_bytes(b"bil")
    (record_dir / "suami_isteri.jpg").write_bytes(b"suami_isteri")
    return record_dir


def _make_record(*, source_record: str) -> ExtractedRecord:
    return ExtractedRecord(
        bil="1",
        nama_suami="MOHAMAD BIN YASMIN",
        ic_lama_suami="A.1192345",
        umur_suami=25,
        nama_isteri="SITI BINTI ALI",
        ic_baru_isteri="900101-10-1234",
        umur_isteri=23,
        mas_kahwin="RM 80.00",
        mas_kahwin_raw="RM 80.00",
        nama_pendaftar="MOHD SALLEH",
        alamat_pendaftar="KAMPUNG BARU",
        nama_wali="ABDUL RAHMAN",
        hubungan_wali="BAPA",
        saksi_1="AHMAD BIN ALI",
        saksi_2="OSMAN BIN DIN",
        tarikh_nikah="27-08-1994",
        tarikh_nikah_raw="27.8.94",
        tarikh_keluar="02-06-1995",
        tarikh_keluar_raw="2.6.95",
        remarks="TIADA",
        confidence=0.93,
        status_review="OK",
        review_reason=[],
        source_file="sample.jpg",
        source_page=1,
        source_record=source_record,
        crop_folder=f"debug/sample/{source_record}",
        raw_bil="1",
        raw_suami_isteri="RAW OCR",
        raw_pendaftar="RAW PENDAFTAR",
        raw_wali="RAW WALI",
        raw_hubungan_wali="RAW HUBUNGAN",
        raw_saksi="RAW SAKSI",
        raw_tarikh_nikah="RAW NIKAH",
        raw_tarikh_keluar="RAW KELUAR",
        raw_remarks="RAW REMARKS",
        raw_ocr_json="{\"mock\":true}",
    )
