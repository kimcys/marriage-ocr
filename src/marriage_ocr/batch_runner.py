# src/marriage_ocr/batch_runner.py

import argparse
from dataclasses import asdict, is_dataclass
import traceback
from pathlib import Path

from marriage_ocr import triage
from marriage_ocr.db_postgres import (
    create_batch,
    fetch_records_for_batch,
    insert_record,
    is_file_done,
    mark_file_done,
    mark_file_failed,
    mark_file_skipped,
)
from marriage_ocr.exporter import export_records_to_xlsx
from marriage_ocr.ocr_cache import file_sha256
from marriage_ocr.pipeline import process_input
from marriage_ocr.typed.pipeline import process_typed_input


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".pdf",
    ".tif",
    ".tiff",
}

# (doc_type, record_type, layout_variant) -> config path, used only when
# run_batch is asked to auto-route (config_path=None). layout_variant is
# None only for typed Nikah (borang_4b has no legacy/modern split); typed
# Cerai/Rujuk carry "legacy"/"modern" the same as handwritten does, per
# triage._classify_headers's enactment-year detection (Tahun 1984 vs 2003).
ROUTING_TABLE: dict[tuple[str, str, str | None], str] = {
    ("handwritten", "nikah", "legacy"): "config/handwritten.yaml",
    ("handwritten", "cerai", "legacy"): "config/handwritten_cerai_legacy.yaml",
    ("handwritten", "cerai", "modern"): "config/handwritten_cerai_modern.yaml",
    ("handwritten", "rujuk", "legacy"): "config/handwritten_rujuk_legacy.yaml",
    ("handwritten", "rujuk", "modern"): "config/handwritten_rujuk_modern.yaml",
    ("typed", "nikah", None): "config/typed_borang4b.yaml",
    ("typed", "cerai", "legacy"): "config/typed_cerai_legacy.yaml",
    ("typed", "cerai", "modern"): "config/typed_cerai_modern.yaml",
    ("typed", "rujuk", "legacy"): "config/typed_rujuk_legacy.yaml",
    ("typed", "rujuk", "modern"): "config/typed_rujuk_modern.yaml",
}


def list_input_files(input_dir: str):
    input_path = Path(input_dir)

    files = [
        p for p in input_path.rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    return sorted(files)


def normalize_record(record):
    if isinstance(record, dict):
        return dict(record)

    if hasattr(record, "model_dump"):
        return record.model_dump()

    if hasattr(record, "dict"):
        return record.dict()

    if hasattr(record, "to_dict"):
        return record.to_dict()

    if is_dataclass(record):
        return asdict(record)

    return dict(record)


def run_batch(
    input_dir: str,
    batch_name: str,
    output_dir: str,
    config_path: str | None = None,
):
    """Process every supported file under input_dir.

    If config_path is given, every file is forced through that one config,
    exactly as before -- use this when you already know the batch is
    homogeneous (e.g. one already-sorted folder of legacy-layout Cerai
    ledgers). If config_path is None, each file is classified first
    (src/marriage_ocr/triage.py: doc_type/record_type/layout_variant/Jawi)
    and routed automatically via ROUTING_TABLE -- this is the mode meant for
    an unpredictable dump of mixed typed/handwritten, mixed Nikah/Cerai/Rujuk
    files. A page assessed as essentially entirely Jawi, or a file whose
    classification has no ROUTING_TABLE entry, is recorded via
    mark_file_skipped and never reaches a pipeline or a paid Gemini call.
    """
    files = list_input_files(input_dir)
    batch_id = create_batch(batch_name, input_dir, len(files))

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    typed_output_path = Path(output_dir) / "typed_records.csv"

    for file_index, file_path in enumerate(files, start=1):
        file_path_str = str(file_path)

        if is_file_done(file_path_str):
            print(f"[SKIP] Already processed: {file_path_str}")
            continue

        print(f"[{file_index}/{len(files)}] Processing {file_path_str}")

        try:
            file_hash = file_sha256(file_path_str)
            classification = None

            if config_path is not None:
                config_file = Path(config_path)
            else:
                classification = triage.classify_file(
                    file_path, allowed_extensions=sorted(SUPPORTED_EXTENSIONS)
                )

                if classification.is_jawi:
                    print(
                        f"[SKIPPED_JAWI] {file_path_str} "
                        f"({classification.jawi_proportion:.0%} Jawi)"
                    )
                    mark_file_skipped(
                        batch_id,
                        file_path_str,
                        "SKIPPED_JAWI",
                        doc_type=classification.doc_type,
                        record_type=classification.record_type,
                        layout_variant=classification.layout_variant,
                        is_jawi=True,
                        jawi_proportion=classification.jawi_proportion,
                        notes=classification.notes,
                    )
                    continue

                route_key = (
                    classification.doc_type,
                    classification.record_type,
                    classification.layout_variant,
                )
                routed_config = ROUTING_TABLE.get(route_key)
                if routed_config is None:
                    status = (
                        "BLOCKED_NO_TEMPLATE"
                        if classification.doc_type == "typed"
                        else "NEEDS_MANUAL_CLASSIFICATION"
                    )
                    print(f"[{status}] {file_path_str} classified as {route_key}")
                    mark_file_skipped(
                        batch_id,
                        file_path_str,
                        status,
                        doc_type=classification.doc_type,
                        record_type=classification.record_type,
                        layout_variant=classification.layout_variant,
                        is_jawi=False,
                        jawi_proportion=classification.jawi_proportion,
                        notes=classification.notes,
                    )
                    continue

                config_file = Path(routed_config)

            if classification is not None and classification.doc_type == "typed":
                process_typed_input(
                    input_path=file_path,
                    output_path=typed_output_path,
                    debug_path=Path(output_dir) / "typed_debug",
                    config_path=config_file,
                )
                mark_file_done(batch_id, file_path_str, file_hash)
                continue

            result = process_input(
                input_path=file_path,
                output_path=None,
                debug_path=Path(output_dir),
                config_path=config_file,
            )

            records = getattr(result, "records", [])

            for record_index, record in enumerate(records, start=1):
                record_dict = normalize_record(record)

                insert_record(
                    batch_id=batch_id,
                    source_file=file_path_str,
                    source_page=record_dict.get("source_page") or 1,
                    source_record=record_dict.get("source_record") or record_index,
                    record=record_dict,
                )

            mark_file_done(batch_id, file_path_str, file_hash)

        except Exception:
            error_text = traceback.format_exc()
            print(f"[FAILED] {file_path_str}")
            print(error_text)

            mark_file_failed(
                batch_id=batch_id,
                file_path=file_path_str,
                error_message=error_text,
            )

    merged_records = fetch_records_for_batch(batch_id)
    merged_output_path = Path(output_dir) / "exports" / f"{batch_name}_merged.xlsx"
    merged_output_path.parent.mkdir(parents=True, exist_ok=True)
    export_records_to_xlsx(
        merged_records,
        merged_output_path,
        {"append": False, "dedupe": False, "sheet_name": "Records"},
        reset_output=True,
        skip_existing=False,
    )
    print(f"Merged XLSX exported to {merged_output_path}")

    print("Batch completed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--batch-name", required=True)
    parser.add_argument("--output-dir", default="runs/batch_output")
    parser.add_argument(
        "--config-path",
        default=None,
        help=(
            "Force every file through this one config. Omit to auto-classify "
            "and route each file individually (doc_type/record_type/"
            "layout_variant/Jawi) -- see src/marriage_ocr/triage.py."
        ),
    )

    args = parser.parse_args()

    run_batch(
        input_dir=args.input_dir,
        batch_name=args.batch_name,
        output_dir=args.output_dir,
        config_path=args.config_path,
    )


if __name__ == "__main__":
    main()
