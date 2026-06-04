# src/marriage_ocr/batch_runner.py

import argparse
from dataclasses import asdict, is_dataclass
import traceback
from pathlib import Path

from marriage_ocr.db_postgres import (
    create_batch,
    fetch_records_for_batch,
    insert_record,
    is_file_done,
    mark_file_done,
    mark_file_failed,
)
from marriage_ocr.exporter import export_records_to_xlsx
from marriage_ocr.ocr_cache import file_sha256
from marriage_ocr.pipeline import process_input


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".pdf",
    ".tif",
    ".tiff",
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
    files = list_input_files(input_dir)
    batch_id = create_batch(batch_name, input_dir, len(files))

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    config_file = Path(config_path) if config_path is not None else Path("config/default.yaml")

    for file_index, file_path in enumerate(files, start=1):
        file_path_str = str(file_path)

        if is_file_done(file_path_str):
            print(f"[SKIP] Already processed: {file_path_str}")
            continue

        print(f"[{file_index}/{len(files)}] Processing {file_path_str}")

        try:
            file_hash = file_sha256(file_path_str)

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
    parser.add_argument("--config-path", default=None)

    args = parser.parse_args()

    run_batch(
        input_dir=args.input_dir,
        batch_name=args.batch_name,
        output_dir=args.output_dir,
        config_path=args.config_path,
    )


if __name__ == "__main__":
    main()
