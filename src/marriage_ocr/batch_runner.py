# src/marriage_ocr/batch_runner.py

import argparse
from dataclasses import asdict, is_dataclass
import traceback
from pathlib import Path

from marriage_ocr import triage
from marriage_ocr.db_postgres import (
    create_batch,
    fetch_records_for_batch,
    find_done_file_by_hash,
    find_duplicate_record,
    get_existing_record_identity,
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


def sync_gemini_batch_results(validated_records) -> dict[str, object]:
    """Write llm.run_batch_extraction's merged records back into the same
    `records` table run_batch() populated.

    Gemini Batch Mode (see `marriage-ocr gemini-batch run`) is intentionally
    a disk-only tool -- it discovers debug artifacts, submits/waits/merges,
    and overwrites validated_record.json, with no Postgres dependency at
    all. That's the right shape for a standalone offline re-extraction
    tool, but it means those merged values never reach the production DB on
    their own: a real production run is `run_batch(...)` with
    `llm.enabled: false` (so Gemini is never called synchronously) followed
    by `gemini-batch run --sync-db`, and without this step the second half
    of that workflow would silently do nothing to the database the first
    half populated.

    Matches existing rows by (source_file, source_page, source_record) --
    the same UNIQUE key insert_record()'s ON CONFLICT relies on -- and
    preserves each row's existing is_duplicate/duplicate_of_record_id
    rather than letting insert_record's ON CONFLICT DO UPDATE reset them to
    the ExtractedRecord dataclass defaults (see
    get_existing_record_identity's docstring for why).
    """
    synced = 0
    skipped_no_match: list[str] = []

    for record in validated_records:
        record_dict = normalize_record(record)
        source_file = record_dict.get("source_file")
        source_page = record_dict.get("source_page") or 1
        source_record = record_dict.get("source_record") or 1

        if not source_file:
            skipped_no_match.append(f"(missing source_file) record {source_record}")
            continue

        identity = get_existing_record_identity(source_file, source_page, source_record)
        if identity is None:
            skipped_no_match.append(f"{source_file}#{source_page}.{source_record}")
            continue

        record_dict["is_duplicate"] = identity["is_duplicate"]
        record_dict["duplicate_of_record_id"] = identity["duplicate_of_record_id"]

        insert_record(
            batch_id=identity["batch_id"],
            source_file=source_file,
            source_page=source_page,
            source_record=source_record,
            record=record_dict,
        )
        synced += 1

    return {"synced": synced, "skipped_no_match": skipped_no_match}


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

            duplicate_path = find_done_file_by_hash(file_hash)
            if duplicate_path is not None and duplicate_path != file_path_str:
                print(f"[SKIPPED_DUPLICATE_FILE] {file_path_str} (same content as {duplicate_path})")
                mark_file_skipped(
                    batch_id,
                    file_path_str,
                    "DUPLICATE_FILE",
                    notes=[f"duplicate of {duplicate_path} (identical file hash)"],
                )
                continue

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
                source_page = record_dict.get("source_page") or 1
                source_record = record_dict.get("source_record") or record_index

                record_type = str(record_dict.get("record_type") or "NIKAH").upper()
                ic_a, ic_b = (
                    (record_dict.get("ic_baru_suami"), record_dict.get("ic_baru_isteri"))
                    if record_type == "NIKAH"
                    else (record_dict.get("ic_suami"), record_dict.get("ic_isteri"))
                )
                duplicate_of_record_id = find_duplicate_record(
                    record_type=record_type,
                    bil=record_dict.get("bil"),
                    ic_a=ic_a,
                    ic_b=ic_b,
                    exclude_source_file=file_path_str,
                    exclude_source_page=source_page,
                    exclude_source_record=source_record,
                )
                if duplicate_of_record_id is not None:
                    print(
                        f"[DUPLICATE_RECORD] {file_path_str} record {record_index} "
                        f"matches existing record id={duplicate_of_record_id}"
                    )
                record_dict["is_duplicate"] = duplicate_of_record_id is not None
                record_dict["duplicate_of_record_id"] = duplicate_of_record_id

                insert_record(
                    batch_id=batch_id,
                    source_file=file_path_str,
                    source_page=source_page,
                    source_record=source_record,
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
