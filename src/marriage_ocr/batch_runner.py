# src/marriage_ocr/batch_runner.py

import argparse
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, is_dataclass
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
from marriage_ocr.typed.models import ProcessingStatus
from marriage_ocr.typed.pipeline import process_typed_documents_no_csv

# TypedDocumentResult.processing_status uses its own vocabulary (see
# src/marriage_ocr/typed/models.py); the records table's status column uses
# the one src/marriage_ocr/review_store.py's ALLOWED_REVIEW_STATUSES
# defines. FAILED_OCR (not a generic "FAILED") is what the handwritten path
# already uses for an OCR-side failure, so typed's FAILED maps to the same
# value rather than inventing a new status string.
TYPED_STATUS_TO_DB_STATUS: dict[ProcessingStatus, str] = {
    ProcessingStatus.SUCCESS: "OK",
    ProcessingStatus.SUCCESS_WITH_RETRY: "OK",
    ProcessingStatus.REVIEW_REQUIRED: "REVIEW",
    ProcessingStatus.FAILED: "FAILED_OCR",
}


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
    # Nikah now splits legacy/modern the same as Cerai/Rujuk (see
    # NIKAH_LEGACY_REGIONS/NIKAH_MODERN_REGIONS in typed/template.py) --
    # triage.py never emits ("typed", "nikah", None) any more.
    ("typed", "nikah", "legacy"): "config/typed_nikah_legacy.yaml",
    ("typed", "nikah", "modern"): "config/typed_nikah_modern.yaml",
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


def _insert_record_with_dedup(
    *,
    batch_id: int,
    file_path_str: str,
    record_dict: dict,
    record_index: int,
    source_page: int,
    source_record: int,
) -> None:
    """Content-identity duplicate check + insert, shared by both the
    general (process_input) and typed (process_typed_documents_no_csv)
    paths -- same rule either way: same record_type + Bil + a matching IC
    as an existing, non-duplicate record from a different source file.
    """
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


def _process_one_file(
    file_path: Path,
    *,
    batch_id: int,
    output_dir: str,
    config_path: str | None,
) -> None:
    """Route, OCR, and DB-insert one file. Split out of run_batch so it can
    be submitted to a thread pool -- see run_batch's docstring for the
    concurrency model.
    """
    file_path_str = str(file_path)

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
            return

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
                return

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
                return

            config_file = Path(routed_config)

        if classification is not None and classification.doc_type == "typed":
            # process_typed_documents_no_csv has no shared mutable state
            # (unlike the old TypedCsvStore-based path), so it's safe to
            # call concurrently for different files -- each result is
            # inserted straight into `records` the same way the general
            # path does, via insert_record()'s upsert-on-unique-key.
            typed_results = process_typed_documents_no_csv(
                input_path=file_path,
                debug_path=Path(output_dir) / "typed_debug",
                config_path=config_file,
            )

            for record_index, typed_result in enumerate(typed_results, start=1):
                record_dict = normalize_record(typed_result.record)
                record_dict["status_review"] = TYPED_STATUS_TO_DB_STATUS[typed_result.processing_status]
                reasons = list(record_dict.get("review_reason") or [])
                reasons.extend(f"typed field failed: {field}" for field in typed_result.failed_fields)
                if typed_result.error_message:
                    reasons.append(typed_result.error_message)
                record_dict["review_reason"] = reasons

                # Typed's source_record is the PDF filename's stem (often a
                # long numeric reference code, e.g. "01000122140838082020")
                # -- kept as-is in raw_record for reference, but it would
                # overflow the records table's INTEGER source_record column
                # if passed straight through. Each typed PDF is exactly one
                # record on one page, and source_file already uniquely
                # identifies it, so a constant works fine here.
                _insert_record_with_dedup(
                    batch_id=batch_id,
                    file_path_str=file_path_str,
                    record_dict=record_dict,
                    record_index=record_index,
                    source_page=1,
                    source_record=1,
                )

            mark_file_done(batch_id, file_path_str, file_hash)
            return

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

            _insert_record_with_dedup(
                batch_id=batch_id,
                file_path_str=file_path_str,
                record_dict=record_dict,
                record_index=record_index,
                source_page=source_page,
                source_record=source_record,
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


def run_batch(
    input_dir: str,
    batch_name: str,
    output_dir: str,
    config_path: str | None = None,
    workers: int = 4,
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

    `workers` files are processed concurrently (each file's OCR/parse/DB-
    insert work is independent of every other file's) -- this now applies
    equally to handwritten/general and typed files. Both paths insert
    straight into the `records` table via insert_record(), which upserts on
    a unique (source_file, source_page, source_record) key; db_postgres.py's
    connection pool is designed for concurrent callers. (Typed used to go
    through a shared CSV file with its own load-modify-flush cycle, which
    wasn't safe to parallelize -- see git history on process_typed_input /
    process_typed_documents_no_csv if you need the old behavior.)

    Keep `workers` comfortably under DATABASE_POOL_MAX_SIZE (default 10,
    see db_postgres.py) -- raise both together if you want more throughput.
    """
    files = list_input_files(input_dir)
    batch_id = create_batch(batch_name, input_dir, len(files))

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    pending_files = []
    for file_path in files:
        file_path_str = str(file_path)
        if is_file_done(file_path_str):
            print(f"[SKIP] Already processed: {file_path_str}")
            continue
        pending_files.append(file_path)

    total = len(pending_files)
    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(
                _process_one_file,
                file_path,
                batch_id=batch_id,
                output_dir=output_dir,
                config_path=config_path,
            ): file_path
            for file_path in pending_files
        }
        for future in as_completed(futures):
            completed += 1
            print(f"[{completed}/{total}] Processed {futures[future]}")
            future.result()  # re-raise anything _process_one_file didn't already catch/log

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
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help=(
            "Number of files to process concurrently. Keep comfortably under "
            "DATABASE_POOL_MAX_SIZE (default 10) -- raise both together for "
            "more throughput. See run_batch()'s docstring for what does and "
            "doesn't parallelize well."
        ),
    )

    args = parser.parse_args()

    run_batch(
        input_dir=args.input_dir,
        batch_name=args.batch_name,
        output_dir=args.output_dir,
        workers=args.workers,
        config_path=args.config_path,
    )


if __name__ == "__main__":
    main()
