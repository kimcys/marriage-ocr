# src/marriage_ocr/export_from_postgres.py

import argparse
from pathlib import Path

import pandas as pd

from marriage_ocr.db_postgres import get_connection


EXPORT_COLUMNS = (
    "id",
    "source_file",
    "source_page",
    "source_record",
    "bil",
    "nama_suami",
    "ic_baru_suami",
    "nama_isteri",
    "ic_baru_isteri",
    "tarikh_nikah",
    "mas_kahwin",
    "wali",
    "status",
    "confidence",
    "validation_errors",
)
EXPORT_COLUMNS_SQL = ",\n".join(EXPORT_COLUMNS)


def export_xlsx_parts(output_dir: str, rows_per_file: int = 500_000):
    if rows_per_file <= 0:
        raise ValueError(f"rows_per_file must be positive, got {rows_per_file}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with get_connection() as conn:
        for part, df in enumerate(_iter_record_chunks(conn, rows_per_file), start=1):
            output_path = Path(output_dir) / f"records_part_{part:03d}.xlsx"
            df.to_excel(output_path, index=False)
            print(f"Exported {output_path}")


def export_csv(output_path: str):
    output_path = Path(output_path)
    if output_path.parent:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    wrote_any = False

    with get_connection() as conn:
        for df in _iter_record_chunks(conn, 100_000):
            df.to_csv(
                output_path,
                mode="w" if not wrote_any else "a",
                index=False,
                header=not wrote_any,
            )
            wrote_any = True

    if not wrote_any:
        pd.DataFrame(columns=EXPORT_COLUMNS).to_csv(output_path, index=False)

    print(f"Exported CSV to {output_path}")


def _iter_record_chunks(conn, rows_per_chunk: int):
    query = f"""
    SELECT {EXPORT_COLUMNS_SQL}
    FROM records
    ORDER BY id
    """

    with conn.cursor() as cur:
        cur.execute(query)

        while True:
            rows = cur.fetchmany(rows_per_chunk)
            if not rows:
                break

            yield pd.DataFrame(rows, columns=EXPORT_COLUMNS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="exports/final_xlsx")
    parser.add_argument("--rows-per-file", type=int, default=500_000)
    parser.add_argument("--csv-path", default=None)

    args = parser.parse_args()

    export_xlsx_parts(
        output_dir=args.output_dir,
        rows_per_file=args.rows_per_file,
    )

    if args.csv_path:
        export_csv(args.csv_path)


if __name__ == "__main__":
    main()
