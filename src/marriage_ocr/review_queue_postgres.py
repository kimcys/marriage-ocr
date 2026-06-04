# src/marriage_ocr/review_queue_postgres.py

import argparse
from pathlib import Path

import pandas as pd

from marriage_ocr.db_postgres import get_connection


def export_review_queue(output_path: str):
    query = """
    SELECT
        id,
        source_file,
        source_page,
        source_record,
        bil,
        nama_suami,
        ic_baru_suami,
        nama_isteri,
        ic_baru_isteri,
        tarikh_nikah,
        mas_kahwin,
        wali,
        status,
        confidence,
        validation_errors
    FROM records
    WHERE status = 'REVIEW'
    ORDER BY id
    """

    with get_connection() as conn:
        df = pd.read_sql_query(query, conn)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if output_path.endswith(".xlsx"):
        df.to_excel(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)

    print(f"Exported review queue to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", default="exports/review_queue.xlsx")

    args = parser.parse_args()

    export_review_queue(args.output_path)


if __name__ == "__main__":
    main()