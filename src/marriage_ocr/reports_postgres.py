# src/marriage_ocr/reports_postgres.py

import pandas as pd

from marriage_ocr.db_postgres import count_records, get_connection


def print_report():
    total_count = count_records()

    with get_connection() as conn:
        records_by_status = pd.read_sql_query(
            """
            SELECT status, COUNT(*) AS count
            FROM records
            GROUP BY status
            ORDER BY count DESC
            """,
            conn,
        )

        files_by_status = pd.read_sql_query(
            """
            SELECT status, COUNT(*) AS count
            FROM processed_files
            GROUP BY status
            ORDER BY count DESC
            """,
            conn,
        )

        batches = pd.read_sql_query(
            """
            SELECT
                batch_name,
                status,
                total_files,
                processed_files,
                ok_records,
                review_records,
                failed_records,
                started_at,
                completed_at
            FROM batches
            ORDER BY id DESC
            """,
            conn,
        )

    print("\n=== OCR REPORT ===")
    print(f"Total records: {total_count}")

    print("\nRecords by status:")
    print(records_by_status.to_string(index=False))

    print("\nFiles by status:")
    print(files_by_status.to_string(index=False))

    print("\nBatches:")
    print(batches.to_string(index=False))


def main():
    print_report()


if __name__ == "__main__":
    main()
