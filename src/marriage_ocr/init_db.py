# src/marriage_ocr/init_db.py

from marriage_ocr.db_postgres import init_db


def main():
    init_db()
    print("PostgreSQL database initialized.")


if __name__ == "__main__":
    main()