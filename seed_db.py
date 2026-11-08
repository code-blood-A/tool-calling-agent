"""
seed_db.py — create and populate the sample SQLite expenses database.

Run once before any agent version:
    python seed_db.py

The database schema:
    expenses(id INTEGER PRIMARY KEY,
             city     TEXT,
             category TEXT,
             amount   REAL,
             date     TEXT)

References & tutorials
─────────────────────────────────────────────────────────────────────────────
Python sqlite3 module
  → https://docs.python.org/3/library/sqlite3.html

SQLite tutorial (create table, insert, etc.)
  → https://www.sqlitetutorial.net/sqlite-python/creating-tables/

SQLite data types
  → https://www.sqlite.org/datatype3.html
─────────────────────────────────────────────────────────────────────────────
"""

import sqlite3

# ── Database file path ────────────────────────────────────────────────────────
# SQLite stores everything in a single file — no server needed.
# Docs: https://www.sqlite.org/index.html
DB_PATH = "expenses.db"


def seed() -> None:
    # connect() creates the file if it doesn't exist
    # Reference: https://docs.python.org/3/library/sqlite3.html#sqlite3.connect
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # ── CREATE TABLE ──────────────────────────────────────────────────────────
    # IF NOT EXISTS prevents errors on re-runs
    # SQL reference: https://www.w3schools.com/sql/sql_create_table.asp
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS expenses (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            city     TEXT    NOT NULL,
            category TEXT    NOT NULL,
            amount   REAL    NOT NULL,
            date     TEXT    NOT NULL
        )
    """)

    # ── CLEAR OLD DATA ────────────────────────────────────────────────────────
    cursor.execute("DELETE FROM expenses")

    # ── SAMPLE DATA ───────────────────────────────────────────────────────────
    # Intentional variety: multiple cities, categories, and amount ranges.
    # The agent will later query this to find the highest-spending city.
    rows = [
        # city,       category,       amount,   date
        ("Delhi",    "Travel",        12500.00, "2024-01-10"),
        ("Delhi",    "Food",           3200.00, "2024-01-11"),
        ("Delhi",    "Accommodation",  8900.00, "2024-01-12"),
        ("Mumbai",   "Travel",         9800.00, "2024-02-05"),
        ("Mumbai",   "Food",           4100.00, "2024-02-06"),
        ("Mumbai",   "Entertainment",  6700.00, "2024-02-07"),
        ("Mumbai",   "Accommodation",  7500.00, "2024-02-08"),
        ("Bangalore","Travel",        11000.00, "2024-03-01"),
        ("Bangalore","Food",           2900.00, "2024-03-02"),
        ("Bangalore","Accommodation",  6200.00, "2024-03-03"),
        ("Chennai",  "Travel",         7400.00, "2024-04-10"),
        ("Chennai",  "Food",           2600.00, "2024-04-11"),
        ("Chennai",  "Entertainment",  3300.00, "2024-04-12"),
        ("Kolkata",  "Travel",         5500.00, "2024-05-15"),
        ("Kolkata",  "Food",           1800.00, "2024-05-16"),
        ("Kolkata",  "Accommodation",  4000.00, "2024-05-17"),
    ]

    # executemany inserts all rows in one call — efficient for bulk inserts
    # Reference: https://docs.python.org/3/library/sqlite3.html#sqlite3.Cursor.executemany
    cursor.executemany(
        "INSERT INTO expenses (city, category, amount, date) VALUES (?, ?, ?, ?)",
        rows
    )

    conn.commit()  # write changes to disk
    conn.close()

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"[OK] Database '{DB_PATH}' seeded with {len(rows)} rows.")
    print("\nCity totals (should match what the agent finds):")

    # Quick verification query
    verify_conn = sqlite3.connect(DB_PATH)
    for city_row in verify_conn.execute(
        "SELECT city, SUM(amount) FROM expenses GROUP BY city ORDER BY SUM(amount) DESC"
    ):
        print(f"  {city_row[0]:<12}  Rs.{city_row[1]:,.2f}")
    verify_conn.close()


if __name__ == "__main__":
    seed()

