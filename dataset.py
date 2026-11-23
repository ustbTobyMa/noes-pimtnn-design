"""Read the Table S4 example and export the same records to SQLite."""

import argparse
import csv
import hashlib
import json
import math
import sqlite3
from contextlib import closing
from pathlib import Path


DATA = Path(__file__).resolve().parent / "data" / "table_s4.csv"
FEATURES = (
    "c_wt_pct", "si_wt_pct", "mn_wt_pct", "al_wt_pct", "p_wt_pct",
    "s_wt_pct", "cu_wt_pct", "n_wt_pct", "ti_wt_pct", "sheet_thickness_mm",
    "final_anneal_temp_c", "final_anneal_time_s", "normalizing_temp_c",
)
TARGETS = ("b50_t", "p1_5_50_w_kg", "uts_mpa")
COLUMNS = ("record_id", *FEATURES, *TARGETS)
# Ordered numeric cells and nulls of the released SI table, independent of CSV line endings.
TABLE_S4_DIGEST = "ffe95b35bd7b7080e3408a784ea430c3ba1ba42ab4ce376104ab10992dcc4dd9"


def record_digest(records):
    digest = hashlib.sha256()
    for record in records:
        values = [record[column] for column in COLUMNS]
        line = json.dumps(values, separators=(",", ":"), allow_nan=False)
        digest.update((line + "\n").encode("ascii"))
    return digest.hexdigest()


def read_records(path=DATA):
    """Read numeric values in physical units. A blank cell is returned as None."""
    records = []
    seen = set()
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        header = reader.fieldnames or []
        if len(header) != len(set(header)) or not set(COLUMNS).issubset(header):
            raise ValueError("The CSV must contain each Table S4 column exactly once.")
        for row in reader:
            if None in row or any(row[name] is None for name in COLUMNS):
                raise ValueError(f"Malformed CSV at line {reader.line_num}.")
            key = int(row["record_id"])
            if key <= 0 or key in seen:
                raise ValueError(f"Invalid or duplicate record_id {key}.")
            seen.add(key)
            record = {"record_id": key}
            for name in COLUMNS[1:]:
                token = row[name].strip()
                number = float(token) if token else None
                if number is not None and not math.isfinite(number):
                    raise ValueError(f"Non-finite {name} for record {key}. Use a blank for missing data.")
                record[name] = number
            records.append(record)
    if not records:
        raise ValueError("The dataset contains no records.")
    return records


def check_sqlite(records, path):
    """Compare the CSV columns and values with the companion database in read-only mode."""
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as db:
        if db.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
            raise ValueError("The SQLite integrity check failed.")
        columns = tuple(row[1] for row in db.execute("PRAGMA table_info(conditions)"))
        if columns != COLUMNS:
            raise ValueError("The SQLite conditions table has different columns from the CSV.")
        sql_rows = db.execute("SELECT * FROM conditions ORDER BY record_id").fetchall()
    if len(sql_rows) != len(records):
        raise ValueError(f"Row counts differ: CSV {len(records)}, SQLite {len(sql_rows)}.")
    for record, sql_row in zip(sorted(records, key=lambda row: row["record_id"]), sql_rows):
        for column, value in zip(COLUMNS, sql_row):
            if record[column] != value:
                raise ValueError(f"CSV/SQLite difference at record {record['record_id']}, column {column}.")


def write_sqlite(records, destination):
    """Create a new single-table database. Existing files are protected."""
    destination = Path(destination)
    destination.touch(exist_ok=False)
    numeric_fields = ", ".join(f"{name} REAL" for name in COLUMNS[1:])
    bindings = ", ".join(f":{name}" for name in COLUMNS)
    try:
        with closing(sqlite3.connect(destination)) as db, db:
            db.execute(f"CREATE TABLE conditions (record_id INTEGER PRIMARY KEY CHECK (record_id > 0), {numeric_fields})")
            db.executemany(f"INSERT INTO conditions VALUES ({bindings})", records)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DATA)
    parser.add_argument("--sqlite", type=Path, help="Create the database at a new path.")
    parser.add_argument("--check", action="store_true", help="Compare the CSV with its existing .sqlite3 companion.")
    args = parser.parse_args()
    rows = read_records(args.csv)
    if args.check:
        db_path = args.csv.with_suffix(".sqlite3")
        try:
            check_sqlite(rows, db_path)
        except (ValueError, sqlite3.Error) as error:
            parser.exit(1, f"Database check failed ({db_path.name}): {error}\n")
        if args.csv.resolve() == DATA.resolve() and record_digest(rows) != TABLE_S4_DIGEST:
            parser.exit(1, "Bundled records differ from the released Supplementary Table S4.\n")
        missing = sum(value is None for row in rows for value in row.values())
        print(f"CSV and SQLite agree on {len(rows) * len(COLUMNS)} cells, including {missing} missing values.")
    print(f"{len(rows)} records, {len(FEATURES)} input columns, {len(TARGETS)} targets")
    for target in TARGETS:
        present = sum(row[target] is not None for row in rows)
        print(f"{target}: {present} populated")
    if args.sqlite:
        write_sqlite(rows, args.sqlite)
        print(f"Wrote {args.sqlite}")
