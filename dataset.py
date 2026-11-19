"""Read the Table S4 example and export the same records to SQLite."""

import argparse
import csv
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
    args = parser.parse_args()
    rows = read_records(args.csv)
    print(f"{len(rows)} records, {len(FEATURES)} input columns, {len(TARGETS)} targets")
    for target in TARGETS:
        present = sum(row[target] is not None for row in rows)
        print(f"{target}: {present} populated")
    if args.sqlite:
        write_sqlite(rows, args.sqlite)
        print(f"Wrote {args.sqlite}")
