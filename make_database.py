# make_database.py
# Creates a simple SQLite DB (data/supply_chain.db) from the CSV used in the project.

import pandas as pd
import sqlite3
from pathlib import Path

ROOT = Path(__file__).parent
CSV = ROOT / "data" / "supply_chain_data.csv"
DB = ROOT / "data" / "supply_chain.db"
TABLE_NAME = "supply_chain"

if not CSV.exists():
    raise SystemExit(f"CSV not found: {CSV}")

print("Loading CSV:", CSV)
df = pd.read_csv(CSV)

print("Rows, cols:", df.shape)
print("Creating SQLite DB at:", DB)

conn = sqlite3.connect(DB)
df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
conn.close()

print(f"Done — table `{TABLE_NAME}` written to {DB}")
