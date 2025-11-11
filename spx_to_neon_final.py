import re
import math
import json
import requests
import pandas as pd
from datetime import datetime, timezone
import psycopg2
from psycopg2.extras import execute_values
import streamlit as st

NEON_URL = st.secrets["N_URL"]
TABLE = "spx_chain"
CHAIN_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json,text/plain,*/*",
    "Referer": "https://www.cboe.com/delayed_quotes/spx/quote_table",
}

SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    run_ts TIMESTAMPTZ NOT NULL,
    expiration_date DATE,
    strike NUMERIC,
    cp TEXT,
    last NUMERIC,
    bid NUMERIC,
    ask NUMERIC,
    volume NUMERIC(20,0),
    oi NUMERIC(20,0),
    iv NUMERIC
);
"""

OPRA = re.compile(r"SPXW?(\d{6})([CP])(\d{8})")

def parse_opra(code):
    m = OPRA.fullmatch(code or "")
    if not m:
        return None, None, None
    yymmdd, cp, strike_raw = m.groups()
    exp = f"20{yymmdd[:2]}-{yymmdd[2:4]}-{yymmdd[4:6]}"
    strike = int(strike_raw) / 1000.0
    return exp, strike, cp

def to_float(x):
    if x is None:
        return None
    s = str(x).strip()
    s = (s.replace("\u2212", "-")
         .replace("\u2013", "-")
         .replace("\u2014", "-")
         .replace("\u00A0", "")
         .replace(",", ""))
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    if s.endswith("%"):
        s = s[:-1]
    try:
        v = float(s)
        return None if math.isnan(v) or math.isinf(v) else v
    except:
        return None

def to_int(x):
    v = to_float(x)
    return int(v) if v is not None else None

def load(df):
    with psycopg2.connect(NEON_URL) as conn, conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
        cur.execute(f"TRUNCATE {TABLE};")
        execute_values(cur,
            f"INSERT INTO {TABLE} VALUES %s",
            [tuple(r) for r in df.itertuples(index=False)]
        )
    print(f"✅ Inserted {len(df)} rows into Neon (raw format).")

def main():
    print("Fetching data...")
    r = requests.get(CHAIN_URL, headers=HEADERS, timeout=60)
    data = r.json()
    options = data.get("data", {}).get("options", [])
    if not options:
        raise Exception("No options returned.")

    rows = []
    now = datetime.now(timezone.utc)

    for o in options:
        exp, strike, cp = parse_opra(o.get("option"))
        if not exp or strike is None or cp not in ("C", "P"):
            continue

        rows.append([
            now,
            exp,
            strike,
            cp,
            to_float(o.get("last_trade_price")),
            to_float(o.get("bid")),
            to_float(o.get("ask")),
            to_int(o.get("volume")),
            to_int(o.get("open_interest")),
            to_float(o.get("iv")),
        ])

    df = pd.DataFrame(rows, columns=[
        "run_ts","expiration_date","strike","cp",
        "last","bid","ask","volume","oi","iv"
    ])

    print(f"Parsed rows: {len(df)}")
    load(df)

if __name__ == "__main__":
    main()
