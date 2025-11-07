import os
import pandas as pd
from datetime import datetime, timezone
import psycopg2
from psycopg2.extras import execute_values
import requests

NEON_URL = "postgresql://neondb_owner:npg_JQ4KUj8XoNTm@ep-small-fire-agsc86td-pooler.c-2.eu-central-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
TABLE = "spx_chain"

CHAIN_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json"

SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    run_ts TIMESTAMPTZ NOT NULL,
    expiration_date TEXT,
    strike NUMERIC,
    call_last NUMERIC,
    call_bid NUMERIC,
    call_ask NUMERIC,
    call_volume BIGINT,
    call_oi BIGINT,
    call_iv NUMERIC,
    put_last NUMERIC,
    put_bid NUMERIC,
    put_ask NUMERIC,
    put_volume BIGINT,
    put_oi BIGINT,
    put_iv NUMERIC
);
"""

REPLACE_OLD = True  # set False if you ever want to keep history

def load_into_neon(df):
    if df.empty:
        print("No rows to insert; skipping.")
        return

    with psycopg2.connect(NEON_URL) as conn, conn.cursor() as cur:
        # Ensure schema
        cur.execute(SCHEMA_SQL)

        if REPLACE_OLD:
            # Remove everything in one shot (fast) before inserting the fresh snapshot
            cur.execute(f"TRUNCATE {TABLE};")

        cols = df.columns.tolist()
        rows = [tuple(None if pd.isna(x) else x for x in r) for r in df.itertuples(index=False)]
        execute_values(cur, f"INSERT INTO {TABLE} ({','.join(cols)}) VALUES %s", rows)

    print(f"✅ Inserted {len(df)} rows into Neon (old data {'removed' if REPLACE_OLD else 'kept'}).")


def main():
    r = requests.get(CHAIN_URL, timeout=30)
    r.raise_for_status()
    data = r.json()

    # ✅ Correct JSON structure
    options = data["data"]["options"]

    rows = []
    for o in options:
        rows.append({
            "run_ts": datetime.now(timezone.utc),
            "expiration_date": o.get("expirationDate"),
            "strike": o.get("strikePrice"),
            "call_last": o.get("call", {}).get("last"),
            "call_bid": o.get("call", {}).get("bid"),
            "call_ask": o.get("call", {}).get("ask"),
            "call_volume": o.get("call", {}).get("volume"),
            "call_oi": o.get("call", {}).get("open_interest"),
            "call_iv": o.get("call", {}).get("iv"),
            "put_last": o.get("put", {}).get("last"),
            "put_bid": o.get("put", {}).get("bid"),
            "put_ask": o.get("put", {}).get("ask"),
            "put_volume": o.get("put", {}).get("volume"),
            "put_oi": o.get("put", {}).get("open_interest"),
            "put_iv": o.get("put", {}).get("iv"),
        })

    df = pd.DataFrame(rows)
    load_into_neon(df)

if __name__ == "__main__":
    main()

