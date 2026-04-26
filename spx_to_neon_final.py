# spx_to_neon_final.py

import os
import re
import math
import requests
import pandas as pd
from datetime import datetime, timezone

import psycopg2
from psycopg2.extras import execute_values

try:
    import streamlit as st
except Exception:
    st = None


# ============================================================
# CONFIG
# ============================================================

TABLE = "options_chain"
RETENTION_DAYS = 4

SYMBOLS_TO_RUN = ["SPX"]

SOURCES = {
    "SPX": {
        "url": "https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json",
        "referer": "https://www.cboe.com/delayed_quotes/spx/quote_table",
        "opra_regex": re.compile(r"SPXW?(\d{6})([CP])(\d{8})"),
    },
    "NDX": {
        "url": "https://cdn.cboe.com/api/global/delayed_quotes/options/_NDX.json",
        "referer": "https://www.cboe.com/delayed_quotes/ndx/quote_table",
        "opra_regex": re.compile(r"NDX\w?(\d{6})([CP])(\d{8})"),
    },
    "VIX": {
        "url": "https://cdn.cboe.com/api/global/delayed_quotes/options/VIX.json",
        "referer": "https://www.cboe.com/delayed_quotes/vix/quote_table",
        "opra_regex": re.compile(r"VIX\w?(\d{6})([CP])(\d{8})"),
    },
}

INSERT_COLUMNS = [
    "symbol",
    "run_ts",
    "underlying_px",
    "expiration_date",
    "strike",
    "cp",
    "last",
    "bid",
    "ask",
    "volume",
    "oi",
    "iv",
    "delta",
    "gamma",
    "option_net",
]


SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    id BIGSERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    run_ts TIMESTAMPTZ NOT NULL,
    underlying_px NUMERIC,
    expiration_date DATE NOT NULL,
    strike NUMERIC NOT NULL,
    cp TEXT NOT NULL,
    last NUMERIC,
    bid NUMERIC,
    ask NUMERIC,
    volume NUMERIC(20,0),
    oi NUMERIC(20,0),
    iv NUMERIC,
    delta NUMERIC,
    gamma NUMERIC,
    option_net NUMERIC
);
"""


UNIQUE_INDEX_SQL = f"""
CREATE UNIQUE INDEX IF NOT EXISTS uq_options_chain_snapshot_contract
ON {TABLE}(symbol, run_ts, expiration_date, strike, cp);
"""


# ============================================================
# SECRETS / CONNECTION
# ============================================================

def get_neon_url() -> str:
    """
    Works both locally/GitHub Actions via environment variable
    and in Streamlit via st.secrets.
    """
    env_url = os.getenv("NEON_URL")
    if env_url:
        return env_url

    if st is not None:
        try:
            return st.secrets["NEON_URL"]
        except Exception:
            pass

    raise RuntimeError(
        "NEON_URL not found. Set it as an environment variable or in Streamlit secrets."
    )


NEON_URL = get_neon_url()


# ============================================================
# PARSING HELPERS
# ============================================================

def to_float(x):
    if x is None:
        return None

    s = str(x).strip()
    s = (
        s.replace("\u2212", "-")
         .replace("\u2013", "-")
         .replace("\u2014", "-")
         .replace("\u00A0", "")
         .replace(",", "")
    )

    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]

    if s.endswith("%"):
        s = s[:-1]

    try:
        value = float(s)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    except Exception:
        return None


def to_int(x):
    value = to_float(x)
    return int(value) if value is not None else None


def parse_opra(code, opra_regex):
    """
    Parses Cboe OPRA-style option code.

    Example:
    SPXW260417C05000000
    -> expiration_date, strike, cp
    """
    code = str(code or "").strip()
    match = opra_regex.fullmatch(code)

    if not match:
        return None, None, None

    yymmdd, cp, strike_raw = match.groups()

    exp = f"20{yymmdd[:2]}-{yymmdd[2:4]}-{yymmdd[4:6]}"
    strike = int(strike_raw) / 1000.0

    return exp, strike, cp


def build_headers(referer):
    return {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json,text/plain,*/*",
        "Referer": referer,
    }


# ============================================================
# FETCH / NORMALIZE
# ============================================================

def fetch_cboe_json(symbol, config):
    print(f"Fetching {symbol} data...")

    response = requests.get(
        config["url"],
        headers=build_headers(config["referer"]),
        timeout=60,
    )

    response.raise_for_status()
    return response.json()


def extract_underlying_px(data):
    data_block = data.get("data", {})

    return to_float(
        data_block.get("underlying_price")
        or data_block.get("underlying_last")
        or data_block.get("underlying")
        or data_block.get("last")
    )


def parse_cboe_options(symbol, data, opra_regex):
    data_block = data.get("data", {})
    options = data_block.get("options", [])

    if not options:
        raise RuntimeError(f"No options returned for {symbol}.")

    underlying_px = extract_underlying_px(data)

    if underlying_px is None:
        print(f"⚠️ Could not detect underlying_px for {symbol}; inserting NULL.")

    run_ts = datetime.now(timezone.utc)

    rows = []

    for option in options:
        exp, strike, cp = parse_opra(option.get("option"), opra_regex)

        if not exp or strike is None or cp not in ("C", "P"):
            continue

        rows.append([
            symbol,
            run_ts,
            underlying_px,
            exp,
            strike,
            cp,
            to_float(option.get("last_trade_price")),
            to_float(option.get("bid")),
            to_float(option.get("ask")),
            to_int(option.get("volume")),
            to_int(option.get("open_interest")),
            to_float(option.get("iv")),
            to_float(option.get("delta")),
            to_float(option.get("gamma")),
            to_float(option.get("option_net")),
        ])

    df = pd.DataFrame(rows, columns=INSERT_COLUMNS)

    return df


# ============================================================
# DATABASE LOAD
# ============================================================

def load_to_neon(df, symbol):
    if df.empty:
        print(f"⚠️ No rows to insert for {symbol}.")
        return

    with psycopg2.connect(NEON_URL) as conn:
        with conn.cursor() as cur:
            # Schema and indexes are managed manually in Neon.
            # Do not create indexes here because the ETL user may not own the table.
            # cur.execute(SCHEMA_SQL)
            # cur.execute(UNIQUE_INDEX_SQL)

            # Rolling retention, per symbol only.
            cur.execute(
                f"""
                DELETE FROM {TABLE}
                WHERE symbol = %s
                  AND run_ts < NOW() - INTERVAL '{RETENTION_DAYS} days';
                """,
                (symbol,),
            )

            insert_sql = f"""
                INSERT INTO {TABLE} ({",".join(INSERT_COLUMNS)})
                VALUES %s
                ON CONFLICT (symbol, run_ts, expiration_date, strike, cp)
                DO UPDATE SET
                    underlying_px = EXCLUDED.underlying_px,
                    last = EXCLUDED.last,
                    bid = EXCLUDED.bid,
                    ask = EXCLUDED.ask,
                    volume = EXCLUDED.volume,
                    oi = EXCLUDED.oi,
                    iv = EXCLUDED.iv,
                    delta = EXCLUDED.delta,
                    gamma = EXCLUDED.gamma,
                    option_net = EXCLUDED.option_net;
                            """

            execute_values(
                cur,
                insert_sql,
                [tuple(row) for row in df[INSERT_COLUMNS].itertuples(index=False, name=None)],
            )

    print(f"✅ Inserted/updated {len(df):,} rows for {symbol} into {TABLE}.")


# ============================================================
# RUNNER
# ============================================================

def run_symbol(symbol):
    if symbol not in SOURCES:
        raise ValueError(f"No source configured for symbol: {symbol}")

    config = SOURCES[symbol]

    data = fetch_cboe_json(symbol, config)
    df = parse_cboe_options(symbol, data, config["opra_regex"])

    print(f"{symbol} parsed rows: {len(df):,}")

    if not df.empty:
        print(df.head())

    load_to_neon(df, symbol)


def main():
    print(f"Running ETL for: {', '.join(SYMBOLS_TO_RUN)}")

    for symbol in SYMBOLS_TO_RUN:
        run_symbol(symbol)

    print("✅ ETL complete.")


if __name__ == "__main__":
    main()