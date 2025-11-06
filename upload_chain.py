import sys, re, io
from datetime import datetime, timezone
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

# ---- EDIT ONLY IF YOU WANT A DIFFERENT TABLE NAME ----
DB_CONN = "postgresql://neondb_owner:npg_JQ4KUj8XoNTm@ep-small-fire-agsc86td-pooler.c-2.eu-central-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
TABLE = "spx_chain"

SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    run_ts           TIMESTAMPTZ NOT NULL,
    expiration_date  TEXT,
    strike           NUMERIC,
    call_last        NUMERIC,
    call_bid         NUMERIC,
    call_ask         NUMERIC,
    call_volume      BIGINT,
    call_oi          BIGINT,
    call_iv          NUMERIC,
    put_last         NUMERIC,
    put_bid          NUMERIC,
    put_ask          NUMERIC,
    put_volume       BIGINT,
    put_oi           BIGINT,
    put_iv           NUMERIC
);
"""

def read_text_with_fallback(path):
    for enc in ("utf-8-sig", "utf-8", "latin-1", "utf-16", "utf-16le", "utf-16be"):
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                return f.read()
        except Exception:
            continue
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()

def find_header_index(lines):
    """
    Find the first line that looks like the header row.
    Your screenshot shows the header at visual row 4 (0-based index 3),
    containing 'Expiration Date,Calls,Last Sale,...,Strike,Puts,...'
    """
    for i, line in enumerate(lines[:50]):  # scan first 50 lines
        if ("Expiration Date" in line) and ("Strike" in line):
            return i
    # fallback to row 3 if not found
    return 3

def load_csv(path):
    raw = read_text_with_fallback(path)
    lines = raw.splitlines()
    hdr_idx = find_header_index(lines)
    body = "\n".join(lines[hdr_idx:])

    # Let pandas infer delimiter; your file is comma separated, but this is robust
    df = pd.read_csv(io.StringIO(body), engine="python", sep=None)

    # Expected columns come in duplicates for Calls vs Puts:
    # 'Last Sale', 'Bid', 'Ask', 'Volume', 'IV', 'Open Interest'
    # and their '.1' versions for puts.
    # We'll build a safe column-getter that tolerates tiny name differences.
    def pick(colname, alt=None):
        for c in df.columns:
            if c.strip().lower() == colname.strip().lower():
                return c
        if alt:
            for c in df.columns:
                if c.strip().lower() == alt.strip().lower():
                    return c
        return None

    # Map canonical names
    cols = {
        "expiration_date": pick("Expiration Date"),
        "strike":          pick("Strike"),
        "call_last":       pick("Last Sale"),
        "call_bid":        pick("Bid"),
        "call_ask":        pick("Ask"),
        "call_volume":     pick("Volume"),
        "call_iv":         pick("IV"),
        "call_oi":         pick("Open Interest"),
        "put_last":        pick("Last Sale.1", "Last Sale_1"),
        "put_bid":         pick("Bid.1", "Bid_1"),
        "put_ask":         pick("Ask.1", "Ask_1"),
        "put_volume":      pick("Volume.1", "Volume_1"),
        "put_iv":          pick("IV.1", "IV_1"),
        "put_oi":          pick("Open Interest.1", "Open Interest_1"),
    }

    # If pandas renamed duplicates differently (e.g., '.1' became '.2'),
    # fall back by position using the well-known layout:
    # [Expiration Date, Calls, Last Sale, Net, Bid, Ask, Volume, IV, Delta, Gamma, Open Interest,
    #  Strike, Puts, Last Sale, Net, Bid, Ask, Volume, IV, Delta, Gamma, Open Interest]
    if any(v is None for v in cols.values()):
        # try positional rescue
        try:
            pos = {name: None for name in cols}
            headers = list(df.columns)

            def find_index_exact(h):
                try:
                    return headers.index(h)
                except ValueError:
                    return None

            idx_exp = find_index_exact("Expiration Date")
            idx_strike = find_index_exact("Strike")
            # calls block indices (relative to header)
            # Exp Date, Calls, Last Sale, Net, Bid, Ask, Volume, IV, Delta, Gamma, Open Interest
            if idx_exp is not None and idx_strike is not None and idx_strike > idx_exp:
                pos["expiration_date"] = headers[idx_exp]
                pos["call_last"]  = headers[idx_exp + 2] if len(headers) > idx_exp + 2 else None
                pos["call_bid"]   = headers[idx_exp + 4] if len(headers) > idx_exp + 4 else None
                pos["call_ask"]   = headers[idx_exp + 5] if len(headers) > idx_exp + 5 else None
                pos["call_volume"]= headers[idx_exp + 6] if len(headers) > idx_exp + 6 else None
                pos["call_iv"]    = headers[idx_exp + 7] if len(headers) > idx_exp + 7 else None
                pos["call_oi"]    = headers[idx_exp +10] if len(headers) > idx_exp +10 else None
                pos["strike"]     = headers[idx_strike]

                # puts block starts right after 'Puts' label (idx_strike+1 is Puts, +2 Last Sale, +4 Bid, +5 Ask, +6 Volume, +7 IV, +10 OI)
                pos["put_last"]   = headers[idx_strike + 2] if len(headers) > idx_strike + 2 else None
                pos["put_bid"]    = headers[idx_strike + 4] if len(headers) > idx_strike + 4 else None
                pos["put_ask"]    = headers[idx_strike + 5] if len(headers) > idx_strike + 5 else None
                pos["put_volume"] = headers[idx_strike + 6] if len(headers) > idx_strike + 6 else None
                pos["put_iv"]     = headers[idx_strike + 7] if len(headers) > idx_strike + 7 else None
                pos["put_oi"]     = headers[idx_strike +10] if len(headers) > idx_strike +10 else None

                for k, v in pos.items():
                    if cols.get(k) is None and v is not None:
                        cols[k] = v
        except Exception:
            pass

    # Keep only the columns we have
    keep_pairs = [(k, v) for k, v in cols.items() if v in df.columns]
    if not keep_pairs:
        raise RuntimeError(f"Could not find expected headers. Got: {list(df.columns)}")

    tidy = pd.DataFrame()
    for k, v in keep_pairs:
        tidy[k] = df[v]

    # Convert numeric columns; tolerate commas and blanks
    num_cols = [
        "strike","call_last","call_bid","call_ask","call_volume","call_oi","call_iv",
        "put_last","put_bid","put_ask","put_volume","put_oi","put_iv"
    ]
    for c in num_cols:
        if c in tidy.columns:
            tidy[c] = (
                tidy[c]
                .astype(str)
                .str.replace(",", "", regex=False)
                .replace({"": None, "nan": None, "None": None})
            )
            tidy[c] = pd.to_numeric(tidy[c], errors="coerce")

    tidy.insert(0, "run_ts", datetime.now(timezone.utc))
    return tidy

def insert_neon(df):
    with psycopg2.connect(DB_CONN) as conn, conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
        cols = list(df.columns)
        rows = [tuple(None if pd.isna(x) else x for x in r) for r in df.itertuples(index=False, name=None)]
        execute_values(cur, f"INSERT INTO {TABLE} ({','.join(cols)}) VALUES %s", rows)
    print(f"✅ Inserted {len(df)} rows into Neon → {TABLE}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python upload_chain.py <path_to_csv>")
        sys.exit(1)
    path = sys.argv[1]
    df = load_csv(path)
    if df.empty:
        raise RuntimeError("Parsed DataFrame is empty. Check the file/headers.")
    insert_neon(df)

if __name__ == "__main__":
    main()
