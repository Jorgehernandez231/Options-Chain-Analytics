import os, json
import pandas as pd
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import psycopg2
from psycopg2.extras import execute_values

NEON_URL = os.environ["NEON_URL"]  # set as GitHub Secret
TABLE = os.environ.get("NEON_TABLE", "spx_chain")

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

PAGE_URL = "https://www.cboe.com/delayed_quotes/spx/quote_table"
JSON_URL = "https://data.cboe.com/api/v1/marketdata/options/spx/chains?all=true"

def should_run_now() -> bool:
    """Only run around 16:10 America/New_York (market close buffer)."""
    now_ny = datetime.now(ZoneInfo("America/New_York"))
    return now_ny.hour == 16 and 5 <= now_ny.minute <= 25

def json_to_df(data: dict) -> pd.DataFrame:
    rows = []
    for item in data.get("options", []):
        rows.append({
            "expiration_date": item.get("expirationDate"),
            "strike": item.get("strikePrice"),
            "call_last": item.get("call", {}).get("last"),
            "call_bid": item.get("call", {}).get("bid"),
            "call_ask": item.get("call", {}).get("ask"),
            "call_volume": item.get("call", {}).get("volume"),
            "call_oi": item.get("call", {}).get("openInterest"),
            "call_iv": item.get("call", {}).get("iv"),
            "put_last": item.get("put", {}).get("last"),
            "put_bid": item.get("put", {}).get("bid"),
            "put_ask": item.get("put", {}).get("ask"),
            "put_volume": item.get("put", {}).get("volume"),
            "put_oi": item.get("put", {}).get("openInterest"),
            "put_iv": item.get("put", {}).get("iv"),
        })
    df = pd.DataFrame(rows)
    df.insert(0, "run_ts", datetime.now(timezone.utc))
    return df

def fetch_via_requests() -> dict:
    import requests
    from requests.adapters import HTTPAdapter, Retry

    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/124.0.0.0 Safari/537.36"),
        "Accept": "application/json,text/plain,*/*",
        "Origin": "https://www.cboe.com",
        "Referer": PAGE_URL,
        "Cache-Control": "no-cache",
    }
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=0.5, status_forcelist=[403,429,500,502,503,504])))
    r = s.get(JSON_URL, headers=headers, timeout=45)
    r.raise_for_status()
    return r.json()

def fetch_via_playwright() -> dict:
    # Fallback if CDN blocks raw requests
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context()
        page = ctx.new_page()
        page.goto(PAGE_URL, wait_until="domcontentloaded")
        # accept cookies if present
        for sel in ["#onetrust-accept-btn-handler","button#onetrust-accept-btn-handler","button:has-text('Accept All')"]:
            loc = page.locator(sel)
            if loc.count():
                try:
                    loc.first.click(timeout=1500)
                    break
                except Exception:
                    pass
        text = page.evaluate(
            """async (url) => {
                const r = await fetch(url, { headers: { "accept": "application/json" }, credentials: "include" });
                if (!r.ok) throw new Error("fetch failed " + r.status);
                return await r.text();
            }""",
            JSON_URL
        )
        browser.close()
    return json.loads(text)

def load_into_neon(df: pd.DataFrame):
    with psycopg2.connect(NEON_URL) as conn, conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
        cols = list(df.columns)
        vals = [tuple(None if pd.isna(x) else x for x in r) for r in df.itertuples(index=False, name=None)]
        execute_values(cur, f"INSERT INTO {TABLE} ({','.join(cols)}) VALUES %s", vals)
    print(f"✅ Inserted {len(df)} rows into Neon table {TABLE}")

def main():
    if not should_run_now():
        print("⏭️ Not market close window (NY). Exiting.")
        return
    try:
        data = fetch_via_requests()
    except Exception as e:
        print(f"requests failed ({e}); trying Playwright fallback…")
        data = fetch_via_playwright()

    df = json_to_df(data)
    if df.empty:
        raise RuntimeError("Empty DataFrame from API.")
    load_into_neon(df)

if __name__ == "__main__":
    main()
