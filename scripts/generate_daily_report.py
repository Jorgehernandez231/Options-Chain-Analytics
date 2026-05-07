import os
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import pandas as pd
from sqlalchemy import create_engine, text


# =========================
# CONFIG
# =========================

DATABASE_URL = (
    os.getenv("DATABASE_URL")
    or os.getenv("NEON_URL")
    or os.getenv("N_URL")
)

EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_TO = os.getenv("EMAIL_TO")

TABLE_NAME = os.getenv("OPTIONS_TABLE")

OPTIONS_SYMBOLS = os.getenv("OPTIONS_SYMBOLS", "SPX,NDX,VIX")
SYMBOLS = [s.strip().upper() for s in OPTIONS_SYMBOLS.split(",") if s.strip()]


# =========================
# DATABASE
# =========================

def get_engine():
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL, NEON_URL or N_URL is missing.")
    return create_engine(DATABASE_URL)


def load_latest_snapshot_for_symbol(engine, symbol):
    query_latest_run = text(f"""
        SELECT MAX(run_ts) AS latest_run
        FROM {TABLE_NAME}
        WHERE symbol = :symbol
    """)

    latest_run = pd.read_sql(
        query_latest_run,
        engine,
        params={"symbol": symbol}
    )["latest_run"].iloc[0]

    if pd.isna(latest_run):
        print(f"[WARNING] No run_ts found for {symbol}. Skipping.")
        return pd.DataFrame(), None

    query_data = text(f"""
        SELECT 
            id,
            symbol,
            run_ts,
            underlying_px,
            expiration_date,
            strike,
            cp,
            last,
            bid,
            ask,
            volume,
            oi,
            iv,
            delta,
            gamma,
            option_net
        FROM {TABLE_NAME}
        WHERE run_ts = :latest_run
          AND symbol = :symbol
        ORDER BY expiration_date, strike, cp
    """)

    df = pd.read_sql(
        query_data,
        engine,
        params={
            "latest_run": latest_run,
            "symbol": symbol
        }
    )

    return df, latest_run


# =========================
# CALC HELPERS
# =========================

def safe_divide(a, b):
    if b == 0 or pd.isna(b):
        return None
    return a / b


def get_value(row, column):
    if row.empty:
        return None
    return row[column].iloc[0]


def get_underlying_price(df):
    px = df["underlying_px"].dropna()
    if not px.empty:
        return float(px.iloc[0])
    return None


def calculate_atm_iv(df, spot):
    if spot is None:
        return None

    tmp = df.copy()
    tmp = tmp[tmp["iv"].notna()]
    tmp = tmp[tmp["iv"] > 0]

    if tmp.empty:
        return None

    tmp["distance_to_spot"] = (tmp["strike"] - spot).abs()

    atm_rows = tmp.sort_values("distance_to_spot").head(10)

    if atm_rows.empty:
        return None

    return float(atm_rows["iv"].mean())


def calculate_gamma_exposure(df):
    """
    Approximate gamma exposure.

    Formula:
    GEX ≈ gamma * OI * 100 * S² * 0.01

    Puts are treated as negative gamma contribution for positioning view.
    """
    tmp = df.copy()

    spot = get_underlying_price(tmp)

    if spot is None:
        return None, None, None

    tmp["gamma_exposure"] = tmp["gamma"] * tmp["oi"] * 100 * (spot ** 2) * 0.01

    tmp.loc[tmp["cp"] == "P", "gamma_exposure"] *= -1

    gex_by_strike = (
        tmp.groupby("strike", as_index=False)["gamma_exposure"]
        .sum()
        .sort_values("strike")
    )

    if gex_by_strike.empty:
        return None, None, None

    total_gex = gex_by_strike["gamma_exposure"].sum()

    max_positive_gex_row = gex_by_strike.sort_values(
        "gamma_exposure",
        ascending=False
    ).head(1)

    max_negative_gex_row = gex_by_strike.sort_values(
        "gamma_exposure",
        ascending=True
    ).head(1)

    max_positive_gex_strike = get_value(max_positive_gex_row, "strike")
    max_negative_gex_strike = get_value(max_negative_gex_row, "strike")

    return total_gex, max_positive_gex_strike, max_negative_gex_strike


def calculate_expected_move(df, spot, atm_iv):
    """
    Simple expected move approximation using nearest expiration:

    1σ move = spot * ATM IV * sqrt(days_to_exp / 365)

    This is useful for daily report levels.
    """
    if spot is None or atm_iv is None:
        return None, None, None, None

    expirations = sorted(df["expiration_date"].dropna().unique())

    if not expirations:
        return None, None, None, None

    nearest_exp = pd.to_datetime(expirations[0]).date()
    today = datetime.utcnow().date()

    days_to_exp = max((nearest_exp - today).days, 1)

    one_sigma_move = spot * atm_iv * ((days_to_exp / 365) ** 0.5)

    one_sigma_low = spot - one_sigma_move
    one_sigma_high = spot + one_sigma_move

    return days_to_exp, one_sigma_move, one_sigma_low, one_sigma_high


# =========================
# METRICS
# =========================

def calculate_metrics_for_symbol(df, latest_run, symbol):
    df = df.copy()

    numeric_cols = [
        "underlying_px", "strike", "last", "bid", "ask",
        "volume", "oi", "iv", "delta", "gamma", "option_net"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["volume"] = df["volume"].fillna(0)
    df["oi"] = df["oi"].fillna(0)
    df["gamma"] = df["gamma"].fillna(0)

    calls = df[df["cp"] == "C"]
    puts = df[df["cp"] == "P"]

    call_volume = calls["volume"].sum()
    put_volume = puts["volume"].sum()

    call_oi = calls["oi"].sum()
    put_oi = puts["oi"].sum()

    put_call_volume_ratio = safe_divide(put_volume, call_volume)
    put_call_oi_ratio = safe_divide(put_oi, call_oi)

    spot = get_underlying_price(df)
    atm_iv = calculate_atm_iv(df, spot)

    days_to_exp, one_sigma_move, one_sigma_low, one_sigma_high = calculate_expected_move(
        df,
        spot,
        atm_iv
    )

    max_call_volume_row = calls.sort_values("volume", ascending=False).head(1)
    max_put_volume_row = puts.sort_values("volume", ascending=False).head(1)

    max_call_oi_row = calls.sort_values("oi", ascending=False).head(1)
    max_put_oi_row = puts.sort_values("oi", ascending=False).head(1)

    expirations = sorted(df["expiration_date"].dropna().unique())

    total_gex, max_positive_gex_strike, max_negative_gex_strike = calculate_gamma_exposure(df)

    metrics = {
        "symbol": symbol,
        "latest_run": latest_run,
        "spot": spot,
        "atm_iv": atm_iv,
        "days_to_exp": days_to_exp,
        "one_sigma_move": one_sigma_move,
        "one_sigma_low": one_sigma_low,
        "one_sigma_high": one_sigma_high,
        "total_rows": len(df),
        "num_expirations": len(expirations),
        "first_expiration": expirations[0] if expirations else None,
        "last_expiration": expirations[-1] if expirations else None,
        "call_volume": call_volume,
        "put_volume": put_volume,
        "call_oi": call_oi,
        "put_oi": put_oi,
        "put_call_volume_ratio": put_call_volume_ratio,
        "put_call_oi_ratio": put_call_oi_ratio,
        "max_call_volume_strike": get_value(max_call_volume_row, "strike"),
        "max_call_volume": get_value(max_call_volume_row, "volume"),
        "max_put_volume_strike": get_value(max_put_volume_row, "strike"),
        "max_put_volume": get_value(max_put_volume_row, "volume"),
        "max_call_oi_strike": get_value(max_call_oi_row, "strike"),
        "max_call_oi": get_value(max_call_oi_row, "oi"),
        "max_put_oi_strike": get_value(max_put_oi_row, "strike"),
        "max_put_oi": get_value(max_put_oi_row, "oi"),
        "total_gex": total_gex,
        "max_positive_gex_strike": max_positive_gex_strike,
        "max_negative_gex_strike": max_negative_gex_strike,
    }

    return metrics


def collect_all_metrics(engine, symbols):
    all_metrics = []

    for symbol in symbols:
        print(f"Loading latest snapshot for {symbol}...")

        df, latest_run = load_latest_snapshot_for_symbol(engine, symbol)

        if df.empty or latest_run is None:
            continue

        metrics = calculate_metrics_for_symbol(df, latest_run, symbol)
        all_metrics.append(metrics)

        print(f"{symbol}: {len(df)} rows analysed. Latest run_ts: {latest_run}")

    if not all_metrics:
        raise ValueError("No data found for any configured symbol.")

    return all_metrics


# =========================
# FORMATTING
# =========================

def fmt_number(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:,.0f}"


def fmt_price(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:,.2f}"


def fmt_decimal(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.2f}"


def fmt_percent(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value * 100:.2f}%"


def fmt_gex(value):
    if value is None or pd.isna(value):
        return "N/A"

    abs_value = abs(value)

    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"

    return f"{value:,.0f}"


def fmt_datetime(value):
    if value is None:
        return "N/A"
    if isinstance(value, str):
        return value
    return value.strftime("%Y-%m-%d %H:%M:%S")


# =========================
# INTERPRETATION
# =========================

def build_interpretation(metrics):
    comments = []

    symbol = metrics["symbol"]
    pc_volume = metrics["put_call_volume_ratio"]
    pc_oi = metrics["put_call_oi_ratio"]
    total_gex = metrics["total_gex"]

    if pc_volume is not None:
        if pc_volume > 1.2:
            comments.append(
                f"{symbol}: put volume is meaningfully higher than call volume, "
                "suggesting stronger downside hedging or bearish activity in this snapshot."
            )
        elif pc_volume < 0.8:
            comments.append(
                f"{symbol}: call volume is higher than put volume, "
                "suggesting stronger upside activity in this snapshot."
            )
        else:
            comments.append(
                f"{symbol}: put and call volume are relatively balanced."
            )

    if pc_oi is not None:
        if pc_oi > 1:
            comments.append(
                "Total put open interest is higher than call open interest."
            )
        else:
            comments.append(
                "Total call open interest is higher than put open interest."
            )

    if total_gex is not None:
        if total_gex > 0:
            comments.append(
                "Net gamma exposure is positive, which can be associated with more stable, "
                "mean-reverting market behaviour around key strikes."
            )
        elif total_gex < 0:
            comments.append(
                "Net gamma exposure is negative, which can be associated with higher "
                "directional sensitivity and potentially larger intraday moves."
            )

    if not comments:
        return "No interpretation available due to missing metrics."

    return " ".join(comments)


# =========================
# HTML REPORT
# =========================

def build_summary_table(all_metrics):
    rows = ""

    for m in all_metrics:
        rows += f"""
        <tr>
            <td><strong>{m["symbol"]}</strong></td>
            <td>{fmt_price(m["spot"])}</td>
            <td>{fmt_percent(m["atm_iv"])}</td>
            <td>{fmt_decimal(m["put_call_volume_ratio"])}</td>
            <td>{fmt_decimal(m["put_call_oi_ratio"])}</td>
            <td>{fmt_gex(m["total_gex"])}</td>
            <td>{fmt_number(m["max_call_oi_strike"])}</td>
            <td>{fmt_number(m["max_put_oi_strike"])}</td>
        </tr>
        """

    return f"""
    <h2>Cross-Symbol Summary</h2>

    <table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%;">
        <tr style="background-color: #f2f2f2;">
            <th>Symbol</th>
            <th>Underlying</th>
            <th>ATM IV</th>
            <th>Put/Call Vol</th>
            <th>Put/Call OI</th>
            <th>Net GEX</th>
            <th>Max Call OI Strike</th>
            <th>Max Put OI Strike</th>
        </tr>
        {rows}
    </table>
    """


def build_symbol_section(metrics):
    interpretation = build_interpretation(metrics)

    return f"""
    <hr style="margin-top: 30px; margin-bottom: 30px;">

    <h2>{metrics["symbol"]} Options Report</h2>

    <p>
        <strong>Latest snapshot:</strong> {fmt_datetime(metrics["latest_run"])}<br>
        <strong>Underlying price:</strong> {fmt_price(metrics["spot"])}<br>
        <strong>ATM IV approximation:</strong> {fmt_percent(metrics["atm_iv"])}<br>
        <strong>Rows analysed:</strong> {fmt_number(metrics["total_rows"])}<br>
        <strong>Expirations analysed:</strong> {metrics["num_expirations"]}<br>
        <strong>Expiration range:</strong> {metrics["first_expiration"]} to {metrics["last_expiration"]}
    </p>

    <h3>Expected Move</h3>

    <table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse;">
        <tr style="background-color: #f2f2f2;">
            <th>Metric</th>
            <th>Value</th>
        </tr>
        <tr>
            <td>Nearest expiration days</td>
            <td>{fmt_number(metrics["days_to_exp"])}</td>
        </tr>
        <tr>
            <td>1σ Expected Move</td>
            <td>±{fmt_price(metrics["one_sigma_move"])}</td>
        </tr>
        <tr>
            <td>1σ Expected Range</td>
            <td>{fmt_price(metrics["one_sigma_low"])} – {fmt_price(metrics["one_sigma_high"])}</td>
        </tr>
    </table>

    <h3>Market Positioning</h3>

    <table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse;">
        <tr style="background-color: #f2f2f2;">
            <th>Metric</th>
            <th>Value</th>
        </tr>
        <tr>
            <td>Call Volume</td>
            <td>{fmt_number(metrics["call_volume"])}</td>
        </tr>
        <tr>
            <td>Put Volume</td>
            <td>{fmt_number(metrics["put_volume"])}</td>
        </tr>
        <tr>
            <td>Put/Call Volume Ratio</td>
            <td>{fmt_decimal(metrics["put_call_volume_ratio"])}</td>
        </tr>
        <tr>
            <td>Call Open Interest</td>
            <td>{fmt_number(metrics["call_oi"])}</td>
        </tr>
        <tr>
            <td>Put Open Interest</td>
            <td>{fmt_number(metrics["put_oi"])}</td>
        </tr>
        <tr>
            <td>Put/Call OI Ratio</td>
            <td>{fmt_decimal(metrics["put_call_oi_ratio"])}</td>
        </tr>
    </table>

    <h3>Key Strikes</h3>

    <table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse;">
        <tr style="background-color: #f2f2f2;">
            <th>Category</th>
            <th>Strike</th>
            <th>Value</th>
        </tr>
        <tr>
            <td>Highest Call Volume</td>
            <td>{fmt_number(metrics["max_call_volume_strike"])}</td>
            <td>{fmt_number(metrics["max_call_volume"])}</td>
        </tr>
        <tr>
            <td>Highest Put Volume</td>
            <td>{fmt_number(metrics["max_put_volume_strike"])}</td>
            <td>{fmt_number(metrics["max_put_volume"])}</td>
        </tr>
        <tr>
            <td>Highest Call OI</td>
            <td>{fmt_number(metrics["max_call_oi_strike"])}</td>
            <td>{fmt_number(metrics["max_call_oi"])}</td>
        </tr>
        <tr>
            <td>Highest Put OI</td>
            <td>{fmt_number(metrics["max_put_oi_strike"])}</td>
            <td>{fmt_number(metrics["max_put_oi"])}</td>
        </tr>
    </table>

    <h3>Gamma Exposure Approximation</h3>

    <table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse;">
        <tr style="background-color: #f2f2f2;">
            <th>Metric</th>
            <th>Value</th>
        </tr>
        <tr>
            <td>Net Gamma Exposure</td>
            <td>{fmt_gex(metrics["total_gex"])}</td>
        </tr>
        <tr>
            <td>Highest Positive GEX Strike</td>
            <td>{fmt_number(metrics["max_positive_gex_strike"])}</td>
        </tr>
        <tr>
            <td>Highest Negative GEX Strike</td>
            <td>{fmt_number(metrics["max_negative_gex_strike"])}</td>
        </tr>
    </table>

    <h3>Interpretation</h3>

    <p>{interpretation}</p>
    """


def build_html_report(all_metrics):
    today = datetime.now().strftime("%Y-%m-%d")

    sections = ""

    for metrics in all_metrics:
        sections += build_symbol_section(metrics)

    html = f"""
    <html>
    <body style="font-family: Arial, sans-serif; color: #222; line-height: 1.5;">
        <h1>Daily Options Report — {today}</h1>

        <p>
            Symbols analysed: <strong>{", ".join([m["symbol"] for m in all_metrics])}</strong>
        </p>

        {build_summary_table(all_metrics)}

        {sections}

        <hr style="margin-top: 30px;">

        <p style="font-size: 12px; color: #666;">
            Automated report generated from your Neon PostgreSQL options database.
            This report is for analysis and educational purposes, not financial advice.
        </p>
    </body>
    </html>
    """

    return html


# =========================
# SEND EMAIL
# =========================

def send_email(subject, html_body):
    if not EMAIL_USER:
        raise ValueError("EMAIL_USER is missing.")
    if not EMAIL_PASSWORD:
        raise ValueError("EMAIL_PASSWORD is missing.")
    if not EMAIL_TO:
        raise ValueError("EMAIL_TO is missing.")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = EMAIL_USER
    msg["To"] = EMAIL_TO

    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_USER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_USER, EMAIL_TO, msg.as_string())


# =========================
# MAIN
# =========================

def main():
    if not SYMBOLS:
        raise ValueError("No symbols configured. Set OPTIONS_SYMBOLS, for example: SPX,NDX,VIX")

    print(f"Configured symbols: {SYMBOLS}")

    engine = get_engine()

    all_metrics = collect_all_metrics(engine, SYMBOLS)

    html = build_html_report(all_metrics)

    today = datetime.now().strftime("%Y-%m-%d")
    subject_symbols = ", ".join([m["symbol"] for m in all_metrics])
    subject = f"Daily Options Report — {subject_symbols} — {today}"

    send_email(subject, html)

    print("Daily options report sent successfully.")
    print(f"Symbols included: {subject_symbols}")


if __name__ == "__main__":
    main()