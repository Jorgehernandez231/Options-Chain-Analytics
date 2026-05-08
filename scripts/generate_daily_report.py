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

TABLE_NAME = (os.getenv("OPTIONS_TABLE") or "options_chain").strip()

OPTIONS_SYMBOLS = os.getenv("OPTIONS_SYMBOLS", "SPX,NDX,VIX")
SYMBOLS = [s.strip().upper() for s in OPTIONS_SYMBOLS.split(",") if s.strip()]

CONTRACT_MULTIPLIER = 100


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


def load_previous_snapshot_for_symbol(engine, symbol, latest_run):
    query_previous_run = text(f"""
        SELECT MAX(run_ts) AS previous_run
        FROM {TABLE_NAME}
        WHERE symbol = :symbol
          AND run_ts < :latest_run
    """)

    previous_run = pd.read_sql(
        query_previous_run,
        engine,
        params={
            "symbol": symbol,
            "latest_run": latest_run
        }
    )["previous_run"].iloc[0]

    if pd.isna(previous_run):
        print(f"[WARNING] No previous run_ts found for {symbol}. Daily comparison unavailable.")
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
        WHERE run_ts = :previous_run
          AND symbol = :symbol
        ORDER BY expiration_date, strike, cp
    """)

    df_prev = pd.read_sql(
        query_data,
        engine,
        params={
            "previous_run": previous_run,
            "symbol": symbol
        }
    )

    return df_prev, previous_run


# =========================
# BASIC HELPERS
# =========================

def normalize_dataframe(df):
    df = df.copy()

    numeric_cols = [
        "underlying_px", "strike", "last", "bid", "ask",
        "volume", "oi", "iv", "delta", "gamma", "option_net"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["volume"] = df["volume"].fillna(0)
    df["oi"] = df["oi"].fillna(0)
    df["gamma"] = df["gamma"].fillna(0)

    df["cp"] = df["cp"].astype(str).str.upper().str.strip()

    return df


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


def days_to_expiration(expiration_date):
    if expiration_date is None or pd.isna(expiration_date):
        return None

    exp_date = pd.to_datetime(expiration_date).date()
    today = datetime.utcnow().date()

    return max((exp_date - today).days, 1)


# =========================
# IV / EXPECTED MOVE
# =========================

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


def calculate_atm_iv_for_expiration(df, spot, expiration_date):
    if spot is None:
        return None

    tmp = df[df["expiration_date"] == expiration_date].copy()
    tmp = tmp[tmp["iv"].notna()]
    tmp = tmp[tmp["iv"] > 0]

    if tmp.empty:
        return None

    tmp["distance_to_spot"] = (tmp["strike"] - spot).abs()
    atm_rows = tmp.sort_values("distance_to_spot").head(6)

    if atm_rows.empty:
        return None

    return float(atm_rows["iv"].mean())


def calculate_expected_move_for_expiration(spot, atm_iv, expiration_date):
    if spot is None or atm_iv is None:
        return None, None, None, None

    dte = days_to_expiration(expiration_date)

    if dte is None:
        return None, None, None, None

    one_sigma_move = spot * atm_iv * ((dte / 365) ** 0.5)

    one_sigma_low = spot - one_sigma_move
    one_sigma_high = spot + one_sigma_move

    return dte, one_sigma_move, one_sigma_low, one_sigma_high


def calculate_expected_move(df, spot, atm_iv):
    if spot is None or atm_iv is None:
        return None, None, None, None

    expirations = sorted(df["expiration_date"].dropna().unique())

    if not expirations:
        return None, None, None, None

    nearest_exp = expirations[0]

    return calculate_expected_move_for_expiration(
        spot=spot,
        atm_iv=atm_iv,
        expiration_date=nearest_exp
    )


def calculate_term_structure(df, spot, max_expirations=8):
    expirations = sorted(df["expiration_date"].dropna().unique())

    rows = []

    for exp in expirations[:max_expirations]:
        atm_iv = calculate_atm_iv_for_expiration(df, spot, exp)
        dte, move, low, high = calculate_expected_move_for_expiration(
            spot=spot,
            atm_iv=atm_iv,
            expiration_date=exp
        )

        exp_df = df[df["expiration_date"] == exp]
        call_volume = exp_df[exp_df["cp"] == "C"]["volume"].sum()
        put_volume = exp_df[exp_df["cp"] == "P"]["volume"].sum()
        call_oi = exp_df[exp_df["cp"] == "C"]["oi"].sum()
        put_oi = exp_df[exp_df["cp"] == "P"]["oi"].sum()

        rows.append({
            "expiration_date": exp,
            "days_to_exp": dte,
            "atm_iv": atm_iv,
            "expected_move": move,
            "one_sigma_low": low,
            "one_sigma_high": high,
            "put_call_volume_ratio": safe_divide(put_volume, call_volume),
            "put_call_oi_ratio": safe_divide(put_oi, call_oi),
        })

    return rows


# =========================
# GAMMA
# =========================

def calculate_gamma_profile(df):
    """
    Approximate gamma exposure.

    GEX ≈ gamma * OI * 100 * S² * 0.01

    Puts are treated as negative gamma contribution for positioning view.
    """
    tmp = df.copy()
    spot = get_underlying_price(tmp)

    if spot is None:
        return {
            "total_gex": None,
            "max_positive_gex_strike": None,
            "max_negative_gex_strike": None,
            "gamma_flip": None,
            "distance_to_gamma_flip": None,
            "gex_by_strike": pd.DataFrame(),
        }

    tmp["gamma_exposure"] = tmp["gamma"] * tmp["oi"] * CONTRACT_MULTIPLIER * (spot ** 2) * 0.01
    tmp.loc[tmp["cp"] == "P", "gamma_exposure"] *= -1

    gex_by_strike = (
        tmp.groupby("strike", as_index=False)["gamma_exposure"]
        .sum()
        .sort_values("strike")
        .reset_index(drop=True)
    )

    if gex_by_strike.empty:
        return {
            "total_gex": None,
            "max_positive_gex_strike": None,
            "max_negative_gex_strike": None,
            "gamma_flip": None,
            "distance_to_gamma_flip": None,
            "gex_by_strike": gex_by_strike,
        }

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

    gex_by_strike["cumulative_gex"] = gex_by_strike["gamma_exposure"].cumsum()

    gamma_flip = None

    for i in range(1, len(gex_by_strike)):
        prev_gex = gex_by_strike.loc[i - 1, "cumulative_gex"]
        curr_gex = gex_by_strike.loc[i, "cumulative_gex"]

        if prev_gex == 0:
            gamma_flip = float(gex_by_strike.loc[i - 1, "strike"])
            break

        if (prev_gex < 0 and curr_gex > 0) or (prev_gex > 0 and curr_gex < 0):
            prev_strike = float(gex_by_strike.loc[i - 1, "strike"])
            curr_strike = float(gex_by_strike.loc[i, "strike"])

            denominator = abs(prev_gex) + abs(curr_gex)

            if denominator == 0:
                gamma_flip = curr_strike
            else:
                weight = abs(prev_gex) / denominator
                gamma_flip = prev_strike + weight * (curr_strike - prev_strike)

            break

    distance_to_gamma_flip = None

    if gamma_flip is not None and spot is not None:
        distance_to_gamma_flip = spot - gamma_flip

    return {
        "total_gex": total_gex,
        "max_positive_gex_strike": max_positive_gex_strike,
        "max_negative_gex_strike": max_negative_gex_strike,
        "gamma_flip": gamma_flip,
        "distance_to_gamma_flip": distance_to_gamma_flip,
        "gex_by_strike": gex_by_strike,
    }


# =========================
# TOP STRIKES / OI CHANGE
# =========================

def get_top_strikes(df, cp_value, metric, top_n=5):
    tmp = df[df["cp"] == cp_value].copy()

    if tmp.empty or metric not in tmp.columns:
        return []

    grouped = (
        tmp.groupby("strike", as_index=False)[metric]
        .sum()
        .sort_values(metric, ascending=False)
        .head(top_n)
    )

    return [
        {
            "strike": row["strike"],
            "value": row[metric]
        }
        for _, row in grouped.iterrows()
    ]


def calculate_oi_change_by_strike(current_df, previous_df, cp_value, top_n=5):
    current = current_df[current_df["cp"] == cp_value].copy()
    previous = previous_df[previous_df["cp"] == cp_value].copy()

    if current.empty or previous.empty:
        return {
            "top_increases": [],
            "top_decreases": []
        }

    current_oi = (
        current.groupby("strike", as_index=False)["oi"]
        .sum()
        .rename(columns={"oi": "current_oi"})
    )

    previous_oi = (
        previous.groupby("strike", as_index=False)["oi"]
        .sum()
        .rename(columns={"oi": "previous_oi"})
    )

    merged = current_oi.merge(
        previous_oi,
        on="strike",
        how="outer"
    )

    merged["current_oi"] = merged["current_oi"].fillna(0)
    merged["previous_oi"] = merged["previous_oi"].fillna(0)
    merged["oi_change"] = merged["current_oi"] - merged["previous_oi"]

    increases = (
        merged[merged["oi_change"] > 0]
        .sort_values("oi_change", ascending=False)
        .head(top_n)
    )

    decreases = (
        merged[merged["oi_change"] < 0]
        .sort_values("oi_change", ascending=True)
        .head(top_n)
    )

    top_increases = [
        {
            "strike": row["strike"],
            "change": row["oi_change"],
            "current_oi": row["current_oi"],
            "previous_oi": row["previous_oi"],
        }
        for _, row in increases.iterrows()
    ]

    top_decreases = [
        {
            "strike": row["strike"],
            "change": row["oi_change"],
            "current_oi": row["current_oi"],
            "previous_oi": row["previous_oi"],
        }
        for _, row in decreases.iterrows()
    ]

    return {
        "top_increases": top_increases,
        "top_decreases": top_decreases
    }


# =========================
# METRICS
# =========================

def calculate_metrics_for_symbol(df, latest_run, symbol):
    df = normalize_dataframe(df)

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

    gamma_profile = calculate_gamma_profile(df)

    top_call_oi = get_top_strikes(df, "C", "oi", top_n=5)
    top_put_oi = get_top_strikes(df, "P", "oi", top_n=5)
    top_call_volume = get_top_strikes(df, "C", "volume", top_n=5)
    top_put_volume = get_top_strikes(df, "P", "volume", top_n=5)

    term_structure = calculate_term_structure(df, spot, max_expirations=8)

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
        "total_gex": gamma_profile["total_gex"],
        "max_positive_gex_strike": gamma_profile["max_positive_gex_strike"],
        "max_negative_gex_strike": gamma_profile["max_negative_gex_strike"],
        "gamma_flip": gamma_profile["gamma_flip"],
        "distance_to_gamma_flip": gamma_profile["distance_to_gamma_flip"],
        "top_call_oi": top_call_oi,
        "top_put_oi": top_put_oi,
        "top_call_volume": top_call_volume,
        "top_put_volume": top_put_volume,
        "term_structure": term_structure,
    }

    return metrics


def calculate_metric_change(current, previous, key):
    current_value = current.get(key)
    previous_value = previous.get(key)

    if current_value is None or previous_value is None:
        return None

    if pd.isna(current_value) or pd.isna(previous_value):
        return None

    return current_value - previous_value


def enrich_metrics_with_previous(current_metrics, previous_metrics):
    if previous_metrics is None:
        current_metrics["has_previous"] = False
        return current_metrics

    current_metrics["has_previous"] = True
    current_metrics["previous_run"] = previous_metrics.get("latest_run")

    comparison_keys = [
        "spot",
        "atm_iv",
        "call_volume",
        "put_volume",
        "put_call_volume_ratio",
        "call_oi",
        "put_oi",
        "put_call_oi_ratio",
        "total_gex",
        "gamma_flip",
        "distance_to_gamma_flip",
    ]

    for key in comparison_keys:
        current_metrics[f"{key}_change"] = calculate_metric_change(
            current_metrics,
            previous_metrics,
            key
        )

    return current_metrics


def enrich_metrics_with_oi_change(current_metrics, current_df, previous_df):
    if previous_df is None or previous_df.empty:
        current_metrics["has_oi_change"] = False
        return current_metrics

    current_df = normalize_dataframe(current_df)
    previous_df = normalize_dataframe(previous_df)

    call_oi_change = calculate_oi_change_by_strike(
        current_df=current_df,
        previous_df=previous_df,
        cp_value="C",
        top_n=5
    )

    put_oi_change = calculate_oi_change_by_strike(
        current_df=current_df,
        previous_df=previous_df,
        cp_value="P",
        top_n=5
    )

    current_metrics["has_oi_change"] = True
    current_metrics["call_oi_top_increases"] = call_oi_change["top_increases"]
    current_metrics["call_oi_top_decreases"] = call_oi_change["top_decreases"]
    current_metrics["put_oi_top_increases"] = put_oi_change["top_increases"]
    current_metrics["put_oi_top_decreases"] = put_oi_change["top_decreases"]

    return current_metrics


def collect_all_metrics(engine, symbols):
    all_metrics = []

    for symbol in symbols:
        print(f"Loading latest snapshot for {symbol}...")

        df, latest_run = load_latest_snapshot_for_symbol(engine, symbol)

        if df.empty or latest_run is None:
            continue

        current_metrics = calculate_metrics_for_symbol(df, latest_run, symbol)

        print(f"Loading previous snapshot for {symbol}...")

        df_prev, previous_run = load_previous_snapshot_for_symbol(
            engine,
            symbol,
            latest_run
        )

        if df_prev.empty or previous_run is None:
            current_metrics = enrich_metrics_with_previous(
                current_metrics,
                previous_metrics=None
            )

            current_metrics = enrich_metrics_with_oi_change(
                current_metrics,
                current_df=df,
                previous_df=None
            )
        else:
            previous_metrics = calculate_metrics_for_symbol(
                df_prev,
                previous_run,
                symbol
            )

            current_metrics = enrich_metrics_with_previous(
                current_metrics,
                previous_metrics
            )

            current_metrics = enrich_metrics_with_oi_change(
                current_metrics,
                current_df=df,
                previous_df=df_prev
            )

        all_metrics.append(current_metrics)

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


def fmt_change(value):
    if value is None or pd.isna(value):
        return "N/A"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:,.0f}"


def fmt_price_change(value):
    if value is None or pd.isna(value):
        return "N/A"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:,.2f}"


def fmt_decimal_change(value):
    if value is None or pd.isna(value):
        return "N/A"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.2f}"


def fmt_percent_point_change(value):
    if value is None or pd.isna(value):
        return "N/A"
    sign = "+" if value > 0 else ""
    return f"{sign}{value * 100:.2f} pp"


def fmt_gex_change(value):
    if value is None or pd.isna(value):
        return "N/A"

    sign = "+" if value > 0 else ""
    abs_value = abs(value)

    if abs_value >= 1_000_000_000:
        return f"{sign}{value / 1_000_000_000:.2f}B"
    if abs_value >= 1_000_000:
        return f"{sign}{value / 1_000_000:.2f}M"

    return f"{sign}{value:,.0f}"


def fmt_date(value):
    if value is None or pd.isna(value):
        return "N/A"
    return str(pd.to_datetime(value).date())


# =========================
# EXECUTIVE SUMMARY / ALERTS
# =========================

def classify_market_regime(metrics):
    total_gex = metrics.get("total_gex")
    atm_iv_change = metrics.get("atm_iv_change")
    pc_volume = metrics.get("put_call_volume_ratio")

    if total_gex is not None and total_gex > 0:
        gamma_regime = "positive gamma"
    elif total_gex is not None and total_gex < 0:
        gamma_regime = "negative gamma"
    else:
        gamma_regime = "unclear gamma"

    if atm_iv_change is not None and not pd.isna(atm_iv_change):
        if atm_iv_change > 0.01:
            vol_regime = "volatility expansion"
        elif atm_iv_change < -0.01:
            vol_regime = "volatility compression"
        else:
            vol_regime = "stable implied volatility"
    else:
        vol_regime = "volatility trend unavailable"

    if pc_volume is not None:
        if pc_volume > 1.3:
            flow_regime = "put-heavy flow"
        elif pc_volume < 0.75:
            flow_regime = "call-heavy flow"
        else:
            flow_regime = "balanced flow"
    else:
        flow_regime = "flow unavailable"

    return f"{gamma_regime}, {vol_regime}, {flow_regime}"


def generate_symbol_alerts(metrics):
    alerts = []
    symbol = metrics["symbol"]

    pc_volume = metrics.get("put_call_volume_ratio")
    pc_oi = metrics.get("put_call_oi_ratio")
    atm_iv_change = metrics.get("atm_iv_change")
    total_gex = metrics.get("total_gex")
    total_gex_change = metrics.get("total_gex_change")
    gamma_flip = metrics.get("gamma_flip")
    distance_to_gamma_flip = metrics.get("distance_to_gamma_flip")
    spot = metrics.get("spot")

    if pc_volume is not None and pc_volume >= 1.5:
        alerts.append(f"{symbol}: Put/Call Volume Ratio is elevated at {fmt_decimal(pc_volume)}.")

    if pc_volume is not None and pc_volume <= 0.6:
        alerts.append(f"{symbol}: Call volume strongly dominates, with Put/Call Volume Ratio at {fmt_decimal(pc_volume)}.")

    if pc_oi is not None and pc_oi >= 1.5:
        alerts.append(f"{symbol}: Put open interest is meaningfully higher than call open interest. Put/Call OI Ratio: {fmt_decimal(pc_oi)}.")

    if atm_iv_change is not None and not pd.isna(atm_iv_change):
        if atm_iv_change >= 0.02:
            alerts.append(f"{symbol}: ATM IV increased by {fmt_percent_point_change(atm_iv_change)}.")
        elif atm_iv_change <= -0.02:
            alerts.append(f"{symbol}: ATM IV decreased by {fmt_percent_point_change(atm_iv_change)}.")

    if total_gex is not None and total_gex < 0:
        alerts.append(f"{symbol}: Net GEX is negative at {fmt_gex(total_gex)}.")

    if total_gex_change is not None and not pd.isna(total_gex_change):
        if abs(total_gex_change) >= 1_000_000_000:
            alerts.append(f"{symbol}: Net GEX changed significantly by {fmt_gex_change(total_gex_change)}.")

    if gamma_flip is not None and distance_to_gamma_flip is not None and spot is not None:
        distance_pct = abs(distance_to_gamma_flip) / spot

        if distance_pct <= 0.005:
            alerts.append(
                f"{symbol}: Price is close to gamma flip. Spot {fmt_price(spot)}, gamma flip {fmt_price(gamma_flip)}."
            )

    if metrics.get("has_oi_change"):
        call_inc = metrics.get("call_oi_top_increases") or []
        put_inc = metrics.get("put_oi_top_increases") or []

        if call_inc:
            top_call = call_inc[0]
            if abs(top_call.get("change", 0)) >= 10_000:
                alerts.append(
                    f"{symbol}: Large Call OI increase at strike {fmt_number(top_call.get('strike'))}: "
                    f"{fmt_change(top_call.get('change'))} contracts."
                )

        if put_inc:
            top_put = put_inc[0]
            if abs(top_put.get("change", 0)) >= 10_000:
                alerts.append(
                    f"{symbol}: Large Put OI increase at strike {fmt_number(top_put.get('strike'))}: "
                    f"{fmt_change(top_put.get('change'))} contracts."
                )

    return alerts


def generate_all_alerts(all_metrics):
    alerts = []

    for metrics in all_metrics:
        alerts.extend(generate_symbol_alerts(metrics))

    if not alerts:
        alerts.append("No major alerts triggered based on the current rule set.")

    return alerts


def build_symbol_summary_sentence(metrics):
    symbol = metrics["symbol"]
    spot = metrics.get("spot")
    atm_iv = metrics.get("atm_iv")
    total_gex = metrics.get("total_gex")
    gamma_flip = metrics.get("gamma_flip")
    pc_volume = metrics.get("put_call_volume_ratio")
    regime = classify_market_regime(metrics)

    sentence = (
        f"{symbol}: spot {fmt_price(spot)}, ATM IV {fmt_percent(atm_iv)}, "
        f"Put/Call Volume Ratio {fmt_decimal(pc_volume)}, Net GEX {fmt_gex(total_gex)}, "
        f"Gamma Flip {fmt_price(gamma_flip)}. Regime: {regime}."
    )

    return sentence


def build_executive_summary(all_metrics):
    sentences = [build_symbol_summary_sentence(m) for m in all_metrics]
    alerts = generate_all_alerts(all_metrics)

    priority_alerts = alerts[:5]

    alert_html = "".join([f"<li>{alert}</li>" for alert in priority_alerts])
    summary_html = "".join([f"<li>{sentence}</li>" for sentence in sentences])

    return f"""
    <h2>Executive Summary</h2>

    <ul>
        {summary_html}
    </ul>

    <h3>Top Alerts</h3>

    <ul>
        {alert_html}
    </ul>
    """


# =========================
# INTERPRETATION
# =========================

def build_interpretation(metrics):
    comments = []

    symbol = metrics["symbol"]
    pc_volume = metrics["put_call_volume_ratio"]
    pc_oi = metrics["put_call_oi_ratio"]
    total_gex = metrics["total_gex"]
    gamma_flip = metrics.get("gamma_flip")
    distance_to_gamma_flip = metrics.get("distance_to_gamma_flip")

    comments.append(f"Market regime: {classify_market_regime(metrics)}.")

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
            comments.append("Total put open interest is higher than call open interest.")
        else:
            comments.append("Total call open interest is higher than put open interest.")

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

    if gamma_flip is not None:
        if distance_to_gamma_flip is not None:
            if distance_to_gamma_flip > 0:
                comments.append(
                    f"Spot is above the estimated gamma flip by {fmt_price(distance_to_gamma_flip)} points."
                )
            elif distance_to_gamma_flip < 0:
                comments.append(
                    f"Spot is below the estimated gamma flip by {fmt_price(abs(distance_to_gamma_flip))} points."
                )
            else:
                comments.append("Spot is currently at the estimated gamma flip level.")

    if metrics.get("has_previous"):
        spot_change = metrics.get("spot_change")
        atm_iv_change = metrics.get("atm_iv_change")
        total_gex_change = metrics.get("total_gex_change")
        gamma_flip_change = metrics.get("gamma_flip_change")

        if spot_change is not None and not pd.isna(spot_change):
            if spot_change > 0:
                comments.append(
                    f"The underlying price increased by {fmt_price_change(spot_change)} versus the previous snapshot."
                )
            elif spot_change < 0:
                comments.append(
                    f"The underlying price decreased by {fmt_price_change(spot_change)} versus the previous snapshot."
                )

        if atm_iv_change is not None and not pd.isna(atm_iv_change):
            if atm_iv_change > 0:
                comments.append(
                    f"ATM IV increased by {fmt_percent_point_change(atm_iv_change)}, suggesting higher implied uncertainty."
                )
            elif atm_iv_change < 0:
                comments.append(
                    f"ATM IV decreased by {fmt_percent_point_change(atm_iv_change)}, suggesting lower implied uncertainty."
                )

        if total_gex_change is not None and not pd.isna(total_gex_change):
            if total_gex_change > 0:
                comments.append(
                    f"Net GEX increased by {fmt_gex_change(total_gex_change)}, pointing to stronger positive gamma positioning."
                )
            elif total_gex_change < 0:
                comments.append(
                    f"Net GEX decreased by {fmt_gex_change(total_gex_change)}, pointing to weaker or more negative gamma positioning."
                )

        if gamma_flip_change is not None and not pd.isna(gamma_flip_change):
            comments.append(
                f"The estimated gamma flip changed by {fmt_price_change(gamma_flip_change)} points."
            )

    if metrics.get("has_oi_change"):
        call_increases = metrics.get("call_oi_top_increases") or []
        put_increases = metrics.get("put_oi_top_increases") or []

        if call_increases:
            top_call = call_increases[0]
            comments.append(
                f"The largest new call OI increase appeared around strike "
                f"{fmt_number(top_call.get('strike'))}, with "
                f"{fmt_change(top_call.get('change'))} contracts."
            )

        if put_increases:
            top_put = put_increases[0]
            comments.append(
                f"The largest new put OI increase appeared around strike "
                f"{fmt_number(top_put.get('strike'))}, with "
                f"{fmt_change(top_put.get('change'))} contracts."
            )

    if not comments:
        return "No interpretation available due to missing metrics."

    return " ".join(comments)


# =========================
# HTML REPORT SECTIONS
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
            <td>{fmt_price(m["gamma_flip"])}</td>
            <td>{fmt_price_change(m.get("distance_to_gamma_flip"))}</td>
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
            <th>Gamma Flip</th>
            <th>Distance to Flip</th>
            <th>Max Call OI Strike</th>
            <th>Max Put OI Strike</th>
        </tr>
        {rows}
    </table>
    """


def build_alerts_section(all_metrics):
    alerts = generate_all_alerts(all_metrics)
    rows = "".join([f"<li>{alert}</li>" for alert in alerts])

    return f"""
    <h2>Alerts</h2>
    <ul>
        {rows}
    </ul>
    """


def build_daily_change_section(metrics):
    if not metrics.get("has_previous"):
        return """
        <h3>Daily Change</h3>
        <p>No previous snapshot available for comparison.</p>
        """

    return f"""
    <h3>Daily Change vs Previous Snapshot</h3>

    <p>
        <strong>Previous snapshot:</strong> {fmt_datetime(metrics.get("previous_run"))}
    </p>

    <table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse;">
        <tr style="background-color: #f2f2f2;">
            <th>Metric</th>
            <th>Change</th>
        </tr>
        <tr>
            <td>Underlying Price</td>
            <td>{fmt_price_change(metrics.get("spot_change"))}</td>
        </tr>
        <tr>
            <td>ATM IV</td>
            <td>{fmt_percent_point_change(metrics.get("atm_iv_change"))}</td>
        </tr>
        <tr>
            <td>Call Volume</td>
            <td>{fmt_change(metrics.get("call_volume_change"))}</td>
        </tr>
        <tr>
            <td>Put Volume</td>
            <td>{fmt_change(metrics.get("put_volume_change"))}</td>
        </tr>
        <tr>
            <td>Put/Call Volume Ratio</td>
            <td>{fmt_decimal_change(metrics.get("put_call_volume_ratio_change"))}</td>
        </tr>
        <tr>
            <td>Call Open Interest</td>
            <td>{fmt_change(metrics.get("call_oi_change"))}</td>
        </tr>
        <tr>
            <td>Put Open Interest</td>
            <td>{fmt_change(metrics.get("put_oi_change"))}</td>
        </tr>
        <tr>
            <td>Put/Call OI Ratio</td>
            <td>{fmt_decimal_change(metrics.get("put_call_oi_ratio_change"))}</td>
        </tr>
        <tr>
            <td>Net Gamma Exposure</td>
            <td>{fmt_gex_change(metrics.get("total_gex_change"))}</td>
        </tr>
        <tr>
            <td>Gamma Flip</td>
            <td>{fmt_price_change(metrics.get("gamma_flip_change"))}</td>
        </tr>
    </table>
    """


def build_term_structure_section(metrics):
    rows = ""

    for item in metrics.get("term_structure", []):
        rows += f"""
        <tr>
            <td>{fmt_date(item.get("expiration_date"))}</td>
            <td>{fmt_number(item.get("days_to_exp"))}</td>
            <td>{fmt_percent(item.get("atm_iv"))}</td>
            <td>±{fmt_price(item.get("expected_move"))}</td>
            <td>{fmt_price(item.get("one_sigma_low"))} – {fmt_price(item.get("one_sigma_high"))}</td>
            <td>{fmt_decimal(item.get("put_call_volume_ratio"))}</td>
            <td>{fmt_decimal(item.get("put_call_oi_ratio"))}</td>
        </tr>
        """

    if not rows:
        return """
        <h3>Term Structure</h3>
        <p>No term structure data available.</p>
        """

    return f"""
    <h3>Term Structure</h3>

    <table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse;">
        <tr style="background-color: #f2f2f2;">
            <th>Expiration</th>
            <th>DTE</th>
            <th>ATM IV</th>
            <th>1σ Move</th>
            <th>1σ Range</th>
            <th>Put/Call Vol</th>
            <th>Put/Call OI</th>
        </tr>
        {rows}
    </table>
    """


def build_top_strikes_table(title, rows):
    if not rows:
        return f"""
        <h4>{title}</h4>
        <p>No data available.</p>
        """

    html_rows = ""

    for item in rows:
        html_rows += f"""
        <tr>
            <td>{fmt_number(item.get("strike"))}</td>
            <td>{fmt_number(item.get("value"))}</td>
        </tr>
        """

    return f"""
    <h4>{title}</h4>

    <table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; margin-bottom: 16px;">
        <tr style="background-color: #f2f2f2;">
            <th>Strike</th>
            <th>Value</th>
        </tr>
        {html_rows}
    </table>
    """


def build_top_strikes_section(metrics):
    return f"""
    <h3>Top 5 Strikes</h3>

    <table width="100%" cellpadding="0" cellspacing="0">
        <tr>
            <td valign="top" width="50%">
                {build_top_strikes_table("Top 5 Call OI", metrics.get("top_call_oi"))}
            </td>
            <td valign="top" width="50%">
                {build_top_strikes_table("Top 5 Put OI", metrics.get("top_put_oi"))}
            </td>
        </tr>
        <tr>
            <td valign="top" width="50%">
                {build_top_strikes_table("Top 5 Call Volume", metrics.get("top_call_volume"))}
            </td>
            <td valign="top" width="50%">
                {build_top_strikes_table("Top 5 Put Volume", metrics.get("top_put_volume"))}
            </td>
        </tr>
    </table>
    """


def build_oi_change_table(title, rows):
    if not rows:
        return f"""
        <h4>{title}</h4>
        <p>No relevant OI change detected.</p>
        """

    html_rows = ""

    for item in rows:
        html_rows += f"""
        <tr>
            <td>{fmt_number(item.get("strike"))}</td>
            <td>{fmt_change(item.get("change"))}</td>
            <td>{fmt_number(item.get("previous_oi"))}</td>
            <td>{fmt_number(item.get("current_oi"))}</td>
        </tr>
        """

    return f"""
    <h4>{title}</h4>

    <table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; margin-bottom: 16px;">
        <tr style="background-color: #f2f2f2;">
            <th>Strike</th>
            <th>OI Change</th>
            <th>Previous OI</th>
            <th>Current OI</th>
        </tr>
        {html_rows}
    </table>
    """


def build_oi_change_section(metrics):
    if not metrics.get("has_oi_change"):
        return """
        <h3>OI Change by Strike</h3>
        <p>No previous snapshot available for OI change analysis.</p>
        """

    return f"""
    <h3>OI Change by Strike</h3>

    <table width="100%" cellpadding="0" cellspacing="0">
        <tr>
            <td valign="top" width="50%">
                {build_oi_change_table("Top Call OI Increases", metrics.get("call_oi_top_increases"))}
            </td>
            <td valign="top" width="50%">
                {build_oi_change_table("Top Put OI Increases", metrics.get("put_oi_top_increases"))}
            </td>
        </tr>
        <tr>
            <td valign="top" width="50%">
                {build_oi_change_table("Top Call OI Decreases", metrics.get("call_oi_top_decreases"))}
            </td>
            <td valign="top" width="50%">
                {build_oi_change_table("Top Put OI Decreases", metrics.get("put_oi_top_decreases"))}
            </td>
        </tr>
    </table>
    """


def build_symbol_section(metrics):
    interpretation = build_interpretation(metrics)
    daily_change_section = build_daily_change_section(metrics)
    term_structure_section = build_term_structure_section(metrics)
    top_strikes_section = build_top_strikes_section(metrics)
    oi_change_section = build_oi_change_section(metrics)

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

    {daily_change_section}

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

    {term_structure_section}

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

    {top_strikes_section}

    {oi_change_section}

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
            <td>Gamma Flip Level</td>
            <td>{fmt_price(metrics["gamma_flip"])}</td>
        </tr>
        <tr>
            <td>Distance to Gamma Flip</td>
            <td>{fmt_price_change(metrics["distance_to_gamma_flip"])}</td>
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
        <h1>Daily Options Intelligence Report — {today}</h1>

        <p>
            Symbols analysed: <strong>{", ".join([m["symbol"] for m in all_metrics])}</strong>
        </p>

        {build_executive_summary(all_metrics)}

        {build_summary_table(all_metrics)}

        {build_alerts_section(all_metrics)}

        {sections}

        <hr style="margin-top: 30px;">

        <h3>Methodology Notes</h3>

        <p style="font-size: 12px; color: #666;">
            Gamma Exposure is an approximation based on available option gamma, open interest, contract multiplier and underlying price.
            Puts are treated as negative gamma contribution for positioning analysis.
            Gamma Flip is estimated from the cumulative gamma exposure by strike and should be interpreted as an approximate positioning level.
            Expected move uses ATM IV and the square-root-of-time approximation.
        </p>

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
    if not TABLE_NAME:
        raise ValueError("OPTIONS_TABLE is empty. Set it to 'options_chain'.")

    if not SYMBOLS:
        raise ValueError("No symbols configured. Set OPTIONS_SYMBOLS, for example: SPX,NDX,VIX")

    print(f"Using table: {TABLE_NAME}")
    print(f"Configured symbols: {SYMBOLS}")

    engine = get_engine()

    all_metrics = collect_all_metrics(engine, SYMBOLS)

    html = build_html_report(all_metrics)

    today = datetime.now().strftime("%Y-%m-%d")
    subject_symbols = ", ".join([m["symbol"] for m in all_metrics])
    subject = f"Daily Options Intelligence Report — {subject_symbols} — {today}"

    send_email(subject, html)

    print("Daily options intelligence report sent successfully.")
    print(f"Symbols included: {subject_symbols}")


if __name__ == "__main__":
    main()