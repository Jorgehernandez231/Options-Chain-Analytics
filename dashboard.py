# dashboard.py

import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psycopg2
import streamlit as st
from datetime import date, datetime, timezone, timedelta
from statsmodels.nonparametric.smoothers_lowess import lowess
from urllib.parse import quote, unquote
from sqlalchemy import create_engine
from plotly.subplots import make_subplots



# ---------- Config ----------
PX_TEMPLATE = "plotly_dark"
OPTIONS_TABLE = "options_chain"

UNDERLYINGS = {
    "SPX": {
        "label": "S&P 500 Index Options",
        "table_symbol": "SPX",
        "contract_multiplier": 100.0,
        "asset_class": "equity_index",
    },
    "NDX": {
        "label": "Nasdaq-100 Index Options",
        "table_symbol": "NDX",
        "contract_multiplier": 100.0,
        "asset_class": "equity_index",
    },
    "VIX": {
        "label": "Cboe Volatility Index Options",
        "table_symbol": "VIX",
        "contract_multiplier": 100.0,
        "asset_class": "volatility_index",
    },
}

# ---------- Page / Theme ----------
st.set_page_config(page_title="Options Terminal", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<style>div.block-container{padding-top:1.0rem;padding-bottom:0.5rem;}</style>", unsafe_allow_html=True)

# ---------- Underlying selector ----------
with st.sidebar:
    st.subheader("Underlying")

    selected_symbol = st.selectbox(
        "Select product",
        list(UNDERLYINGS.keys()),
        index=0,
        format_func=lambda x: f"{x} — {UNDERLYINGS[x]['label']}",
    )

symbol_config = UNDERLYINGS[selected_symbol]
symbol = symbol_config["table_symbol"]
CONTRACT_MULT = symbol_config["contract_multiplier"]

# ---------- Page title ----------
st.title(f"{selected_symbol} Terminal • Options")
st.caption(symbol_config["label"])

if selected_symbol == "VIX":
    st.info(
        "VIX options are volatility-index options. Interpret gamma, probable levels, "
        "and skew differently from SPX/NDX equity-index options."
    )

# ---------- DB using SQLAlchemy ----------

@st.cache_resource
def get_engine():
    user = st.secrets["DB_USER"]
    pwd  = st.secrets["DB_PASS"]
    host = st.secrets["DB_HOST"]
    db   = st.secrets["DB_NAME"]

    url = f"postgresql+psycopg2://{user}:{pwd}@{host}:5432/{db}"
    return create_engine(url, pool_pre_ping=True)

@st.cache_data(ttl=180)
def q(sql, params=None):
    engine = get_engine()
    return pd.read_sql(sql, engine, params=params)


# ---------- URL state (permalink) ----------
def _to_bool(v, default=False):
    if v is None: return default
    s = str(v).lower()
    return s in ("1","true","t","yes","y","on")

def _to_float(v, default=None):
    try: return float(v)
    except: return default

def _to_list(v):
    if v is None: return []
    if isinstance(v, list): return v
    return [v]

def get_url_state():
    try:
        qp = dict(st.query_params)        # Streamlit ≥1.30
    except Exception:
        qp = st.experimental_get_query_params()  # older
    s = {}
    s["use_hist"]   = _to_bool(qp.get("hist", ["1"])[0], True)
    s["run_date"]   = qp.get("d", [None])[0]
    s["spot"]       = _to_float(qp.get("S", [None])[0], None)
    s["r"]          = _to_float(qp.get("r", [None])[0], None)
    s["calls"]      = _to_bool(qp.get("C", ["1"])[0], True)
    s["puts"]       = _to_bool(qp.get("P", ["1"])[0], True)
    s["exp"]        = qp.get("exp", [None])[0]
    s["k_low"]      = _to_float(qp.get("kL", [None])[0], None)
    s["k_high"]     = _to_float(qp.get("kH", [None])[0], None)
    s["ovr_x"]      = qp.get("ovx", ["strike"])[0]
    s["ovr_avgcp"]  = _to_bool(qp.get("ovavg", ["1"])[0], True)
    s["ovr_mode"]   = qp.get("ovm", ["raw"])[0]
    s["ovr_frac"]   = _to_float(qp.get("ovf", [0.15])[0], 0.15)
    s["ovr_exps"]   = [unquote(x) for x in _to_list(qp.get("ovex"))]
    return s

def set_url_state(state: dict):
    params = {
        "hist": "1" if state.get("use_hist") else "0",
        "d": state.get("run_date") or "",
        "S": f'{state.get("spot"):.4f}' if state.get("spot") is not None else "",
        "r": f'{state.get("r"):.6f}' if state.get("r") is not None else "",
        "C": "1" if state.get("calls", True) else "0",
        "P": "1" if state.get("puts", True) else "0",
        "exp": state.get("exp") or "",
        "kL": state.get("k_low") if state.get("k_low") is not None else "",
        "kH": state.get("k_high") if state.get("k_high") is not None else "",
        "ovx": state.get("ovr_x") or "strike",
        "ovavg": "1" if state.get("ovr_avgcp", True) else "0",
        "ovm": state.get("ovr_mode") or "raw",
        "ovf": state.get("ovr_frac", 0.15),
    }
    ovex = [quote(x) for x in state.get("ovr_exps", [])]
    try:
        st.query_params.clear()
        st.query_params.update(**{k:v for k,v in params.items() if v != ""}, ovex=ovex)
    except Exception:
        st.experimental_set_query_params(**{k:v for k,v in params.items() if v != ""}, ovex=ovex)

URL_DEFAULTS = get_url_state()


# ---------- History helpers ----------
@st.cache_data(ttl=180, show_spinner=False)
def load_run_dates(symbol):
    """
    Return list of distinct run timestamps (run_ts) for one symbol,
    newest first.
    """
    try:
        df = q(
            f"""
            SELECT DISTINCT run_ts
            FROM {OPTIONS_TABLE}
            WHERE symbol = %s
              AND run_ts IS NOT NULL
            ORDER BY run_ts DESC
            """,
            (symbol,),
        )
        return list(pd.to_datetime(df["run_ts"])) if not df.empty else []
    except Exception:
        return []


def prev_run_date(dates, cur_dt):
    if not dates or cur_dt not in dates:
        return None
    idx = dates.index(cur_dt)
    return dates[idx + 1] if idx + 1 < len(dates) else None


@st.cache_data(ttl=180, show_spinner=False)
def load_chain_by_run(symbol, run_ts):
    return q(
        f"""
        SELECT run_ts,
               symbol,
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
        FROM {OPTIONS_TABLE}
        WHERE symbol = %s
          AND run_ts = %s
        ORDER BY expiration_date, strike, cp
        """,
        (symbol, run_ts),
    )


@st.cache_data(ttl=120, show_spinner=False)
def load_latest(symbol):
    return q(
        f"""
        SELECT run_ts,
               symbol,
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
        FROM {OPTIONS_TABLE}
        WHERE symbol = %s
          AND run_ts = (
              SELECT MAX(run_ts)
              FROM {OPTIONS_TABLE}
              WHERE symbol = %s
          )
        ORDER BY expiration_date, strike, cp
        """,
        (symbol, symbol),
    )


@st.cache_data(ttl=180, show_spinner=False)
def load_chain_two_days(symbol, run_date_cur, run_date_prev):
    cur = load_chain_by_run(symbol, run_date_cur)
    prv = load_chain_by_run(symbol, run_date_prev) if run_date_prev else pd.DataFrame(columns=cur.columns)
    return cur, prv

# ---------- Math ----------
def yearfrac(d0: date, d1: date) -> float:
    return max((d1 - d0).days, 0) / 365.0

def bs_d1(S, K, r, sigma, T):
    if S<=0 or K<=0 or sigma<=0 or T<=0: return np.nan
    return (np.log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma*np.sqrt(T))

def bs_gamma(S, K, r, sigma, T):
    if S<=0 or K<=0 or sigma<=0 or T<=0: return 0.0
    d1 = bs_d1(S, K, r, sigma, T)
    pdf = 1.0/np.sqrt(2*np.pi) * np.exp(-0.5*d1*d1)
    return pdf / (S*sigma*np.sqrt(T))

def nearest_strike_iv(df, exp, spot):
    dfe = df[df["expiration_date"].eq(exp)]
    if dfe.empty:
        return np.nan
    idx = (dfe["strike"] - spot).abs().idxmin()
    k = dfe.loc[idx, "strike"]
    ivc = dfe[(dfe["strike"]==k)&(dfe["cp"]=="C")]["iv"].dropna().mean()
    ivp = dfe[(dfe["strike"]==k)&(dfe["cp"]=="P")]["iv"].dropna().mean()
    return np.nanmean([ivc, ivp])

# ---------- Helpers ----------
def tab_help(md_text: str):
    """Standard inline help expander for each tab."""
    with st.expander("What am I seeing?", expanded=False):
        st.markdown(md_text)
        
def get_atm_iv_for_expiry(df, exp_sel, spot, moneyness_band=0.03):
    """
    df: chain for one run date (global df already loaded)
    exp_sel: selected expiration_date
    spot: current underlying price used in the app (float)
    Returns: (underlying_price, atm_iv) or (underlying_price, None)
    """
    dfe = df[df["expiration_date"].eq(exp_sel)].copy()
    if dfe.empty:
        return float(spot), None

    u = float(spot)
    dfe["moneyness"] = dfe["strike"] / u
    dfe = dfe[dfe["moneyness"].between(1 - moneyness_band, 1 + moneyness_band)]

    if dfe.empty:
        return u, None

    iv_atm = float(dfe["iv"].median())
    return u, iv_atm

def compute_probable_ranges(underlying, atm_iv, days_to_exp):
    """
    Returns a dict with 1σ and 2σ price ranges.
    """
    if atm_iv is None or atm_iv <= 0 or days_to_exp <= 0:
        return None

    T = max(days_to_exp, 1) / 365.0  # at least 1 day
    sigma_T = atm_iv * np.sqrt(T)

    move_1s = underlying * sigma_T
    move_2s = 2.0 * move_1s

    ranges = {
        "S0": underlying,
        "T_years": T,
        "atm_iv": atm_iv,
        "one_sigma_low": underlying - move_1s,
        "one_sigma_high": underlying + move_1s,
        "two_sigma_low": underlying - move_2s,
        "two_sigma_high": underlying + move_2s,
    }
    return ranges

def top_gravity_levels(df, exp_sel, low, high, spot, top_n=5):
    """
    Find strikes inside [low, high] with the strongest positioning
    (by gamma*OI if gamma exists, otherwise by OI).
    """
    dfe = df[df["expiration_date"].eq(exp_sel)].copy()
    dfe = dfe[dfe["strike"].between(low, high)]

    if dfe.empty:
        return pd.DataFrame(columns=["strike", "position_metric"])

    # If stored gamma exists, use absolute gamma notional; otherwise fall back to OI.
    if "gamma" in dfe.columns and dfe["gamma"].notna().any():
        u = float(spot)
        dfe["gamma"] = pd.to_numeric(dfe["gamma"], errors="coerce").fillna(0.0)
        dfe["oi"] = pd.to_numeric(dfe["oi"], errors="coerce").fillna(0.0)
        dfe["gamma_notional"] = dfe["gamma"].abs() * dfe["oi"] * CONTRACT_MULT * (u * u)
        metric_col = "gamma_notional"
    else:
        metric_col = "oi"

    out = (
        dfe.groupby("strike")[metric_col]
        .sum()
        .reset_index()
        .sort_values(metric_col, ascending=False)
        .head(top_n)
    )
    out = out.rename(columns={metric_col: "position_metric"})
    return out

def compute_oi_change_by_strike(cur_df, prv_df, exp_sel):
    """
    Returns DataFrame: strike, oiC, oiP  (OI changes for calls/puts)
    """
    key_cols = ["expiration_date", "strike", "cp"]

    cur_e = (
        cur_df[cur_df["expiration_date"].eq(exp_sel)][key_cols + ["oi"]]
        .rename(columns={"oi": "oi_cur"})
    )
    prv_e = (
        prv_df[prv_df["expiration_date"].eq(exp_sel)][key_cols + ["oi"]]
        .rename(columns={"oi": "oi_prev"})
    )

    merged = pd.merge(cur_e, prv_e, on=key_cols, how="left")
    merged["oi_cur"]  = pd.to_numeric(merged["oi_cur"], errors="coerce").fillna(0)
    merged["oi_prev"] = pd.to_numeric(merged["oi_prev"], errors="coerce").fillna(0)
    merged["oi_change"] = merged["oi_cur"] - merged["oi_prev"]

    # Aggregate to strike x side
    hm = merged.groupby(["strike", "cp"], as_index=False)["oi_change"].sum()

    # Pivot to columns oiC/oiP
    out = hm.pivot(index="strike", columns="cp", values="oi_change").fillna(0).reset_index()
    out = out.rename(columns={"C": "oiC", "P": "oiP"})
    if "oiC" not in out.columns: out["oiC"] = 0.0
    if "oiP" not in out.columns: out["oiP"] = 0.0
    return out.sort_values("strike")


def robust_zscore(series: pd.Series) -> pd.Series:
    """Robust z-score using MAD; good for spiky OI-change distributions."""
    x = pd.to_numeric(series, errors="coerce").fillna(0.0)
    med = x.median()
    mad = (x - med).abs().median()
    mad = mad if mad > 0 else 1.0
    return (x - med) / (1.4826 * mad)

# ---------- Load initial data ----------
dates = load_run_dates(symbol)

with st.sidebar:
    st.markdown("---")
    st.subheader("History")

    use_hist = st.checkbox(
        "Use history",
        value=URL_DEFAULTS.get("use_hist", bool(dates)),
        help="Keep always checked to select a past snapshot from history.",
    )

    # Default to the most recent snapshot
    default_idx = 0
    url_run = URL_DEFAULTS.get("run_date")  # we store it as string in the URL

    if (use_hist and dates) and url_run is not None:
        for i, dt in enumerate(dates):
            if str(dt) == url_run:
                default_idx = i
                break

    if dates:
        chosen_date = st.selectbox(
            "Run timestamp (UTC)",
            dates,
            index=default_idx,
            key=f"run_ts_{selected_symbol}",
            format_func=lambda x: x.strftime("%Y-%m-%d %H:%M:%S"),
        )
    else:
        st.caption(f"No history snapshots found for {selected_symbol}.")
        chosen_date = None


df = (
    load_chain_by_run(symbol, chosen_date)
    if (use_hist and chosen_date)
    else load_latest(symbol)
)
if df.empty:
    st.warning("No rows returned. Check tables/permissions.")
    st.stop()

# Dtypes
df["expiration_date"] = pd.to_datetime(df["expiration_date"]).dt.date
for c in ["strike", "last", "bid", "ask", "iv", "volume", "oi", "underlying_px", "delta", "gamma", "option_net"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# ---- Better SPOT detection ----
# Prefer the real underlying price from the data source.
# Fallback to highest-volume strike, then median strike.

def infer_default_spot(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0

    if "underlying_px" in df.columns:
        underlying = pd.to_numeric(df["underlying_px"], errors="coerce").dropna()
        underlying = underlying[underlying > 0]

        if not underlying.empty:
            return float(underlying.median())

    if "volume" in df.columns and df["volume"].notna().any():
        try:
            return float(df.loc[df["volume"].idxmax(), "strike"])
        except Exception:
            pass

    return float(df["strike"].median(skipna=True))


default_spot = infer_default_spot(df)

spot_source = "underlying_px" if (
    "underlying_px" in df.columns
    and pd.to_numeric(df["underlying_px"], errors="coerce").dropna().gt(0).any()
) else "ATM/high-volume strike fallback"
# ---------- Top controls ----------
c1,c2,c3,c4,c5 = st.columns([1,1,1,1,3])

def _safe_default(val, fallback):
    try:
        return float(val) if val is not None else float(fallback)
    except Exception:
        return float(fallback)

spot_default = _safe_default(URL_DEFAULTS.get("spot"), default_spot)
r_default    = _safe_default(URL_DEFAULTS.get("r"), 0.03)

with c1:
    spot = st.number_input("Spot (S)", value=spot_default, step=1.0, format="%.2f")
    st.caption(f"Default source: {spot_source}")
with c2:
    r = st.number_input("Risk-free (r)", value=r_default, step=0.005, format="%.4f")

with c3:
    show_calls = st.toggle("Calls", URL_DEFAULTS.get("calls", True))
with c4:
    show_puts = st.toggle("Puts", URL_DEFAULTS.get("puts", True))
with c5:
    expiries = sorted(df["expiration_date"].unique())
    exp_idx = 0
    if URL_DEFAULTS.get("exp") in expiries:
        exp_idx = expiries.index(URL_DEFAULTS["exp"])
    exp = st.selectbox("Expiration", expiries, index=exp_idx, key="exp_top")


mask = ((df["cp"].eq("C") & show_calls) | (df["cp"].eq("P") & show_puts)) & df["expiration_date"].eq(exp)
dfe = df[mask].copy().sort_values("strike")

min_k, max_k = float(dfe["strike"].min()), float(dfe["strike"].max())
kL = URL_DEFAULTS.get("k_low", min_k);  kH = URL_DEFAULTS.get("k_high", max_k)
if kL is None: kL = min_k
if kH is None: kH = max_k
k_low, k_high = st.slider("Strike range", min_value=float(min_k), max_value=float(max_k),
                          value=(float(kL), float(kH)), step=5.0)
dfe = dfe[(dfe["strike"]>=k_low)&(dfe["strike"]<=k_high)]

# ---------- Sidebar: Permalink ----------
st.sidebar.markdown("---")
st.sidebar.subheader("🔗 Permalink")
if st.sidebar.button("Update URL with current filters"):
    state = {
        "use_hist": use_hist,
        "run_date": chosen_date if (use_hist and chosen_date) else None,
        "spot": spot, "r": r,
        "calls": show_calls, "puts": show_puts,
        "exp": exp, "k_low": k_low, "k_high": k_high,
        # overlay defaults carry-through; can be refined with session_state if desired
        "ovr_x": URL_DEFAULTS.get("ovr_x","strike"),
        "ovr_avgcp": URL_DEFAULTS.get("ovr_avgcp", True),
        "ovr_mode": URL_DEFAULTS.get("ovr_mode", "raw"),
        "ovr_frac": URL_DEFAULTS.get("ovr_frac", 0.15),
        "ovr_exps": URL_DEFAULTS.get("ovr_exps", []),
    }
    set_url_state(state)
    st.sidebar.success("URL updated — copy it from the address bar.")
    
# ---------- Global Plot Helpers ----------

def add_spot_line(fig, spot, x_is_moneyness=False):
    """Add a vertical spot line to any Plotly figure."""
    try:
        s = float(spot)
    except:
        return fig

    x_val = 1.0 if x_is_moneyness else s

    fig.add_vline(
        x=x_val,
        line_dash="dot",
        line_width=1,
        line_color="cyan",
        opacity=0.9,
    )
    return fig
    

# ---------- Tabs ----------
tabs = st.tabs([
    "IV Skew", "OI & Volume", "Gamma (approx)", "Term Structure", "Table",
    "Skew Overlay (Multi-Expiry)", "OI Change (Flows)", "Positioning Tilt",
    "Spread Detector", "Skew: Today vs Yesterday", "3D Vol Surface",
    "Summary", "Help & How To Use", "Probable Levels"
])


# ---- Tab 1: IV Skew ----
with tabs[0]:
    st.subheader("Volatility Skew")
    tab_help("""
**What this shows**

This view plots the **implied volatility (IV)** of options for a single expiration against either **strike** or **moneyness (K/S)**.  
You can see **Calls vs Puts**, optional **smoothed lines** (LOWESS), and a **mid IV** line (average of C and P at each x).

- When IV **increases as strike decreases** (for puts) → classic **equity skew / smirk** (downside protection is expensive).  
- When wings (far OTM) are very high vs ATM → **fat tails priced into the distribution**.  
- Flat skew → market prices a more “normal” distribution, with less fear in the tails.

**How to read it**

- **Spot line** (or K/S = 1 in moneyness):  
  - ATM region — where most delta sits and where hedging flows tend to concentrate.  
- **Calls vs Puts:**  
  - Asymmetry between call and put IV can hint at **call overwriting**, upside panic, or put buying.  
- **Smoothed curves:**  
  - Use them to see the **overall shape** without being distracted by noisy individual quotes.

**Typical questions to answer here**

- Are **downside puts** very expensive vs ATM? (Crash protection bid.)  
- Are **OTM calls** (e.g., K/S > 1.05–1.10) unusually rich? (Chase for upside convexity.)  
- Did the **shape of the skew** change after an event (earnings, macro, etc.)?
""")


    # ── Controls ─────────────────────────────────────────────────────────────
    col_sk1, col_sk2, col_sk3 = st.columns([1, 1, 1])
    with col_sk1:
        x_choice = st.radio(
            "X-axis",
            ["Strike", "Moneyness (K/S)"],
            index=0,
            horizontal=True,
            key="skew_xaxis",
        )
    with col_sk2:
        show_smooth = st.checkbox(
            "Show smoothed curve",
            value=True,
            key="skew_smooth",
        )
    with col_sk3:
        band_pct = st.slider(
            "ATM zoom (±%)",
            min_value=0,
            max_value=60,
            value=0,  # 0 = no zoom
            step=5,
            key="skew_band",
            help="If >0, only show strikes within ±% of Spot.",
        )

    # ── Base data ────────────────────────────────────────────────────────────
    skew_df = dfe.copy()

    # Optional ATM zoom around spot
    if band_pct > 0 and np.isfinite(float(spot)):
        lo = float(spot) * (1 - band_pct / 100.0)
        hi = float(spot) * (1 + band_pct / 100.0)
        skew_df = skew_df[(skew_df["strike"] >= lo) & (skew_df["strike"] <= hi)]

    if skew_df.empty:
        st.info("No strikes available for the selected zoom window.")
        st.stop()

    # Choose x variable (Strike or Moneyness)
    if x_choice.startswith("Moneyness"):
        skew_df["xvar"] = skew_df["strike"].astype(float) / float(spot)
        x_label = "Moneyness (K/S)"
        x_is_moneyness = True
    else:
        skew_df["xvar"] = skew_df["strike"].astype(float)
        x_label = "Strike"
        x_is_moneyness = False

    # Size points by volume (liquidity)
    vol = pd.to_numeric(skew_df["volume"], errors="coerce").fillna(0)
    if vol.max() > 0:
        size = 4 + 8 * (vol / vol.max())  # 4–12
    else:
        size = 6

    # ── Scatter of raw IV points ─────────────────────────────────────────────
    fig = px.scatter(
        skew_df,
        x="xvar",
        y="iv",
        color="cp",
        template=PX_TEMPLATE,
        title=f"IV vs {x_label} · {exp}",
        color_discrete_sequence=["#1E90FF", "#FF4500"],  # C / P
        labels={"iv": "Implied Volatility", "cp": "Side"},
    )
    fig.update_traces(marker=dict(size=size, opacity=0.9))
    
    for t in fig.data:
        if t.mode == "markers":        # raw points
            t.visible = "legendonly"   # hide but keep in legend
    # ── Optional smoothed curves per side ────────────────────────────────────
    if show_smooth:
        for side, g_side in skew_df.groupby("cp"):
            g_side = g_side.dropna(subset=["xvar", "iv"]).sort_values("xvar")
            # Drop zero IVs to avoid vertical spikes
            g_side = g_side[g_side["iv"] > 0]

            if len(g_side) < 5:
                continue

            sm = lowess(
                g_side["iv"].values,
                g_side["xvar"].values,
                frac=0.15,
                return_sorted=True,
            )

            line_name = "C (smooth)" if side == "C" else "P (smooth)"
            fig.add_scatter(
                x=sm[:, 0],
                y=sm[:, 1],
                mode="lines",
                name=line_name,
                line=dict(width=2),
                showlegend=True,
            )

    # ── Optional mid IV line (average of C & P at each x) ────────────────────
    mids = (
        skew_df.dropna(subset=["xvar", "iv"])
        .groupby("xvar", as_index=False)["iv"]
        .mean()
        .sort_values("xvar")
    )
    if len(mids) >= 3:
        fig.add_scatter(
            x=mids["xvar"],
            y=mids["iv"],
            mode="lines",
            name="IV mid",
            line=dict(width=1, dash="dot", color="white"),
            showlegend=True,
            visible="legendonly",
        )

    # ── Layout & helpers ─────────────────────────────────────────────────────
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title="Implied Volatility",
        hovermode="x unified",
        legend_title_text="Side",
    )

    # Spot line (strike or K/S = 1)
    if x_is_moneyness:
        fig = add_spot_line(fig, spot, x_is_moneyness=True)
    else:
        fig = add_spot_line(fig, spot)

    st.plotly_chart(fig, use_container_width=True)



# ---- Tab 2: OI & Volume ----
with tabs[1]:
    st.subheader("Open Interest & Volume")
    tab_help("""
**What this shows**

Two bar charts by **strike and side (C / P)**:

1. **Open Interest (OI)** → how many contracts are currently open.  
2. **Volume** → how many contracts traded in the current snapshot.

Together they show **where positions are sitting** and **where trading is active today**.

**How to read it**

- **High OI, low volume:**  
  - Large “parked” positions, not much trading today. Can still matter for gamma/GEX and pinning.  
- **High OI, high volume:**  
  - Key battleground strikes where traders are actively adjusting big positions.  
- **Low OI, high volume:**  
  - Fresh activity, possible **new structures** forming.

**Controls & interpretation**

- **Log scale:**  
  - Use when a few huge strikes compress the rest of the chart; log reveals **relative structure**.  
- **Percentile clipping:**  
  - Caps extreme bars so mid-sized strikes are still visible.  
- **ATM zoom:**  
  - Focus on the area where Spot is trading; relevant for short-term moves and gamma effects.

Use this tab to quickly answer:  
> “Where are people **positioned** and where is today’s **action**?”
""")

    # ── Controls (local to this tab) ───────────────────────────────────────────
    colc1, colc2, colc3 = st.columns([1, 2, 2])
    with colc1:
        use_log = st.checkbox("Log scale (Y)", value=False,
                              help="Helps when a few strikes dominate.")
    with colc2:
        pct_cap = st.slider("Clip at percentile", 90, 100, 99,
                            help="Clips extreme spikes so the rest is visible. 100 = no clipping.")
    with colc3:
        atm_zoom = st.checkbox("ATM zoom (±%)", value=True,
                               help="Show strikes within ±% of Spot (S).")
        band = st.slider("±% around Spot", 5, 60, 25, 5, disabled=not atm_zoom)

    # ── Base data for this tab (start from your already filtered dfe) ─────────
    df_zoom = dfe.copy()
    if atm_zoom and np.isfinite(float(spot)):
        lo = float(spot) * (1 - band/100.0)
        hi = float(spot) * (1 + band/100.0)
        df_zoom = df_zoom[(df_zoom["strike"] >= lo) & (df_zoom["strike"] <= hi)]
        if df_zoom.empty:
            st.info("No strikes in the selected ATM window; showing full range instead.")
            df_zoom = dfe.copy()

    df_zoom = df_zoom.sort_values(["expiration_date", "strike"])

    # ── Percentile capping helpers ─────────────────────────────────────────────
    def add_capped(series: pd.Series, q: int):
        if series.dropna().empty or q >= 100:
            return series
        hi_val = float(np.nanpercentile(series, q))
        return series.clip(upper=hi_val)

    # ── Aggregate & cap ───────────────────────────────────────────────────────
    oi_df  = df_zoom.groupby(["strike", "cp"], as_index=False)["oi"].sum()
    vol_df = df_zoom.groupby(["strike", "cp"], as_index=False)["volume"].sum()
    oi_df["oi_capped"]   = add_capped(oi_df["oi"], pct_cap)
    vol_df["vol_capped"] = add_capped(vol_df["volume"], pct_cap)

    # ── Plots ─────────────────────────────────────────────────────────────────
    colA, colB = st.columns(2)

    with colA:
        fig_oi = px.bar(
            oi_df, x="strike", y="oi_capped", color="cp", barmode="group",
            template=PX_TEMPLATE, title="Open Interest by Strike", color_discrete_sequence=["#1E90FF", "#FF4500"]
        )
        fig_oi.update_yaxes(type="log" if use_log else "linear", title="OI")
        fig_oi.update_layout(hovermode="x unified")
        fig_oi = add_spot_line(fig_oi, spot)
        st.plotly_chart(fig_oi, use_container_width=True)
        

    with colB:
        fig_vol = px.bar(
            vol_df, x="strike", y="vol_capped", color="cp", barmode="group",
            template=PX_TEMPLATE, title="Volume by Strike (today)", color_discrete_sequence=["#1E90FF", "#FF4500"]
        )
        fig_vol.update_yaxes(type="log" if use_log else "linear", title="Volume")
        fig_vol.update_layout(hovermode="x unified")
        fig_vol = add_spot_line(fig_vol, spot)
        st.plotly_chart(fig_vol, use_container_width=True)


# ---- Tab 3: Gamma (approx) ----
with tabs[2]:
    st.subheader("Dealer Gamma Exposure (approx)")
    tab_help("""
**What this shows**

An approximation of **dealer gamma exposure (GEX)** across strikes:

- Uses **Cboe-provided option gamma** when available, with a Black–Scholes fallback.  
- Multiplies by **Open Interest** and a contract multiplier to estimate a **notional gamma** per strike.  
- Aggregates into **Call Γ**, **Put Γ**, **Total Γ**, plus a **cumulative curve**.

The idea: this is a map of where dealers (in aggregate) are **long or short gamma**.

**How to read it**

- **Total Γ > 0 (long gamma region):**  
  - Dealers tend to **buy dips and sell rips** (hedging dampens moves).  
- **Total Γ < 0 (short gamma region):**  
  - Hedging can **amplify moves** (buying high, selling low), making price more jumpy.  
- **Cumulative Γ:**  
  - Shows how gamma risk accumulates as you move across strikes — useful to see whether gamma is concentrated in one area or spread out.  
- **Flip strike (yellow line):**  
  - Approx strike where total gamma shifts sign → often watched as a **regime line**.  
- **Spot line (cyan):**  
  - Where the market currently trades relative to the gamma structure.

**Practical use**

- If **Spot is well inside a strong long gamma region**, expect **mean-reverting, slower moves**.  
- If **Spot moves into/through a short gamma pocket**, moves can become **sharp and directional**.  
- Combine this with the **OI & Volume** tab to see where large positions and gamma are aligned.
""")


    CONTRACT_MULT = symbol_config["contract_multiplier"]
    today = date.today()

    # --- Controls for style/zoom ---
    colg1, colg2 = st.columns([1, 1])
       
        # --- OI Change overlay controls ---
    st.markdown("##### Overlay")
    colov1, colov2, colov3 = st.columns([1,1,2])
    with colov1:
        overlay_oi = st.checkbox(
            "Overlay OI change (prev vs current run)",
            value=True,
            key="gex_overlay_oi"
        )
    with colov2:
        oi_cap_pct = st.slider(
            "Cap OI Δ spikes (percentile)",
            90, 100, 99, 1,
            key="gex_oi_cap"
        )
    with colov3:
        oi_scale_mode = st.radio(
            "OI Δ scaling",
            ["Auto-fit", "Raw (y2)"],
            horizontal=True,
            key="gex_oi_scale"
        )

    with colg1:
        zoom_atm = st.checkbox("Zoom around Spot (±%)", value=True, key="gex_zoom")
        band = st.slider("±% around Spot", 2, 50, 10, 1,
                         disabled=not zoom_atm, key="gex_band")
    with colg2:
        scale_millions = st.checkbox("Show values in millions",
                                     value=True, key="gex_scale_mio")

    # --- Base gamma calculation ---
    g = dfe.copy()
    g["T"] = g["expiration_date"].apply(lambda d: yearfrac(today, d))
    g["sigma"] = pd.to_numeric(g["iv"], errors="coerce")

    # Prefer Cboe-provided gamma when available.
    # Fallback to Black-Scholes gamma if stored gamma is missing.
    if "gamma" in g.columns and g["gamma"].notna().any():
        g["gamma_calc"] = pd.to_numeric(g["gamma"], errors="coerce")
    else:
        g["gamma_calc"] = np.nan

    missing_gamma = g["gamma_calc"].isna()

    if missing_gamma.any():
        g.loc[missing_gamma, "gamma_calc"] = g.loc[missing_gamma].apply(
            lambda r2: bs_gamma(spot, r2["strike"], r, r2["sigma"], r2["T"])
            if (
                pd.notna(r2["sigma"])
                and r2["sigma"] > 0
                and r2["T"] > 0
            )
            else 0.0,
            axis=1,
        )

    g["gamma_calc"] = g["gamma_calc"].fillna(0.0)
    g["gex"] = -g["gamma_calc"] * g["oi"].fillna(0) * CONTRACT_MULT * (spot ** 2)

    # --- Aggregate by strike / cp ---
    grp = g.groupby(["strike", "cp"], as_index=False)["gex"].sum()
    pivot = grp.pivot(index="strike", columns="cp", values="gex").fillna(0.0)
    pivot = pivot.rename(columns={"C": "gamma_call", "P": "gamma_put"})
    if "gamma_call" not in pivot.columns:
        pivot["gamma_call"] = 0.0
    if "gamma_put" not in pivot.columns:
        pivot["gamma_put"] = 0.0

    pivot = pivot.reset_index().sort_values("strike")
    
        # ---- OI Change overlay (same expiry, prev vs current run) ----
    if overlay_oi:
        all_dates = load_run_dates(symbol)
        date_cur = chosen_date if (use_hist and chosen_date) else (all_dates[0] if all_dates else None)
        date_prev = prev_run_date(all_dates, date_cur) if all_dates else None

        if date_cur is not None and date_prev is not None:
            cur_run, prv_run = load_chain_two_days(symbol, date_cur, date_prev)

            cur_run["expiration_date"] = pd.to_datetime(cur_run["expiration_date"]).dt.date
            prv_run["expiration_date"] = pd.to_datetime(prv_run["expiration_date"]).dt.date

            oi_df = compute_oi_change_by_strike(cur_run, prv_run, exp_sel=exp)

            if oi_cap_pct < 100 and not oi_df.empty:
                abs_all = pd.concat([oi_df["oiC"].abs(), oi_df["oiP"].abs()])
                lim = float(np.nanpercentile(abs_all, oi_cap_pct))
                oi_df["oiC"] = oi_df["oiC"].clip(-lim, lim)
                oi_df["oiP"] = oi_df["oiP"].clip(-lim, lim)

            pivot = pd.merge(pivot, oi_df, on="strike", how="left").fillna(0.0)
        else:
            pivot["oiC"], pivot["oiP"] = 0.0, 0.0
    else:
        pivot["oiC"], pivot["oiP"] = 0.0, 0.0


    # --- Zoom around Spot for plotting ---
    if zoom_atm and np.isfinite(float(spot)):
        lo = float(spot) * (1 - band / 100.0)
        hi = float(spot) * (1 + band / 100.0)
        pivot = pivot[(pivot["strike"] >= lo) & (pivot["strike"] <= hi)]

    if pivot.empty:
        st.info("No strikes in the selected window for Gamma view.")
        st.stop()

    # --- Total gamma & cumulative gamma (in the zoomed window) ---
    pivot["total_gamma"] = pivot["gamma_call"] + pivot["gamma_put"]
    pivot["cum_gamma"] = pivot["total_gamma"].cumsum()

    # --- Flip strike where Total Γ changes sign in this window ---
    flip_strike = None
    tg = pivot["total_gamma"].values
    strikes = pivot["strike"].values

    if len(tg) >= 2:
        sgn = np.sign(tg)
        change_idx = np.where(np.diff(sgn) != 0)[0]
        if len(change_idx):
            i = int(change_idx[0])
            x0, y0 = strikes[i],     tg[i]
            x1, y1 = strikes[i + 1], tg[i + 1]
            if (y1 - y0) != 0:
                flip_strike = float(x0 - y0 * (x1 - x0) / (y1 - y0))
            else:
                flip_strike = float(strikes[i])

    # --- Scale to millions so the axes are readable ---
    y_title = "Gamma Exposure"
    y2_title = "Cumulative Γ"
    if scale_millions:
        pivot[["gamma_call", "gamma_put", "total_gamma", "cum_gamma"]] = (
            pivot[["gamma_call", "gamma_put", "total_gamma", "cum_gamma"]] / 1_000_000.0
        )
        y_title = "Gamma Exposure (millions)"
        y2_title = "Cumulative Γ (millions)"

    # --- Plot (bars + orange curve) ---
    fig_gex = go.Figure()

     # --- Plot (two-panel: Gamma on top, OI Δ on bottom) ---
    fig_gex = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.72, 0.28],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
    )

    # ---------- TOP PANEL: Gamma ----------
    # Make call/put thinner + slightly transparent; keep Total more visible
    fig_gex.add_bar(
        row=1, col=1,
        x=pivot["strike"], y=pivot["gamma_call"],
        name="Γ Call",
        marker_color="green",
        opacity=0.55,
    )
    fig_gex.add_bar(
        row=1, col=1,
        x=pivot["strike"], y=pivot["gamma_put"],
        name="Γ Put",
        marker_color="red",
        opacity=0.55,
    )
    fig_gex.add_bar(
        row=1, col=1,
        x=pivot["strike"], y=pivot["total_gamma"],
        name="Total Γ",
        marker_color="purple",
        opacity=0.85,
    )

    # Cumulative curve on secondary y (right) — TOP PANEL
    fig_gex.add_scatter(
        x=pivot["strike"],
        y=pivot["cum_gamma"],
        name="Curve Γ (cum)",
        mode="lines",
        line=dict(color="orange", width=2),
        yaxis="y2",
    )

    # ---------- BOTTOM PANEL: OI Change ----------
    lim2 = 1.0
    if overlay_oi:
        oiC = pivot.get("oiC", pd.Series(0, index=pivot.index)).astype(float)
        oiP = pivot.get("oiP", pd.Series(0, index=pivot.index)).astype(float)

        # Cap for display (try 97 or 95 if still flat)
        lim2 = float(np.nanpercentile(np.abs(pd.concat([oiC, oiP])), 97))
        lim2 = max(lim2, 1.0)

        pivot["oiC_plot"] = oiC.clip(-lim2, lim2)
        pivot["oiP_plot"] = oiP.clip(-lim2, lim2)

        fig_gex.add_bar(
            row=2, col=1,
            x=pivot["strike"], y=pivot["oiC_plot"],
            name="OI Δ Call (capped)",
            marker_color="rgba(30,144,255,0.75)",
            opacity=0.75,
        )
        fig_gex.add_bar(
            row=2, col=1,
            x=pivot["strike"], y=pivot["oiP_plot"],
            name="OI Δ Put (capped)",
            marker_color="rgba(255,69,0,0.75)",
            opacity=0.75,
        )

        # Add a zero-line to read OI Δ easier
        fig_gex.add_hline(row=2, col=1, y=0, line_dash="dot", line_width=1, opacity=0.5)

    # ---------- Flip line (top panel) ----------
    if flip_strike is not None:
        fig_gex.add_vline(
            x=flip_strike,
            line_dash="dash",
            line_width=2,
            line_color="yellow",
        )

    # ---------- Spot line (both panels) ----------
    try:
        s_val = float(spot)
        fig_gex.add_vline(
            x=s_val,
            line_dash="dot",
            line_width=1,
            line_color="cyan",
        )
    except Exception:
        pass

    # ---------- Make it readable ----------
    # 1) Put the legend on top; 2) Overlay bars within each panel
    fig_gex.update_layout(
        template=PX_TEMPLATE,
        title="Gamma by Strike + OI Change (Two-Panel View)",
        barmode="group",
        bargap=0.10,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),

        yaxis=dict(
            title="Gamma Exposure (per strike)",
            zeroline=True,
        ),
        yaxis2=dict(
            title="Cumulative Gamma",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
    )

    # Axis labels
    fig_gex.update_yaxes(title_text=y_title, row=1, col=1)
    fig_gex.update_yaxes(title_text=y2_title, row=1, col=1, secondary_y=True)
    fig_gex.update_yaxes(title_text="OI Change (contracts)", row=2, col=1)
    fig_gex.update_xaxes(title_text="Strike", row=2, col=1)
    fig_gex.update_yaxes(range=[-lim2, lim2], row=2, col=1)
    
    st.plotly_chart(fig_gex, use_container_width=True)

    st.markdown("### Intraday Decision Tree (Gamma + OI)")

    pv = pivot.sort_values("strike").copy()
    pv["oi_abs"] = pv["oiC"].abs() + pv["oiP"].abs()
    pv["oi_z"] = robust_zscore(pv["oi_abs"])

    short_gamma = (pv["total_gamma"].sum() < 0)
    pv_reset = pv.reset_index(drop=True)
    i = int((pv_reset["strike"] - float(spot)).abs().idxmin())
    slope = pv_reset["cum_gamma"].iloc[min(i+3,len(pv_reset)-1)] - pv_reset["cum_gamma"].iloc[max(i-3,0)]

    regime = short_gamma and slope < 0

    cliff_thr = np.nanpercentile(pv_reset["total_gamma"], 10)
    cliffs = pv_reset[pv_reset["total_gamma"] <= cliff_thr]
    K_star = float(
        cliffs.loc[(cliffs["strike"] - float(spot)).abs().idxmin(), "strike"]
        if not cliffs.empty else pv_reset.iloc[i]["strike"]
    )

    active = pv_reset[pv_reset["oi_z"] >= 2.0]

    up = active[(active["strike"] >= spot) & (active["strike"] <= spot + 150)]
    dn = active[(active["strike"] <= spot) & (active["strike"] >= spot - 150)]

    UpCall, UpPut = up["oiC"].clip(lower=0).sum(), up["oiP"].clip(lower=0).sum()
    DnCall, DnPut = dn["oiC"].clip(lower=0).sum(), dn["oiP"].clip(lower=0).sum()

    if UpCall > UpPut * 1.5:
        bias = "Upside squeeze risk"
    elif DnPut > DnCall * 1.5:
        bias = "Downside continuation risk"
    else:
        bias = "Mixed / neutral"

    cA, cB, cC = st.columns(3)
    cA.metric("Gamma regime", "SHORT GAMMA" if regime else "Not clear")
    cB.metric("Decision strike (K*)", f"{K_star:.0f}")
    cC.metric("Flow bias", bias)


# ---- Tab 4: Term Structure ----
with tabs[3]:
    st.subheader("ATM IV Term Structure")
    tab_help("""
**What this shows**

The **ATM implied volatility** for many expirations vs **days to expiration (DTE)**:

- Each dot = one expiration’s ATM IV.  
- Colors group expirations into maturity buckets (short, medium, long).  
- A line connects them to show the overall **vol curve in time**.

This is the **term structure of volatility**: how expensive options are by horizon.

**How to read it**

- **Upward sloping curve:**  
  - Short-dated vol cheaper than long-dated. Typical in quiet periods or when long-term uncertainty is higher.  
- **Inverted / hump-shaped:**  
  - Short-dated vol elevated (event risk: CPI, FOMC, earnings), with medium/long-dated calmer.  
- **Kinks / jumps:**  
  - Specific expiries pricing in special events (Fed meeting, big macro date, index rebalances).

**Practical questions**

- Is **front vol** expensive vs the rest of the curve? (Event premium.)  
- Are **long-dated options** unusually cheap or rich vs their historical norms?  
- If you trade spreads across time (calendar/diagonal), this view helps identify **which leg is “rich” vs “cheap”** in volatility terms.
""")

    today = date.today()

    # --- Controls ---
    col_ts1, col_ts2 = st.columns([2, 1])
    with col_ts1:
        max_dte = st.slider(
            "Max days to expiration",
            min_value=30,
            max_value=3650,
            value=730,  # default: 2 years
            step=30,
            key="ts_max_dte",
        )
    with col_ts2:
        clip_outliers = st.checkbox(
            "Clip extreme IV outliers (percentile cap)",
            value=True,
            key="ts_clip",
            help="Useful if a single bad IV print distorts the chart.",
        )

    # --- Build ATM IV per expiration ---
    atm_rows = []
    for e in sorted(df["expiration_date"].unique()):
        iv_atm = nearest_strike_iv(df, e, float(spot))
        if np.isfinite(iv_atm):
            dte = max((e - today).days, 0)
            atm_rows.append(
                {"expiration_date": e, "dte": dte, "atm_iv": iv_atm}
            )

    term = pd.DataFrame(atm_rows)
    if term.empty:
        st.info("No ATM IV data available for term structure.")
        st.stop()

    # Drop already-expired / same-day (dte == 0) – they often create that weird spike
    term = term[term["dte"] > 0]

    # Filter by max DTE
    term = term[term["dte"] <= max_dte].sort_values("dte")

    # Optional outlier clipping
    if clip_outliers and term["atm_iv"].notna().any():
        hi = float(np.nanpercentile(term["atm_iv"], 98))
        term["atm_iv"] = term["atm_iv"].clip(upper=hi)

    # --- Maturity buckets for color ---
    def bucket_dte(d):
        if d <= 30:
            return "0–30d"
        elif d <= 90:
            return "30–90d"
        elif d <= 365:
            return "3m–1y"
        else:
            return ">1y"

    term["bucket"] = term["dte"].apply(bucket_dte)

    # --- Main line (overall curve) + colored points by bucket ---
    fig_t = px.scatter(
        term,
        x="dte",
        y="atm_iv",
        color="bucket",
        template=PX_TEMPLATE,
        labels={"dte": "Days to Expiration", "atm_iv": "ATM IV", "bucket": "Maturity"},
        title="ATM IV Across Expirations",
    )

    # Add a smooth connecting line over all expiries
    fig_t.add_scatter(
        x=term["dte"],
        y=term["atm_iv"],
        mode="lines",
        name="ATM IV curve",
        line=dict(width=2),
        showlegend=True,
    )

    # Vertical reference lines (30 / 90 / 180 / 365 days) if in range
    for ref in [30, 90, 180, 365]:
        if ref <= max_dte:
            fig_t.add_vline(
                x=ref,
                line_dash="dot",
                line_width=1,
                line_color="gray",
                opacity=0.4,
            )

    fig_t.update_layout(
        hovermode="x unified",
        legend_title_text="Maturity",
    )

    st.plotly_chart(fig_t, use_container_width=True)

# ---- Tab 5: Table ----
with tabs[4]:
    st.subheader("Contracts")
    tab_help("""
**What this shows**

A **raw table** of the options that feed the charts for the current filters:

- Columns include side (`cp`), strike, prices (`bid`, `ask`, `last`), IV, volume and OI.  
- Each row represents **one contract** at a specific strike, side and expiration.

**How to use it**

- To **inspect a specific strike**, check if the prices and IV look reasonable.  
- To verify **odd points** you see in skew or gamma (e.g., a spike), come here and see if it’s:  
  - A real quote (big volume/OI).  
  - Or a bad/missing print (zero / NaNs / stale price).  
- Use sorting/filtering (via the UI) to focus on:  
  - Highest volume, highest IV, weird bid/ask spreads, etc.

This tab is your **X-ray** for the rest of the dashboard.
""")


    cols = [
        "expiration_date", "cp", "strike",
        "bid", "ask", "last",
        "iv", "delta", "gamma", "option_net",
        "volume", "oi",
    ]
    cols = [c for c in cols if c in dfe.columns]
    st.dataframe(dfe[cols].sort_values(["cp", "strike"]), use_container_width=True, height=420)

# ---- Tab 6: Skew Overlay (Multi-Expiry) ----
with tabs[5]:
    st.subheader("Multi-Expiry Skew Overlay (IV vs Strike or Moneyness)")
    tab_help("""
**What this shows**

IV skews for **multiple expirations** plotted on the same axes:

- Each line = skew for one expiration (possibly averaged across Calls/Puts).  
- You can choose **Strike** or **Moneyness (K/S)** as the x-axis.  
- Smoothing removes noise and lets you see the **shape by tenor**.

**How to read it**

- Compare **front vs back**:  
  - Front-month skew usually responds strongest to near-term risk.  
  - Back-month skew reflects more structural or long-term risk pricing.  
- If **short-dated puts** are much steeper (higher downside IV) than long-dated:  
  - Market expects intense short-term risk but calmer long run.  
- If **long-dated wings** are high:  
  - Tail risk is being priced far out (e.g., structural macro worry).

This tab is great for spotting **which expiries are “doing something different”** in skew compared to the rest.
""")

    exps_all = sorted(df["expiration_date"].unique())

    # If permalink has specific expiries, try to use them first
    url_exps = URL_DEFAULTS.get("ovr_exps") or []
    url_exps = [pd.to_datetime(x).date() for x in url_exps if x]

    if url_exps:
        # keep only those that actually exist in this dataset
        default_multi = [e for e in exps_all if e in url_exps] or exps_all[:min(6, len(exps_all))]
    else:
        # otherwise: pick the 6 expirations with highest total volume
        vol_by_exp = (
            df.groupby("expiration_date")["volume"]
            .sum(min_count=1)
            .sort_values(ascending=False)
        )
        top_exps = list(vol_by_exp.index[:min(6, len(vol_by_exp))])
        default_multi = top_exps

    exps_sel = st.multiselect(
        "Select expirations to overlay",
        exps_all,
        default=default_multi,
        key="exp_overlay"
    )

    x_choice = st.radio("X-axis", ["Strike", "Moneyness (K/S)"], horizontal=True, key="xaxis_overlay",
                        index=(0 if URL_DEFAULTS.get("ovr_x","strike")=="strike" else 1))
    avg_cp = st.checkbox("Average Calls & Puts together", value=URL_DEFAULTS.get("ovr_avgcp", True),
                         help="If off, calls/puts drawn as separate series (line dashes).")

    colS1, colS2 = st.columns([1,2])
    with colS1:
        smooth_mode = st.radio("Smoothing", ["Raw", "Smoothed"],
                               index=(0 if URL_DEFAULTS.get("ovr_mode","raw")=="raw" else 1), horizontal=True)
    with colS2:
        frac = st.slider("Smoothness (LOWESS fraction)", 0.05, 0.5, float(URL_DEFAULTS.get("ovr_frac", 0.15)), 0.01)

    cp_filter = []
    if show_calls: cp_filter.append("C")
    if show_puts:  cp_filter.append("P")
    if not cp_filter:
        st.info("Enable Calls or Puts in the top controls to display overlays.")
        st.stop()

    base = df[df["cp"].isin(cp_filter)].copy()
    base = base[base["expiration_date"].isin(exps_sel)]
    if base.empty:
        st.info("No data for the selected expirations / filters.")
        st.stop()

    if x_choice.startswith("Moneyness"):
        base["xvar"] = base["strike"] / float(spot)
        x_title = "Moneyness (K/S)"
    else:
        base["xvar"] = base["strike"].astype(float)
        x_title = "Strike"

    group_cols = ["expiration_date", "xvar"] if avg_cp else ["expiration_date", "cp", "xvar"]
    iv_grid = (base.groupby(group_cols, as_index=False)["iv"].mean()
                    .dropna(subset=["iv","xvar"])
                    .sort_values(group_cols))

    if smooth_mode == "Raw":
        if avg_cp:
            fig = px.line(iv_grid, x="xvar", y="iv", color="expiration_date",
                          template=PX_TEMPLATE, title="IV Skew Overlay (Raw)")
            fig_scatter = px.scatter(iv_grid, x="xvar", y="iv", color="expiration_date",
                                     template=PX_TEMPLATE)
            for tr in fig_scatter.data:
                tr.mode = "markers"; tr.marker.update(size=6, opacity=0.75)
                fig.add_trace(tr)
        else:
            fig = px.line(iv_grid, x="xvar", y="iv", color="expiration_date",
                          line_dash="cp", template=PX_TEMPLATE, title="IV Skew Overlay (Raw, Calls vs Puts)")
            fig_scatter = px.scatter(iv_grid, x="xvar", y="iv", color="expiration_date",
                                     symbol="cp", template=PX_TEMPLATE)
            for tr in fig_scatter.data:
                tr.mode = "markers"; tr.marker.update(size=6, opacity=0.75)
                fig.add_trace(tr)
    else:
        def smooth_series(df_part):
            df_part = df_part.dropna(subset=["xvar","iv"]).sort_values("xvar")
            if len(df_part) < 5:
                return df_part.rename(columns={"iv": "iv_smooth"})[["xvar","iv_smooth"]]
            sm = lowess(df_part["iv"].values, df_part["xvar"].values, frac=frac, return_sorted=True)
            return pd.DataFrame({"xvar": sm[:,0], "iv_smooth": sm[:,1]})

        smooth_curves = []
        if avg_cp:
            for expi, g in iv_grid.groupby("expiration_date"):
                sm = smooth_series(g); sm["expiration_date"] = expi; smooth_curves.append(sm)
        else:
            for (expi, side), g in iv_grid.groupby(["expiration_date","cp"]):
                sm = smooth_series(g); sm["expiration_date"] = expi; sm["cp"] = side; smooth_curves.append(sm)
        smooth_df = pd.concat(smooth_curves, ignore_index=True) if smooth_curves else pd.DataFrame(columns=["xvar","iv_smooth"])

        if avg_cp:
            fig = px.line(smooth_df, x="xvar", y="iv_smooth", color="expiration_date",
                          template=PX_TEMPLATE, title="IV Skew Overlay (Smoothed)")
        else:
            fig = px.line(smooth_df, x="xvar", y="iv_smooth", color="expiration_date",
                          line_dash="cp", template=PX_TEMPLATE, title="IV Skew Overlay (Smoothed, Calls vs Puts)")

    fig.update_layout(xaxis_title=x_title, yaxis_title="Implied Volatility", legend_title_text="Expiration", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# ---- Tab 7: OI Change (Flows) ----
with tabs[6]:
    st.subheader("Open Interest Change (Day-over-Day)")
    tab_help("""
**What this shows**

The **change in Open Interest** by strike and side between two snapshots (today vs previous run):

- For each strike & side (Call/Put):  
  `OI_change = OI_today − OI_previous`.  
- Bars show how many contracts were **added or removed** at each level.  
- A table below lists the **largest absolute movers** with extra context (volume, IV).

**How to read it**

- **Big positive bar:**  
  - New positions opened → fresh interest there (could be hedging, speculation, or spread legs).  
- **Big negative bar:**  
  - Large positions closed/rolled or exercised/expired.  
- Clusters of changes near certain strikes can indicate **rolls** (e.g., 5000 → 5100).

**Using the controls**

- **Zoom around Spot:**  
  - Focus on changes near the current price.  
- **Percentile cap:**  
  - Avoid one huge change masking smaller but important flows.  
- **Top N movers:**  
  - Jump straight to the most important strikes by flow.

Think of this as your **flow radar**: “Where did the book meaningfully change since last time?”
""")

    all_dates = load_run_dates(symbol)
    if not all_dates:
        st.info("History table not available.")
        st.stop()

    # Same logic as before for current / previous snapshot
    date_cur = chosen_date if (use_hist and chosen_date) else all_dates[0]
    date_prev = prev_run_date(all_dates, date_cur)

    cur, prv = load_chain_two_days(symbol, date_cur, date_prev)
    if cur.empty:
        st.info("No rows for selected date.")
        st.stop()

    exp_sel = st.selectbox(
        "Expiration",
        sorted(cur["expiration_date"].unique()),
        key="exp_oi_change"
    )
    key_cols = ["expiration_date", "strike", "cp"]

    cur_e = (
        cur[cur["expiration_date"].eq(exp_sel)][key_cols + ["oi", "volume", "iv"]]
        .rename(columns={"oi": "oi_cur", "volume": "vol_cur", "iv": "iv_cur"})
    )
    prv_e = (
        prv[prv["expiration_date"].eq(exp_sel)][key_cols + ["oi"]]
        .rename(columns={"oi": "oi_prev"})
    )

    merged = pd.merge(cur_e, prv_e, on=key_cols, how="left")
    for c in ["oi_prev", "oi_cur", "vol_cur", "iv_cur"]:
        merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0)

    merged["oi_change"] = merged["oi_cur"] - merged["oi_prev"]

    # ---- Zoom & clipping controls ----
    colz1, colz2 = st.columns(2)
    with colz1:
        zoom_pct = st.slider(
            "Zoom around Spot (±%)",
            min_value=0,
            max_value=60,
            value=20,
            step=5,
            help="If >0, only show strikes within ±% of Spot.",
            key="oi_change_zoom_pct",
        )
    with colz2:
        cap_pct = st.slider(
            "Cap extreme OI spikes (percentile)",
            min_value=90,
            max_value=100,
            value=99,
            step=1,
            help="Clips huge OI changes so normal moves become visible.",
            key="oi_change_cap_pct",
        )

    hm = merged.groupby(["strike", "cp"], as_index=False)["oi_change"].sum()

    df_plot = hm.copy()

    # X-axis zoom around spot
    if zoom_pct > 0 and np.isfinite(float(spot)):
        lo = float(spot) * (1 - zoom_pct / 100.0)
        hi = float(spot) * (1 + zoom_pct / 100.0)
        df_plot = df_plot[(df_plot["strike"] >= lo) & (df_plot["strike"] <= hi)]

    # Percentile cap on absolute OI change
    if cap_pct < 100 and not df_plot["oi_change"].dropna().empty:
        lim = float(np.nanpercentile(np.abs(df_plot["oi_change"]), cap_pct))
        df_plot["oi_change"] = df_plot["oi_change"].clip(-lim, lim)

    # Pretty timestamps for title
    def fmt_dt(dt):
        return dt.strftime("%Y-%m-%d %H:%M:%S") if dt is not None else "?"

    title = f"OI Change by Strike — {fmt_dt(date_prev)} → {fmt_dt(date_cur)}"

    fig_hm = px.bar(
        df_plot,
        x="strike",
        y="oi_change",
        color="cp",
        barmode="group",
        template=PX_TEMPLATE,
        title=title,
        color_discrete_sequence=[
            "rgba(30,144,255,0.6)",   # lighter blue
            "rgba(255,69,0,0.6)",     # lighter orange/red
        ],
    )
    fig_hm.update_layout(hovermode="x unified")
    fig_hm = add_spot_line(fig_hm, spot)
    st.plotly_chart(fig_hm, use_container_width=True)

    # Top movers table (unchanged)
    topN = st.slider("Show top N absolute movers", 10, 200, 50, 5)
    movers = merged.reindex(
        merged["oi_change"].abs().sort_values(ascending=False).index
    ).head(topN)
    st.dataframe(
        movers[
            [
                "expiration_date",
                "cp",
                "strike",
                "oi_prev",
                "oi_cur",
                "oi_change",
                "vol_cur",
                "iv_cur",
            ]
        ],
        use_container_width=True,
        height=380,
    )


# ---- Tab 8: Positioning Tilt ----
with tabs[7]:
    st.subheader("Net Positioning Tilt (Call OI − Put OI)")
    tab_help("""
**What this shows**

A **net balance** of open interest between calls and puts at each strike:

- For each strike: `Tilt = Call OI − Put OI`.  
- Bars above zero → **call-heavy**; below zero → **put-heavy**.  
- A zero line makes the neutral level obvious; the Spot line shows current price.

**How to read it**

- **Call-heavy regions:**  
  - Often associated with **covered call writing** or speculative upside structures.  
- **Put-heavy regions:**  
  - Can signal **downside hedging**, protective puts, or structured short exposure.  
- Look at where **Spot sits** relative to big tilts:  
  - If Spot is just below a **call-heavy region**, that may act as resistance (supply of calls).  
  - If Spot is above a put-heavy area, that region may reflect hedging / protection below.

Use this tab for a quick visual answer to:  
> “At each strike, is the crowd skewed more towards **upside or downside** exposure?”
""")

    cur_all = df.copy()
    exp_tilt = st.selectbox("Expiration", sorted(cur_all["expiration_date"].unique()), key="tilt_exp")
    cur_e = cur_all[cur_all["expiration_date"].eq(exp_tilt)]

    calls = cur_e[cur_e["cp"].eq("C")].groupby("strike", as_index=False)["oi"].sum().rename(columns={"oi":"call_oi"})
    puts  = cur_e[cur_e["cp"].eq("P")].groupby("strike", as_index=False)["oi"].sum().rename(columns={"oi":"put_oi"})
    tilt = pd.merge(calls, puts, on="strike", how="outer").fillna(0.0)
    tilt["tilt"] = tilt["call_oi"] - tilt["put_oi"]
    tilt = tilt.sort_values("strike")

    fig_tilt = px.bar(tilt, x="strike", y="tilt", template=PX_TEMPLATE, title=f"Net Tilt by Strike — {exp_tilt}", color_discrete_sequence=["#1E90FF", "#FF4500"])
    fig_tilt.add_hline(y=0, line_dash="dash")
    fig_tilt.update_layout(hovermode="x unified")
    fig_tilt = add_spot_line(fig_tilt, spot)
    st.plotly_chart(fig_tilt, use_container_width=True)
    st.dataframe(tilt.tail(100), use_container_width=True, height=320)

# ---- Tab 9: Spread Detector ----
with tabs[8]:
    st.subheader("Spread Detector (heuristic)")
    tab_help("""
**What this shows**

A **rule-based detector** that looks at OI changes and tries to spot:

- **Vertical spread candidates:**  
  - Same side (all Calls or all Puts).  
  - Adjacent strikes with **similar absolute OI change**.  
- **Iron condor candidates:**  
  - One call vertical + one put vertical where the average |OI change| is similar.

It’s not perfect, but it highlights **patterns of OI change** that *look like* spreads.

**How to read it**

- **Vertical spreads table:**  
  - Shows pairs of strikes and the size of OI change in each leg.  
  - You can infer whether it’s likely a **bull** or **bear** spread based on strikes and context.  
- **Iron condors table:**  
  - Groups four strikes into a potential condor structure.  
  - Large `avg_abs_oi_change` hint at size.

Use this as a **starting point**:  
- Identify interesting candidates here, then inspect them more closely in the **OI Change** and **Contracts** tabs to understand intent and pricing.
""")

    all_dates = load_run_dates(symbol)
    if not all_dates:
        st.info("History table not available."); st.stop()
    date_cur = chosen_date if (use_hist and chosen_date) else all_dates[0]
    date_prev = prev_run_date(all_dates, date_cur)

    cur, prv = load_chain_two_days(symbol, date_cur, date_prev)
    if cur.empty:
        st.info("No rows for selected date."); st.stop()

    exp_sel = st.selectbox("Expiration", sorted(cur["expiration_date"].unique()), key="spread_exp")
    key = ["expiration_date","strike","cp"]

    cur_e = cur[cur["expiration_date"].eq(exp_sel)][key + ["oi"]].rename(columns={"oi":"oi_cur"})
    prv_e = prv[prv["expiration_date"].eq(exp_sel)][key + ["oi"]].rename(columns={"oi":"oi_prev"})
    m = pd.merge(cur_e, prv_e, on=key, how="left").fillna({"oi_prev":0})
    m["oi_change"] = pd.to_numeric(m["oi_cur"], errors="coerce").fillna(0) - pd.to_numeric(m["oi_prev"], errors="coerce").fillna(0)

    min_abs = st.slider("Min |OI change| to consider", 10, 2000, 100, 10)
    m = m[m["oi_change"].abs() >= min_abs]

    verts = []
    for cp_side, g in m.groupby("cp"):
        g2 = g.sort_values("strike").reset_index(drop=True)
        for i in range(len(g2)-1):
            r1, r2 = g2.iloc[i], g2.iloc[i+1]
            ratio = abs(abs(r1["oi_change"]) - abs(r2["oi_change"])) / max(abs(r1["oi_change"]), 1)
            if ratio <= 0.2:
                verts.append({
                    "type":"Vertical","cp":cp_side,
                    "k1": r1["strike"], "k2": r2["strike"],
                    "oi_change_1": int(r1["oi_change"]), "oi_change_2": int(r2["oi_change"])
                })
    verts_df = pd.DataFrame(verts)

    condors = []
    if not verts_df.empty:
        calls_v = verts_df[verts_df["cp"]=="C"].copy()
        puts_v  = verts_df[verts_df["cp"]=="P"].copy()
        for _, c in calls_v.iterrows():
            avg_c = np.mean([abs(c["oi_change_1"]), abs(c["oi_change_2"])])
            best = None; best_diff = 1e9
            for _, p in puts_v.iterrows():
                avg_p = np.mean([abs(p["oi_change_1"]), abs(p["oi_change_2"])])
                diff = abs(avg_c - avg_p)
                if diff < best_diff:
                    best = p; best_diff = diff
            if best is not None and best_diff/ max(avg_c,1) < 0.3:
                condors.append({
                    "type":"IronCondor",
                    "call_k1": c["k1"], "call_k2": c["k2"],
                    "put_k1":  best["k1"], "put_k2":  best["k2"],
                    "avg_abs_oi_change": int((avg_c + np.mean([abs(best['oi_change_1']),abs(best['oi_change_2'])]))/2)
                })
    condors_df = pd.DataFrame(condors).drop_duplicates()

    colX, colY = st.columns(2)
    with colX:
        st.markdown("**Vertical Spread candidates**")
        st.dataframe(verts_df if not verts_df.empty else pd.DataFrame(columns=["type","cp","k1","k2","oi_change_1","oi_change_2"]),
                     use_container_width=True, height=320)
    with colY:
        st.markdown("**Iron Condor candidates**")
        st.dataframe(condors_df if not condors_df.empty else pd.DataFrame(columns=["type","call_k1","call_k2","put_k1","put_k2","avg_abs_oi_change"]),
                     use_container_width=True, height=320)
    st.caption("Heuristic detector — use as leads, not definitive classification.")

# ---- Tab 10: Skew Today vs Yesterday ----
with tabs[9]:
    st.subheader("Skew Overlay: Today vs Yesterday")
    tab_help("""
**What this shows**

IV skew for a single expiration on **two different dates** (typically consecutive runs):

- Two curves: **current run vs previous run**.  
- X-axis is strike or moneyness; y-axis is IV.  
- You can either **average C/P** or keep them separate, and optionally smooth.

**How to read it**

- Compare **levels**:  
  - If the whole skew shifted up/down, vol repriced across the board.  
- Compare **shape**:  
  - If downside IV rose more than upside, protection demand increased.  
  - If upside IV popped, there may be chase for upside convexity.  
- It’s especially useful around **events**:  
  - Before vs after CPI/FOMC/earnings to see how the market repriced risk.

This tab answers:  
> “**What changed in skew** between yesterday and today for this expiration?”
""")

    all_dates = load_run_dates(symbol)
    if not all_dates or len(all_dates) < 2:
        st.info("Need at least two run dates in history to compare."); st.stop()

    date_cur = chosen_date if (use_hist and chosen_date) else all_dates[0]
    date_prev = prev_run_date(all_dates, date_cur)
    if not date_prev:
        st.info("No previous run found to compare."); st.stop()

    cur, prv = load_chain_two_days(symbol, date_cur, date_prev)
    if cur.empty or prv.empty:
        st.info("No data for selected comparison dates."); st.stop()

    exp_opts = sorted(set(cur["expiration_date"].unique()).intersection(prv["expiration_date"].unique()))
    exp_cmp = st.selectbox("Expiration", exp_opts, key="exp_cmp")

    x_choice = st.radio("X-axis", ["Strike", "Moneyness (K/S)"], horizontal=True, key="xaxis_compare")
    avg_cp = st.checkbox("Average Calls & Puts together", value=True)
    show_smoothing = st.checkbox("LOWESS smoothing", value=False)
    if show_smoothing:
        frac_cmp = st.slider("Smoothness (LOWESS fraction)", 0.05, 0.5, 0.18, 0.01)

    def prep(df0, label):
        df1 = df0[df0["expiration_date"].eq(exp_cmp)].copy()
        cp_sel = []
        if show_calls: cp_sel.append("C")
        if show_puts:  cp_sel.append("P")
        if cp_sel: df1 = df1[df1["cp"].isin(cp_sel)]
        if x_choice.startswith("Moneyness"):
            df1["xvar"] = df1["strike"].astype(float) / float(spot)
            x_title = "Moneyness (K/S)"
        else:
            df1["xvar"] = df1["strike"].astype(float)
            x_title = "Strike"
        group_cols = ["xvar"] if avg_cp else ["cp","xvar"]
        agg = df1.groupby(group_cols, as_index=False)["iv"].mean().dropna()
        agg["run_label"] = label
        return agg, x_title

    cur_g, x_title = prep(cur, f"{date_cur}")
    prv_g, _      = prep(prv, f"{date_prev}")
    all_g = pd.concat([cur_g, prv_g], ignore_index=True)

    def do_smooth(df_in, fields):
        out = []
        if "cp" in fields:
            for (lab, side), g in df_in.groupby(["run_label","cp"]):
                g = g.sort_values("xvar")
                if len(g) >= 5:
                    sm = lowess(g["iv"].values, g["xvar"].values, frac=frac_cmp, return_sorted=True)
                    out.append(pd.DataFrame({"xvar": sm[:,0], "iv": sm[:,1], "run_label": lab, "cp": side}))
                else:
                    out.append(g[["xvar","iv","run_label","cp"]])
        else:
            for lab, g in df_in.groupby("run_label"):
                g = g.sort_values("xvar")
                if len(g) >= 5:
                    sm = lowess(g["iv"].values, g["xvar"].values, frac=frac_cmp, return_sorted=True)
                    out.append(pd.DataFrame({"xvar": sm[:,0], "iv": sm[:,1], "run_label": lab}))
                else:
                    out.append(g[["xvar","iv","run_label"]])
        return pd.concat(out, ignore_index=True) if out else df_in

    plot_df = all_g.copy()
    if show_smoothing:
        plot_df = do_smooth(plot_df, fields=["cp"] if not avg_cp else [])

    if avg_cp:
        fig = px.line(plot_df, x="xvar", y="iv", color="run_label", template=PX_TEMPLATE,
                      title=f"IV Skew: {date_prev} vs {date_cur} — {exp_cmp}", color_discrete_sequence=["#1E90FF", "#FF4500"])
    else:
        fig = px.line(plot_df, x="xvar", y="iv", color="run_label", line_dash="cp",
                      template=PX_TEMPLATE, title=f"IV Skew: {date_prev} vs {date_cur} — {exp_cmp} (Calls vs Puts)", color_discrete_sequence=["#1E90FF", "#FF4500"])
    fig.update_layout(xaxis_title=x_title, yaxis_title="Implied Volatility", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# ---- Tab 11: 3D Vol Surface ----
with tabs[10]:
    st.subheader("3D Volatility Surface (Moneyness × DTE × IV)")
    tab_help("""
**What this shows**

A **3D surface** of implied volatility across:

- **Moneyness (K/S):** how far strike is from Spot.  
- **Days to Expiration (DTE):** short-dated to long-dated.  
- **IV:** represented by height and color.

This is essentially your **vol surface snapshot**.

**How to read it**

- Look for **ridges and valleys**:  
  - Ridges = areas where options are especially expensive (high IV).  
  - Valleys = relatively cheap regions.  
- Observe how **smile/smirk** changes with time:  
  - Near-term may have a strong downside skew; further out may be flatter.  
- Event-related bumps:  
  - Single DTE regions standing out (like a “vol mountain”) may correspond to event dates.

Use this for an overall **gestalt**:  
> “Where, in (strike, time), is vol rich or cheap right now?”
""")

    surf = df.copy()
    if surf.empty:
        st.info("No data to render."); st.stop()
    today = date.today()
    surf["dte"] = surf["expiration_date"].apply(lambda d: max((d - today).days, 0))
    surf["moneyness"] = surf["strike"] / float(spot)
    surf = surf.dropna(subset=["iv","moneyness","dte"])

    grid = (surf.groupby(["dte","moneyness"], as_index=False)["iv"].mean()
                 .sort_values(["dte","moneyness"]))
    piv = grid.pivot(index="dte", columns="moneyness", values="iv").sort_index()
    X = np.array(piv.columns)   # moneyness
    Y = np.array(piv.index)     # dte
    Z = piv.values              # iv

    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale="Viridis")])
    fig.update_layout(
        template=PX_TEMPLATE, title="IV Surface",
        scene=dict(xaxis_title="Moneyness (K/S)", yaxis_title="Days to Expiration", zaxis_title="Implied Volatility"),
        height=700
    )
    st.plotly_chart(fig, use_container_width=True)

# ---- Tab 12: Summary (Volume, OI, Gamma, GEX) ----
with tabs[11]:
    st.subheader("Daily Summary by Expiration (Volume, OI, Gamma, GEX)")
    tab_help("""
**What this shows**

A **time-series style summary** of key metrics by snapshot and expiration:

- **Volume (Call/Put):** how actively each expiry traded.  
- **OI (Call/Put) & Delta OI:** how positioning size and balance evolve.  
- **Gamma & Net GEX:** aggregate gamma exposure by expiry.  
- **Contracts / ATM close proxy:** rough sense of size and underlying level.

Each row is: **one run timestamp × one expiration**.

**How to read it**

- Track **which expirations** are gaining or losing OI and gamma.  
- See whether **Net GEX** is becoming more positive/negative for front vs back expiries.  
- Spot **regime shifts**: e.g., front month flipping from long to short gamma across days, or big jumps in put OI before events.

Use this tab as your **dashboard of dashboards**:  
> it tells you how the overall options landscape is **drifting over time**, not just in a single snapshot.
""")

    all_dates = load_run_dates(symbol)
    if not all_dates:
        st.info("History table not available.")
        st.stop()

    col1, col2 = st.columns([1, 1])

    with col1:
        max_n = len(all_dates)

        # Safe slider bounds
        min_v = 1
        max_v = max_n
        default_v = min(25, max_v)

        n_rows = st.slider(
            "Number of snapshots (most recent)",
            min_value=min_v,
            max_value=max_v,
            value=default_v,
            step=1,
            key="summary_n",
        )

    with col2:
        exp_mode = st.selectbox(
            "Expiration mode",
            ["All expirations", "Front expiration only"],
            key="summary_exp_mode",
        )

        scale_millions = st.checkbox(
            "Show values in millions",
            value=True,
            key="summary_scale_mio",
        )

    CONTRACT_MULT = symbol_config["contract_multiplier"]
    r_used = r
    spot_used = float(spot)

    rows = []

    selected_dates = all_dates[:n_rows]

    for run_ts in selected_dates:
        df_run = load_chain_by_run(symbol, run_ts).copy()
        if df_run.empty:
            continue

        df_run["expiration_date"] = pd.to_datetime(df_run["expiration_date"]).dt.date
        for c in ["strike", "iv", "volume", "oi", "underlying_px", "delta", "gamma", "option_net"]:
            if c in df_run.columns:
                df_run[c] = pd.to_numeric(df_run[c], errors="coerce")

        if df_run.empty:
            continue

        expirations_all = sorted(df_run["expiration_date"].unique())
        if not expirations_all:
            continue

        if exp_mode == "Front expiration only":
            expirations_use = expirations_all[:1]
        else:
            expirations_use = expirations_all

        snapshot_day = run_ts.date()

        for exp_date in expirations_use:
            df_e = df_run[df_run["expiration_date"].eq(exp_date)].copy()
            if df_e.empty:
                continue

            # ---- Approximate underlying close for this snapshot/expiry ----
            # Prefer real underlying_px if present; fallback to ATM strike proxy
            if "underlying_px" in df_e.columns and df_e["underlying_px"].notna().any():
                try:
                    approx_close = float(df_e["underlying_px"].dropna().iloc[0])
                except Exception:
                    approx_close = float(df_e["strike"].median(skipna=True))
            else:
                if df_e["volume"].notna().any():
                    try:
                        approx_close = float(df_e.loc[df_e["volume"].idxmax(), "strike"])
                    except Exception:
                        approx_close = float(df_e["strike"].median(skipna=True))
                else:
                    approx_close = float(df_e["strike"].median(skipna=True))

            # ---- Volume & OI ----
            vol_call = df_e.loc[df_e["cp"].eq("C"), "volume"].sum()
            vol_put  = df_e.loc[df_e["cp"].eq("P"), "volume"].sum()
            oi_call  = df_e.loc[df_e["cp"].eq("C"), "oi"].sum()
            oi_put   = df_e.loc[df_e["cp"].eq("P"), "oi"].sum()

            delta_oi = oi_call - oi_put
            total_oi = oi_call + oi_put
            ratio_oi = oi_call / total_oi if total_oi > 0 else float("nan")

            # ---- Gamma & GEX ----
            df_e["T"] = df_e["expiration_date"].apply(lambda d: yearfrac(snapshot_day, d))
            df_e["sigma"] = pd.to_numeric(df_e["iv"], errors="coerce")

            if "gamma" in df_e.columns and df_e["gamma"].notna().any():
                df_e["gamma_unit"] = pd.to_numeric(df_e["gamma"], errors="coerce")
            else:
                df_e["gamma_unit"] = np.nan

            missing_gamma = df_e["gamma_unit"].isna()

            if missing_gamma.any():
                df_e.loc[missing_gamma, "gamma_unit"] = df_e.loc[missing_gamma].apply(
                    lambda r2: bs_gamma(spot_used, r2["strike"], r_used, r2["sigma"], r2["T"])
                    if (
                        pd.notna(r2["sigma"])
                        and r2["sigma"] > 0
                        and r2["T"] > 0
                    )
                    else 0.0,
                    axis=1,
                )

            df_e["gamma_unit"] = df_e["gamma_unit"].fillna(0.0)

            df_e["gex"] = -df_e["gamma_unit"] * df_e["oi"].fillna(0) * CONTRACT_MULT * (spot_used ** 2)

            call_gamma = df_e.loc[df_e["cp"].eq("C"), "gex"].sum()
            put_gamma  = df_e.loc[df_e["cp"].eq("P"), "gex"].sum()
            net_gex    = call_gamma + put_gamma

            # Number of contracts (OI is shares → divide by 100)
            contracts = total_oi / CONTRACT_MULT if CONTRACT_MULT else total_oi

            # Gamma ratio
            if abs(put_gamma) > 0:
                gr_pc = abs(call_gamma) / abs(put_gamma)
            else:
                gr_pc = float("nan")

            rows.append({
                "Date": snapshot_day,
                "Expiration": exp_date,
                "Vol.Call": vol_call,
                "Vol.Put": vol_put,
                "OI.Call": oi_call,
                "OI.Put": oi_put,
                "Delta OI": delta_oi,
                "OI Ratio": ratio_oi,
                "Call Gamma": call_gamma,
                "Put Gamma": put_gamma,
                "Net GEX": net_gex,
                "Contracts": contracts,
                "Gamma Ratio (C/P)": gr_pc,
                "Approx Close (ATM)": approx_close,  # <- now always defined
            })


    if not rows:
        st.info("No data available to build the summary.")
        st.stop()

    summary_df = pd.DataFrame(rows).sort_values(["Date", "Expiration"], ascending=[False, True])

    # Scale to millions
    if scale_millions:
        for col in [
            "Vol.Call", "Vol.Put", "OI.Call", "OI.Put",
            "Delta OI", "Call Gamma", "Put Gamma",
            "Net GEX", "Contracts"
        ]:
            summary_df[col] = summary_df[col] / 1_000_000.0

        summary_df = summary_df.rename(columns={
            "Vol.Call": "Vol.Call (M)",
            "Vol.Put": "Vol.Put (M)",
            "OI.Call": "OI.Call (M)",
            "OI.Put": "OI.Put (M)",
            "Delta OI": "Delta OI (M)",
            "Call Gamma": "Call Gamma (M)",
            "Put Gamma": "Put Gamma (M)",
            "Net GEX": "Net GEX (M)",
            "Contracts": "Contracts (M)",
        })

    st.dataframe(summary_df, use_container_width=True, height=540)

# ---- Tab 13: Help & How To Use ----
with tabs[12]:
    st.subheader(f"How to Use {selected_symbol} Terminal")

    st.markdown("""
This dashboard is designed to explore **SPX options positioning, volatility and flows**.

---

### 1️⃣ Global Controls (top & sidebar)

**Sidebar – History**
- **Use history**: when checked, you select a specific snapshot from the database instead of the latest one.
- **Run timestamp (UTC)**: pick which run (download) of the options chain you want to analyze.

**Top controls**
- **Spot (S)**: approximate underlying level. Used for moneyness (K/S), ATM zooms, gamma, GEX, etc.
- **Risk-free (r)**: risk-free rate used in Black–Scholes–based gamma calculations.
- **Calls / Puts toggles**: choose which side is visible across most charts.
- **Expiration**: base expiration used in many tabs for single-expiry views.
- **Strike range slider**: global strike filter for the selected expiration.

**Sidebar – Permalink**
- **Update URL with current filters**: writes the current settings into the URL so you can bookmark or share the exact view.

---

### 2️⃣ Tab-by-Tab Guide

**IV Skew**
- Shows **implied volatility vs strike or moneyness (K/S)** for the selected expiration.
- Use:
  - **X-axis**: switch between Strike or Moneyness.
  - **Show smoothed curve**: LOWESS smoothing for cleaner skew lines.
  - **ATM zoom (±%)**: focus on strikes around Spot.
- The **vertical dotted line** is the Spot (or K/S = 1 if in moneyness).

**OI & Volume**
- Bar charts of **Open Interest** and **Volume** by strike.
- Useful for spotting where **positioning and traded activity** are concentrated.
- Controls:
  - **Log scale (Y)**: helps when a few strikes are very large.
  - **Clip at percentile**: caps extreme spikes so the rest is visible.
  - **ATM zoom (±%)**: focus around Spot.

**Gamma (approx)**
- Estimates **dealer gamma exposure (GEX)** by strike using Black–Scholes gamma.
- Bars:
  - Γ Call, Γ Put, and **Total Γ** (sum).
- Orange line:
  - **Cumulative Γ**, which can highlight the **flip strike** where net gamma changes sign.
- The yellow dashed line marks the **approximate flip level**; cyan dotted line is Spot.

**Term Structure**
- **ATM IV vs Days to Expiration** for all expiries.
- Lets you see if the vol curve is **steep, flat, or inverted**.
- Controls:
  - **Max days to expiration**: limit to short-dated, medium, or long-dated expiries.
  - **Clip extreme IV outliers**: removes bad quotes that distort the curve.
- Reference vertical lines (30/90/180/365 days) help identify maturity buckets.

**Table**
- Raw table of contracts for the selected expiration & filters:
  - `expiration_date, cp, strike, bid, ask, last, iv, delta, gamma, option_net, volume, oi`.
- Good for **sanity checks** and manual inspection.

**Skew Overlay (Multi-Expiry)**
- Compare **IV skews across multiple expirations**.
- Controls:
  - **Select expirations to overlay**: choose which exps to compare (default: highest volume ones).
  - **X-axis**: Strike vs Moneyness.
  - **Average Calls & Puts together**: on = one line per expiry; off = separate C/P lines.
  - **Smoothing**: raw marks or LOWESS-smoothed curves.
- Use this tab to see **term structure of skew** (e.g. front vs back vol behavior).

**OI Change (Flows)**
- Day-over-day **change in open interest** by strike for one expiration.
- Bars show **newly opened or closed positions**:
  - Positive = OI increased; Negative = OI decreased.
- Controls:
  - **Zoom around Spot (±%)**: focus where the action is.
  - **Cap extreme OI spikes**: make normal-sized flows visible.
  - **Top N absolute movers** table lists the largest OI changes with volume and IV.

**Positioning Tilt**
- Net **Call OI − Put OI** by strike for a chosen expiration.
- Above zero: **call-heavy**; below zero: **put-heavy**.
- Useful for quickly seeing **where the market is tilted**.

**Spread Detector**
- Heuristic detector of:
  - **Vertical spreads** (same side, adjacent strikes, similar OI change).
  - **Iron condors** (pair of verticals, calls + puts).
- Output:
  - **Vertical Spread candidates** (strike pairs).
  - **Iron Condor candidates** (4 legs).
- Treat this as a **lead-generation tool**, not strict classification.

**Skew: Today vs Yesterday**
- Overlay IV skews **today vs previous run** for the same expiration.
- Shows how skew **shifted or twisted** over time.
- Controls:
  - **X-axis**: Strike vs Moneyness.
  - **Average Calls & Puts together** or split by C/P.
  - Optional **LOWESS smoothing**.

**3D Vol Surface**
- 3D surface of **IV over Moneyness × Days to Expiration**.
- Helps visualize where vol is **rich or cheap across strikes and maturities**.

**Summary**
- Historical **daily summary by expiration**:
  - Volumes, OI, Delta OI, Call/Put gamma, Net GEX, contracts, approximate close.
- Controls:
  - **Number of snapshots (most recent)**: how many days to include.
  - **Expiration mode**: aggregate all expiries or only the front one.
  - **Show values in millions**: better readability for large numbers.
- Use this to follow **drifts in positioning and gamma over time**.

---

### 3️⃣ Recommended Workflow

1. **Start in Term Structure + IV Skew**  
   Understand the **overall vol level & shape** for a given expiry.

2. **Look at OI & Volume + Positioning Tilt**  
   See where traders are concentrated and whether they are more **call- or put-heavy**.

3. **Check Gamma (approx) & Summary**  
   Identify **gamma flip zones** and the overall **GEX regime** (support / resistance zones).

4. **Use OI Change (Flows) & Spread Detector**  
   See **where fresh positioning is happening** and which **spreads / structures** might be active.

5. **Use Skew Today vs Yesterday**  
   Track **how the skew moves** around events (CPI/FOMC, earnings, etc.).

If you hover around long enough, the charts will talk to you. 😄
""")
    st.markdown("---")
    st.markdown("© 2024 Options Terminal. Built with ❤️ using Streamlit.")

# ---- Tab 14: Probable Levels ---- 

with tabs[13]:  # "Probable Levels"
    st.subheader(f"Most Probable {selected_symbol} Levels (from Options)")

    # 1) Pricing date (today vs chosen history snapshot)
    if use_hist and (chosen_date is not None):
        d0 = chosen_date.date()
    else:
        d0 = date.today()

    # 2) Pick expiration (reuse the global expiries list)
    exps = sorted(df["expiration_date"].unique())
    default_idx = exps.index(exp) if exp in exps else 0
    exp_sel = st.selectbox(
        "Expiration",
        exps,
        index=default_idx,
        key="prob_exp",
    )

    # 3) Days to expiry
    days_to_exp = max((exp_sel - d0).days, 1)

    # 4) ATM IV estimation around current spot
    underlying, atm_iv = get_atm_iv_for_expiry(df, exp_sel, spot)

    if underlying is None:
        st.info("Could not find data for this expiration.")
        st.stop()

    if (atm_iv is None) or np.isnan(atm_iv) or (atm_iv <= 0):
        st.info("Could not estimate ATM IV for this expiration (no near-the-money quotes).")
        st.stop()

    # 5) Compute 1σ and 2σ ranges
    ranges = compute_probable_ranges(underlying, atm_iv, days_to_exp)
    if ranges is None:
        st.info("Insufficient data to compute probable levels.")
        st.stop()

    # 6) Headline metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(f"Current {selected_symbol} (S₀)", f"{ranges['S0']:,.1f}")
    with col2:
        st.metric("ATM IV", f"{ranges['atm_iv']:.2%}")
    with col3:
        st.metric("Days to Expiry", f"{days_to_exp} d")

    st.markdown("### 1σ range (≈68% under normal assumption)")
    col4, col5 = st.columns(2)
    with col4:
        st.metric("1σ Low", f"{ranges['one_sigma_low']:,.1f}")
    with col5:
        st.metric("1σ High", f"{ranges['one_sigma_high']:,.1f}")

    st.markdown("### 2σ range (≈95% under normal assumption)")
    col6, col7 = st.columns(2)
    with col6:
        st.metric("2σ Low", f"{ranges['two_sigma_low']:,.1f}")
    with col7:
        st.metric("2σ High", f"{ranges['two_sigma_high']:,.1f}")

    # 7) Distribution sketch (normal approximation around S0)
    st.markdown("### Distribution Sketch (Normal Approximation on Price)")
    sd_price = ranges["S0"] * ranges["atm_iv"] * np.sqrt(ranges["T_years"])
    price_grid = np.linspace(ranges["two_sigma_low"], ranges["two_sigma_high"], 200)
    pdf = np.exp(-0.5 * ((price_grid - ranges["S0"]) / sd_price) ** 2)

    pdf_df = pd.DataFrame({
        "price": price_grid,
        "density": pdf / pdf.max()  # normalize for plotting
    }).set_index("price")

    st.line_chart(pdf_df)
    st.caption("Illustrative only: assumes normal distribution on price using ATM IV.")

    # 8) Highest OI / Gamma strikes inside the 1σ band
    st.markdown("### Highest OI / Gamma Strikes inside 1σ Range")

    gravity = top_gravity_levels(
        df,
        exp_sel,
        ranges["one_sigma_low"],
        ranges["one_sigma_high"],
        spot,
        top_n=5,
    )

    if gravity.empty:
        st.caption("No strikes in range or no OI/gamma data available.")
    else:
        st.dataframe(gravity, use_container_width=True)
