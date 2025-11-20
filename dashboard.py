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



# ---------- Page / Theme ----------
PX_TEMPLATE = "plotly_dark"
st.set_page_config(page_title="SPX Terminal", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<style>div.block-container{padding-top:1.0rem;padding-bottom:0.5rem;}</style>", unsafe_allow_html=True)
st.title("SPX Terminal • Options")

# ---------- DB ----------
TABLE_LATEST = "spx_chain"       # latest snapshot table (normalized rows: cp in separate rows)
TABLE_HIST   = "spx_chain"  # historical snapshots with run_date
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
def load_run_dates():
    """
    Return list of distinct run timestamps (run_ts) as Python datetime objects,
    newest first.
    """
    try:
        df = q(f"SELECT DISTINCT run_ts FROM {TABLE_HIST} ORDER BY run_ts DESC")
        return list(df["run_ts"])
    except Exception:
        return []

def prev_run_date(dates, cur_dt):
    if not dates or cur_dt not in dates:
        return None
    idx = dates.index(cur_dt)
    return dates[idx + 1] if idx + 1 < len(dates) else None

@st.cache_data(ttl=180, show_spinner=False)
def load_chain_by_run(run_ts):
    return q(f"""
        SELECT run_ts, expiration_date, strike, cp, last, bid, ask, volume, oi, iv
        FROM {TABLE_HIST}
        WHERE run_ts = %s
        ORDER BY expiration_date, strike, cp
    """, (run_ts,))


@st.cache_data(ttl=180, show_spinner=False)
def load_chain_two_days(run_date_cur, run_date_prev):
    cur = load_chain_by_run(run_date_cur)
    prv = load_chain_by_run(run_date_prev) if run_date_prev else pd.DataFrame(columns=cur.columns)
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

# ---------- Load initial data ----------
dates = load_run_dates()

with st.sidebar:
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
            key="run_ts",
            format_func=lambda x: x.strftime("%Y-%m-%d %H:%M:%S"),
        )
    else:
        st.caption("No history snapshots found in table spx_chain_raw_hist.")
        chosen_date = None


@st.cache_data(ttl=120, show_spinner=False)
def load_latest():
    return q(f"SELECT run_ts, expiration_date, strike, cp, last, bid, ask, volume, oi, iv FROM {TABLE_LATEST}")

df = load_chain_by_run(chosen_date) if (use_hist and chosen_date) else load_latest()


df = load_chain_by_run(chosen_date) if (use_hist and chosen_date) else load_latest()
if df.empty:
    st.warning("No rows returned. Check tables/permissions.")
    st.stop()

# Dtypes
df["expiration_date"] = pd.to_datetime(df["expiration_date"]).dt.date
for c in ["strike","last","bid","ask","iv","volume","oi"]:
    if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")

# ---- Better SPOT detection from the options chain (ATM strike proxy) ----

if "volume" in df.columns and df["volume"].notna().any():
    try:
        # ATM = strike of highest volume option
        atm_strike = float(df.loc[df["volume"].idxmax(), "strike"])
    except Exception:
        atm_strike = float(df["strike"].median(skipna=True))
else:
    atm_strike = float(df["strike"].median(skipna=True))

default_spot = atm_strike
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
    "Summary"
])


# ---- Tab 1: IV Skew ----
with tabs[0]:
    st.subheader("Volatility Skew")

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
    CONTRACT_MULT = 100.0
    today = date.today()

    # --- Controls for style/zoom ---
    colg1, colg2 = st.columns([1, 1])
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

    g["gamma"] = g.apply(
        lambda r2: bs_gamma(spot, r2["strike"], r, r2["sigma"], r2["T"])
        if (r2["sigma"] and r2["sigma"] > 0 and r2["T"] > 0) else 0.0,
        axis=1,
    )
    g["gex"] = -g["gamma"] * g["oi"].fillna(0) * CONTRACT_MULT * (spot ** 2)

    # --- Aggregate by strike / cp ---
    grp = g.groupby(["strike", "cp"], as_index=False)["gex"].sum()
    pivot = grp.pivot(index="strike", columns="cp", values="gex").fillna(0.0)
    pivot = pivot.rename(columns={"C": "gamma_call", "P": "gamma_put"})
    if "gamma_call" not in pivot.columns:
        pivot["gamma_call"] = 0.0
    if "gamma_put" not in pivot.columns:
        pivot["gamma_put"] = 0.0

    pivot = pivot.reset_index().sort_values("strike")

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

    # Bars for Call / Put / Total
    fig_gex.add_bar(
        x=pivot["strike"],
        y=pivot["gamma_call"],
        name="Γ Call",
        marker_color="green",
    )
    fig_gex.add_bar(
        x=pivot["strike"],
        y=pivot["gamma_put"],
        name="Γ Put",
        marker_color="red",
    )
    fig_gex.add_bar(
        x=pivot["strike"],
        y=pivot["total_gamma"],
        name="Total Γ",
        marker_color="purple",
        opacity=0.6,
    )

    # Cumulative curve (orange) on second y-axis
    fig_gex.add_scatter(
        x=pivot["strike"],
        y=pivot["cum_gamma"],
        name="Curve Γ (cum)",
        mode="lines",
        line=dict(color="orange", width=2),
        yaxis="y2",
    )

    # Flip vertical line (only if sign change exists)
    if flip_strike is not None:
        fig_gex.add_vline(
            x=flip_strike,
            line_dash="dash",
            line_width=2,
            line_color="yellow",
        )
        fig_gex.add_annotation(
            x=flip_strike,
            y=pivot["cum_gamma"].min(),
            text=f"Flip ~ {flip_strike:.0f}",
            showarrow=False,
            yshift=20,
        )

    # Spot line (cyan dotted)
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

    fig_gex.update_layout(
        template=PX_TEMPLATE,
        title="Gamma by Strike (Calls / Puts / Total) + Cumulative Curve",
        barmode="group",
        hovermode="x unified",
        xaxis=dict(title="Strike"),
        yaxis=dict(title=y_title),
        yaxis2=dict(
            title=y2_title,
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    st.plotly_chart(fig_gex, use_container_width=True)


# ---- Tab 4: Term Structure ----
with tabs[3]:
    st.subheader("ATM IV Term Structure")

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
    cols = ["expiration_date","cp","strike","bid","ask","last","iv","volume","oi"]
    st.dataframe(dfe[cols].sort_values(["cp","strike"]), use_container_width=True, height=420)

# ---- Tab 6: Skew Overlay (Multi-Expiry) ----
with tabs[5]:
    st.subheader("Multi-Expiry Skew Overlay (IV vs Strike or Moneyness)")

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
    all_dates = load_run_dates()
    if not all_dates:
        st.info("History table not available."); st.stop()
    date_cur = chosen_date if (use_hist and chosen_date) else all_dates[0]
    date_prev = prev_run_date(all_dates, date_cur)

    cur, prv = load_chain_two_days(date_cur, date_prev)
    if cur.empty:
        st.info("No rows for selected date."); st.stop()

    exp_sel = st.selectbox("Expiration", sorted(cur["expiration_date"].unique()), key="exp_oi_change")
    key_cols = ["expiration_date","strike","cp"]

    cur_e = cur[cur["expiration_date"].eq(exp_sel)][key_cols + ["oi","volume","iv"]].rename(columns={"oi":"oi_cur","volume":"vol_cur","iv":"iv_cur"})
    prv_e = prv[prv["expiration_date"].eq(exp_sel)][key_cols + ["oi"]].rename(columns={"oi":"oi_prev"})

    merged = pd.merge(cur_e, prv_e, on=key_cols, how="left")
    for c in ["oi_prev","oi_cur","vol_cur","iv_cur"]: merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0)
    merged["oi_change"] = merged["oi_cur"] - merged["oi_prev"]

    hm = merged.groupby(["strike","cp"], as_index=False)["oi_change"].sum()
    fig_hm = px.bar(hm, x="strike", y="oi_change", color="cp", barmode="group",
                    template=PX_TEMPLATE, title=f"OI Change by Strike — {date_prev or '?'} → {date_cur}", color_discrete_sequence=["#1E90FF", "#FF4500"])
    fig_hm.update_layout(hovermode="x unified")
    fig_hm = add_spot_line(fig_hm, spot)
    st.plotly_chart(fig_hm, use_container_width=True)

    topN = st.slider("Show top N absolute movers", 10, 200, 50, 5)
    movers = merged.reindex(merged["oi_change"].abs().sort_values(ascending=False).index).head(topN)
    st.dataframe(movers[["expiration_date","cp","strike","oi_prev","oi_cur","oi_change","vol_cur","iv_cur"]], use_container_width=True, height=380)

# ---- Tab 8: Positioning Tilt ----
with tabs[7]:
    st.subheader("Net Positioning Tilt (Call OI − Put OI)")
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
    all_dates = load_run_dates()
    if not all_dates:
        st.info("History table not available."); st.stop()
    date_cur = chosen_date if (use_hist and chosen_date) else all_dates[0]
    date_prev = prev_run_date(all_dates, date_cur)

    cur, prv = load_chain_two_days(date_cur, date_prev)
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
    all_dates = load_run_dates()
    if not all_dates or len(all_dates) < 2:
        st.info("Need at least two run dates in history to compare."); st.stop()

    date_cur = chosen_date if (use_hist and chosen_date) else all_dates[0]
    date_prev = prev_run_date(all_dates, date_cur)
    if not date_prev:
        st.info("No previous run found to compare."); st.stop()

    cur, prv = load_chain_two_days(date_cur, date_prev)
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

    all_dates = load_run_dates()
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

    CONTRACT_MULT = 100.0
    r_used = r
    spot_used = float(spot)

    rows = []

    selected_dates = all_dates[:n_rows]

    for run_ts in selected_dates:
        df_run = load_chain_by_run(run_ts).copy()
        if df_run.empty:
            continue

        df_run["expiration_date"] = pd.to_datetime(df_run["expiration_date"]).dt.date
        for c in ["strike", "iv", "volume", "oi"]:
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

            # Approximate underlying close using ATM strike (highest volume)
            if df_e["volume"].notna().any():
                try:
                    atm_k = float(df_e.loc[df_e["volume"].idxmax(), "strike"])
                except Exception:
                    atm_k = float(df_e["strike"].median(skipna=True))
            else:
                atm_k = float(df_e["strike"].median(skipna=True))

            # Volume & OI
            vol_call = df_e.loc[df_e["cp"].eq("C"), "volume"].sum()
            vol_put  = df_e.loc[df_e["cp"].eq("P"), "volume"].sum()
            oi_call  = df_e.loc[df_e["cp"].eq("C"), "oi"].sum()
            oi_put   = df_e.loc[df_e["cp"].eq("P"), "oi"].sum()

            delta_oi = oi_call - oi_put
            total_oi = oi_call + oi_put
            ratio_oi = oi_call / total_oi if total_oi > 0 else float("nan")

            # Gamma & GEX
            df_e["T"] = df_e["expiration_date"].apply(lambda d: yearfrac(snapshot_day, d))
            df_e["sigma"] = pd.to_numeric(df_e["iv"], errors="coerce")

            df_e["gamma_unit"] = df_e.apply(
                lambda r2: bs_gamma(spot_used, r2["strike"], r_used, r2["sigma"], r2["T"])
                if (r2["sigma"] and r2["sigma"] > 0 and r2["T"] > 0) else 0.0,
                axis=1,
            )

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
                "Approx Close (ATM)": atm_k,
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

