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

# ---------- Page / Theme ----------
PX_TEMPLATE = "plotly_dark"
st.set_page_config(page_title="SPX Terminal", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<style>div.block-container{padding-top:1.0rem;padding-bottom:0.5rem;}</style>", unsafe_allow_html=True)
st.title("SPX Terminal • Options")

# ---------- DB ----------
TABLE_LATEST = "spx_chain"       # latest snapshot table (normalized rows: cp in separate rows)
TABLE_HIST   = "spx_chain_raw_hist"  # historical snapshots with run_date
def get_conn():
    return psycopg2.connect(
        host=st.secrets["DB_HOST"],
        dbname=st.secrets["DB_NAME"],
        user=st.secrets["DB_USER"],
        password=st.secrets["DB_PASS"],
        sslmode=st.secrets.get("DB_SSLMODE", "require"),
        port=5432,
    )

@st.cache_data(ttl=180, show_spinner=False)
def q(sql, params=None):
    with get_conn() as conn:
        return pd.read_sql(sql, conn, params=params)

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
    s["use_hist"]   = _to_bool(qp.get("hist", ["0"])[0], False)
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
    try:
        df = q(f"SELECT DISTINCT run_date FROM {TABLE_HIST} ORDER BY run_date DESC")
        return [r.strftime("%Y-%m-%d") if hasattr(r, "strftime") else str(r) for r in df["run_date"]]
    except Exception:
        return []

def prev_run_date(dates, cur_date):
    if not dates or cur_date not in dates: return None
    idx = dates.index(cur_date)
    return dates[idx+1] if idx+1 < len(dates) else None

@st.cache_data(ttl=180, show_spinner=False)
def load_chain_by_run(run_date):
    return q(f"""
        SELECT run_ts, run_date, expiration_date, strike, cp, last, bid, ask, volume, oi, iv
        FROM {TABLE_HIST}
        WHERE run_date = %s
    """, (run_date,))

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
c0,c1,c2 = st.columns([1,1,5])
with c0:
    use_hist = st.toggle("History", value=URL_DEFAULTS.get("use_hist", bool(dates)),
                         help="Read from historical table")
with c1:
    default_idx = 0
    if (use_hist and dates) and URL_DEFAULTS.get("run_date") in dates:
        default_idx = dates.index(URL_DEFAULTS["run_date"])
    chosen_date = st.selectbox("Run date (NY)", dates, index=default_idx) if (use_hist and dates) else None

@st.cache_data(ttl=120, show_spinner=False)
def load_latest():
    return q(f"SELECT run_ts, expiration_date, strike, cp, last, bid, ask, volume, oi, iv FROM {TABLE_LATEST}")

df = load_chain_by_run(chosen_date) if (use_hist and chosen_date) else load_latest()
if df.empty:
    st.warning("No rows returned. Check tables/permissions.")
    st.stop()

# Dtypes
df["expiration_date"] = pd.to_datetime(df["expiration_date"]).dt.date
for c in ["strike","last","bid","ask","iv","volume","oi"]:
    if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")

default_spot = df["last"].median(skipna=True)
if not np.isfinite(default_spot):
    default_spot = float(df["strike"].median())

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
    exp = st.selectbox("Expiration", expiries, index=exp_idx)

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

# ---------- Tabs ----------
tabs = st.tabs([
    "IV Skew", "OI & Volume", "Gamma (approx)", "Term Structure", "Table",
    "Skew Overlay (Multi-Expiry)", "OI Change (Flows)", "Positioning Tilt",
    "Spread Detector", "Skew: Today vs Yesterday", "3D Vol Surface"
])

# ---- Tab 1: IV Skew ----
with tabs[0]:
    st.subheader("Volatility Skew")
    fig = px.scatter(dfe, x="strike", y="iv", color="cp", template=PX_TEMPLATE,
                     title=f"IV vs Strike · {exp}")
    fig.update_traces(marker=dict(size=6, opacity=0.9))
    st.plotly_chart(fig, use_container_width=True)


# ---- Tab 2: OI & Volume ----
# ---- Tab 2: OI & Volume ----
with tabs[1]:
    st.subheader("Open Interest & Volume")

    # ── Controls (local to this tab) ───────────────────────────────────────────
    colc1, colc2, colc3 = st.columns([1, 2, 2])
    with colc1:
        use_log = st.checkbox("Log scale (Y)", value=True,
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
            template=PX_TEMPLATE, title="Open Interest by Strike"
        )
        fig_oi.update_yaxes(type="log" if use_log else "linear", title="OI")
        st.plotly_chart(fig_oi, use_container_width=True)

    with colB:
        fig_vol = px.bar(
            vol_df, x="strike", y="vol_capped", color="cp", barmode="group",
            template=PX_TEMPLATE, title="Volume by Strike (today)"
        )
        fig_vol.update_yaxes(type="log" if use_log else "linear", title="Volume")
        st.plotly_chart(fig_vol, use_container_width=True)


# ---- Tab 3: Gamma (approx) ----
with tabs[2]:
    st.subheader("Dealer Gamma Exposure (approx)")
    CONTRACT_MULT = 100.0
    today = date.today()

    g = dfe.copy()
    g["T"] = g["expiration_date"].apply(lambda d: yearfrac(today, d))
    g["sigma"] = pd.to_numeric(g["iv"], errors="coerce")
    g["gamma"] = g.apply(
        lambda r2: bs_gamma(spot, r2["strike"], r, r2["sigma"], r2["T"])
        if (r2["sigma"] and r2["sigma"]>0 and r2["T"]>0) else 0.0,
        axis=1
    )
    g["gex"] = - g["gamma"] * g["oi"].fillna(0) * CONTRACT_MULT * (spot**2)

    curve = g.groupby("strike", as_index=False)["gex"].sum().sort_values("strike")
    curve["cum_gex"] = curve["gex"].cumsum()

    # Flip level ~ zero-cross of cum_gex
    flip_strike = None
    sgn = np.sign(curve["cum_gex"])
    change_idx = np.where(np.diff(sgn) != 0)[0]
    if len(change_idx):
        i = change_idx[0]
        x0, y0 = curve.loc[i,   ["strike","cum_gex"]]
        x1, y1 = curve.loc[i+1, ["strike","cum_gex"]]
        if (y1 - y0) != 0:
            flip_strike = float(x0 - y0*(x1-x0)/(y1-y0))
        else:
            flip_strike = float(curve.loc[i, "strike"])

    colA, colB = st.columns(2)
    with colA:
        fig_g = px.line(curve, x="strike", y="gex", template=PX_TEMPLATE, title="GEX by Strike")
        st.plotly_chart(fig_g, use_container_width=True)
    with colB:
        fig_c = px.line(curve, x="strike", y="cum_gex", template=PX_TEMPLATE, title="Cumulative GEX (zero-cross ≈ flip)")
        if flip_strike is not None:
            fig_c.add_vline(x=flip_strike, line_dash="dash", line_width=2)
            fig_c.add_annotation(x=flip_strike, y=curve["cum_gex"].min(), text=f"Flip ~ {flip_strike:.1f}", showarrow=False, yshift=20)
        st.plotly_chart(fig_c, use_container_width=True)

# ---- Tab 4: Term Structure ----
with tabs[3]:
    st.subheader("ATM IV Term Structure")
    atm_rows = []
    for e in sorted(df["expiration_date"].unique()):
        iv_atm = nearest_strike_iv(df, e, float(spot))
        if np.isfinite(iv_atm):
            atm_rows.append({"expiration_date": e, "atm_iv": iv_atm})
    term = pd.DataFrame(atm_rows).sort_values("expiration_date")
    fig_t = px.line(term, x="expiration_date", y="atm_iv", markers=True,
                    template=PX_TEMPLATE, title="ATM IV Across Expirations")
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
    default_multi = URL_DEFAULTS.get("ovr_exps") or exps_all[:min(6, len(exps_all))]
    exps_sel = st.multiselect("Select expirations to overlay", exps_all, default=default_multi)

    x_choice = st.radio("X-axis", ["Strike", "Moneyness (K/S)"], horizontal=True,
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

    exp_sel = st.selectbox("Expiration", sorted(cur["expiration_date"].unique()))
    key_cols = ["expiration_date","strike","cp"]

    cur_e = cur[cur["expiration_date"].eq(exp_sel)][key_cols + ["oi","volume","iv"]].rename(columns={"oi":"oi_cur","volume":"vol_cur","iv":"iv_cur"})
    prv_e = prv[prv["expiration_date"].eq(exp_sel)][key_cols + ["oi"]].rename(columns={"oi":"oi_prev"})

    merged = pd.merge(cur_e, prv_e, on=key_cols, how="left")
    for c in ["oi_prev","oi_cur","vol_cur","iv_cur"]: merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0)
    merged["oi_change"] = merged["oi_cur"] - merged["oi_prev"]

    hm = merged.groupby(["strike","cp"], as_index=False)["oi_change"].sum()
    fig_hm = px.bar(hm, x="strike", y="oi_change", color="cp", barmode="group",
                    template=PX_TEMPLATE, title=f"OI Change by Strike — {date_prev or '?'} → {date_cur}")
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

    fig_tilt = px.bar(tilt, x="strike", y="tilt", template=PX_TEMPLATE, title=f"Net Tilt by Strike — {exp_tilt}")
    fig_tilt.add_hline(y=0, line_dash="dash")
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
    exp_cmp = st.selectbox("Expiration", exp_opts)

    x_choice = st.radio("X-axis", ["Strike", "Moneyness (K/S)"], horizontal=True)
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
                      title=f"IV Skew: {date_prev} vs {date_cur} — {exp_cmp}")
    else:
        fig = px.line(plot_df, x="xvar", y="iv", color="run_label", line_dash="cp",
                      template=PX_TEMPLATE, title=f"IV Skew: {date_prev} vs {date_cur} — {exp_cmp} (Calls vs Puts)")
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
