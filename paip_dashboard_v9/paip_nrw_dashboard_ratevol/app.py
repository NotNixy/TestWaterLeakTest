"""
PAIP Non-Revenue Water — Leakage Intervention Priority Dashboard
================================================================
Turns PAIP's published monthly production and billing figures into a ranked,
volume-weighted repair schedule.

Run:  streamlit run app.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import spearmanr, kendalltau

import theme as theme_mod
import train_models as tm

DATA = Path(__file__).parent / "data"
ASSETS = Path(__file__).parent / "assets"

st.set_page_config(page_title="PAIP NRW — Intervention Priority",
                   page_icon=":material/water_drop:", layout="wide",
                   initial_sidebar_state="expanded")


def detected_mode() -> str:
    """Read the browser/OS colour scheme. Streamlit exposes the active theme
    through st.context; when no theme is pinned in config.toml that follows
    `prefers-color-scheme`, which is what makes Auto track the system."""
    try:
        t = getattr(st.context, "theme", None)
        val = getattr(t, "type", None) if t is not None else None
        if val in ("light", "dark"):
            return val
    except Exception:
        pass
    return "light"


_detected = detected_mode()
_pref = st.session_state.get("appearance", "Light")
MODE = theme_mod.resolve_mode(_pref, _detected)
T = theme_mod.Theme(MODE)
st.markdown(T.css, unsafe_allow_html=True)
st.markdown("<style>.card-t { font-weight: 700 !important; } "
            ".kpi-l { font-weight: 700 !important; }</style>",
            unsafe_allow_html=True)

PLOT_CFG = {"displayModeBar": False, "responsive": True}

LIPS_COMPONENTS = {
    "nrw_per_km_m3": ("Loss Density", "m³ of NRW lost per km of pipe network"),
    "bursts_per_100km": ("Burst Rate", "Pipe bursts recorded per 100 km"),
    "plant_age_yr": ("Plant Age", "Water treatment plant age in years"),
    "account_density": ("Account Density", "Customer accounts per km of main"),
}

DEFAULT_WEIGHTS = {
    "nrw_per_km_m3": 40,
    "bursts_per_100km": 25,
    "plant_age_yr": 20,
    "account_density": 15,
}


# ==========================================================================
# Data
# ==========================================================================

@st.cache_data
def load():
    m = pd.read_csv(DATA / "nrw_plant_month.csv", parse_dates=["date"])
    y = pd.read_csv(DATA / "nrw_plant_year.csv")
    x = pd.read_csv(DATA / "plant_crosswalk.csv")
    q = pd.read_csv(DATA / "data_quality.csv")
    v = pd.read_csv(DATA / "missing_values.csv")
    pred_path = DATA / "lips_2026_prediction.csv"
    df_priority = pd.read_csv(pred_path) if pred_path.exists() else None

    backtest_path = DATA / "forecast_backtest.csv"
    df_backtest = pd.read_csv(backtest_path) if backtest_path.exists() else None
    if df_backtest is not None and df_backtest.empty:
        df_backtest = None

    return m, y, x, q, v, df_priority, df_backtest

@st.cache_data
def load_ml():
    try:
        p = pd.read_csv(DATA / "ml_plant.csv")
        mm = pd.read_csv(DATA / "ml_monthly.csv", parse_dates=["date"])
        met = json.loads((DATA / "model_metrics.json").read_text())
        return p, mm, met
    except FileNotFoundError:
        return None, None, None


@st.cache_data
def cluster_at(k: int, yr: int):
    py = yearly[yearly.year == yr].copy()
    scored, profile, sil, best = tm.archetypes(py, k=k)
    return scored[["plant", "cluster", "archetype"]], profile, sil, best


@st.cache_data
def load_burst():
    try:
        p = pd.read_csv(DATA / "burst_predictions.csv", parse_dates=["date"])
        h = pd.read_csv(DATA / "burst_history.csv", parse_dates=["date"])
        m = json.loads((DATA / "burst_metrics.json").read_text())
        return p, h, m
    except FileNotFoundError:
        return None, None, None


@st.cache_data
def load_coverage():
    p = DATA / "year_coverage.csv"
    if p.exists():
        return pd.read_csv(p)
    return None


monthly, yearly, crosswalk, quality, missing, prediction, forecast_backtest = load()
ml_plant, ml_monthly, ml_metrics = load_ml()
HAS_ML = ml_plant is not None
burst_pred, burst_hist, burst_metrics = load_burst()
HAS_BURST = burst_pred is not None
coverage = load_coverage()

YEARS = sorted(int(y) for y in monthly.year.unique())
YEAR_MIN, YEAR_MAX = YEARS[0], YEARS[-1]
YEAR_SPAN = f"{YEAR_MIN}" if YEAR_MIN == YEAR_MAX else f"{YEAR_MIN}–{YEAR_MAX}"
ML_YEAR = int(ml_metrics.get("focus_year", YEAR_MAX)) if HAS_ML else None

if coverage is not None:
    MONTHS_BY_YEAR = dict(zip(coverage.year.astype(int), coverage.months.astype(int)))
else:
    MONTHS_BY_YEAR = (monthly.groupby("year").date.apply(lambda s: s.dt.month.nunique())
                      .astype(int).to_dict())


def year_label(y: int) -> str:
    m = MONTHS_BY_YEAR.get(int(y), 12)
    return f"{int(y)}" if m >= 12 else f"{int(y)} · {m} of 12 months"


def is_partial(y: int) -> bool:
    return MONTHS_BY_YEAR.get(int(y), 12) < 12


def percentile_rank(s):
    return s.rank(pct=True, method="average") * 100


@st.cache_data
def score_lips(df: pd.DataFrame, weights: tuple) -> pd.DataFrame:
    w = dict(weights)
    total = sum(w.values()) or 1
    out = df.copy()

    if "account_density" not in out.columns and "customer_accounts" in out.columns and "pipe_length_km" in out.columns:
        out["account_density"] = out.customer_accounts / out.pipe_length_km

    score = pd.Series(0.0, index=out.index)
    for col, wt in w.items():
        if col in out.columns:
            pr = percentile_rank(out[col])
            out[f"pr_{col}"] = pr
            score += pr * (wt / total)

    out["lips"] = score.round(2)
    out = out.sort_values(["lips", "nrw_per_km_m3", "nrw_m3"], ascending=False)
    out["lips_rank"] = np.arange(1, len(out) + 1)
    out["volume_rank"] = out.nrw_m3.rank(ascending=False, method="first").astype(int)
    out["rate_rank"] = out.nrw_pct.rank(ascending=False, method="first").astype(int)
    out["rank_gap"] = out.rate_rank - out.volume_rank
    return out


def chart(fig, **kw):
    fig.update_layout(paper_bgcolor=T.SURFACE, plot_bgcolor=T.SURFACE)
    kw.setdefault("width", "stretch")
    kw.setdefault("config", PLOT_CFG)
    kw.setdefault("theme", None)
    return st.plotly_chart(fig, **kw)


def card(title, sub_=None):
    c = st.container(border=True)
    with c:
        st.markdown(f'<div class="card-t">{title}</div>'
                    + (f'<div class="card-s">{sub_}</div>' if sub_ else ""),
                    unsafe_allow_html=True)
    return c


def table(df, cols, height=520, bar=None, bar_max=100.0):
    """A palette-styled HTML table.

    `cols` is a list of (column, header, formatter). `bar` names one column to
    draw as a progress bar rather than a number — the equivalent of
    st.column_config.ProgressColumn, but in colours we control.
    """
    head = "".join(f"<th>{h}</th>" for _c, h, _f in cols)
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c, _h, f in cols:
            v = r[c]
            if c == bar:
                pct = max(0.0, min(100.0, float(v) / bar_max * 100.0))
                cells.append(
                    f'<td class="num"><span class="tbar">'
                    f'<span class="tbar-f" style="width:{pct:.1f}%"></span>'
                    f'</span><span class="tbar-v">{f(v)}</span></td>')
            else:
                txt = f(v)
                cls = "num" if isinstance(v, (int, float)) else ""
                cells.append(f'<td class="{cls}">{txt}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    # A size class, not just an inline height: Streamlit measures a markdown
    # container before layout and collapses it, so whatever follows the table
    # paints on top of it. The class gives CSS something to reserve height
    # against.
    cls = ("tbl-xs" if height <= 230 else
           "tbl-s" if height <= 330 else
           "tbl-m" if height <= 470 else "tbl-l")
    st.markdown(
        f'<div class="tblwrap {cls}" style="max-height:{height}px">'
        f'<table class="tbl"><thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>',
        unsafe_allow_html=True)


def fmt(n, dp=0):
    return f"{n:,.{dp}f}"


def m3(n):
    if abs(n) >= 1e9:
        return f"{n/1e9:,.2f}B"
    if abs(n) >= 1e6:
        return f"{n/1e6:,.1f}M"
    if abs(n) >= 1e3:
        return f"{n/1e3:,.0f}k"
    return f"{n:,.0f}"


def rm(n):
    if abs(n) >= 1e9:
        return f"{n/1e9:,.2f}B"
    if abs(n) >= 1e6:
        return f"{n/1e6:,.1f}M"
    return f"{n:,.0f}"


# ==========================================================================
# Topbar (Branding Moved Here)
# ==========================================================================

top_col1, top_col2 = st.columns([0.08, 0.92])
with top_col1:
    st.image(ASSETS / "logo.png", width=54)
with top_col2:
    st.markdown('<div class="sb-brand" style="font-size: 1.5rem; font-weight: 700 !important;">PENGURUSAN AIR PAHANG BERHAD</div>'
                '<div class="sb-sub" style="font-size: 0.9rem;">Non-Revenue Water intervention targeting</div>',
                unsafe_allow_html=True)


# ==========================================================================
# Sidebar (Filters & Nav)
# ==========================================================================

NAV = [
    ("OVERVIEW", [(":material/dashboard: At a glance", "cmd")]),
    ("PRIORITY", [(":material/leaderboard: Ranking", "rank"),
                  (":material/table_eye: Full schedule", "sched"),
                  (":material/stacked_line_chart: Recovery curve", "curve"),
                  (":material/online_prediction: Prediction", "pred")]),
    ("PLANT PROFILE", [(":material/article: Summary", "psum"),
                       (":material/diagnosis: Diagnosis", "pmodel"),
                       (":material/history: History", "phist"),
                      (":material/compare: Comparison", "pcomp")]),
]
SEC_PRIORITY = {"rank", "sched", "curve", "pred"}
SEC_LOSS = {"ratevol", "comp"}
SEC_PLANT = {"psum", "pmodel", "phist", "pcomp"}

if "view" not in st.session_state:
    st.session_state.view = "cmd"

with st.sidebar:
    _b = st.columns([0.36, 0.64])
    with _b[0]:
        st.image(ASSETS / "logo.png", width=54)
    with _b[1]:
        st.markdown('<div class="sb-brand">PAIP</div>'
                    '<div class="sb-sub">Non-Revenue Water<br>intervention targeting</div>',
                    unsafe_allow_html=True)
    st.markdown('<div class="waverule"></div>', unsafe_allow_html=True)

    year = st.selectbox("Reporting year", sorted(YEARS, reverse=True), index=0,
                        format_func=year_label)

    _p_all = sorted(yearly.plant.unique())
    _d_all = sorted(yearly.district.unique())

    def _picked(prefix, options):
        on = [o for o in options if st.session_state.get(f"{prefix}{o}", False)]
        return on or list(options)

    _n_on = sum(1 for pre, opts in (("f_p_", _p_all), ("f_d_", _d_all))
                for o in opts if st.session_state.get(f"{pre}{o}", False))
    _label = f"Filters · {_n_on}" if _n_on else "Filters"

    with st.popover(_label, width="stretch"):
        st.markdown('<div class="pop-h">Tick to narrow. Nothing ticked means '
                    'everything.</div>', unsafe_allow_html=True)

        filters = st.radio("Filter by:", ["Plant name", "Districts"], horizontal=True, key="filter")
      
        def _clear():
            for pre, opts in (("f_p_", _p_all), ("f_d_", _d_all)):
                for o in opts:
                    st.session_state[f"{pre}{o}"] = False

        st.button("Clear all", key="f_clear", type="tertiary", on_click=_clear)

        if filters == "Plant name":
            target = [("Plant Name", "f_p_", _p_all, 4)]
        elif filters == "Districts":
            target = [("District", "f_d_", _d_all, 4)]
        else:
            target = []

        for _title, _pre, _opts, _cols in target:
            st.markdown(f'<div class="pop-s">{_title}</div>',
                        unsafe_allow_html=True)
            _cc = st.columns(_cols)
            for _i, _o in enumerate(_opts):
                with _cc[_i % _cols]:
                    st.checkbox(_o, key=f"{_pre}{_o}")

    plants = _picked("f_p_", _p_all)
    districts = _picked("f_d_", _d_all)

    st.markdown('<div class="waverule"></div>', unsafe_allow_html=True)

    def _go(k):
        st.session_state.view = k

    for _sec, _items in NAV:
        st.markdown(f'<div class="nav-h">{_sec}</div>', unsafe_allow_html=True)
        for _label, _key in _items:
            _active = st.session_state.view == _key
            st.markdown(
                f'<div class="nav-row{" nav-on" if _active else ""}"></div>',
                unsafe_allow_html=True)
            st.button(_label, key=f"nav_{_key}", width="stretch",
                      type="primary" if _active else "tertiary",
                      on_click=_go, args=(_key,))

    st.markdown('<div class="waverule"></div>', unsafe_allow_html=True)
    st.radio("Appearance", ["Light", "Dark", "Auto"], horizontal=True,
             key="appearance",
             help=f"Auto follows your system setting (detected: {_detected}).")
    st.caption(f"Source: Pengurusan Air Pahang Berhad (PAIP).")

if is_partial(year):
    st.markdown(
        f'<div class="caption" style="margin:2px 0 6px 0">⚠ {year} is still in '
        f'progress ({MONTHS_BY_YEAR[year]} of 12 months). Volume totals are '
        f'actuals; charts comparing years annualise them. Rates are unaffected.'
        f'</div>', unsafe_allow_html=True)

mask = (yearly.year == year) & yearly.plant.isin(plants) & yearly.district.isin(districts)
sel = yearly[mask].copy()
mmask = (monthly.year == year) & monthly.plant.isin(plants) & monthly.district.isin(districts)
msel = monthly[mmask].copy()

if sel.empty:
    st.error("No plants match the current filters. Widen the selection under "
             "Scope in the sidebar.")
    st.stop()

ML_COLS = ["criticality", "criticality_rank", "unexplained_pp", "unexplained_m3", "expected_nrw_pct", "actual_nrw_pct","trend_pp_yr", "trend_p", "trend_recent_pp_yr", "step_shift_pp","step_p", "anomaly_months", "worst_z", "anomaly_score",
            "archetype", "cluster", "projected_nrw_pct_12m",
           "projected_extra_m3", "volatility_pp", "latest_nrw_pct",
           "pr_unexplained", "pr_deterioration", "pr_trend"]
ML_MATCHES_YEAR = HAS_ML and year == ML_YEAR
if ML_MATCHES_YEAR:
    sel = sel.merge(ml_plant[["plant"] + ML_COLS], on="plant", how="left")
elif HAS_ML:
    for c in ML_COLS:
        sel[c] = np.nan

weights = dict(DEFAULT_WEIGHTS)
lips_weights = dict(weights)
sel = score_lips(sel, tuple(sorted(lips_weights.items())))

tot_prod = sel.production_m3.sum()
tot_nrw = sel.nrw_m3.sum()
tot_val = sel.nrw_value_rm.sum()
tot_phys = sel.physical_loss_m3.sum()
sys_pct = tot_nrw / tot_prod * 100
n_plants = len(sel)

HIGH_RISK_THRESHOLD = 75  # plants with LIPS score above this count as "high risk"

prev = yearly[(yearly.year == year - 1) & yearly.plant.isin(sel.plant)]
prev_pct = (prev.nrw_m3.sum() / prev.production_m3.sum() * 100) if len(prev) else np.nan

high_risk = int((sel.lips > HIGH_RISK_THRESHOLD).sum()) if n_plants else 0

lips_q1 = sel.lips.quantile(0.25) if n_plants else 0.0
lips_q3 = sel.lips.quantile(0.75) if n_plants else 0.0
lips_iqr = lips_q3 - lips_q1

delta = ""
if not np.isnan(prev_pct):
    _d = sys_pct - prev_pct
    _cls = "kpi-good" if _d < 0 else "kpi-bad"
    delta = (f'<span class="{_cls}">{"↓" if _d < 0 else "↑"} '
             f'{abs(_d):.2f} pp</span> vs {year-1}')

_top10_share = (sel.nsmallest(min(10, n_plants), "lips_rank").nrw_m3.sum()
                / tot_nrw * 100) if tot_nrw else 0.0
_flagged = int(burst_pred.flag.sum()) if HAS_BURST and "flag" in burst_pred else 0

_kpis = [
    ("System loss rate", f"{sys_pct:.1f}%", delta or f"{n_plants} plants"),
    ("Water lost", f"{m3(tot_nrw)} m³", f"{m3(tot_nrw/365)} m³ per day"),
    ("High risk plants", f"{high_risk}", f"LIPS > {HIGH_RISK_THRESHOLD} · of {n_plants} plants"),
    ("LIPS interquartile range", f"{lips_iqr:.1f}", f"Q1 {lips_q1:.1f} – Q3 {lips_q3:.1f}"),
]
_cells = "".join(
    f'<div class="kpi"><div class="kpi-l"><span class="drop"></span>'
    f'<span style="font-weight:700 !important">{l}</span></div>'
    f'<div class="kpi-v">{v}</div><div class="kpi-s">{s_}</div></div>'
    for l, v, s_ in _kpis)
st.markdown(f'<div class="kpistrip">{_cells}</div>', unsafe_allow_html=True)
st.markdown('<div class="waverule" style="margin-bottom: 1rem;"></div>', unsafe_allow_html=True)

VIEW = st.session_state.view


def mini(fig, h=214, ylab=None, xlab=None, legend=False):
    fig.update_layout(
        height=h, margin=dict(l=2, r=10, t=22 if legend else 4, b=2),
        title=None,
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0,
                    font=dict(size=10)),
        font=dict(size=10.5),
        xaxis=dict(title=xlab, title_font=dict(size=10),
                   tickfont=dict(size=10)),
        yaxis=dict(title=ylab, title_font=dict(size=10),
                   tickfont=dict(size=10)))
    return fig


# ==========================================================================
# Overview tab (Limited to 6 Graphs with Bullet Explanations)
# ==========================================================================
if VIEW == "cmd":
    st.markdown("""
    <style>
    /* Tighten page padding and container gaps */
    .block-container { padding-top: 0.5rem !important; padding-bottom: 0.5rem !important; }
    div[data-testid="stVerticalBlockBorderWrapper"] { padding: 0.35rem 0.5rem !important; overflow: hidden; }
    div[data-testid="stVerticalBlock"] { gap: 0.2rem !important; }
    
    /* Card headers */
    .card-t { margin-bottom: 0 !important; font-size: 0.95rem !important; line-height: 1.1 !important; }
    /* No opacity here: it multiplies whatever colour the theme sets and was
       what made these read as translucent. Contrast is handled in theme.py
       (INK_2, ~6.5:1 light / ~11:1 dark). */
    .card-s { margin-bottom: 0.2rem !important; font-size: 0.72rem !important; }
    .card-s.card-s-lg { font-size: 0.82rem !important; }
    
    /* Explanation Box & Text styling */
    .ov-line {
        margin: 0.2rem 0 0 0 !important;
        padding: 0.3rem 0.5rem !important;
        font-size: 0.68rem !important;
        line-height: 1.2 !important;
        background-color: rgba(127, 127, 127, 0.08);
        border-radius: 4px;
        border-left: 3px solid #0066cc;
        box-sizing: border-box;
        width: 100%;
        white-space: normal;
        word-wrap: break-word;
        overflow-wrap: break-word;
    }
    </style>
    """, unsafe_allow_html=True)
    _n8 = min(8, n_plants)
    # Stacked bars need more vertical room per row than a plain bar, so the
    # two stacked cards show six plants rather than eight.
    _n6 = min(6, n_plants)
    _top10 = sel.nsmallest(min(10, n_plants), "lips_rank")
    _share10 = _top10.nrw_m3.sum() / tot_nrw * 100 if tot_nrw else 0.0

    # Row 1: Graphs 1, 2 and 3
    r1 = st.columns(3)

    with r1[0]:
        with st.container(border=True, vertical_alignment="center"):
            st.markdown(f'<div class="card-t">Top Priority Queue</div>'
                        f'<div class="card-s card-s-lg">Highest rank needs repair first</div>',
                        unsafe_allow_html=True)
            d_ = sel.nsmallest(_n8, "lips_rank").sort_values("lips")
            f = go.Figure(go.Bar(
                x=d_.lips, y=d_.plant, orientation="h",
                marker=dict(color=d_.lips, colorscale=T.SEQ,
                            line=dict(color=T.SURFACE, width=2), showscale=False),
                text=[f"{v:.0f}" for v in d_.lips], textposition="outside",
                textfont=dict(size=10.5, color=T.INK_2), customdata=d_.district,
                hovertemplate=("<b>%{y}</b> · %{customdata}<br>"
                               "LIPS %{x:.1f}<extra></extra>")))
            f.update_xaxes(range=[0, 116], showgrid=True)
            f.update_yaxes(showgrid=False)
            chart(mini(f, h=284))

    with r1[1]:
        with st.container(border=True, vertical_alignment="center"):
            st.markdown('<div class="card-t">Rate vs Volume</div>'
                        '<div class="card-s card-s-lg">Comparing percentage loss against total volume</div>',
                        unsafe_allow_html=True)
            tr_ = sel.nsmallest(10, "rate_rank")
            tv_ = sel.nsmallest(10, "volume_rank")
            pl_ = sel.copy()
            pl_["grp"] = np.select(
                [pl_.plant.isin(tv_.plant), pl_.plant.isin(tr_.plant)],
                ["Top 10 volume", "Top 10 rate"], default="Neither")
            f = go.Figure()
            for g_, c_ in [("Neither", T.NEUTRAL), ("Top 10 volume", T.BLUE),
                           ("Top 10 rate", T.ORANGE)]:
                q = pl_[pl_.grp == g_]
                if q.empty:
                    continue
                f.add_trace(go.Scatter(
                    x=q.production_m3, y=q.nrw_pct, mode="markers", name=g_,
                    marker=dict(size=np.sqrt(q.nrw_m3 / pl_.nrw_m3.max()) * 18 + 6,
                                color=c_, opacity=0.85,
                                line=dict(color=T.SURFACE, width=2)),
                    customdata=q.plant,
                    hovertemplate=("<b>%{customdata}</b><br>%{x:,.0f} m³ produced"
                                   "<br>%{y:.1f}% loss<extra></extra>")))
            f.update_xaxes(type="log", dtick=1)
            f.update_yaxes(ticksuffix="%")
            chart(mini(f, h=284, legend=True))

    with r1[2]:
        with st.container(border=True, vertical_alignment="center"):
            st.markdown('<div class="card-t">LIPS Component Percentiles</div>'
                        '<div class="card-s card-s-lg">Shows what factors drive each plant\'s score</div>',
                        unsafe_allow_html=True)
            d_ = sel.nsmallest(_n6, "lips_rank").sort_values("lips")
            f = go.Figure()
            for col, lab, c_ in [("pr_nrw_per_km_m3", "Loss density", T.BLUE),
                                 ("pr_bursts_per_100km", "Burst rate", T.ORANGE),
                                 ("pr_plant_age_yr", "Plant age", T.AQUA),
                                 ("pr_account_density", "Accounts", T.YELLOW)]:
                if col in d_.columns:
                    f.add_trace(go.Bar(
                        x=d_[col], y=d_.plant, orientation="h", name=lab,
                        marker=dict(color=c_, line=dict(color=T.SURFACE, width=2)),
                        hovertemplate=(f"<b>%{{y}}</b><br>{lab} "
                                       "percentile %{x:.0f}<extra></extra>")))
            f.update_layout(barmode="stack")
            f.update_yaxes(showgrid=False)
            chart(mini(f, h=284, legend=True))

    # Row 2 : Graphs 4, 5 and 6
    r2 = st.columns(3)
    with r2[0]:
        with st.container(border=True, vertical_alignment="center"):
            st.markdown(f'<div class="card-t">Monthly Loss Rate ({year})</div>'
                        f'<div class="card-s card-s-lg">Tracks Pahang\'s water loss rate in {year}</div>',
                        unsafe_allow_html=True)
            mo = (msel.groupby("month", as_index=False)
                      .agg(p=("production_m3", "sum"), nn=("nrw_m3", "sum")))
            mo["pct"] = mo.nn / mo.p * 100
            f = go.Figure(go.Scatter(
                x=mo.month, y=mo.pct, mode="lines",
                line=dict(color=T.BLUE, width=2, shape="spline", smoothing=0.5),
                fill="tozeroy", fillcolor=T.TILE_WASH,
                hovertemplate="Month %{x}<br>%{y:.2f}% loss<extra></extra>"))
            f.update_yaxes(ticksuffix="%",
                           range=[mo.pct.min() - 1.0, mo.pct.max() + 1.0])
            f.update_xaxes(dtick=2, showgrid=False)
            chart(mini(f, h=284))

    with r2[1]:
        with st.container(border=True, vertical_alignment="center"):
            st.markdown('<div class="card-t">Loss Concentration</div>'
                        f'<div class="card-s card-s-lg">Prioritizing top plants can save tons of water </div>',
                        unsafe_allow_html=True)
            sv = sel.sort_values("nrw_m3", ascending=False).reset_index(drop=True)
            sv["cum"] = sv.nrw_m3.cumsum() / sel.nrw_m3.sum() * 100
            n10 = min(10, len(sv))
            f = go.Figure(go.Scatter(
                x=np.arange(1, len(sv) + 1), y=sv.cum, mode="lines",
                line=dict(color=T.BLUE, width=2), fill="tozeroy",
                fillcolor=T.TILE_WASH,
                hovertemplate="Top %{x} plants<br>%{y:.0f}% of water<extra></extra>"))
            f.add_vline(x=n10, line=dict(color=T.BASELINE, width=1))
            f.add_annotation(x=n10, y=sv.cum.iloc[n10 - 1],
                             text=f"<b>{sv.cum.iloc[n10-1]:.0f}%</b> in {n10}",
                             showarrow=False, xshift=46, yshift=-12,
                             font=dict(size=11, color=T.INK))
            f.update_yaxes(range=[0, 104], ticksuffix="%")
            f.update_xaxes(showgrid=False)
            chart(mini(f, h=284))

    with r2[2]:
        with st.container(border=True, vertical_alignment="center"):
            st.markdown('<div class="card-t">Top NRW losses</div>'
                        '<div class="card-s card-s-lg">Split into physical and commercial</div>',
                        unsafe_allow_html=True)
            d_ = sel.nlargest(_n6, "nrw_m3").sort_values("nrw_m3")
            f = go.Figure()
            f.add_trace(go.Bar(
                x=d_.physical_loss_m3, y=d_.plant, orientation="h",
                name="Physical", marker=dict(color=T.BLUE,
                                             line=dict(color=T.SURFACE, width=2)),
                hovertemplate="<b>%{y}</b><br>%{x:,.0f} m³<extra></extra>"))
            f.add_trace(go.Bar(
                x=d_.commercial_loss_m3, y=d_.plant, orientation="h",
                name="Commercial",
                marker=dict(color=T.NEUTRAL,
                            line=dict(color=T.SURFACE, width=2)),
                hovertemplate="<b>%{y}</b><br>%{x:,.0f} m³<extra></extra>"))
            f.update_layout(barmode="stack")
            f.update_yaxes(showgrid=False)
            chart(mini(f, h=284, legend=True))


# ====================================================================
# Priority Tab
# ====================================================================

if VIEW in SEC_PRIORITY:
    if VIEW == "rank":
        st.markdown(" ### :material/leaderboard: Leakage Intervention Priority Score (4-Factor LIPS)")
        st.markdown("#### :material/info: LIPS blends loss density, burst rate, plant age, and account density into one 0–100 priority score.", unsafe_allow_html=True)
        st.space()

        c1, c2 = st.columns([1, 1])
        n_show = min(15, n_plants)
        top = sel.nsmallest(n_show, "lips_rank").sort_values("lips")

        with c1:
            fig = go.Figure(go.Bar(
                x=top.lips, y=top.plant, orientation="h",
                marker=dict(color=top.lips, colorscale=T.SEQ,
                            line=dict(color=T.SURFACE, width=2), showscale=False),
                text=[f"{v:.1f}" for v in top.lips],
                textposition="outside", textfont=dict(size=11, color=T.INK_2),
                customdata=np.stack([top.district, top.nrw_per_km_m3, top.bursts_per_100km, top.lips_rank, top.plant_age_yr, top.account_density], -1),
                hovertemplate=("<b>%{y}</b> · %{customdata[0]} · ✪ %{customdata[3]}<br>"
                               "LIPS Score: %{x:.1f}<br>"
                               "Loss Density: %{customdata[1]:,.0f} m³/km<br>"
                               "Burst Rate: %{customdata[2]:.1f} /100km<br>"
                               "Plant Age: %{customdata[4]:.0f} years old<br>"
                               "Account density: %{customdata[5]:.1f} acc/km<extra></extra>")))
            fig.update_layout(
                title=f"<b>Top {n_show} Plants by LIPS Priority</b>", height=572,
                bargap=0.3,
                xaxis=dict(title="LIPS Score (0–100)", range=[0, 115]),
                yaxis=dict(title=None, tickfont=dict(size=11)))
            chart(fig)

        with c2:
            fig = go.Figure()
            comp_cols = {
                "pr_nrw_per_km_m3": ("Loss Density (40%)", T.BLUE),
                "pr_bursts_per_100km": ("Burst Rate (25%)", T.ORANGE),
                "pr_plant_age_yr": ("Plant Age (20%)", T.AQUA),
                "pr_account_density": ("Account Density (15%)", T.YELLOW)
            }
            for col, (label, color) in comp_cols.items():
                if col in top.columns:
                    fig.add_trace(go.Bar(
                        x=top[col], y=top.plant, orientation="h", name=label,
                        marker=dict(color=color,
                                    line=dict(color=T.SURFACE, width=2)),
                        hovertemplate=(f"<b>%{{y}}</b><br>{label} "
                                       "percentile %{x:.0f}<extra></extra>")
                    ))
            fig.update_layout(
                title="<b>LIPS Component Percentile Profile</b>",
                height=572, barmode="stack", bargap=0.3,
                xaxis=dict(title="Weighted Component Contribution"),
                yaxis=dict(title=None, tickfont=dict(size=11)))
            chart(fig)

    if VIEW == "curve":
        st.markdown("### :material/stacked_line_chart: Recovery curve — how far a crew programme gets")

        order_lips = sel.sort_values("lips_rank")
        order_rate = sel.sort_values("rate_rank")
        total_nrw = sel.nrw_m3.sum()

        def curve(df):
            return df.nrw_m3.cumsum() / total_nrw * 100

        x = np.arange(1, n_plants + 1)
        lips_curve = curve(order_lips).to_numpy()
        rate_curve = curve(order_rate).to_numpy()

        # Crew-effort checkpoint: first ~25% of the plant list (at least 3 stops)
        k = int(np.clip(round(n_plants * 0.25), min(3, n_plants), n_plants))
        lips_at_k = lips_curve[k - 1]
        rate_at_k = rate_curve[k - 1]
        gap_at_k = lips_at_k - rate_at_k

        st.markdown(
            f"#### :material/info: In the first <b>{k} plant{'s' if k != 1 else ''}</b> "
            f"(~{k / n_plants * 100:.0f}% of the programme), LIPS/volume order "
            f"recovers <b>{lips_at_k:.1f}%</b> of total NRW versus "
            f"<b>{rate_at_k:.1f}%</b> for rate order — a "
            f"<b>{gap_at_k:.1f} pp</b> gap for the same crew effort.",
            unsafe_allow_html=True)

        fig = go.Figure()
        for name, df_, col in [("LIPS / volume order", order_lips, T.BLUE),
                               ("Rate order", order_rate, T.ORANGE)]:
            fig.add_trace(go.Scatter(
                x=x, y=curve(df_), mode="lines", name=name,
                line=dict(color=col, width=2.5),
                hovertemplate=(f"<b>{name}</b><br>First %{{x}} plants<br>"
                                "cover %{y:.1f}% of NRW<extra></extra>")))
        fig.update_layout(
            title="Share of total NRW covered, by queue ordering", height=612,
            xaxis=dict(title="Plants visited, in queue order"),
            yaxis=dict(title="% of total NRW covered", ticksuffix="%",
                        range=[0, 102]))
        chart(fig)

    if VIEW == "sched":
        st.markdown("### :material/table_eye: Full intervention schedule")
        st.space()
        sched = sel.sort_values("lips_rank")[
            ["lips_rank", "plant", "district", "area_type", "lips",
             "nrw_per_km_m3", "bursts_per_100km", "plant_age_yr", "account_density",
             "nrw_m3", "nrw_pct", "volume_rank", "rate_rank"]]
        table(sched, [
            ("lips_rank", "Priority", lambda v: f"{int(v)}"),
            ("plant", "Plant", str),
            ("district", "District", str),
            ("area_type", "Area", str),
            ("lips", "LIPS Score", lambda v: f"{v:.1f}"),
            ("nrw_per_km_m3", "Loss Density (m³/km)", lambda v: f"{v:,.0f}"),
            ("bursts_per_100km", "Bursts /100km", lambda v: f"{v:.1f}"),
            ("plant_age_yr", "Plant Age (yr)", lambda v: f"{v:.0f}"),
            ("account_density", "Acc Density (/km)", lambda v: f"{v:.1f}"),
            ("nrw_m3", "NRW m³", lambda v: f"{v:,.0f}"),
            ("nrw_pct", "Rate", lambda v: f"{v:.1f}%"),
            ("volume_rank", "Vol Rank", lambda v: f"{int(v)}"),
            ("rate_rank", "Rate Rank", lambda v: f"{int(v)}"),
        ], height=530, bar="lips")
        st.download_button("Download schedule (CSV)", sched.to_csv(index=False),
                           f"paip_lips_schedule_{year}.csv", "text/csv")

    if VIEW == "pred":
        st.markdown("### :material/online_prediction: **LIPS 2026 Prediction**")
        # Matched to the Recovery curve caption: one line, plain language,
        # the numbers that matter. The full reasoning lives in
        # prepare_data.predict_2026_lips for anyone who wants it.
        # YEAR_SPAN, not backtest train_years: the backtest trains on
        # 2023-2024 to score itself against 2025, but the projection itself
        # averages every observed year.
        _pj = forecast_backtest[forecast_backtest.method == "projection"].iloc[0]
        _bl = forecast_backtest[
            forecast_backtest.method.str.startswith("persistence")].iloc[0]
        st.markdown(
            f"#### :material/info: Only the burst rate is really forecast — "
            f"averaged over <b>{YEAR_SPAN}</b> instead "
            f"of taken from the last year alone. That gets the ranking within "
            f"<b>{_pj.rank_mae_places:.1f} places</b>, against "
            f"<b>{_bl.rank_mae_places:.1f}</b> if you just assume next year "
            f"looks like this one.",
            unsafe_allow_html=True)
        
        # Priority Summary Metrics
        escalated = len(prediction[prediction['rank_change'] > 1])
        deescalated = len(prediction[prediction['rank_change'] < 1])
        top_risk = prediction.sort_values('lips_rank_2026').iloc[0]['plant']
        lips_pred = prediction.sort_values('lips_rank_2026').iloc[0]['lips_2026']
        
        m = st.columns(4)

        m[0].markdown(T.tile('<span style="font-weight:800 !important">2026 #1 Priority Plant</span>', top_risk, "with", f"projected LIPS score {lips_pred}"), unsafe_allow_html=True)

        m[1].markdown(T.tile('<span style="font-weight:800 !important">Escalating Plants</span>', f"{escalated} Plants"), unsafe_allow_html=True)

        m[2].markdown(T.tile('<span style="font-weight:800 !important">Improving / Stable</span>', f"{deescalated} Plants"), unsafe_allow_html=True)

        if forecast_backtest is not None:
            _proj = forecast_backtest[
                forecast_backtest.method == "projection"].iloc[0]
            _base = forecast_backtest[
                forecast_backtest.method.str.startswith("persistence")].iloc[0]
            # Rank agreement (Spearman) rather than top-N precision.
            # Both are percentages and both read "higher is better", but
            # projection and persistence score an identical 80.0% on top-10
            # precision - a 0.0pp gap - so that number cannot show whether
            # the model beats doing nothing. Rank agreement separates them
            # (92.3% vs 90.4%) and uses the whole ranking, not just the top
            # ten. The baseline stays in the subtitle so the gap is visible
            # rather than implied.
            m[3].markdown(T.tile(
                '<span style="font-weight:800 !important">Forecast Accuracy</span>',
                f"{_proj.rank_spearman * 100:.1f}%",
                "",
                f"rank agreement with actual · "
                f"{_base.rank_spearman * 100:.1f}% if you assume no change · "
                f"backtest {_proj.train_years} → {int(_proj.holdout_year)}"),
                unsafe_allow_html=True)
        else:
            m[3].markdown(T.tile('<span style="font-weight:800 !important">Forecast Accuracy</span>', "N/A", "",
                                 "Needs 2+ years of history to backtest"),
                          unsafe_allow_html=True)


        # 2026 Priority Table Display
        st.markdown("#### **Predicted 2026 LIPS Priority Schedule**")
        
        # Prepare display columns
        disp_df = prediction[[
            'lips_rank_2026', 'plant', 'lips_2026', 'rank_change', 
            'predicted_nrw_2026', 'predicted_bursts_2026', 'pipe_age_2026'
        ]].sort_values('lips_rank_2026')
        
        disp_df.columns = [
            '2026 Rank', 'Plant Name', 'Projected LIPS Score', 'Rank Shift', 
            'Est. 2026 NRW (%)', 'Est. Bursts / 100km', 'Pipe Age (2026)'
        ]
        
        # Format Rank Shift output
        def style_shift(val):
            if val > 0:
                return f"▲ +{val}"
            elif val < 0:
                return f"▼ {val}"
            return "➖"
            
        disp_df['Rank Shift'] = disp_df['Rank Shift'].apply(style_shift)

        table(disp_df, [
            ("2026 Rank", "2026 Rank", lambda v: f"{int(v)}"),
            ("Plant Name", "Plant Name", str),
            ("Projected LIPS Score", "Projected LIPS Score",
             lambda v: f"{v:.1f}"),
            ("Rank Shift", "Rank Shift", str),
            ("Est. 2026 NRW (%)", "Est. 2026 NRW (%)", lambda v: f"{v:.1f}%"),
            ("Est. Bursts / 100km", "Est. Bursts / 100km",
             lambda v: f"{v:.1f}"),
            ("Pipe Age (2026)", "Pipe Age (2026)", lambda v: f"{v:.0f}"),
        ], height=430, bar="Projected LIPS Score")

        # Plotly Comparison: Current vs 2026 Rank Shift
        st.markdown("#### **2026 Priority Escalation Matrix**")
        # Built with graph_objects rather than px so it goes through chart(),
        # which forces paper/plot background. Going straight to
        # st.plotly_chart left Streamlit to fill those in, which is why this
        # one panel rendered on black while everything around it was light.
        _m = prediction.copy()
        _lim = max(1.0, float(_m.rank_change.abs().max()))
        fig_shift = go.Figure()
        fig_shift.add_shape(type="line",
                            line=dict(dash="dash", color=T.BASELINE, width=1),
                            x0=0, x1=100, y0=0, y1=100)
        fig_shift.add_trace(go.Scatter(
            x=_m.lips, y=_m.lips_2026, mode="markers",
            marker=dict(
                size=np.clip(_m.predicted_bursts_2026 /
                             max(_m.predicted_bursts_2026.max(), 1) * 22, 7, 26),
                # Diverging blue<->red with a neutral midpoint: rank change has
                # a real zero and two opposite directions. The previous
                # red-yellow-green ramp put the two ends on the classic
                # red/green pair, which is the hardest for colour-blind
                # readers, and had a hue at the midpoint where "no change"
                # should read as nothing.
                color=_m.rank_change, cmin=-_lim, cmax=_lim,
                colorscale=[[0.0, T.AQUA], [0.5, T.NEUTRAL], [1.0, T.CRITICAL]],
                colorbar=dict(title=dict(text="Rank<br>change", side="top"),
                              thickness=11, len=0.75,
                              tickfont=dict(size=10, color=T.INK_2),
                              outlinewidth=0),
                line=dict(color=T.SURFACE, width=1.5)),
            customdata=np.stack([_m.plant, _m.lips_rank,
                                 _m.lips_rank_2026, _m.rank_change], -1),
            hovertemplate=("<b>%{customdata[0]}</b><br>"
                           "LIPS %{x:.1f} → %{y:.1f}<br>"
                           "rank %{customdata[1]} → %{customdata[2]} "
                           "(%{customdata[3]:+d})<extra></extra>")))

        # Label only the plants that actually move. A name on all 74 points
        # was unreadable, and a value on every mark is the anti-pattern.
        _lab = _m.reindex(_m.rank_change.abs().sort_values(ascending=False)
                          .index).head(8)
        fig_shift.add_trace(go.Scatter(
            x=_lab.lips, y=_lab.lips_2026, mode="text", text=_lab.plant,
            textposition="top center",
            textfont=dict(size=10.5, color=T.INK), showlegend=False,
            hoverinfo="skip"))

        fig_shift.update_layout(
            height=520, showlegend=False,
            title=dict(text="Current vs projected 2026 LIPS score",
                       font=dict(size=15, color=T.INK)),
            xaxis=dict(title="Current LIPS score", range=[0, 104]),
            yaxis=dict(title="Projected 2026 LIPS score", range=[0, 104]))
        chart(fig_shift)
        st.markdown(
            '<div class="caption">Points above the dashed line are projected '
            'to become more urgent, below it less. Marker size is the '
            'projected burst rate. Only the eight largest movers are labelled '
            '&mdash; the rest are in the tooltip.</div>',
            unsafe_allow_html=True)

# ==========================================================================
# Plant Profile tab
# ==========================================================================

if VIEW in SEC_PLANT:
    plant_list = sel.sort_values("lips_rank").plant.tolist()
    if VIEW == "pcomp":
        # 1. Define top row columns matching both selectboxes and captions
        c0a, c0b = st.columns(2)
    
        with c0a:
            plant = st.selectbox(":material/water_drop: Plant", plant_list, index=0)
            p = sel[sel.plant == plant].iloc[0]
            pm = msel[msel.plant == plant].sort_values("date")
            hist = monthly[monthly.plant == plant].sort_values("date")
            st.markdown(
                f'<div class="caption" style="padding-top:10px;">'
                f'<b>{plant}</b> · {p.district} · {p.area_type}<br>'
                f'{p.plant_age_yr:.0f}-year-old plant · '
                f'{p.pipe_length_km:,.0f} km of main · '
                f'{p.customer_accounts:,.0f} connections · '
                f'{p.population_served:,.0f} people served</div>',
                unsafe_allow_html=True)

        with c0b:
            other_plants = [p_item for p_item in plant_list if p_item != plant]

            # Making sure that Plant 2 stays when Plant 1 changes
            saved_p2 = st.session_state.get("p2_name")
            p2_idx = other_plants.index(saved_p2) if saved_p2 in other_plants else 0

            p2_name = st.selectbox(":material/compare_arrows: Compare with", other_plants, index=p2_idx, key="p2_name")
        
            p2 = sel[sel.plant == p2_name].iloc[0]
            hist2 = monthly[monthly.plant == p2_name].sort_values("date")

            st.markdown(
                f'<div class="caption" style="padding-top:10px;">'
                f'<b>{p2_name}</b> · {p2.district} · {p2.area_type}<br>'
                f'{p2.plant_age_yr:.0f}-year-old plant · '
                f'{p2.pipe_length_km:,.0f} km of main · '
                f'{p2.customer_accounts:,.0f} connections · '
                f'{p2.population_served:,.0f} people served</div>',
                unsafe_allow_html=True)

        st.space()

    else:
        # Standard single-plant layout for psum, pmodel, phist
        c0a, c0b = st.columns([1, 2])
        with c0a:
            plant = st.selectbox(":material/water_drop: Plant", plant_list, index=0)
            p = sel[sel.plant == plant].iloc[0]
            pm = msel[msel.plant == plant].sort_values("date")
            hist = monthly[monthly.plant == plant].sort_values("date")
        with c0b:
            st.markdown(
                f'<div class="caption" style="padding-top:39px">'
                f'<b>{plant}</b> · {p.district} · {p.area_type} · '
                f'{p.plant_age_yr:.0f}-year-old plant · '
                f'{p.pipe_length_km:,.0f} km of main · '
                f'{p.customer_accounts:,.0f} connections · '
                f'{p.population_served:,.0f} people served</div>',
                unsafe_allow_html=True)
        
        st.space()
    
    st.markdown('<div class="waverule" style="margin-bottom: 1rem;"></div>', unsafe_allow_html=True)

    # Calculate fleet median dynamically
    burst_med = sel.bursts_per_100km.median()
    burst_ratio = p.bursts_per_100km / burst_med if burst_med else 1.0

    if VIEW == "psum":
        st.markdown(f"### :material/article: Summary for {plant}")
        st.space()

        k = st.columns(4)
        k[0].markdown(T.tile('<span style="font-weight:800 !important">LIPS</span>', f"{p.lips:.1f}", f"rank {p.lips_rank}", f"of {n_plants} plants in the current selection"), unsafe_allow_html=True)

        k[1].markdown(T.tile('<span style="font-weight:800 !important">Water loss rate</span>', f"{p.nrw_pct:.1f}", "%", f"Rank {p.rate_rank} · system average {sys_pct:.1f}%"), unsafe_allow_html=True)

        k[2].markdown(T.tile('<span style="font-weight:800 !important">Water lost</span>', m3(p.nrw_m3), "m³", f"Rank {p.volume_rank} · {p.nrw_m3/tot_nrw*100:.1f}% of selection total"), unsafe_allow_html=True)

        k[3].markdown(T.tile('<span style="font-weight:800 !important">Burst rate</span>', f"{p.bursts_per_100km:.1f}", "/100km", f"{burst_ratio:.1f}x fleet median ({burst_med:.1f})"), unsafe_allow_html=True)

        st.space()
        c1, c2 = st.columns([1.4, 1])
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=hist.date, y=hist.physical_loss_m3, name="Physical leakage",
                marker=dict(color=T.BLUE, line=dict(color=T.SURFACE, width=1)),
                hovertemplate="%{x|%b %Y}<br>Physical  %{y:,.0f} m³<extra></extra>"))
            fig.add_trace(go.Bar(
                x=hist.date, y=hist.commercial_loss_m3, name="Commercial loss",
                marker=dict(color=T.ORANGE, line=dict(color=T.SURFACE, width=1)),
                hovertemplate="%{x|%b %Y}<br>Commercial  %{y:,.0f} m³<extra></extra>"))
            fig.update_layout(
                title=f"{plant} — monthly loss volume, {YEAR_SPAN}", height=300,
                barmode="stack", bargap=0.15,
                xaxis=dict(title=None), yaxis=dict(title="Volume lost (m³)"))
            chart(fig)

        with c2:
            peer = sel[(sel.area_type == p.area_type)]
            metrics = [
                ("Water loss rate %", "nrw_pct", "%"),
                ("NRW per km", "nrw_per_km_m3", "m³"),
                ("Bursts /100km", "bursts_per_100km", "/100km"),
                ("Plant age", "plant_age_yr", "yrs"),
                ("Meter age", "meter_age_yr", "yrs")
            ]
    
            labels, pvals, medians, raw_vals, text_labels = [], [], [], [], []
            for label, col, unit in metrics:
                med = peer[col].median()
                val = p[col]
                if med and not np.isnan(med):
                    pct = (val / med) * 100
                    labels.append(label)
                    pvals.append(pct)
                    medians.append(med)
                    raw_vals.append(val)
                    text_labels.append(f"<b>{val:,.1f} {unit}</b> ({pct:.0f}%)")

            fig = go.Figure()

            # Bar Trace
            fig.add_trace(go.Bar(
                x=pvals,
                y=labels,
                orientation="h",
                marker=dict(
                    color=[
                        T.CRITICAL if v > 130 else T.BLUE if v > 70 else T.GOOD 
                        for v in pvals
                    ],
                    line=dict(color=T.SURFACE, width=1.5)
                ),
                text=text_labels,
                textposition="inside",
                cliponaxis=False,
                customdata=list(zip(medians, raw_vals)),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Plant Value: <b>%{customdata[1]:,.1f}</b><br>"
                    "Peer Median: <b>%{customdata[0]:,.1f}</b><br>"
                    "Ratio: <b>%{x:.0f}%</b> of median<extra></extra>"
                )
            ))

            # Baseline at 100% (Peer Median)
            fig.add_vline(
                x=100, 
                line=dict(color=T.BASELINE, width=1.5, dash="dash"),
                annotation_text="Peer Median (100%)",
                annotation_position="top right",
                annotation_font=dict(size=10, color=T.MUTED)
            )

            fig.update_layout(
                title=dict(text=f"<b>Vs. {p.area_type} Peers</b> (n={len(peer)})", font=dict(size=14)),
                height=320,
                margin=dict(l=10, r=60, t=40, b=30),
                bargap=0.3,
                xaxis=dict(
                    title="% of peer median",
                    range=[0, max(max(pvals, default=100) * 1.3, 130)],
                    showgrid=True,
                    gridcolor="rgba(0,0,0,0.05)"
                ),
                yaxis=dict(autorange="reversed", title=None) # Keeps top metric at top
            )
            chart(fig)

    if VIEW == "pmodel":
        if not (HAS_ML and not pd.isna(p.get("criticality", np.nan))):
            st.info(f"No model output for {plant} in {year}. The expected-loss "
                    f"model is fitted for {ML_YEAR}.")
        else:
            st.markdown("### :material/diagnosis: Root-Cause Diagnosis")
            pm_ml = ml_monthly[ml_monthly.plant == plant].sort_values("date")
            
            # --- 1. Operational KPI Strip (Driven by LIPS + ML Diagnostics) ---
            gap_pp = float(p.unexplained_pp)
            gap_m3 = float(p.unexplained_m3)
            daily_m3_lost = max(0, gap_m3 / 365) if gap_m3 > 0 else 0

            # Diagnostic Diagnosis based on ML baseline vs actual gap
            if gap_pp > 3 and p.bursts_per_100km > 12:
                verdict_title = "DIAGNOSIS: ACTIVE LEAKAGE"
                rec_action = "Urgent dispatch — acoustic leak detection and pressure reduction."
                action_bg = "rgba(239, 68, 68, 0.12)"
                border_col = "#ef4444"
            elif gap_pp > 3:
                verdict_title = "DIAGNOSIS: COMMERCIAL ANOMALY"
                rec_action = "Unaccounted volume — audit bulk meters and large accounts."
                action_bg = "rgba(245, 158, 11, 0.12)"
                border_col = "#f59e0b"
            else:
                verdict_title = "DIAGNOSIS: STRUCTURAL AGING"
                rec_action = "Expected for network age — plan pipe replacement, not emergency response."
                action_bg = "rgba(16, 185, 129, 0.12)"
                border_col = "#10b981"

            k1, k2, k3 = st.columns(3)
            # Standard HTML superscript with a hover tooltip for k1
            pp_label = 'pp <sup title="Percentage Points: Absolute difference between percentage values" style="cursor:pointer; color:#6b7280; font-weight:bold; font-size:18px">?</sup>'

            k1.markdown(T.tile('<span style="font-weight:800 !important">Unexplained deviation</span>', f"{gap_pp:+.1f} {pp_label}", 
                        "vs actual loss"), unsafe_allow_html=True)
            k2.markdown(T.tile('<span style="font-weight:800 !important">Unaccounted volume</span>', f"{m3(gap_m3)} m³", 
                               "volume annually"), unsafe_allow_html=True)
            k3.markdown(T.tile('<span style="font-weight:800 !important">Daily recovery potential</span>', f"{m3(daily_m3_lost)} m³", 
                               "volume per day"), unsafe_allow_html=True)

            # Diagnostic Action Banner
            st.markdown(f'''
            <div style="background:{action_bg}; padding: 0.8rem 1rem; border-radius: 6px; margin: 1rem 0; border-left: 5px solid {border_col};">
                <span style="font-weight:700; font-size:0.95rem; color:{border_col};">{verdict_title}</span><br>
                <span style="font-size:0.9rem;">{rec_action}</span>
            </div>
            ''', unsafe_allow_html=True)

            # --- 2. Chart & Diagnostic Breakdown ---
            c7, c8 = st.columns([1.6, 1])
            with c7:
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=pm_ml.date, y=pm_ml.predicted_nrw_pct, mode="lines",
                    name="Expected loss",
                    line=dict(color=T.ORANGE, width=2, dash="dash"),
                    hovertemplate="%{x|%b %Y}<br>Expected loss: %{y:.1f}%<extra></extra>"))
                fig.add_trace(go.Scatter(
                    x=pm_ml.date, y=pm_ml.nrw_pct, mode="lines",
                    name="Actual loss", line=dict(color=T.BLUE, width=2.5),
                    hovertemplate="%{x|%b %Y}<br>Actual loss: %{y:.1f}%<extra></extra>"))
                an = pm_ml[pm_ml.is_anomaly.fillna(False).astype(bool)]
                if len(an):
                    fig.add_trace(go.Scatter(
                        x=an.date, y=an.nrw_pct, mode="markers",
                        name="Unexplained loss spike",
                        marker=dict(size=11, color=T.CRITICAL, symbol="circle-open",
                                    line=dict(width=2.5)),
                        hovertemplate="%{x|%b %Y}<br><b>Operational Anomaly Detected</b><br>Loss Rate: %{y:.1f}%<extra></extra>"))
                  
                fig.update_layout(
                    title="<b>Actual Loss vs. Expected Loss</b>",
                    height=380, xaxis=dict(title=None),
                    yaxis=dict(title="NRW Loss Rate (%)", ticksuffix="%"),
                    legend=dict(orientation="h", y=1.12))
                chart(fig)
                
            with c8:
                st.markdown("##### **Model Diagnostic Breakdown**")
                
                trend_desc = "Rapid Deterioration" if p.trend_pp_yr > 1.0 else ("Gradual Rise" if p.trend_pp_yr > 0.2 else ("Improving" if p.trend_pp_yr < -0.2 else "Stable"))
                shift_desc = "Sudden Burst / Loss Shift" if abs(p.step_shift_pp) > 2 else "Consistent Pattern"
                sig = pd.DataFrame({
                    "Diagnostic Factor": [
                        "Asset Behavior Profile",
                        "Multi-Year Trajectory",
                        "Recent Step Change (6 Mo)",
                        "Flagged Anomaly Months",
                        "Month-to-Month Stability"
                    ],
                    "Model Finding": [
                        f"**{p.archetype}**",
                        f"{trend_desc} ({p.trend_pp_yr:+.1f}%/yr)",
                        f"{shift_desc}",
                        f"{int(p.anomaly_months)} Month(s)",
                        "Volatile" if p.volatility_pp > 3 else "Steady"
                    ]
                })
                
                table(sig, [(c, c, str) for c in sig.columns], height=210)
                
                st.info(
                    """💡 **How it works:** Learns each plant's expected loss from its profile (age, length, connections), 
                    then flags drift from it — separating aging from real leaks or billing errors."""
                )

    if VIEW == "phist":
        st.markdown("### :material/history: <b>History</b>", unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist.date, y=hist.nrw_pct, mode="lines+markers", name="Loss rate",
                line=dict(color=T.BLUE, width=2),
                marker=dict(size=5, color=T.BLUE),
                hovertemplate="%{x|%b %Y}<br>Loss rate  %{y:.1f}%<extra></extra>"))
            fig.update_layout(title="Loss rate history", height=500,
                              showlegend=False, xaxis=dict(title=None),
                              yaxis=dict(title="NRW (%)", ticksuffix="%"))
            chart(fig)

        with c4:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=hist.date, y=hist.pipe_bursts, name="Bursts",
                marker=dict(color=T.ORANGE, line=dict(color=T.SURFACE, width=1)),
                hovertemplate="%{x|%b %Y}<br>%{y:.0f} bursts<extra></extra>"))
            fig.update_layout(title="Recorded pipe bursts", height=500,
                              showlegend=False, bargap=0.15,
                              xaxis=dict(title=None),
                              yaxis=dict(title="Bursts in month"))
            chart(fig)

        with st.expander(f"Monthly records for {plant}"):
            cols = ["date", "production_m3", "billed_m3", "nrw_m3", "nrw_pct",
                    "physical_loss_m3", "commercial_loss_m3", "pipe_bursts", "pressure_bar"]
            table(hist[cols], [
                ("date", "Month",
                 lambda v: pd.to_datetime(v).strftime("%b %Y")),
                ("production_m3", "Production m³", lambda v: f"{v:,.0f}"),
                ("billed_m3", "Billed m³", lambda v: f"{v:,.0f}"),
                ("nrw_m3", "NRW m³", lambda v: f"{v:,.0f}"),
                ("nrw_pct", "Rate", lambda v: f"{v:.1f}%"),
                ("physical_loss_m3", "Physical NRW m³", lambda v: f"{v:,.0f}"),
                ("commercial_loss_m3", "Commercial NRW m³",
                 lambda v: f"{v:,.0f}"),
                ("pipe_bursts", "Bursts", lambda v: f"{int(v)}"),
                ("pressure_bar", "Pressure bar", lambda v: f"{v:.2f}"),
            ], height=420)
    if VIEW == "pcomp":
        st.markdown("### :material/compare: <b>Side-by-Side Plant Comparison</b>", unsafe_allow_html=True)

        # --- Key Metrics Comparison Table/Tiles ---
        # Small inline SVGs, not :material/...: shortcodes — those only expand
        # inside Streamlit's own markdown parser, not raw unsafe_allow_html HTML
        # (T.tile() returns a plain HTML string). Neutral/muted color on purpose:
        # this is a factual "which number is bigger" comparison, not a verdict —
        # for water loss / burst rate, the *lower* number is actually the better one.
        _CMP_STYLE = 'style="vertical-align:-0.1em;margin-left:4px;"'
        _ICO_UP = (f'<svg width="0.85em" height="0.85em" viewBox="0 0 16 16" fill="none" '
                   f'stroke="currentColor" stroke-width="2" {_CMP_STYLE}>'
                   f'<path d="M4 10l4-4 4 4" stroke-linecap="round" stroke-linejoin="round"/></svg>')
        _ICO_DOWN = (f'<svg width="0.85em" height="0.85em" viewBox="0 0 16 16" fill="none" '
                     f'stroke="currentColor" stroke-width="2" {_CMP_STYLE}>'
                     f'<path d="M4 6l4 4 4-4" stroke-linecap="round" stroke-linejoin="round"/></svg>')
        _ICO_EQUAL = (f'<svg width="0.85em" height="0.85em" viewBox="0 0 16 16" fill="none" '
                      f'stroke="currentColor" stroke-width="2" {_CMP_STYLE}>'
                      f'<path d="M3 6h10M3 10h10" stroke-linecap="round"/></svg>')

        def _cmp_icon(val_a, val_b, tol=1e-9):
            """Icon on val_a's tile showing how it compares to val_b: higher, lower, or equal."""
            if abs(val_a - val_b) <= tol:
                return _ICO_EQUAL
            return _ICO_UP if val_a > val_b else _ICO_DOWN

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"#### **{plant}** (Selected)")
            st.markdown(T.tile(f"LIPS Score{_cmp_icon(p.lips, p2.lips)}", f"{p.lips:.1f}", f"Rank {p.lips_rank}"), unsafe_allow_html=True)
            st.markdown(T.tile(f"Water Loss Rate{_cmp_icon(p.nrw_pct, p2.nrw_pct)}", f"{p.nrw_pct:.1f}%", f"Rank {p.rate_rank}"), unsafe_allow_html=True)
            st.markdown(T.tile(f"Water Lost{_cmp_icon(p.nrw_m3, p2.nrw_m3)}", f"{m3(p.nrw_m3)} m³", f"Rank {p.volume_rank}"), unsafe_allow_html=True)
            st.markdown(T.tile(f"Burst Rate{_cmp_icon(p.bursts_per_100km, p2.bursts_per_100km)}", f"{p.bursts_per_100km:.1f}", "/100km"), unsafe_allow_html=True)

        with c2:
            st.markdown(f"#### **{p2_name}** (Comparison)")
            st.markdown(T.tile(f"LIPS Score{_cmp_icon(p2.lips, p.lips)}", f"{p2.lips:.1f}", f"Rank {p2.lips_rank}"), unsafe_allow_html=True)
            st.markdown(T.tile(f"Water Loss Rate{_cmp_icon(p2.nrw_pct, p.nrw_pct)}", f"{p2.nrw_pct:.1f}%", f"Rank {p2.rate_rank}"), unsafe_allow_html=True)
            st.markdown(T.tile(f"Water Lost{_cmp_icon(p2.nrw_m3, p.nrw_m3)}", f"{m3(p2.nrw_m3)} m³", f"Rank {p2.volume_rank}"), unsafe_allow_html=True)
            st.markdown(T.tile(f"Burst Rate{_cmp_icon(p2.bursts_per_100km, p.bursts_per_100km)}", f"{p2.bursts_per_100km:.1f}", "/100km"), unsafe_allow_html=True)

        st.markdown("")
        st.markdown("")
        st.markdown('<div class="waverule" style="margin-bottom: 1rem;"></div>', unsafe_allow_html=True)

        # --- Comparative Charts ---
        st.markdown("### :material/monitoring: <b>Comparative Charts</b>", unsafe_allow_html=True)
        st.space()
        chart_col1, chart_col2 = st.columns(2)

        # Chart 1: Historical NRW Rate Comparison
        with chart_col1:
            fig_rate = go.Figure()
            fig_rate.add_trace(go.Scatter(
                x=hist.date, y=hist.nrw_pct, mode="lines+markers", name=plant,
                line=dict(color=T.BLUE, width=2)
            ))
            fig_rate.add_trace(go.Scatter(
                x=hist2.date, y=hist2.nrw_pct, mode="lines+markers", name=p2_name,
                line=dict(color=T.ORANGE, width=2)
            ))
            fig_rate.update_layout(
                title="Loss Rate Trend Comparison (%)",
                height=350,
                xaxis=dict(title=None),
                yaxis=dict(title="NRW Rate (%)", ticksuffix="%"),
                legend=dict(orientation="h", y=1.1)
            )
            chart(fig_rate)

        # Chart 2: Physical vs Commercial Loss Volume Comparison
        with chart_col2:
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Bar(
                x=["Physical Loss", "Commercial Loss"],
                y=[p.physical_loss_m3, p.commercial_loss_m3],
                name=plant,
                marker=dict(color=T.BLUE)
            ))
            fig_vol.add_trace(go.Bar(
                x=["Physical Loss", "Commercial Loss"],
                y=[p2.physical_loss_m3, p2.commercial_loss_m3],
                name=p2_name,
                marker=dict(color=T.ORANGE)
            ))
            fig_vol.update_layout(
                title="Loss Breakdown Comparison (m³)",
                height=350,
                barmode="group",
                yaxis=dict(title="Volume (m³)"),
                legend=dict(orientation="h", y=1.1)
            )
            chart(fig_vol)
