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
                   page_icon="◔", layout="wide",
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
# The appearance control lives in the top strip now; it is read from session
# state here because the theme must be resolved before any CSS is emitted.
# Light by default rather than Auto: a dashboard shown to an examiner, or
# printed into a report, should not change appearance with whatever the
# viewing machine happens to be set to. Auto is still offered.
_pref = st.session_state.get("appearance", "Light")
MODE = theme_mod.resolve_mode(_pref, _detected)
T = theme_mod.Theme(MODE)
st.markdown(T.css, unsafe_allow_html=True)

PLOT_CFG = {"displayModeBar": False, "responsive": True}

# LIPS balances volume density, pipe condition, asset age, and account exposure:
#   nrw_per_km_m3 (40%)    : Combined Loss Density (NRW volume concentration)
#   bursts_per_100km (25%) : Burst Rate (Proxy for physical pipe failure)
#   plant_age_yr (20%)     : Asset Condition / Deterioration risk
#   account_density (15%)  : Commercial & Metering risk exposure
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
    return m, y, x, q, v


@st.cache_data
def load_ml():
    """Model artefacts produced by train_models.py. Training is done offline so
    the outputs are reproducible and auditable rather than refitted per click."""
    try:
        p = pd.read_csv(DATA / "ml_plant.csv")
        mm = pd.read_csv(DATA / "ml_monthly.csv", parse_dates=["date"])
        met = json.loads((DATA / "model_metrics.json").read_text())
        return p, mm, met
    except FileNotFoundError:
        return None, None, None


@st.cache_data
def cluster_at(k: int, yr: int):
    """Re-run KMeans at a chosen k for a given year. Cheap (74 rows), and
    separation is modest at every k, so the operator should be able to try
    alternatives."""
    py = yearly[yearly.year == yr].copy()
    scored, profile, sil, best = tm.archetypes(py, k=k)
    return scored[["plant", "cluster", "archetype"]], profile, sil, best


@st.cache_data
def load_burst():
    """Artefacts from train_burst_model.py. Training happens offline so the
    dashboard shows a fixed, auditable model rather than refitting per click."""
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


monthly, yearly, crosswalk, quality, missing = load()
ml_plant, ml_monthly, ml_metrics = load_ml()
HAS_ML = ml_plant is not None
burst_pred, burst_hist, burst_metrics = load_burst()
HAS_BURST = burst_pred is not None
coverage = load_coverage()

# Everything below derives its year range from the data, so a refresh that adds
# 2026 needs no code change.
YEARS = sorted(int(y) for y in monthly.year.unique())
YEAR_MIN, YEAR_MAX = YEARS[0], YEARS[-1]
YEAR_SPAN = f"{YEAR_MIN}" if YEAR_MIN == YEAR_MAX else f"{YEAR_MIN}–{YEAR_MAX}"
ML_YEAR = int(ml_metrics.get("focus_year", YEAR_MAX)) if HAS_ML else None

# Months observed per year, so partial years can be labelled and annualised.
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
    """Rank plants using the 4-Factor Leakage Intervention Priority Score.

    Calculates percentile ranks (0–100) for loss density, burst rate,
    plant age, and account density within the active cohort.
    """
    w = dict(weights)
    total = sum(w.values()) or 1
    out = df.copy()

    # Ensure derived metric exists if not pre-computed in CSV
    if "account_density" not in out.columns and "customer_accounts" in out.columns and "pipe_length_km" in out.columns:
        out["account_density"] = out.customer_accounts / out.pipe_length_km

    score = pd.Series(0.0, index=out.index)
    for col, wt in w.items():
        if col in out.columns:
            pr = percentile_rank(out[col])
            out[f"pr_{col}"] = pr
            score += pr * (wt / total)

    out["lips"] = score.round(2)

    # Tie-breaking priority: LIPS Score -> NRW Loss Density -> Raw NRW Volume
    out = out.sort_values(["lips", "nrw_per_km_m3", "nrw_m3"], ascending=False)
    out["lips_rank"] = np.arange(1, len(out) + 1)
    out["volume_rank"] = out.nrw_m3.rank(ascending=False, method="first").astype(int)
    out["rate_rank"] = out.nrw_pct.rank(ascending=False, method="first").astype(int)
    out["rank_gap"] = out.rate_rank - out.volume_rank
    return out


def chart(fig, **kw):
    """Render a figure with its chrome forced, then hand it to Streamlit."""
    fig.update_layout(paper_bgcolor=T.SURFACE, plot_bgcolor=T.SURFACE)
    kw.setdefault("width", "stretch")
    kw.setdefault("config", PLOT_CFG)
    kw.setdefault("theme", None)
    return st.plotly_chart(fig, **kw)


def card(title, sub_=None):
    """A real bordered container with a title row, for the figure to sit in."""
    c = st.container(border=True)
    with c:
        st.markdown(f'<div class="card-t">{title}</div>'
                    + (f'<div class="card-s">{sub_}</div>' if sub_ else ""),
                    unsafe_allow_html=True)
    return c


def fmt(n, dp=0):
    return f"{n:,.{dp}f}"


def m3(n):
    """Volumes span five orders of magnitude, so units are scaled per value.
    Uppercase M for millions — a lowercase 'm' beside 'm³' reads as metres."""
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
# Topbar
# ==========================================================================

# The whole view map in one place. Section label, then (label, key) rows.
# "Overview" is the single-screen summary; everything under it is the same
# material at full size with the reasoning attached.
NAV = [
    ("OVERVIEW", [("Command centre", "cmd")]),
    ("PRIORITY", [("Ranking", "rank"),
                  ("Full schedule", "sched"),
                  ("Recovery curve", "curve")]),
    ("LOSS DYNAMIC", [("Rate vs volume", "ratevol"),
                      ("Loss composition", "comp")]),
    ("BURST RISK", [("Risk ranking", "brisk"),
                    ("Performance", "bperf"),
                    ("Model evidence", "bevid"),
                    ("Validation", "bval"),
                    ("Register", "breg"),
                    ("Limitations", "blim")]),
    ("PLANT PROFILE", [("Summary", "psum"),
                       ("What the model sees", "pmodel"),
                       ("History", "phist")]),
]
SEC_PRIORITY = {"rank", "sched", "curve"}
SEC_LOSS = {"ratevol", "comp"}
SEC_BURST = {"brisk", "bperf", "bevid", "bval", "breg", "blim"}
SEC_PLANT = {"psum", "pmodel", "phist"}

if "view" not in st.session_state:
    st.session_state.view = "cmd"

# Filters live in the sidebar so the main plane keeps every one of its ~900
# usable vertical pixels for content, and so one filter row scopes every view
# rather than each card carrying its own controls.
with st.sidebar:
    _b = st.columns([0.36, 0.64])
    with _b[0]:
        st.image(ASSETS / "logo.png", width=54)
    with _b[1]:
        st.markdown('<div class="sb-brand">PAIP</div>'
                    '<div class="sb-sub">Non-Revenue Water<br>intervention targeting</div>',
                    unsafe_allow_html=True)
    st.markdown('<div class="waverule"></div>', unsafe_allow_html=True)

    # Labels are the widgets' own, not markdown divs above them: Streamlit
    # collapses a markdown container that sits directly before a widget, and
    # the widget then paints over the heading.
    year = st.selectbox("Reporting year", sorted(YEARS, reverse=True), index=0,
                        format_func=year_label)

    # One compact trigger that opens a checkbox panel, rather than three tall
    # multiselects stacked down the rail. Nothing ticked means no filter, so
    # the common case ("everything") needs no state and shows no chips.
    _r_all = sorted(yearly.region.unique())
    _d_all = sorted(yearly.district.unique())
    _a_all = sorted(yearly.area_type.unique())

    def _picked(prefix, options):
        """Ticked boxes, or every option when none is ticked."""
        on = [o for o in options if st.session_state.get(f"{prefix}{o}", False)]
        return on or list(options)

    _n_on = sum(1 for pre, opts in (("f_r_", _r_all), ("f_d_", _d_all),
                                    ("f_a_", _a_all))
                for o in opts if st.session_state.get(f"{pre}{o}", False))
    _label = f"Filters · {_n_on}" if _n_on else "Filters"

    with st.popover(_label, width="stretch"):
        st.markdown('<div class="pop-h">Tick to narrow. Nothing ticked means '
                    'everything.</div>', unsafe_allow_html=True)

        def _clear():
            # Same reason as the nav buttons: a body-level st.rerun() here
            # would abort before the Appearance radio registers and silently
            # reset the theme.
            for pre, opts in (("f_r_", _r_all), ("f_d_", _d_all),
                              ("f_a_", _a_all)):
                for o in opts:
                    st.session_state[f"{pre}{o}"] = False

        st.button("Clear all", key="f_clear", type="tertiary", on_click=_clear)

        for _title, _pre, _opts, _cols in (("Region", "f_r_", _r_all, 4),
                                           ("District", "f_d_", _d_all, 4),
                                           ("Area type", "f_a_", _a_all, 4)):
            st.markdown(f'<div class="pop-s">{_title}</div>',
                        unsafe_allow_html=True)
            _cc = st.columns(_cols)
            for _i, _o in enumerate(_opts):
                with _cc[_i % _cols]:
                    st.checkbox(_o, key=f"{_pre}{_o}")

    regions = _picked("f_r_", _r_all)
    # District options are not narrowed by region here: the panel shows every
    # district at once, and a district outside the chosen regions simply
    # intersects to nothing, which the empty-selection guard already reports.
    districts = _picked("f_d_", _d_all)
    areas = _picked("f_a_", _a_all)

    st.markdown('<div class="waverule"></div>', unsafe_allow_html=True)

    # Navigation. Buttons rather than a radio group because the sections need
    # headings between them, and a single radio cannot carry those; this also
    # gives the active row its own styling hook.
    def _go(k):
        st.session_state.view = k

    for _sec, _items in NAV:
        st.markdown(f'<div class="nav-h">{_sec}</div>', unsafe_allow_html=True)
        for _label, _key in _items:
            _active = st.session_state.view == _key
            st.markdown(
                f'<div class="nav-row{" nav-on" if _active else ""}"></div>',
                unsafe_allow_html=True)
            # on_click, NOT a body-level st.rerun(). st.rerun() aborts the
            # script at this point, so the Appearance radio further down never
            # registers on that run — Streamlit then garbage-collects its
            # widget state, and the next run falls back to the "Light" default.
            # Navigating silently reset the theme. A callback runs before the
            # script body instead, so every widget still registers.
            st.button(_label, key=f"nav_{_key}", width="stretch",
                      type="primary" if _active else "tertiary",
                      on_click=_go, args=(_key,))

    st.markdown('<div class="waverule"></div>', unsafe_allow_html=True)
    st.radio("Appearance", ["Light", "Dark", "Auto"], horizontal=True,
             key="appearance",
             help=f"Auto follows your system setting (detected: {_detected}).")
    st.caption(f"Source: PAIP monthly records, {YEAR_SPAN}.")

if is_partial(year):
    st.markdown(
        f'<div class="caption" style="margin:2px 0 6px 0">⚠ {year} is still in '
        f'progress ({MONTHS_BY_YEAR[year]} of 12 months). Volume totals are '
        f'actuals; charts comparing years annualise them. Rates are unaffected.'
        f'</div>', unsafe_allow_html=True)

mask = (yearly.year == year) & yearly.region.isin(regions) \
       & yearly.district.isin(districts) & yearly.area_type.isin(areas)
sel = yearly[mask].copy()
mmask = (monthly.year == year) & monthly.region.isin(regions) \
        & monthly.district.isin(districts) & monthly.area_type.isin(areas)
msel = monthly[mmask].copy()

if sel.empty:
    st.error("No plants match the current filters. Widen the selection under "
             "Scope in the sidebar.")
    st.stop()

# The models are fitted for one focus year. Merging those scores onto another
# year would mislabel them, so they attach only when the year matches.
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

# LIPS weights are fixed in DEFAULT_WEIGHTS rather than exposed as sliders: a
# priority score that changes when the viewer drags a control is not a queue,
# it is an opinion generator. The model-based Criticality Index is deliberately
# NOT folded in either — it measures how abnormal a plant looks, not how urgent
# it is, and the Plant Profile keeps it as a separate signal.
weights = dict(DEFAULT_WEIGHTS)
lips_weights = dict(weights)
sel = score_lips(sel, tuple(sorted(lips_weights.items())))

tot_prod = sel.production_m3.sum()
tot_nrw = sel.nrw_m3.sum()
tot_val = sel.nrw_value_rm.sum()
tot_phys = sel.physical_loss_m3.sum()
sys_pct = tot_nrw / tot_prod * 100
n_plants = len(sel)

prev = yearly[(yearly.year == year - 1) & yearly.plant.isin(sel.plant)]
prev_pct = (prev.nrw_m3.sum() / prev.production_m3.sum() * 100) if len(prev) else np.nan


# ==========================================================================
# Header
# ==========================================================================

# One slim strip instead of a title block plus tall tiles: on a 1080p screen
# the old header consumed 320px, over a third of the viewport, before any
# content appeared. This does the same job in about 110px.
above = int((sel.nrw_pct > T.POLICY_TARGET_PCT).sum())
delta = ""
if not np.isnan(prev_pct):
    _d = sys_pct - prev_pct
    _cls = "kpi-good" if _d < 0 else "kpi-bad"
    delta = (f'<span class="{_cls}">{"↓" if _d < 0 else "↑"} '
             f'{abs(_d):.2f} pp</span> vs {year-1}')

# Five tiles, one rail — the shape the reference dashboards use. Each is a
# stat tile rather than a chart because each is a single number: a one-bar bar
# chart would be the anti-pattern.
_top10_share = (sel.nsmallest(min(10, n_plants), "lips_rank").nrw_m3.sum()
                / tot_nrw * 100) if tot_nrw else 0.0
_flagged = int(burst_pred.flag.sum()) if HAS_BURST and "flag" in burst_pred else 0

_kpis = [
    ("System loss rate", f"{sys_pct:.1f}%", delta or f"{n_plants} plants"),
    ("Water lost", f"{m3(tot_nrw)} m³", f"{m3(tot_nrw/365)} m³ per day"),
    ("Value forgone", f"RM {rm(tot_val)}", f"across {n_plants} plants"),
    ("Above 25% target", f"{above} of {n_plants}",
     f"{above/n_plants*100:.0f}% of the estate"),
    ("Top-10 priority", f"{_top10_share:.0f}%",
     f"of all NRW, in 10 plants" if not _flagged
     else f"of all NRW · {_flagged} burst-flagged"),
]
_cells = "".join(
    f'<div class="kpi"><div class="kpi-l"><span class="drop"></span>{l}</div>'
    f'<div class="kpi-v">{v}</div><div class="kpi-s">{s_}</div></div>'
    for l, v, s_ in _kpis)
st.markdown(f'<div class="kpistrip">{_cells}</div>', unsafe_allow_html=True)


VIEW = st.session_state.view


def mini(fig, h=214, ylab=None, xlab=None, legend=False):
    """Size a figure for a card. The title lives in the card, not the figure."""
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
# TAB 0 — Command screen
# ==========================================================================
if VIEW == "cmd":
    # Nine cards, but deliberately NOT a uniform 3x3. Column widths vary and
    # one slot is a number rather than a plot, so the eye has somewhere to
    # land first; an even grid of equal cards reads as a spreadsheet.
    #
    # Colour is cut to three roles here. BLUE is "this is the measure", ORANGE
    # appears only where a genuine second series is being contrasted, and grey
    # carries context. The one exception is the component card, where the four
    # LIPS factors are true categories and need their own hues, and the burst
    # card, where the colours mean good/bad and come from the status palette.
    _n8 = min(8, n_plants)
    _top10 = sel.nsmallest(min(10, n_plants), "lips_rank")
    _share10 = _top10.nrw_m3.sum() / tot_nrw * 100 if tot_nrw else 0.0

    r1 = st.columns([0.78, 1.42, 1.2])

    with r1[0]:
        # The hero. A single number is not a chart — a one-bar bar chart here
        # would be the anti-pattern.
        with st.container(border=True):
            st.markdown(
                f'<div class="card-t">Where the water is</div>'
                f'<div class="card-s">Top 10 priority plants, share of all NRW</div>'
                f'<div class="hero-v">{_share10:.0f}<span class="hero-u">%</span></div>'
                f'<div class="hero-s">{m3(_top10.nrw_m3.sum())} m³ of '
                f'{m3(tot_nrw)} m³, in {len(_top10)} of {n_plants} plants.<br>'
                f'Worst plant: <b>{_top10.iloc[0].plant}</b>, LIPS '
                f'{_top10.iloc[0].lips:.0f}.</div>',
                unsafe_allow_html=True)

    with r1[1]:
        with st.container(border=True):
            st.markdown('<div class="card-t">Priority queue</div>'
                        f'<div class="card-s">Top {_n8} plants by LIPS</div>',
                        unsafe_allow_html=True)
            d_ = sel.nsmallest(_n8, "lips_rank").sort_values("lips")
            f = go.Figure(go.Bar(
                x=d_.lips, y=d_.plant, orientation="h",
                # One series, one colour. Shading each bar darker-where-bigger
                # would double-encode length as hue and spend the only free
                # channel on information the bar already carries.
                marker=dict(color=T.BLUE, line=dict(width=0)),
                text=[f"{v:.0f}" for v in d_.lips], textposition="outside",
                textfont=dict(size=10.5, color=T.INK_2), customdata=d_.district,
                hovertemplate=("<b>%{y}</b> · %{customdata}<br>"
                               "LIPS %{x:.1f}<extra></extra>")))
            f.update_xaxes(range=[0, 116], showgrid=True)
            f.update_yaxes(showgrid=False)
            chart(mini(f, h=196))

    with r1[2]:
        with st.container(border=True):
            st.markdown('<div class="card-t">Rate vs volume</div>'
                        '<div class="card-s">The two measures rank plants '
                        'oppositely</div>', unsafe_allow_html=True)
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
            chart(mini(f, h=178, legend=True))

    r2 = st.columns([1.2, 1, 1.2])

    with r2[0]:
        with st.container(border=True):
            st.markdown('<div class="card-t">What drives each score</div>'
                        '<div class="card-s">LIPS component percentiles, '
                        'weighted</div>', unsafe_allow_html=True)
            d_ = sel.nsmallest(_n8, "lips_rank").sort_values("lips")
            f = go.Figure()
            for col, lab, c_ in [("pr_nrw_per_km_m3", "Loss density", T.BLUE),
                                 ("pr_bursts_per_100km", "Burst rate", T.ORANGE),
                                 ("pr_plant_age_yr", "Plant age", T.AQUA),
                                 ("pr_account_density", "Accounts", T.YELLOW)]:
                if col in d_.columns:
                    f.add_trace(go.Bar(
                        x=d_[col], y=d_.plant, orientation="h", name=lab,
                        # A 2px surface gap separates the segments; a drawn
                        # border around each would be the anti-pattern.
                        marker=dict(color=c_,
                                    line=dict(color=T.SURFACE, width=2)),
                        hovertemplate=(f"<b>%{{y}}</b><br>{lab} "
                                       "percentile %{x:.0f}<extra></extra>")))
            f.update_layout(barmode="stack")
            f.update_yaxes(showgrid=False)
            chart(mini(f, h=178, legend=True))

    with r2[1]:
        with st.container(border=True):
            st.markdown(f'<div class="card-t">Monthly loss rate</div>'
                        f'<div class="card-s">{year}, system-wide</div>',
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
            chart(mini(f, h=196))

    with r2[2]:
        with st.container(border=True):
            st.markdown('<div class="card-t">Loss concentration</div>'
                        '<div class="card-s">Cumulative share of NRW, plants '
                        'ranked by volume</div>', unsafe_allow_html=True)
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
            chart(mini(f, h=196))

    r3 = st.columns([1, 1.15, 1.25])

    with r3[0]:
        with st.container(border=True):
            st.markdown('<div class="card-t">Burst risk</div>'
                        '<div class="card-s">Predicted band, next month</div>',
                        unsafe_allow_html=True)
            if HAS_BURST:
                bp = burst_pred.copy()
                bands = [("Low", 0.0, 0.25, T.GOOD),
                         ("Moderate", 0.25, 0.5, T.WARNING),
                         ("High", 0.5, 0.75, T.SERIOUS),
                         ("Critical", 0.75, 1.01, T.CRITICAL)]
                xs = [b[0] for b in bands]
                ys = [int(((bp.risk >= lo) & (bp.risk < hi)).sum())
                      for _, lo, hi, _c in bands]
                f = go.Figure(go.Bar(
                    x=xs, y=ys,
                    marker=dict(color=[b[3] for b in bands], line=dict(width=0)),
                    text=ys, textposition="outside",
                    textfont=dict(size=10.5, color=T.INK_2),
                    hovertemplate="<b>%{x}</b><br>%{y} plants<extra></extra>"))
                # Thin bars: these are the only saturated blocks on the screen
                # and at full width they shouted over everything else.
                f.update_layout(bargap=0.66)
                f.update_yaxes(range=[0, max(ys) * 1.3 if max(ys) else 1],
                               showgrid=False, showticklabels=False)
                chart(mini(f, h=196))
            else:
                st.markdown('<div class="card-s">No burst model artefacts in '
                            'this build.</div>', unsafe_allow_html=True)

    with r3[1]:
        with st.container(border=True):
            st.markdown('<div class="card-t">Loss composition</div>'
                        '<div class="card-s">Largest plants &middot; the split '
                        'is PAIP&rsquo;s assumed apportionment</div>',
                        unsafe_allow_html=True)
            d_ = sel.nlargest(_n8, "nrw_m3").sort_values("nrw_m3")
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
            chart(mini(f, h=178, legend=True))

    with r3[2]:
        with st.container(border=True):
            st.markdown('<div class="card-t">Water lost by district</div>'
                        '<div class="card-s">Total NRW volume</div>',
                        unsafe_allow_html=True)
            d_ = (sel.groupby("district", as_index=False)
                     .agg(v=("nrw_m3", "sum")).sort_values("v"))
            f = go.Figure(go.Bar(
                x=d_.v, y=d_.district, orientation="h",
                marker=dict(color=T.BLUE, line=dict(width=0)),
                hovertemplate="<b>%{y}</b><br>%{x:,.0f} m³<extra></extra>"))
            f.update_yaxes(showgrid=False)
            chart(mini(f, h=196))


if VIEW in SEC_PRIORITY:
    if VIEW == "rank":
        st.markdown("#### Leakage Intervention Priority Score (4-Factor LIPS)")
        st.markdown(T.callout(
            "LIPS prioritizes intervention by combining four operational indicators into a balanced 0–100 score:<br>"
            "• <b>Loss Density (40%)</b>: Concentration of NRW volume per kilometer of pipe.<br>"
            "• <b>Burst Rate (25%)</b>: Bursts per 100 km (proxy for pipe structural failure).<br>"
            "• <b>Plant Age (20%)</b>: Deterioration and asset condition risk.<br>"
            "• <b>Account Density (15%)</b>: Customer connections per km (commercial/metering exposure)."
        ), unsafe_allow_html=True)

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
                customdata=np.stack([top.district, top.nrw_per_km_m3, top.bursts_per_100km, top.lips_rank], -1),
                hovertemplate=("<b>%{y}</b> · %{customdata[0]}<br>"
                               "LIPS Score: %{x:.1f} (Priority %{customdata[3]})<br>"
                               "Loss Density: %{customdata[1]:,.0f} m³/km<br>"
                               "Burst Rate: %{customdata[2]:.1f} /100km<extra></extra>")))
            fig.update_layout(
                title=f"Top {n_show} Plants by LIPS Priority", height=572,
                bargap=0.3,
                xaxis=dict(title="LIPS Score (0–100)", range=[0, 115]),
                yaxis=dict(title=None, tickfont=dict(size=11)))
            chart(fig)

        with c2:
            # Component breakdown chart showing percentiles for each factor
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
                title="LIPS Component Percentile Profile",
                height=572, barmode="stack", bargap=0.3,
                xaxis=dict(title="Weighted Component Contribution"),
                yaxis=dict(title=None, tickfont=dict(size=11)))
            chart(fig)

    if VIEW == "curve":
        st.markdown("#### Recovery curve — how far a crew programme gets")

        order_lips = sel.sort_values("lips_rank")
        order_rate = sel.sort_values("rate_rank")
        total_nrw = sel.nrw_m3.sum()

        def curve(df):
            return df.nrw_m3.cumsum() / total_nrw * 100

        x = np.arange(1, n_plants + 1)
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
        st.markdown("#### Full intervention schedule")
        sched = sel.sort_values("lips_rank")[
            ["lips_rank", "plant", "district", "area_type", "lips",
             "nrw_per_km_m3", "bursts_per_100km", "plant_age_yr", "account_density",
             "nrw_m3", "nrw_pct", "volume_rank", "rate_rank"]]
        st.dataframe(
            sched, width='stretch', hide_index=True, height=545,
            column_config={
                "lips_rank": st.column_config.NumberColumn("Priority", width="small"),
                "plant": "Plant", "district": "District", "area_type": "Area",
                "lips": st.column_config.ProgressColumn("LIPS Score", min_value=0, max_value=100, format="%.1f"),
                "nrw_per_km_m3": st.column_config.NumberColumn("Loss Density (m³/km)", format="%,d"),
                "bursts_per_100km": st.column_config.NumberColumn("Bursts /100km", format="%.1f"),
                "plant_age_yr": st.column_config.NumberColumn("Plant Age (yr)", format="%.0f"),
                "account_density": st.column_config.NumberColumn("Acc Density (/km)", format="%.1f"),
                "nrw_m3": st.column_config.NumberColumn("NRW m³", format="%,d"),
                "nrw_pct": st.column_config.NumberColumn("Rate", format="%.1f%%"),
                "volume_rank": st.column_config.NumberColumn("Vol Rank"),
                "rate_rank": st.column_config.NumberColumn("Rate Rank")})
        st.download_button("Download schedule (CSV)", sched.to_csv(index=False),
                           f"paip_lips_schedule_{year}.csv", "text/csv")

if VIEW in SEC_BURST:
    if not HAS_BURST:
        st.warning("Burst-risk artefacts not found. Run "
                   "`python train_burst_model.py`, then reload.")
        st.stop()

    bm = burst_metrics
    BURST_MIN = bm["target"].split(">= ")[-1]
    res = bm["results"]
    best = bm["selected_model"]
    cmx = bm["confusion_matrix"]
    tp, fp, fn, tn = cmx["tp"], cmx["fp"], cmx["fn"], cmx["tn"]
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    NAMES = {"majority": "Majority class", "persistence": "Persistence rule",
             "logistic": "Logistic regression", "forest": "Random forest",
             "boosting": "Gradient boosting"}

    # Sub-tab strip replaced by the sidebar nav.
    if VIEW == "brisk":

        st.markdown(f"#### Which plants will suffer bursts next month? · {bm['horizon']}")
        k = st.columns(5)
        k[0].markdown(T.tile("Discrimination", f"{res[best]['test_roc_auc']:.3f}", "AUC",
                             f"{NAMES[best]} on 6 unseen months"), unsafe_allow_html=True)
        k[1].markdown(T.tile("Precision", f"{prec*100:.0f}", "%",
                             f"Of plants flagged, {prec*100:.0f}% did have an elevated month"),
                      unsafe_allow_html=True)
        k[2].markdown(T.tile("Recall", f"{rec*100:.0f}", "%",
                             f"Of elevated months that occurred, {rec*100:.0f}% were caught"),
                      unsafe_allow_html=True)
        k[3].markdown(T.tile("Beats persistence by",
                             f"+{(res[best]['test_roc_auc']-res['persistence']['test_roc_auc'])*100:.0f}",
                             "pts AUC",
                             f"Naive rule scores {res['persistence']['test_roc_auc']:.3f}"),
                      unsafe_allow_html=True)
        k[4].markdown(T.tile("Flagged next month", f"{bm['n_flagged']}",
                             f"of {len(burst_pred)}",
                             f"At the {bm['operating_threshold']:.2f} decision threshold"),
                      unsafe_allow_html=True)

        st.markdown("")
        c1, c2 = st.columns([1.55, 1])

        with c1:
            st.markdown("###### Ranked risk for next month")
            bp = burst_pred.copy()
            if len(districts) < yearly.district.nunique():
                bp = bp[bp.district.isin(districts)]
            n_show = min(16, len(bp))
            top = bp.nsmallest(n_show, "risk_rank").sort_values("risk_pct")
            BAND_COLOR = {"Critical": T.CRITICAL, "High": T.SERIOUS,
                          "Moderate": T.WARNING, "Low": T.GOOD}
            fig = go.Figure(go.Bar(
                x=top.risk_pct, y=top.plant, orientation="h",
                marker=dict(color=[BAND_COLOR.get(b, T.BLUE) for b in top.risk_band],
                            line=dict(color=T.SURFACE, width=2)),
                text=[f"{v:.0f}%  {b}" for v, b in zip(top.risk_pct, top.risk_band)],
                textposition="outside", textfont=dict(size=10.5, color=T.INK_2),
                customdata=np.stack([top.district, top.pipe_bursts,
                                     top.bursts_roll3, top.risk_rank], -1),
                hovertemplate=("<b>%{y}</b> · %{customdata[0]}<br>"
                               "Risk  %{x:.1f}%  (rank %{customdata[3]})<br>"
                               "Bursts this month  %{customdata[1]:.0f}<br>"
                               "3-month average  %{customdata[2]:.1f}"
                               "<extra></extra>")))
            fig.add_vline(x=bm["operating_threshold"] * 100,
                          line=dict(color=T.BASELINE, width=1.5, dash="dash"),
                          annotation_text="dispatch threshold",
                          annotation_position="top",
                          annotation_font=dict(size=10.5, color=T.MUTED))
            fig.update_layout(
                title=f"Probability of an elevated-burst month · {bm['horizon']}",
                height=520, bargap=0.3,
                xaxis=dict(title="Predicted probability (%)", range=[0, 118]),
                yaxis=dict(title=None, tickfont=dict(size=11)))
            chart(fig)
            st.markdown(
                '<div class="caption">Every bar names its band, so the ranking '
                'never depends on colour alone.</div>', unsafe_allow_html=True)

        with c2:
            st.markdown("###### How to read this")
            st.markdown(
                f'<div class="caption">A supervised classifier trained on a '
                f'<b>real outcome</b> — the burst count PAIP already records. '
                f'Features come from month <i>t</i>, the label from <i>t+1</i>, '
                f'so it predicts a future event and can be scored against '
                f'whether it happened.<br><br>The target is an <b>elevated '
                f'month ({BURST_MIN}+ bursts)</b>, not "any burst": 98.1% of '
                f'plant-months already record at least one, so "any burst" is a '
                f'constant rather than a prediction. Elevated months occur in '
                f'{bm["positive_rate"]:.1%} of cases.<br><br>Bars past the '
                f'dashed line are the plants the model would dispatch a crew '
                f'to. See <b>Performance</b> for how well it did on months it '
                f'never saw.</div>', unsafe_allow_html=True)

    if VIEW == "bperf":
        st.markdown("###### Was it right? Last 6 months, held out")
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=bm["roc_curve"]["fpr"], y=bm["roc_curve"]["tpr"], mode="lines",
                name=f"{NAMES[best]} (AUC {res[best]['test_roc_auc']:.3f})",
                line=dict(color=T.BLUE, width=2.5), fill="tozeroy",
                fillcolor=T.TILE_WASH,
                hovertemplate=("False positive rate  %{x:.2f}<br>"
                               "True positive rate  %{y:.2f}<extra></extra>")))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", name="Random guessing",
                line=dict(color=T.MUTED, width=1.5, dash="dash"), hoverinfo="skip"))
            fig.update_layout(
                title="ROC curve", height=610,
                xaxis=dict(title="False positive rate", range=[0, 1]),
                yaxis=dict(title="True positive rate", range=[0, 1.02]))
            chart(fig)

        with c2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=bm["pr_curve"]["recall"], y=bm["pr_curve"]["precision"],
                mode="lines", name=f"Model (PR-AUC {res[best]['test_pr_auc']:.3f})",
                line=dict(color=T.BLUE, width=2.5),
                hovertemplate="Recall  %{x:.2f}<br>Precision  %{y:.2f}<extra></extra>"))
            fig.add_hline(y=bm["positive_rate"],
                          line=dict(color=T.MUTED, width=1.5, dash="dash"),
                          annotation_text=f"base rate {bm['positive_rate']:.0%}",
                          annotation_position="bottom right",
                          annotation_font=dict(size=10.5, color=T.MUTED))
            fig.update_layout(
                title="Precision–recall curve", height=610,
                xaxis=dict(title="Recall", range=[0, 1]),
                yaxis=dict(title="Precision", range=[0, 1.02]))
            chart(fig)
            st.markdown(
                f'<div class="caption">Precision–recall is the honest curve when '
                f'classes are uneven: a model that flagged everything would sit at '
                f'the dashed base rate of {bm["positive_rate"]:.0%}.</div>',
                unsafe_allow_html=True)

    if VIEW == "bevid":
        st.markdown("#### Model evidence")
        c3, c4, c5 = st.columns([1, 1, 1])

        with c3:
            st.markdown("###### Confusion matrix")
            z = [[tn, fp], [fn, tp]]
            labels = [["True negative", "False positive"],
                      ["False negative", "True positive"]]
            fig = go.Figure(go.Heatmap(
                z=z, x=["Predicted quiet", "Predicted elevated"],
                y=["Actually quiet", "Actually elevated"],
                colorscale=T.SEQ, showscale=False,
                text=[[f"{labels[i][j]}<br><b>{z[i][j]}</b>" for j in range(2)]
                      for i in range(2)],
                texttemplate="%{text}", textfont=dict(size=13),
                hovertemplate="%{y} · %{x}<br>%{z} cases<extra></extra>"))
            fig.update_layout(title=f"At threshold {bm['operating_threshold']:.2f}",
                              height=560, xaxis=dict(side="bottom"),
                              yaxis=dict(autorange="reversed"))
            chart(fig)
            st.markdown(
                f'<div class="caption">{fp} false alarms cost a wasted inspection; '
                f'{fn} misses cost an unanticipated burst. The threshold was tuned '
                f'on training data to favour recall — missing a burst is the more '
                f'expensive error.</div>', unsafe_allow_html=True)

        with c4:
            st.markdown("###### Are the probabilities honest?")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                line=dict(color=T.MUTED, width=1.5, dash="dash"), hoverinfo="skip"))
            fig.add_trace(go.Scatter(
                x=bm["calibration"]["predicted"], y=bm["calibration"]["observed"],
                mode="lines+markers", name="Observed",
                line=dict(color=T.BLUE, width=2.5),
                marker=dict(size=9, color=T.BLUE,
                            line=dict(color=T.SURFACE, width=2)),
                hovertemplate=("Predicted  %{x:.0%}<br>Actually happened  %{y:.0%}"
                               "<extra></extra>")))
            fig.update_layout(
                title="Calibration", height=560,
                xaxis=dict(title="Predicted probability", range=[0, 1], tickformat=".0%"),
                yaxis=dict(title="Observed frequency", range=[0, 1], tickformat=".0%"))
            chart(fig)
            st.markdown(
                f'<div class="caption">The points sit on the diagonal, so a stated '
                f'80% really does mean roughly 80%. Brier score '
                f'<b>{res[best]["brier"]:.3f}</b>. This is what makes the risk '
                f'percentages usable as probabilities rather than just a ranking.'
                f'</div>', unsafe_allow_html=True)

        with c5:
            st.markdown("###### What the model relies on")
            gi = pd.DataFrame(bm["grouped_importances"]).sort_values("importance")
            fig = go.Figure(go.Bar(
                x=gi.importance, y=gi.group, orientation="h",
                error_x=dict(type="data", array=gi["std"], color=T.MUTED,
                             thickness=1, width=3),
                marker=dict(color=T.BLUE, line=dict(color=T.SURFACE, width=2)),
                customdata=gi.n_features,
                hovertemplate=("<b>%{y}</b><br>AUC drop  %{x:.4f}<br>"
                               "%{customdata} features<extra></extra>")))
            fig.add_vline(x=0, line=dict(color=T.BASELINE, width=1.5))
            fig.update_layout(
                title="Grouped permutation importance", height=560,
                xaxis=dict(title="Drop in AUC when the group is shuffled"),
                yaxis=dict(title=None, tickfont=dict(size=10.5)))
            chart(fig)
            st.markdown(
                '<div class="caption">Whole families are shuffled together. '
                'Shuffling one burst-history column alone proves nothing — six '
                'others carry the same information and cover for it, which is how '
                'a single-feature ranking put <i>calendar month</i> on top despite '
                'the elevated-burst rate varying only between 35% and 47% across '
                'the year.</div>', unsafe_allow_html=True)

    if VIEW == "bval":
        c6, c7 = st.columns([1, 1])
        with c6:
            st.markdown("###### Every model tested")
            rows = []
            for kind in ["majority", "persistence", "logistic", "forest", "boosting"]:
                r = res[kind]
                rows.append({
                    "Model": NAMES[kind] + ("  ✓ selected" if kind == best else ""),
                    "CV AUC": r.get("cv_roc_auc", float("nan")),
                    "Test AUC": r["test_roc_auc"],
                    "PR-AUC": r["test_pr_auc"],
                    "F1": r["test_f1"],
                    "Brier": r["brier"]})
            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True,
                         column_config={
                             "CV AUC": st.column_config.NumberColumn(format="%.3f"),
                             "Test AUC": st.column_config.NumberColumn(format="%.3f"),
                             "PR-AUC": st.column_config.NumberColumn(format="%.3f"),
                             "F1": st.column_config.NumberColumn(format="%.3f"),
                             "Brier": st.column_config.NumberColumn(format="%.3f")})
            st.markdown(f"""
    <div class="caption">

    <b>Two baselines, both beaten.</b> <i>Majority class</i> predicts "quiet" for
    everyone and scores AUC 0.500 — the floor. <i>Persistence</i> is the rule a
    planner would use without any model: "next month looks like this month". It
    reaches {res['persistence']['test_roc_auc']:.3f}, which is genuinely
    informative — so the model has to beat that, not just beat chance. It does, by
    {(res[best]['test_roc_auc']-res['persistence']['test_roc_auc'])*100:.0f} AUC
    points.

    <b>Selected on cross-validation, reported on a held-out window.</b> The winner
    was chosen by expanding-window CV inside the training period; the final six
    months were never used to pick the model, only to report it.

    </div>""", unsafe_allow_html=True)

        with c7:
            st.markdown("###### How it was validated")
            st.markdown(f"""
    <div class="caption">

    <b>Splits are chronological, never random.</b> Every fold trains strictly on
    months that precede the ones it is scored on. A random split would let the model
    see {YEAR_MAX} while predicting {YEAR_MIN}, which inflates every metric.

    <b>Note this differs from the expected-loss model deliberately.</b> That one
    grouped by plant, because the question was "does this work on a plant we have
    never seen". Here the plants are fixed and known, and the question is "does this
    work on a month that has not happened yet" — so time, not plant, is what must be
    held out.

    <b>No leakage.</b> Features come from month <i>t</i>; the label is month
    <i>t+1</i>. Rolling windows look only backwards. `shift(-1)` appears once in the
    whole module, on the target.

    <b>The threshold was tuned on training data</b> ({bm['operating_threshold']:.3f},
    maximising F1). Tuning it on the test window would make the precision and recall
    above flattering rather than honest.

    <b>Train</b> {bm['n_train']:,} rows to {bm['train_end']} ·
    <b>Test</b> {bm['n_test']:,} rows over {bm['test_months']} months ·
    <b>Base rate</b> {bm['positive_rate']:.1%}

    </div>""", unsafe_allow_html=True)

    if VIEW == "breg":
        st.markdown("#### Full risk register")
        reg = burst_pred.sort_values("risk_rank")[
            ["risk_rank", "plant", "district", "area_type", "risk_pct", "risk_band",
             "pipe_bursts", "bursts_roll3", "bursts_per_100km", "elevated_share6",
             "plant_age_yr", "pipe_length_km", "nrw_pct"]]
        st.dataframe(
            reg, width='stretch', hide_index=True, height=560,
            column_config={
                "risk_rank": st.column_config.NumberColumn("#", width="small"),
                "plant": "Plant", "district": "District", "area_type": "Area",
                "risk_pct": st.column_config.ProgressColumn(
                    "Risk", min_value=0, max_value=100, format="%.1f%%"),
                "risk_band": "Band",
                "pipe_bursts": st.column_config.NumberColumn("Bursts now"),
                "bursts_roll3": st.column_config.NumberColumn("3-mo avg", format="%.1f"),
                "bursts_per_100km": st.column_config.NumberColumn("Per 100km", format="%.1f"),
                "elevated_share6": st.column_config.NumberColumn("Elevated 6mo", format="%.2f"),
                "plant_age_yr": st.column_config.NumberColumn("Age yr", format="%.0f"),
                "pipe_length_km": st.column_config.NumberColumn("Network km", format="%.0f"),
                "nrw_pct": st.column_config.NumberColumn("NRW", format="%.1f%%")})
        st.download_button("Download risk register (CSV)", reg.to_csv(index=False),
                           f"paip_burst_risk_{bm['horizon'].replace(' ', '_')}.csv",
                           "text/csv")

    if VIEW == "blim":
        st.markdown("#### Limitations")
        st.markdown(f"""
<div class="caption">

<b>It predicts one month ahead, not a specific pipe.</b> The unit is the plant,
so the output says "this plant is likely to see multiple bursts next month" —
not where on the network, and not when in the month.

<b>Burst counts are a reporting artefact as well as a physical one.</b> A plant
with more staff or better reporting records more bursts. The model partly learns
reporting behaviour, and no field in this dataset can separate the two.

<b>Structure beats history here — which is worth knowing.</b> Grouped importance
puts asset condition and network shape well above burst history. Recent bursts
do predict (the persistence rule reaches
{res['persistence']['test_roc_auc']:.2f} on its own), but once the model knows a
plant is large, old and long-networked, last month's count adds little. Burst
risk on this estate is mostly <i>structural</i>.

<b>One month of horizon, {bm['test_months']} months of test data.</b> A longer
horizon would need re-training and would almost certainly score worse.

<b>The dataset appears synthetic</b> — every published identity holds to the
rounding digit — so these relationships may be partly manufactured. The method
transfers to real PAIP data; the exact AUC may not.

</div>""", unsafe_allow_html=True)


# ==========================================================================
# TAB 2 — Loss Dynamic (rate vs volume, then loss composition)
# ==========================================================================

if VIEW in SEC_LOSS:
    # A single view: the divergence IS the rate-versus-volume argument, so it
    # no longer sits behind a sub-tab of its own. The two ranked queue tables
    # were removed; the dumbbell already shows every plant's position in both
    # rankings, and the full ordering is downloadable from the Priority
    # Schedule.
    if VIEW == "ratevol":
        st.markdown("#### Two measures, two different repair queues")

        rho = spearmanr(sel.nrw_pct, sel.nrw_m3).statistic
        tau = kendalltau(sel.rate_rank, sel.volume_rank).statistic
        n_top = min(10, n_plants)
        top_rate = sel.nsmallest(n_top, "rate_rank")
        top_vol = sel.nsmallest(n_top, "volume_rank")
        overlap = len(set(top_rate.plant) & set(top_vol.plant))
        w_rate, w_vol = top_rate.nrw_m3.sum(), top_vol.nrw_m3.sum()
        ratio = w_vol / w_rate if w_rate else np.nan

        st.markdown(T.callout(
            f"Ranking the same {n_plants} plants by loss <b>rate</b> and by loss "
            f"<b>volume</b> produces a rank correlation of <b>ρ = {rho:.2f}</b> "
            f"(Kendall τ = {tau:.2f}). The two top-{n_top} queues share "
            f"<b>{overlap} plant{'s' if overlap != 1 else ''}</b>. The volume queue "
            f"covers <b>{m3(w_vol)} m³</b> of losses against <b>{m3(w_rate)} m³</b> "
            f"for the rate queue — <b>{ratio:.1f}× more water</b> for the same ten "
            f"crew deployments.",
            "crit" if overlap <= 2 else "warn"), unsafe_allow_html=True)

        c1, c2 = st.columns([1.25, 1])

        with c1:
            pl = sel.copy()
            cond = [pl.plant.isin(set(top_rate.plant) & set(top_vol.plant)),
                    pl.plant.isin(top_vol.plant), pl.plant.isin(top_rate.plant)]
            pl["queue"] = np.select(
                cond, [f"Both queues", f"Top {n_top} by volume", f"Top {n_top} by rate"],
                default="Neither")

            order = [f"Top {n_top} by volume", f"Top {n_top} by rate", "Both queues", "Neither"]
            colors = {f"Top {n_top} by volume": T.BLUE, f"Top {n_top} by rate": T.ORANGE,
                    "Both queues": T.AQUA, "Neither": T.NEUTRAL}

            fig = go.Figure()
            for grp in order:
                g = pl[pl.queue == grp]
                if g.empty:
                    continue
                fig.add_trace(go.Scatter(
                    x=g.production_m3, y=g.nrw_pct, mode="markers", name=grp,
                    marker=dict(size=np.sqrt(g.nrw_m3 / pl.nrw_m3.max()) * 44 + 7,
                                color=colors[grp], opacity=0.85,
                                line=dict(color=T.SURFACE, width=2)),
                    customdata=np.stack([g.plant, g.district, g.nrw_m3,
                                         g.volume_rank, g.rate_rank], -1),
                    hovertemplate=("<b>%{customdata[0]}</b> · %{customdata[1]}<br>"
                                   "Production  %{x:,.0f} m³<br>"
                                   "Loss rate  %{y:.1f}%  (rank %{customdata[4]})<br>"
                                   "NRW volume  %{customdata[2]:,.0f} m³  "
                                   "(rank %{customdata[3]})<extra></extra>")))

            # Direct-label only the largest few, offset below each bubble by its
            # own radius so the text clears the mark.
            lab = pl.nlargest(3, "nrw_m3")
            for _, r in lab.iterrows():
                radius = (np.sqrt(r.nrw_m3 / pl.nrw_m3.max()) * 44 + 7) / 2
                fig.add_annotation(x=np.log10(r.production_m3), y=r.nrw_pct,
                                   text=r.plant, showarrow=False,
                                   yshift=-(radius + 14),
                                   font=dict(size=10.5, color=T.INK_2),
                                   bgcolor=T.SURFACE, opacity=0.9, borderpad=2)
            fig.add_hline(y=T.POLICY_TARGET_PCT,
                          line=dict(color=T.GOOD, width=1.2, dash="dash"),
                          annotation_text="25% target",
                          annotation_position="top left",
                          annotation_font=dict(size=10.5, color=T.SUCCESS_TEXT))
            fig.update_layout(
                title="Loss rate against plant size — bubble area is NRW volume",
                height=612,
                xaxis=dict(title="Annual production (m³, log scale)", type="log",
                           dtick=1, minor=dict(showgrid=False)),
                yaxis=dict(title="Loss rate (% of production)", ticksuffix="%"))
            chart(fig)

        with c2:
            st.markdown("###### How far plants move between the two rankings")
            # A dumbbell rather than a slope chart: giving every plant its own
            # row keeps the labels legible, where a two-column slope chart packs
            # volume ranks 1..n on top of each other.
            n_dumb = min(12, n_plants)
            mv = sel.nsmallest(n_dumb, "volume_rank")[
                ["plant", "rate_rank", "volume_rank", "nrw_m3", "nrw_pct"]].copy()
            mv = mv.sort_values("volume_rank", ascending=False)

            fig = go.Figure()
            for _, r in mv.iterrows():
                fig.add_trace(go.Scatter(
                    x=[r.volume_rank, r.rate_rank], y=[r.plant, r.plant],
                    mode="lines", line=dict(color=T.NEUTRAL, width=2.5),
                    showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(
                x=mv.volume_rank, y=mv.plant, mode="markers", name="Rank by volume",
                marker=dict(size=11, color=T.BLUE,
                            line=dict(color=T.SURFACE, width=2)),
                customdata=mv.nrw_m3,
                hovertemplate=("<b>%{y}</b><br>Volume rank  %{x}<br>"
                               "NRW  %{customdata:,.0f} m³<extra></extra>")))
            fig.add_trace(go.Scatter(
                x=mv.rate_rank, y=mv.plant, mode="markers+text", name="Rank by rate",
                marker=dict(size=11, color=T.ORANGE,
                            line=dict(color=T.SURFACE, width=2)),
                text=[f"  {v}" for v in mv.rate_rank], textposition="middle right",
                textfont=dict(size=10.5, color=T.MUTED),
                customdata=mv.nrw_pct,
                hovertemplate=("<b>%{y}</b><br>Rate rank  %{x}<br>"
                               "Loss rate  %{customdata:.1f}%<extra></extra>")))
            fig.update_layout(
                title=f"The {len(mv)} largest-volume plants in each ranking",
                height=612,
                xaxis=dict(title="Rank among all plants (1 = highest priority)",
                           range=[0, n_plants + 5]),
                yaxis=dict(title=None, tickfont=dict(size=11)))
            chart(fig)

    if VIEW == "comp":
        st.markdown("#### Physical leakage versus commercial loss")
        st.markdown(T.callout(
            "Physical (real) losses are water escaping the network — pipe repair "
            "and pressure management recover them. Commercial (apparent) losses are "
            "water delivered but not billed — meter under-registration, "
            "unauthorised connections, billing lag — and they need metering and "
            "enforcement instead. The two demand entirely different interventions, "
            "so the split determines <i>which</i> crew to send, not just where."),
            unsafe_allow_html=True)

        c1, c2 = st.columns([1.3, 1])
        with c1:
            n_show = min(16, n_plants)
            comp = sel.nlargest(n_show, "nrw_m3").sort_values("nrw_m3")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=comp.physical_loss_m3, y=comp.plant, orientation="h",
                name="Physical leakage", marker=dict(color=T.BLUE,
                                                     line=dict(color=T.SURFACE, width=2)),
                customdata=comp.physical_share_pct,
                hovertemplate=("<b>%{y}</b><br>Physical  %{x:,.0f} m³ "
                               "(%{customdata:.0f}% of loss)<extra></extra>")))
            fig.add_trace(go.Bar(
                x=comp.commercial_loss_m3, y=comp.plant, orientation="h",
                name="Commercial loss", marker=dict(color=T.ORANGE,
                                                    line=dict(color=T.SURFACE, width=2)),
                customdata=100 - comp.physical_share_pct,
                hovertemplate=("<b>%{y}</b><br>Commercial  %{x:,.0f} m³ "
                               "(%{customdata:.0f}% of loss)<extra></extra>")))
            fig.update_layout(
                title=f"Loss composition, {n_show} largest-loss plants", height=600,
                barmode="stack", bargap=0.3,
                xaxis=dict(title="Volume lost (m³)"),
                yaxis=dict(title=None, tickfont=dict(size=11)))
            chart(fig)


# ==========================================================================
# TAB 4 — Plant Profile
# ==========================================================================

if VIEW in SEC_PLANT:
    st.markdown("#### Plant profile")
    plant_list = sel.sort_values("lips_rank").plant.tolist()
    c0a, c0b = st.columns([1, 2])
    with c0a:
        plant = st.selectbox("Plant", plant_list, index=0)
    p = sel[sel.plant == plant].iloc[0]
    pm = msel[msel.plant == plant].sort_values("date")
    hist = monthly[monthly.plant == plant].sort_values("date")

    with c0b:
        st.markdown(
            f'<div class="caption" style="padding-top:30px">'
            f'<b>{plant}</b> · {p.district} · {p.area_type} · '
            f'{p.plant_age_yr:.0f}-year-old plant · '
            f'{p.pipe_length_km:,.0f} km of main · '
            f'{p.customer_accounts:,.0f} connections · '
            f'{p.population_served:,.0f} people served</div>',
            unsafe_allow_html=True)

    if VIEW == "psum":
        k = st.columns(5)
        k[0].markdown(T.tile("LIPS", f"{p.lips:.1f}", f"rank {p.lips_rank}",
                             f"of {n_plants} plants in the current selection"),
                      unsafe_allow_html=True)
        k[1].markdown(T.tile("Loss rate", f"{p.nrw_pct:.1f}", "%",
                             f"Rank {p.rate_rank} · system average {sys_pct:.1f}%"),
                      unsafe_allow_html=True)
        k[2].markdown(T.tile("Water lost", m3(p.nrw_m3), "m³",
                             f"Rank {p.volume_rank} · {p.nrw_m3/tot_nrw*100:.1f}% of selection total"),
                      unsafe_allow_html=True)
        k[3].markdown(T.tile("Physical share", f"{p.physical_share_pct:.0f}", "%",
                             f"{m3(p.physical_loss_m3)} m³ addressable by repair"),
                      unsafe_allow_html=True)
        k[4].markdown(T.tile("Value at stake", "RM " + rm(p.nrw_value_rm), "",
                             f"{p.bursts_per_100km:.1f} bursts per 100 km recorded"),
                      unsafe_allow_html=True)

        # Burst-risk line for this plant, from the supervised classifier.
        if HAS_BURST:
            _b = burst_pred[burst_pred.plant == plant]
            if len(_b):
                _b = _b.iloc[0]
                _band = str(_b.risk_band)
                _col = {"Critical": T.CRITICAL, "High": T.SERIOUS,
                        "Moderate": T.WARNING, "Low": T.SUCCESS_TEXT}.get(_band, T.INK)
                _icon = {"Critical": "▲", "High": "▲", "Moderate": "■",
                         "Low": "●"}.get(_band, "■")
                st.markdown(
                    f'<div class="callout" style="border-left-color:{_col}">'
                    f'<b>Burst risk for {burst_metrics["horizon"]}: '
                    f'<span style="color:{_col}">{_icon} {_b.risk_pct:.0f}% · '
                    f'{_band}</span></b> — ranked {int(_b.risk_rank)} of '
                    f'{len(burst_pred)} plants. '
                    f'{"Above" if _b.will_flag else "Below"} the dispatch threshold '
                    f'of {burst_metrics["operating_threshold"]*100:.0f}%. '
                    f'This plant recorded {int(_b.pipe_bursts)} burst(s) in the '
                    f'latest month and averaged {_b.bursts_roll3:.1f} over the last '
                    f'three. The band is stated in words as well as colour, and the '
                    f'probability is calibrated — 80% here means it happened about '
                    f'80% of the time in testing.</div>',
                    unsafe_allow_html=True)

        st.markdown("")
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
            metrics = [("Loss rate %", "nrw_pct"), ("NRW per km", "nrw_per_km_m3"),
                       ("Bursts /100km", "bursts_per_100km"),
                       ("Plant age", "plant_age_yr"), ("Meter age", "meter_age_yr")]
            labels, pvals, medians = [], [], []
            for label, col in metrics:
                med = peer[col].median()
                if med and not np.isnan(med):
                    labels.append(label)
                    pvals.append(p[col] / med * 100)
                    medians.append(med)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=pvals, y=labels, orientation="h",
                marker=dict(color=[T.CRITICAL if v > 130 else
                                   T.BLUE if v > 70 else T.GOOD for v in pvals],
                            line=dict(color=T.SURFACE, width=2)),
                text=[f"{v:.0f}" for v in pvals], textposition="outside",
                textfont=dict(size=11, color=T.INK_2),
                customdata=medians,
                hovertemplate=("<b>%{y}</b><br>%{x:.0f}% of peer median<br>"
                               "Peer median  %{customdata:,.1f}<extra></extra>")))
            fig.add_vline(x=100, line=dict(color=T.BASELINE, width=1.5, dash="dash"),
                          annotation_text="peer median",
                          annotation_position="bottom right",
                          annotation_font=dict(size=10.5, color=T.MUTED))
            fig.update_layout(
                title=f"Against {p.area_type} peers (n={len(peer)}), median = 100",
                height=300, bargap=0.35,
                xaxis=dict(title="% of peer median",
                           range=[0, max(pvals) * 1.25 if pvals else 200]),
                yaxis=dict(title=None))
            chart(fig)

    if VIEW == "pmodel":
        if not (HAS_ML and not pd.isna(p.get("criticality", np.nan))):
            st.info(f"No model output for {plant} in {year}. The expected-loss "
                    f"model is fitted for {ML_YEAR}.")
        else:
            st.markdown("###### What the model sees at this plant")
            pm_ml = ml_monthly[ml_monthly.plant == plant].sort_values("date")
            c7, c8 = st.columns([1.5, 1])
            with c7:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=pm_ml.date, y=pm_ml.predicted_nrw_pct, mode="lines",
                    name="Expected from plant characteristics",
                    line=dict(color=T.ORANGE, width=2, dash="dash"),
                    hovertemplate="%{x|%b %Y}<br>Expected  %{y:.1f}%<extra></extra>"))
                fig.add_trace(go.Scatter(
                    x=pm_ml.date, y=pm_ml.nrw_pct, mode="lines",
                    name="Actual", line=dict(color=T.BLUE, width=2.5),
                    hovertemplate="%{x|%b %Y}<br>Actual  %{y:.1f}%<extra></extra>"))
                an = pm_ml[pm_ml.is_anomaly.fillna(False).astype(bool)]
                if len(an):
                    fig.add_trace(go.Scatter(
                        x=an.date, y=an.nrw_pct, mode="markers",
                        name="Anomalous month",
                        marker=dict(size=13, color=T.CRITICAL, symbol="circle-open",
                                    line=dict(width=2.5)),
                        hovertemplate=("%{x|%b %Y}<br>Anomaly · robust z "
                                       "%{customdata:.1f}<extra></extra>"),
                        customdata=an.robust_z))
                fig.update_layout(
                    title=f"{plant} — actual loss against model expectation",
                    height=560, xaxis=dict(title=None),
                    yaxis=dict(title="NRW (%)", ticksuffix="%"))
                chart(fig)
                gap = float(p.unexplained_pp)
                verdict = ("above" if gap > 0 else "below")
                st.markdown(
                    f'<div class="caption">This plant runs <b>{abs(gap):.1f} pp '
                    f'{verdict}</b> what its network characteristics predict — '
                    f'{abs(float(p.unexplained_m3)):,.0f} m³ a year '
                    f'{"unaccounted for" if gap > 0 else "better than expected"}. '
                    f'Pattern: <b>{p.archetype}</b>.</div>',
                    unsafe_allow_html=True)
            with c8:
                sig = pd.DataFrame({
                    "Signal": ["Criticality rank", "Unexplained loss",
                               "Trend (36 mo)", "Recent trend (12 mo)",
                               "Step change (last 6 mo)", "Anomalous months",
                               "Month-to-month volatility"],
                    "Value": [f"{int(p.criticality_rank)} of {n_plants}",
                              f"{p.unexplained_pp:+.1f} pp",
                              f"{p.trend_pp_yr:+.2f} pp/yr (p={p.trend_p:.3f})",
                              f"{p.trend_recent_pp_yr:+.2f} pp/yr",
                              f"{p.step_shift_pp:+.2f} pp (p={p.step_p:.3f})",
                              f"{int(p.anomaly_months)}",
                              f"{p.volatility_pp:.2f} pp"]})
                st.dataframe(sig, width='stretch', hide_index=True,
                             height=300)
                st.markdown('<div class="caption">A negative trend means '
                            'improving. p-values above 0.05 mean the movement is '
                            'not distinguishable from noise.</div>',
                            unsafe_allow_html=True)

    if VIEW == "phist":
        c3, c4 = st.columns(2)
        with c3:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist.date, y=hist.nrw_pct, mode="lines+markers", name="Loss rate",
                line=dict(color=T.BLUE, width=2),
                marker=dict(size=5, color=T.BLUE),
                hovertemplate="%{x|%b %Y}<br>Loss rate  %{y:.1f}%<extra></extra>"))
            fig.add_hline(y=T.POLICY_TARGET_PCT,
                          line=dict(color=T.GOOD, width=1.2, dash="dash"),
                          annotation_text="25% target",
                          annotation_font=dict(size=10.5, color=T.SUCCESS_TEXT))
            fig.update_layout(title="Loss rate history", height=560,
                              showlegend=False, xaxis=dict(title=None),
                              yaxis=dict(title="NRW (%)", ticksuffix="%"))
            chart(fig)

        with c4:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=hist.date, y=hist.pipe_bursts, name="Bursts",
                marker=dict(color=T.ORANGE, line=dict(color=T.SURFACE, width=1)),
                hovertemplate="%{x|%b %Y}<br>%{y:.0f} bursts<extra></extra>"))
            fig.update_layout(title="Recorded pipe bursts", height=560,
                              showlegend=False, bargap=0.15,
                              xaxis=dict(title=None),
                              yaxis=dict(title="Bursts in month"))
            chart(fig)

        with st.expander(f"Monthly records for {plant}"):
            cols = ["date", "production_m3", "billed_m3", "nrw_m3", "nrw_pct",
                    "physical_loss_m3", "commercial_loss_m3", "pipe_bursts",
                    "complaints", "pressure_bar", "rainfall_mm", "nrw_value_rm"]
            st.dataframe(hist[cols], width='stretch', hide_index=True,
                         column_config={
                             "date": st.column_config.DateColumn("Month", format="MMM YYYY"),
                             "production_m3": st.column_config.NumberColumn("Production m³", format="%,d"),
                             "billed_m3": st.column_config.NumberColumn("Billed m³", format="%,d"),
                             "nrw_m3": st.column_config.NumberColumn("NRW m³", format="%,d"),
                             "nrw_pct": st.column_config.NumberColumn("Rate", format="%.1f%%"),
                             "physical_loss_m3": st.column_config.NumberColumn("Physical m³", format="%,d"),
                             "commercial_loss_m3": st.column_config.NumberColumn("Commercial m³", format="%,d"),
                             "pipe_bursts": st.column_config.NumberColumn("Bursts"),
                             "complaints": st.column_config.NumberColumn("Complaints"),
                             "pressure_bar": st.column_config.NumberColumn("Pressure bar", format="%.2f"),
                             "rainfall_mm": st.column_config.NumberColumn("Rain mm", format="%.0f"),
                             "nrw_value_rm": st.column_config.NumberColumn("Value RM", format="%,d")})
