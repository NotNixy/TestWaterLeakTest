"""
PAIP NRW — pattern recognition and early-warning models
=======================================================

This module answers a different question from LIPS. LIPS asks "where is the most
water?". This asks "where is something *wrong*?" — which plants are losing more
than their physical characteristics can account for, which have deteriorated
suddenly, and which are getting worse fastest.

There are no ground-truth labels in this dataset: no record of which plants
actually had a defect found, or which repairs succeeded. Nothing here is a
trained failure classifier, and it is not presented as one. What is modelled is
*expected loss given plant characteristics*; plants far above expectation are
flagged as carrying loss their network does not explain.

Three signals, combined into a Criticality Index:

  1. UNEXPLAINED LOSS   A regression model predicts NRW% from asset, network,
                        operational and environmental features. Three candidates
                        (gradient boosting, ridge, mean baseline) compete and the
                        winner on grouped CV is used — on this estate that is
                        ridge, not the booster. Predictions are out-of-fold under
                        GroupKFold *by plant*, so a plant's expected loss always
                        comes from a model that never saw that plant. The
                        residual is loss the network cannot account for.
  2. SUDDEN DETERIORATION  Robust per-plant z-scores, a global Isolation Forest,
                        and a step-change test comparing the last 6 months
                        against the preceding history.
  3. ACCELERATING TREND Per-plant OLS slope of NRW% over 36 months.

Plus KMeans archetypes, which sort plants into failure patterns that imply
*different* interventions.

Outputs (./data):
    ml_plant.csv          per-plant scores, ranks, archetypes
    ml_monthly.csv        per-plant-month predictions, residuals, anomaly flags
    model_metrics.json    validation table, feature importance, cluster profiles
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore", category=UserWarning)

HERE = Path(__file__).parent
DATA = HERE / "data"
SEED = 42
TARGET = "nrw_pct"
# The focus year is derived from the data, never hard-coded: scoring must follow
# whatever the latest refresh contains. A partial year is still scored — it is
# the one operators most want to act on — but the anomaly and trend tests need a
# minimum history, enforced in main().
MIN_MONTHS_FOR_TRENDS = 12

# ---------------------------------------------------------------------------
# Feature set — LEAKAGE IS THE CENTRAL RISK HERE
# ---------------------------------------------------------------------------
# NRW is defined as production minus billed volume. Any feature carrying billed
# volume therefore encodes the target algebraically, and a model fed one would
# score near-perfectly while learning nothing.
#
# Two columns in the source workbook are exactly this trap and are EXCLUDED:
#   consumption_per_capita_l_day = billed / population   -> leaks
#   revenue_per_account_rm       = billed x tariff / accounts -> leaks
#
# These were verified numerically rather than assumed (see verify.py). Also
# excluded: every NRW-derived measure (nrw_m3, physical/commercial loss,
# nrw_per_km, physical_share, nrw_value, operating margin).
#
# Retained and confirmed safe — each divides by PRODUCTION, never by billed:
#   capacity_utilisation_pct = production / (capacity x days)
#   energy_intensity_kwh_m3  = kWh / production
#   cost_per_m3_rm           = opex / production

NUMERIC_FEATURES = [
    # Asset condition
    "plant_age_yr", "meter_age_yr", "capacity_m3_day", "pipe_length_km",
    # Network scale and density
    "customer_accounts", "population_served", "production_m3",
    "connections_per_km", "pipe_m_per_connection", "population_per_connection",
    # Operations
    "pipe_bursts", "bursts_per_100km", "pressure_bar",
    "capacity_utilisation_pct", "supply_interruption_hr", "complaints",
    "complaints_per_1000_accounts", "staff_count", "accounts_per_staff",
    "energy_intensity_kwh_m3", "cost_per_m3_rm",
    # Environment and raw water
    "rainfall_mm", "temperature_c", "raw_turbidity_ntu",
    "water_quality_compliance_pct",
    # Calendar
    "month",
]
CATEGORICAL_FEATURES = ["area_type", "region"]

# Explicitly banned, asserted at runtime.
BANNED = {
    "nrw_m3", "nrw_pct", "billed_m3", "billed_domestic_m3",
    "billed_commercial_m3", "billed_industrial_m3", "physical_loss_m3",
    "commercial_loss_m3", "physical_share_pct", "commercial_share_pct",
    "nrw_per_km_m3", "physical_loss_per_km_m3", "nrw_value_rm",
    "physical_loss_value_rm", "nrw_sunk_cost_rm", "nrw_per_account_m3",
    "loss_per_connection_l_day", "billed_revenue_rm", "operating_margin_pct",
    "revenue_per_account_rm", "consumption_per_capita_l_day", "nrw_m3_per_day",
    "tariff_rm_m3",
}


def engineer(m: pd.DataFrame) -> pd.DataFrame:
    """Network-density features. Leakage per km of main is strongly shaped by how
    much pipe serves each customer, which none of the raw columns state directly."""
    d = m.copy()
    d["connections_per_km"] = d.customer_accounts / d.pipe_length_km
    d["pipe_m_per_connection"] = d.pipe_length_km * 1000 / d.customer_accounts
    d["population_per_connection"] = d.population_served / d.customer_accounts
    d["complaints_per_1000_accounts"] = d.complaints / d.customer_accounts * 1000
    d["accounts_per_staff"] = d.customer_accounts / d.staff_count
    return d


def assert_no_leakage():
    overlap = (set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES)) & BANNED
    if overlap:
        raise AssertionError(f"Target leakage — banned features present: {overlap}")


# ---------------------------------------------------------------------------
# 1. Expected-loss model
# ---------------------------------------------------------------------------

def make_model(kind: str):
    # The imputer sits INSIDE the pipeline so its medians are learned on each
    # training fold alone — imputing before splitting would leak held-out data.
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    if kind == "gb":
        # HistGradientBoosting handles the raw numerics natively and tolerates
        # the NaNs in pressure/turbidity without imputation.
        pre_gb = ColumnTransformer([
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ])
        # Heavily regularised on purpose. With only 74 plant groups and mostly
        # static per-plant features, an unconstrained booster memorises plant
        # identity: a depth-6 / 400-iteration configuration scored R2 0.21 on
        # grouped CV against 0.38 for this one. Shallow stumps with a large leaf
        # floor and strong L2 generalise to unseen plants far better.
        return Pipeline([("pre", pre_gb),
                         ("m", HistGradientBoostingRegressor(
                             max_iter=60, learning_rate=0.08, max_depth=2,
                             min_samples_leaf=120, l2_regularization=30.0,
                             max_features=0.6, early_stopping=False,
                             random_state=SEED))])
    if kind == "ridge":
        return Pipeline([("pre", pre), ("m", Ridge(alpha=5.0))])
    return Pipeline([("pre", pre), ("m", DummyRegressor(strategy="mean"))])


def fit_expected_loss(d: pd.DataFrame):
    """Out-of-fold predictions under GroupKFold BY PLANT.

    Grouping by plant is the point. Static characteristics (pipe length, age,
    capacity) are near-constant within a plant, so a random split would let the
    model memorise each plant's own loss level and the residual would collapse
    toward zero. Grouping forces every prediction to come from a model that has
    never seen the plant it is scoring, which is what makes the residual
    "unexplained by characteristics" rather than "unexplained by noise".
    """
    X = d[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = d[TARGET].values
    groups = d.plant.values
    gkf = GroupKFold(n_splits=5)

    results, oof = {}, {}
    for kind in ["gb", "ridge", "mean"]:
        pred = np.zeros(len(d))
        for tr, te in gkf.split(X, y, groups):
            mdl = make_model(kind)
            mdl.fit(X.iloc[tr], y[tr])
            pred[te] = mdl.predict(X.iloc[te])
        oof[kind] = pred
        results[kind] = {
            "r2": float(r2_score(y, pred)),
            "mae": float(mean_absolute_error(y, pred)),
            "rmse": float(root_mean_squared_error(y, pred)),
        }

    # Select on grouped-CV R2 rather than defaulting to the most complex model.
    # On this estate the linear model wins: the relationship between plant
    # characteristics and loss rate is largely additive, and with 74 groups the
    # booster spends its capacity on plant-specific structure that does not
    # transfer. Residuals are taken from whichever model actually generalises.
    best = max(("gb", "ridge"), key=lambda k: results[k]["r2"])
    results["selected"] = best
    final = make_model(best).fit(X, y)
    return final, oof, results, X, y, groups, best


def temporal_check(d: pd.DataFrame, focus_year: int) -> dict:
    """Second, independent generalisation test: train on every prior year and
    predict the latest one.

    This answers a different question from the grouped CV — not "does it
    generalise to unseen plants" but "does it generalise forward in time".
    Needs at least one earlier year; returns empty if the data is single-year.
    """
    tr, te = d[d.year < focus_year], d[d.year == focus_year]
    if tr.empty or te.empty:
        return {}
    out = {}
    for kind in ["gb", "ridge", "mean"]:
        mdl = make_model(kind).fit(tr[NUMERIC_FEATURES + CATEGORICAL_FEATURES],
                                   tr[TARGET])
        p = mdl.predict(te[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
        out[kind] = {"r2": float(r2_score(te[TARGET], p)),
                     "mae": float(mean_absolute_error(te[TARGET], p)),
                     "rmse": float(root_mean_squared_error(te[TARGET], p))}
    return out


def importances(kind, X, y, groups) -> pd.DataFrame:
    """Permutation importance on a held-out plant group — measured on data the
    fitted model did not train on, so it reflects genuine predictive use."""
    gkf = GroupKFold(n_splits=5)
    tr, te = next(gkf.split(X, y, groups))
    mdl = make_model(kind).fit(X.iloc[tr], y[tr])
    r = permutation_importance(mdl, X.iloc[te], y[te], n_repeats=12,
                               random_state=SEED, scoring="r2")
    return (pd.DataFrame({"feature": X.columns,
                          "importance": r.importances_mean,
                          "std": r.importances_std})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True))


# ---------------------------------------------------------------------------
# 2. Sudden deterioration
# ---------------------------------------------------------------------------

def robust_z(s: pd.Series) -> pd.Series:
    """Median/MAD z-score. With 36 observations a mean/SD z-score would be
    dragged by the very outliers it is meant to detect; MAD is not."""
    med = s.median()
    mad = (s - med).abs().median()
    scale = mad * 1.4826  # MAD -> sigma for a normal distribution
    if scale < 1e-9:
        scale = s.std(ddof=0) or 1e-9
    return (s - med) / scale


def step_change(s: pd.Series, recent: int = 6) -> tuple:
    """Welch's t-test, last `recent` months against everything before.

    A step change is the signature of a burst main that was never repaired: the
    plant settles at a new, worse baseline rather than spiking and recovering.
    Returns a null result when history is too short to support the test rather
    than reporting a spurious shift.
    """
    s = s.dropna()
    if len(s) < recent + 6:
        return 0.0, 1.0
    a, b = s.iloc[-recent:], s.iloc[:-recent]
    shift = float(a.mean() - b.mean())
    p = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
    return shift, p


def deterioration(monthly: pd.DataFrame) -> pd.DataFrame:
    """Per-plant anomaly, step-change and trend signals over the full 36 months."""
    rows, flags = [], []
    for plant, g in monthly.sort_values("date").groupby("plant"):
        s = g.set_index("date")[TARGET].dropna()
        if len(s) < 3:
            # Too short for any of these tests to mean anything. Emit neutral
            # values so the plant still appears, rather than dropping it.
            rows.append({"plant": plant, "anomaly_months": 0, "worst_z": 0.0,
                         "recent_z": 0.0, "step_shift_pp": 0.0, "step_p": 1.0,
                         "trend_pp_yr": 0.0, "trend_p": 1.0,
                         "trend_recent_pp_yr": 0.0, "volatility_pp": 0.0,
                         "latest_nrw_pct": float(s.iloc[-1]) if len(s) else np.nan})
            continue
        z = robust_z(s)
        shift, p_shift = step_change(s)

        x = np.arange(len(s))
        lr = stats.linregress(x, s.values)
        trend_pp_yr = float(lr.slope * 12)

        # Trend restricted to the most recent 12 months — a plant that has just
        # started deteriorating is invisible in a multi-year slope. Falls back
        # to the full series when fewer than 12 months exist.
        win = min(12, len(s))
        lr12 = stats.linregress(np.arange(win), s.values[-win:])

        rows.append({
            "plant": plant,
            "anomaly_months": int((z.abs() > 3.5).sum()),
            "worst_z": float(z.abs().max()),
            "recent_z": float(z.iloc[-1]),
            "step_shift_pp": shift,
            "step_p": p_shift,
            "trend_pp_yr": trend_pp_yr,
            "trend_p": float(lr.pvalue),
            "trend_recent_pp_yr": float(lr12.slope * 12),
            "volatility_pp": float(s.std(ddof=0)),
            "latest_nrw_pct": float(s.iloc[-1]),
        })
        flags.append(pd.DataFrame({"plant": plant, "date": s.index,
                                   "robust_z": z.values,
                                   "is_anomaly": (z.abs() > 3.5).values}))
    return pd.DataFrame(rows), pd.concat(flags, ignore_index=True)


def isolation_forest(d: pd.DataFrame) -> np.ndarray:
    """Multivariate outlier score over the operating signature of each month.

    The per-plant z-score only sees NRW%. This sees the combination — a month
    with ordinary NRW but abnormal bursts, pressure and complaints together is
    an operating state worth flagging.
    """
    cols = ["nrw_pct", "bursts_per_100km", "pressure_bar",
            "supply_interruption_hr", "complaints_per_1000_accounts",
            "capacity_utilisation_pct", "energy_intensity_kwh_m3"]
    X = d[cols].copy()
    X = X.fillna(X.median())
    Xs = StandardScaler().fit_transform(X)
    iso = IsolationForest(n_estimators=300, contamination=0.05,
                          random_state=SEED)
    iso.fit(Xs)
    # Negate so that larger = more anomalous.
    return -iso.score_samples(Xs)


# ---------------------------------------------------------------------------
# 3. Failure archetypes
# ---------------------------------------------------------------------------

CLUSTER_FEATURES = ["nrw_pct", "physical_share_pct", "bursts_per_100km",
                    "plant_age_yr", "meter_age_yr", "nrw_per_km_m3",
                    "connections_per_km", "pressure_bar"]


def archetypes(py: pd.DataFrame, k: int = None):
    """KMeans over loss signature. Clusters are chosen by silhouette, then named
    from where each centroid sits relative to the estate median — the point is to
    say which *kind* of intervention a plant needs, not merely that it is bad."""
    py = py.copy()
    py["connections_per_km"] = py.customer_accounts / py.pipe_length_km
    X = py[CLUSTER_FEATURES].copy()
    X = X.fillna(X.median())
    Xs = StandardScaler().fit_transform(X)

    scores = {}
    for kk in range(2, 7):
        km = KMeans(n_clusters=kk, n_init=25, random_state=SEED).fit(Xs)
        scores[kk] = float(silhouette_score(Xs, km.labels_))
    # Silhouette picks the default, but the caller may override: separation on
    # this estate is modest at every k, so the "right" number of archetypes is a
    # judgement about operational usefulness, not a value the data dictates.
    best_k = int(k) if k else max(scores, key=scores.get)

    km = KMeans(n_clusters=best_k, n_init=25, random_state=SEED).fit(Xs)
    py["cluster"] = km.labels_

    # Names are derived from the data, not from hand-written rules: each cluster
    # is labelled by the two features whose centroid deviates most from the
    # estate, so labels are always distinct and always backed by the numbers in
    # the profile table.
    LABELS = {
        "nrw_pct": ("high loss rate", "low loss rate"),
        "physical_share_pct": ("physical-leakage dominant", "commercial-loss dominant"),
        "bursts_per_100km": ("burst-prone", "few bursts"),
        "plant_age_yr": ("ageing plant", "newer plant"),
        "meter_age_yr": ("ageing meters", "newer meters"),
        "nrw_per_km_m3": ("concentrated leakage", "dispersed leakage"),
        "connections_per_km": ("dense network", "sparse network"),
        "pressure_bar": ("high pressure", "low pressure"),
    }
    ACTIONS = {
        "physical_share_pct_low": "Meter replacement, billing audit, enforcement",
        "physical_share_pct_high": "Active leak detection and mains repair",
        "bursts_per_100km_high": "Mains rehabilitation and pressure management",
        "nrw_per_km_m3_high": "District metering areas, step-testing",
        "plant_age_yr_high": "Asset renewal programme",
        "pressure_bar_high": "Pressure management and PRV installation",
        "nrw_pct_high": "Network survey and step-testing",
    }

    centroids = pd.DataFrame(km.cluster_centers_, columns=CLUSTER_FEATURES)
    profile, names = [], {}
    for c in sorted(py.cluster.unique()):
        sub = py[py.cluster == c]
        z = centroids.loc[c]
        top = z.abs().sort_values(ascending=False).index[:2]
        parts = [LABELS[f][0 if z[f] > 0 else 1] for f in top]
        name = " · ".join(p.capitalize() if i == 0 else p
                          for i, p in enumerate(parts))
        lead = top[0]
        action = ACTIONS.get(f"{lead}_{'high' if z[lead] > 0 else 'low'}",
                             "Network survey and monitoring")
        names[c] = name
        profile.append({
            "cluster": int(c), "name": name, "action": action,
            "plants": int(len(sub)),
            "median_nrw_pct": float(sub.nrw_pct.median()),
            "median_physical_share": float(sub.physical_share_pct.median()),
            "median_bursts_per_100km": float(sub.bursts_per_100km.median()),
            "median_plant_age": float(sub.plant_age_yr.median()),
            "median_nrw_per_km": float(sub.nrw_per_km_m3.median()),
            "total_nrw_m3": float(sub.nrw_m3.sum()),
            "silhouette": float(scores[best_k]),
        })
    py["archetype"] = py.cluster.map(names)
    return py, profile, scores, best_k


# ---------------------------------------------------------------------------
# 4. Criticality Index
# ---------------------------------------------------------------------------

CRIT_WEIGHTS = {"unexplained": 40, "deterioration": 30, "trend": 30}


def pr(s: pd.Series) -> pd.Series:
    return s.rank(pct=True, method="average") * 100


def criticality(p: pd.DataFrame) -> pd.DataFrame:
    p = p.copy()
    # Only *positive* surprises matter: a plant performing better than predicted
    # is not a problem, so the residual is floored at zero before ranking.
    p["unexplained_signal"] = p.unexplained_pp.clip(lower=0)
    # A step change only counts as evidence when it is statistically supported.
    p["step_signal"] = np.where(p.step_p < 0.05, p.step_shift_pp.clip(lower=0), 0.0)
    p["deterioration_signal"] = (
        pr(p.step_signal) * 0.5 + pr(p.anomaly_score) * 0.3
        + pr(p.worst_z.clip(lower=0)) * 0.2)
    # Trend is measured RELATIVE TO THE ESTATE. Every plant in this dataset is
    # improving, so an absolute "is it worsening" test flags nobody and the
    # component would collapse to a constant. Improving materially slower than
    # your peers is the real signal: the plant is falling behind.
    p["trend_signal"] = (p.trend_pp_yr - p.trend_pp_yr.median()).clip(lower=0)

    comp = {
        "unexplained": pr(p.unexplained_signal),
        "deterioration": pr(p.deterioration_signal),
        "trend": pr(p.trend_signal),
    }
    total = sum(CRIT_WEIGHTS.values())
    score = sum(comp[k] * (w / total) for k, w in CRIT_WEIGHTS.items())
    for k, v in comp.items():
        p[f"pr_{k}"] = v.round(2)
    p["criticality"] = score.round(2)
    # Strict order: ties break on unexplained volume, the most actionable signal.
    p = p.sort_values(["criticality", "unexplained_m3"], ascending=False)
    p["criticality_rank"] = np.arange(1, len(p) + 1)
    return p


# ---------------------------------------------------------------------------

def main():
    assert_no_leakage()
    monthly = engineer(pd.read_csv(DATA / "nrw_plant_month.csv",
                                   parse_dates=["date"]))
    yearly = pd.read_csv(DATA / "nrw_plant_year.csv")

    # Focus on the most recent year present, whatever that is.
    focus_year = int(monthly.year.max())
    py = yearly[yearly.year == focus_year].copy()
    n_months_focus = int(monthly[monthly.year == focus_year].date.dt.month.nunique())
    total_months = int(monthly.date.nunique())
    print(f"Focus year {focus_year} ({n_months_focus} of 12 months) · "
          f"{total_months} months of history")
    if total_months < MIN_MONTHS_FOR_TRENDS:
        print(f"  WARNING: only {total_months} months of history. Trend and "
              f"step-change tests need at least {MIN_MONTHS_FOR_TRENDS} to be "
              f"meaningful; their outputs will be weak or null.")

    print("Fitting expected-loss model (GroupKFold by plant)...")
    model, oof, cv, X, y, groups, best = fit_expected_loss(monthly)
    temporal = temporal_check(monthly, focus_year)
    imp = importances(best, X, y, groups)

    monthly["predicted_nrw_pct"] = oof[best]
    monthly["residual_pp"] = monthly[TARGET] - monthly.predicted_nrw_pct
    monthly["anomaly_score"] = isolation_forest(monthly)

    print("Detecting deterioration...")
    det, flags = deterioration(monthly)
    monthly = monthly.merge(flags, on=["plant", "date"], how="left")

    # Per-plant summary for the focus year.
    focus = monthly[monthly.year == focus_year]
    agg = focus.groupby("plant", as_index=False).agg(
        actual_nrw_pct=(TARGET, "mean"),
        expected_nrw_pct=("predicted_nrw_pct", "mean"),
        unexplained_pp=("residual_pp", "mean"),
        anomaly_score=("anomaly_score", "mean"),
        production_m3=("production_m3", "sum"))
    # Convert the residual into water: how many m3 the plant loses beyond what
    # its characteristics predict.
    agg["unexplained_m3"] = agg.unexplained_pp / 100 * agg.production_m3

    print("Clustering archetypes...")
    py, profile, sil, best_k = archetypes(py)

    p = (py.merge(agg.drop(columns=["production_m3"]), on="plant")
           .merge(det, on="plant"))
    p["projected_nrw_pct_12m"] = (p.latest_nrw_pct + p.trend_pp_yr).clip(0, 100)
    p["projected_extra_m3"] = (p.trend_pp_yr / 100 * p.production_m3).clip(lower=0)
    p = criticality(p)

    p.to_csv(DATA / "ml_plant.csv", index=False)
    monthly[["plant", "date", "year", "month", TARGET, "predicted_nrw_pct",
             "residual_pp", "robust_z", "is_anomaly", "anomaly_score",
             "production_m3", "pipe_bursts"]].to_csv(
        DATA / "ml_monthly.csv", index=False)

    # The deterioration detectors returned a null result on this estate. That is
    # a finding, not a failure, and it is recorded explicitly so the dashboard
    # can state it rather than implying the detectors found something.
    det_summary = {
        "plants_with_worsening_trend": int((p.trend_pp_yr > 0).sum()),
        "plants_with_significant_worsening": int(
            ((p.trend_p < 0.10) & (p.trend_pp_yr > 0)).sum()),
        "plants_with_significant_step_increase": int(
            ((p.step_p < 0.05) & (p.step_shift_pp > 0)).sum()),
        "plants_with_anomaly_months": int((p.anomaly_months > 0).sum()),
        "total_anomaly_months": int(p.anomaly_months.sum()),
        "median_trend_pp_yr": float(p.trend_pp_yr.median()),
        "estate_is_improving": bool(p.trend_pp_yr.median() < 0),
    }

    metrics = {
        "target": TARGET,
        "n_records": int(len(monthly)),
        "n_plants": int(monthly.plant.nunique()),
        "n_features": len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES),
        "excluded_for_leakage": sorted(BANNED),
        "cv_grouped_by_plant": cv,
        "selected_model": best,
        "temporal_holdout_prior_years_to_focus": temporal,
        "focus_year": focus_year,
        "focus_year_months": n_months_focus,
        "years_covered": sorted(int(y) for y in monthly.year.unique()),
        "history_months": total_months,
        "temporal_caveat": (
            "The temporal split shares plants between train and test, so a "
            "flexible model can memorise each plant's own loss level. Its R2 is "
            "therefore optimistic and is NOT the number used for the "
            "unexplained-loss residual. The plant-grouped CV above is the valid "
            "estimate, because there every prediction comes from a model that "
            "never saw that plant."),
        "importances": imp.head(18).to_dict("records"),
        "cluster_silhouette": sil,
        "cluster_k": int(best_k),
        "cluster_profile": profile,
        "criticality_weights": CRIT_WEIGHTS,
        "deterioration_summary": det_summary,
        "seed": SEED,
    }
    (DATA / "model_metrics.json").write_text(json.dumps(metrics, indent=2))

    print("\n--- Expected-loss model, out-of-fold (grouped by plant) ---")
    for k, v in cv.items():
        if not isinstance(v, dict):
            continue
        star = "  <- selected" if k == best else ""
        print(f"  {k:6s} R2={v['r2']:7.3f}  MAE={v['mae']:6.3f} pp  "
              f"RMSE={v['rmse']:6.3f} pp{star}")
    if temporal:
        print(f"\n--- Temporal holdout (train <{focus_year} -> test {focus_year}) ---")
        for k, v in temporal.items():
            print(f"  {k:6s} R2={v['r2']:7.3f}  MAE={v['mae']:6.3f} pp")
    else:
        print("\n--- Temporal holdout: skipped (needs more than one year) ---")
    print(f"\n--- Archetypes (k={best_k}, silhouette={sil[best_k]:.3f}) ---")
    for c in profile:
        print(f"  {c['name']:34s} n={c['plants']:2d}  "
              f"NRW {c['median_nrw_pct']:.1f}%  bursts/100km {c['median_bursts_per_100km']:.0f}")
    print("\n--- Top 10 most critical ---")
    print(p.head(10)[["criticality_rank", "plant", "district", "criticality",
                      "unexplained_pp", "trend_pp_yr", "step_shift_pp",
                      "archetype"]].to_string(index=False))
    print("\n--- Top 8 features ---")
    print(imp.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
