"""
PAIP NRW — burst risk classifier
================================

Predicts which plants will experience an ELEVATED BURST MONTH next month, so
crews can be sent before the failure rather than after it.

Why this target, and not "will there be a burst":
    98.1% of plant-months already record at least one burst, so "any burst" is
    a constant, not a prediction. The operationally meaningful signal is a month
    with MULTIPLE bursts (>= 2), which occurs in 39.8% of plant-months. That is
    a genuinely balanced target and it is the one a maintenance planner acts on.

Why this is a different kind of model from the expected-loss one:
    That model had no ground truth — it inferred "unusual" from a residual.
    This one has a real label already in the data (Kejadian_Pecah_Paip). We
    predict a future event, then check whether it actually happened. That makes
    it supervised learning in the strict sense, and it is why the usual
    classification diagnostics (ROC, precision-recall, confusion matrix,
    calibration) are available here and were not there.

Validation design — TEMPORAL, not grouped:
    The expected-loss model used GroupKFold by plant because the question was
    "does this generalise to a plant we have never seen". Here the question is
    different: the plants are known and fixed, and we forecast FORWARD IN TIME.
    So every split is chronological. Nothing from the future ever reaches the
    training set, and the final test window is the last 6 months, untouched
    until the model was chosen.

Outputs (./data):
    burst_predictions.csv   per-plant risk for the next month, ranked
    burst_history.csv       per plant-month predicted probability vs outcome
    burst_metrics.json      validation table, curves, importances, operating point
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             brier_score_loss, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
DATA = HERE / "data"
SEED = 42
BURST_THRESHOLD = 2      # >= this many bursts = "elevated burst month"
TEST_MONTHS = 6          # final held-out window
CV_FOLDS = 4

NUMERIC = [
    # Burst history — the strongest signal, and legitimately autoregressive:
    # everything here is observed at month t, predicting month t+1.
    "pipe_bursts", "bursts_per_100km", "bursts_roll3", "bursts_roll6",
    "bursts_max6", "elevated_last_month", "elevated_share6",
    # Asset condition
    "plant_age_yr", "meter_age_yr", "pipe_length_km", "capacity_m3_day",
    # Hydraulics — pressure is the classic physical driver of pipe failure
    "pressure_bar", "capacity_utilisation_pct",
    # Loss state
    "nrw_pct", "nrw_per_km_m3",
    # Service symptoms
    "supply_interruption_hr", "complaints", "complaints_per_1000_accounts",
    # Network shape
    "customer_accounts", "population_served", "connections_per_km",
    # Environment and season
    "rainfall_mm", "rainfall_roll3", "temperature_c", "raw_turbidity_ntu",
    "month",
]
CATEGORICAL = ["area_type", "region"]

TARGET = "elevated_next"


def build_features(d: pd.DataFrame) -> pd.DataFrame:
    """All features are observed at month t; the label is month t+1.

    Every rolling window uses only past and present months. `shift(-1)` appears
    exactly once, on the target, and nowhere in the features.
    """
    d = d.sort_values(["plant", "date"]).copy()
    g = d.groupby("plant")

    d["elevated"] = (d.pipe_bursts >= BURST_THRESHOLD).astype(int)
    d["bursts_roll3"] = g.pipe_bursts.transform(lambda s: s.rolling(3, min_periods=1).mean())
    d["bursts_roll6"] = g.pipe_bursts.transform(lambda s: s.rolling(6, min_periods=1).mean())
    d["bursts_max6"] = g.pipe_bursts.transform(lambda s: s.rolling(6, min_periods=1).max())
    d["elevated_last_month"] = g.elevated.shift(1).fillna(0)
    d["elevated_share6"] = g.elevated.transform(lambda s: s.rolling(6, min_periods=1).mean())
    d["rainfall_roll3"] = g.rainfall_mm.transform(lambda s: s.rolling(3, min_periods=1).mean())
    d["connections_per_km"] = d.customer_accounts / d.pipe_length_km
    d["complaints_per_1000_accounts"] = d.complaints / d.customer_accounts * 1000

    # The label: does an elevated-burst month follow?
    d[TARGET] = g.elevated.shift(-1)
    return d


def make_model(kind: str):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    pre_raw = ColumnTransformer([
        ("num", "passthrough", NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    if kind == "logistic":
        return Pipeline([("pre", pre),
                         ("m", LogisticRegression(max_iter=3000, C=1.0,
                                                  random_state=SEED))])
    if kind == "forest":
        return Pipeline([("pre", pre),
                         ("m", RandomForestClassifier(
                             n_estimators=400, min_samples_leaf=8,
                             max_features="sqrt", random_state=SEED, n_jobs=-1))])
    if kind == "boosting":
        return Pipeline([("pre", pre_raw),
                         ("m", HistGradientBoostingClassifier(
                             max_iter=250, learning_rate=0.06, max_depth=4,
                             min_samples_leaf=30, l2_regularization=1.0,
                             random_state=SEED))])
    if kind == "persistence":
        return "persistence"
    return Pipeline([("pre", pre),
                     ("m", DummyClassifier(strategy="prior", random_state=SEED))])


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    """The persistence baseline is a rule, not an estimator: it simply says
    'next month looks like this month'. Included because a model that cannot
    beat it has learned nothing worth deploying."""
    if model == "persistence":
        return X.elevated.astype(float).values
    return model.predict_proba(X)[:, 1]


def temporal_cv(tr: pd.DataFrame, kind: str, folds: int = CV_FOLDS) -> dict:
    """Expanding-window validation over calendar months.

    A plain KFold would let the model train on months that come after the ones
    it is scored on. Every fold here trains strictly on the past.
    """
    months = np.sort(tr.date.unique())
    if len(months) < folds + 4:
        return {}
    edges = np.array_split(months[4:], folds)  # keep a minimum warm-up
    aucs, aps = [], []
    for edge in edges:
        cut = edge[0]
        a, b = tr[tr.date < cut], tr[tr.date.isin(edge)]
        if len(b) == 0 or a[TARGET].nunique() < 2 or b[TARGET].nunique() < 2:
            continue
        m = make_model(kind)
        if m != "persistence":
            m.fit(a, a[TARGET])
        p = predict_proba(m, b)
        aucs.append(roc_auc_score(b[TARGET], p))
        aps.append(average_precision_score(b[TARGET], p))
    if not aucs:
        return {}
    return {"cv_roc_auc": float(np.mean(aucs)), "cv_roc_auc_sd": float(np.std(aucs)),
            "cv_pr_auc": float(np.mean(aps)), "folds": len(aucs)}


def main():
    d = build_features(pd.read_csv(DATA / "nrw_plant_month.csv",
                                   parse_dates=["date"]))
    data = d.dropna(subset=[TARGET]).copy()
    data[TARGET] = data[TARGET].astype(int)

    cutoff = data.date.max() - pd.DateOffset(months=TEST_MONTHS)
    train, test = data[data.date <= cutoff], data[data.date > cutoff]
    print(f"Target: >= {BURST_THRESHOLD} bursts next month "
          f"({data[TARGET].mean():.1%} of plant-months)")
    print(f"Train {len(train):,} rows to {cutoff.date()} | "
          f"Test {len(test):,} rows after")

    # ---- model selection on CV, reported on the untouched test window -----
    results, fitted = {}, {}
    for kind in ["majority", "persistence", "logistic", "forest", "boosting"]:
        cv = temporal_cv(train, kind)
        m = make_model(kind)
        if m != "persistence":
            m.fit(train, train[TARGET])
        fitted[kind] = m
        p = predict_proba(m, test)
        pred = (p >= 0.5).astype(int)
        results[kind] = {
            **cv,
            "test_roc_auc": float(roc_auc_score(test[TARGET], p))
            if len(np.unique(p)) > 1 else 0.5,
            "test_pr_auc": float(average_precision_score(test[TARGET], p)),
            "test_accuracy": float(accuracy_score(test[TARGET], pred)),
            "test_precision": float(precision_score(test[TARGET], pred, zero_division=0)),
            "test_recall": float(recall_score(test[TARGET], pred, zero_division=0)),
            "test_f1": float(f1_score(test[TARGET], pred, zero_division=0)),
            "brier": float(brier_score_loss(test[TARGET], np.clip(p, 0, 1))),
        }

    # Selected on cross-validated AUC among real models — never on the test
    # window, which stays untouched until the choice is made.
    candidates = ["logistic", "forest", "boosting"]
    best = max(candidates, key=lambda k: results[k].get("cv_roc_auc", 0))
    model = fitted[best]
    p_test = predict_proba(model, test)

    # ---- operating point --------------------------------------------------
    # Chosen on the TRAINING data, not the test window, so the threshold is not
    # tuned on the same rows used to report performance.
    p_train = predict_proba(model, train)
    prec, rec, thr = precision_recall_curve(train[TARGET], p_train)
    f1s = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    best_thr = float(thr[int(np.nanargmax(f1s[:-1]))]) if len(thr) else 0.5

    pred_test = (p_test >= best_thr).astype(int)
    cm = confusion_matrix(test[TARGET], pred_test)
    tn, fp, fn, tp = cm.ravel()

    fpr, tpr, _ = roc_curve(test[TARGET], p_test)
    pr_p, pr_r, _ = precision_recall_curve(test[TARGET], p_test)
    frac_pos, mean_pred = calibration_curve(test[TARGET], np.clip(p_test, 0, 1),
                                            n_bins=8, strategy="quantile")

    # ---- feature importance ----------------------------------------------
    # Single-feature permutation is misleading here. Six burst-history features
    # carry nearly the same information, so permuting any one of them leaves the
    # others to compensate and every importance collapses toward zero — which is
    # how `month` ended up ranked first despite the elevated-burst rate varying
    # only between 0.35 and 0.47 across the calendar. Permuting whole FAMILIES
    # together measures what the model actually relies on.
    GROUPS = {
        "Burst history": ["pipe_bursts", "bursts_per_100km", "bursts_roll3",
                          "bursts_roll6", "bursts_max6", "elevated_last_month",
                          "elevated_share6"],
        "Hydraulics": ["pressure_bar", "capacity_utilisation_pct"],
        "Asset condition": ["plant_age_yr", "meter_age_yr", "pipe_length_km",
                            "capacity_m3_day"],
        "Loss state": ["nrw_pct", "nrw_per_km_m3"],
        "Service symptoms": ["supply_interruption_hr", "complaints",
                             "complaints_per_1000_accounts"],
        "Network shape": ["customer_accounts", "population_served",
                          "connections_per_km"],
        "Environment": ["rainfall_mm", "rainfall_roll3", "temperature_c",
                        "raw_turbidity_ntu"],
        "Season": ["month"],
        "Location": ["area_type", "region"],
    }
    rng = np.random.default_rng(SEED)
    base_auc = roc_auc_score(test[TARGET], p_test)
    grouped = []
    for gname, gcols in GROUPS.items():
        drops = []
        for _ in range(10):
            shuffled = test.copy()
            idx = rng.permutation(len(shuffled))
            for c in gcols:
                shuffled[c] = shuffled[c].values[idx]
            drops.append(base_auc - roc_auc_score(
                shuffled[TARGET], predict_proba(model, shuffled)))
        grouped.append({"group": gname, "importance": float(np.mean(drops)),
                        "std": float(np.std(drops)), "n_features": len(gcols)})
    grouped = sorted(grouped, key=lambda r: -r["importance"])

    imp = permutation_importance(model, test, test[TARGET], n_repeats=10,
                                 random_state=SEED, scoring="roc_auc")
    cols = NUMERIC + CATEGORICAL
    importances = (pd.DataFrame({"feature": cols,
                                 "importance": imp.importances_mean[:len(cols)],
                                 "std": imp.importances_std[:len(cols)]})
                   .sort_values("importance", ascending=False).reset_index(drop=True))

    # ---- risk for the month after the data ends ---------------------------
    # The final observed month for each plant carries the features that predict
    # the month that has not happened yet.
    latest = d.sort_values("date").groupby("plant").tail(1).copy()
    latest["risk"] = predict_proba(model, latest)
    latest["risk_pct"] = (latest.risk * 100).round(1)
    latest["will_flag"] = (latest.risk >= best_thr).astype(int)
    latest = latest.sort_values("risk", ascending=False)
    latest["risk_rank"] = np.arange(1, len(latest) + 1)
    latest["risk_band"] = pd.cut(latest.risk, [-0.01, 0.25, 0.5, 0.75, 1.01],
                                 labels=["Low", "Moderate", "High", "Critical"])
    horizon = (latest.date.max() + pd.DateOffset(months=1)).strftime("%B %Y")

    out_cols = ["plant", "district", "region", "area_type", "date", "risk",
                "risk_pct", "risk_rank", "risk_band", "will_flag",
                "pipe_bursts", "bursts_roll3", "bursts_per_100km",
                "elevated_share6", "pressure_bar", "plant_age_yr",
                "pipe_length_km", "nrw_pct", "nrw_m3"]
    latest[out_cols].to_csv(DATA / "burst_predictions.csv", index=False)

    hist = data[["plant", "district", "date", "year", "month", "pipe_bursts",
                 "elevated", TARGET]].copy()
    hist["predicted"] = predict_proba(model, data)
    hist["split"] = np.where(data.date <= cutoff, "train", "test")
    hist.to_csv(DATA / "burst_history.csv", index=False)

    metrics = {
        "target": f"bursts_next_month >= {BURST_THRESHOLD}",
        "target_rationale": (
            "98.1% of plant-months already record at least one burst, so 'any "
            "burst' is a constant rather than a prediction. An elevated month "
            "(>= 2 bursts) occurs in 39.8% of plant-months and is the event a "
            "maintenance planner can act on."),
        "positive_rate": float(data[TARGET].mean()),
        "n_train": int(len(train)), "n_test": int(len(test)),
        "train_end": str(cutoff.date()), "test_months": TEST_MONTHS,
        "validation": "expanding-window temporal CV; final 6 months held out",
        "selected_model": best,
        "selected_on": "cross-validated ROC-AUC (test window never used for selection)",
        "results": results,
        "operating_threshold": best_thr,
        "operating_point_note": (
            "Threshold maximises F1 on the TRAINING data. Tuning it on the test "
            "window would make the reported precision and recall optimistic."),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "pr_curve": {"precision": pr_p.tolist(), "recall": pr_r.tolist()},
        "calibration": {"predicted": mean_pred.tolist(), "observed": frac_pos.tolist()},
        "importances": importances.head(15).to_dict("records"),
        "grouped_importances": grouped,
        "importance_note": (
            "Grouped permutation is the headline measure. Single-feature "
            "permutation is unreliable here because seven burst-history "
            "features carry nearly the same information and substitute for one "
            "another when any single one is shuffled."),
        "base_test_auc": float(base_auc),
        "horizon": horizon,
        "n_flagged": int(latest.will_flag.sum()),
        "seed": SEED,
    }
    (DATA / "burst_metrics.json").write_text(json.dumps(metrics, indent=2))

    print("\n--- Model comparison ---")
    print(f"{'model':14s}{'CV AUC':>9s}{'test AUC':>10s}{'PR-AUC':>9s}"
          f"{'prec':>7s}{'recall':>8s}{'F1':>7s}")
    for k, v in results.items():
        star = "  <- selected" if k == best else ""
        print(f"{k:14s}{v.get('cv_roc_auc', float('nan')):>9.3f}"
              f"{v['test_roc_auc']:>10.3f}{v['test_pr_auc']:>9.3f}"
              f"{v['test_precision']:>7.2f}{v['test_recall']:>8.2f}"
              f"{v['test_f1']:>7.2f}{star}")
    print(f"\nOperating threshold {best_thr:.3f}  |  confusion matrix "
          f"TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"Precision {tp/max(tp+fp,1):.2f} · Recall {tp/max(tp+fn,1):.2f}")
    print(f"\nForecast horizon: {horizon} · {latest.will_flag.sum()} of "
          f"{len(latest)} plants flagged")
    print("\n--- Top 10 at risk ---")
    print(latest.head(10)[["risk_rank", "plant", "district", "risk_pct",
                           "risk_band", "pipe_bursts"]].to_string(index=False))
    print("\n--- Feature groups (permuted together) ---")
    for r in grouped:
        bar = "#" * max(0, int(r["importance"] * 300))
        print(f"  {r['group']:20s} {r['importance']:+.4f} AUC  {bar}")
    print("\n--- Single features (unreliable under collinearity) ---")
    print(importances.head(6).to_string(index=False))


if __name__ == "__main__":
    main()
