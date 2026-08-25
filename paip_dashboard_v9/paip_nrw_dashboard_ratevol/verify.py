"""Independent verification.

Recomputes every headline figure straight from the RAW CSV, using a separate
parsing path from prepare_data.py, and asserts agreement with the prepared
artefacts the dashboard reads. A bug shared by both paths would have to be
introduced twice to survive this.
"""
import re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Read the same raw inputs the pipeline reads, and take the focus year from the
# data. Nothing here may be pinned to a particular year: a refresh that adds a
# new year must be verifiable without editing this file.
from pathlib import Path as _P
RAW_DIR = _P(__file__).parent / "data" / "raw"
RAW_FILES = sorted(RAW_DIR.glob("*.csv"))
if not RAW_FILES:
    raise SystemExit("No raw CSV found in data/raw/ — run prepare_data.py first.")
fails, checks = [], 0


def ok(name, a, b, tol=1e-6, rel=False):
    global checks
    checks += 1
    d = abs(a - b)
    good = (d / max(abs(b), 1e-9) < tol) if rel else (d <= tol)
    print(f"  {'PASS' if good else 'FAIL'}  {name:52s} {a:>18,.4f} vs {b:>18,.4f}")
    if not good:
        fails.append(name)


# --- Independent parse: regex-strip, no shared helper --------------------
# Deliberately does NOT import dataloader: a bug shared by both paths would have
# to be written twice to survive. Duplicate plant-months are dropped the same
# way the loader does (last wins) so row counts are comparable.
raw = pd.concat([pd.read_csv(f, dtype=str) for f in RAW_FILES], ignore_index=True)
raw = raw.drop_duplicates(["Nama_Loji", "Tarikh"], keep="last").reset_index(drop=True)


def num(col):
    return raw[col].map(lambda v: float(re.sub(r"[,%]", "", str(v))))


YEAR = int(num("Tahun").max())
print(f"Verifying focus year {YEAR} from {len(RAW_FILES)} raw file(s)")
year_mask = num("Tahun") == YEAR
prod = num("Pengeluaran_m3")[year_mask]
billed = num("Jumlah_Dibilkan_m3")[year_mask]
nrw = num("NRW_m3")[year_mask]
phys = num("Kehilangan_Fizikal_m3")[year_mask]
tariff = num("Tarif_Purata_RM_m3")[year_mask]
plants = raw["Nama_Loji"][year_mask]

prep_m = pd.read_csv("data/nrw_plant_month.csv")
prep_y = pd.read_csv("data/nrw_plant_year.csv")
py = prep_y[prep_y.year == YEAR]

print(f"\n=== Totals, {YEAR} ===")
ok("total production m3", prod.sum(), py.production_m3.sum(), 1)
ok("total billed m3", billed.sum(), py.billed_m3.sum(), 1)
ok("total NRW m3", nrw.sum(), py.nrw_m3.sum(), 1)
ok("total physical loss m3", phys.sum(), py.physical_loss_m3.sum(), 1)
ok("system NRW pct", nrw.sum() / prod.sum() * 100, py.nrw_m3.sum() / py.production_m3.sum() * 100, 1e-9)
ok("NRW value RM", (nrw * tariff).sum(), py.nrw_value_rm.sum(), 1.0)
ok("physical share pct", phys.sum() / nrw.sum() * 100,
   py.physical_loss_m3.sum() / py.nrw_m3.sum() * 100, 1e-9)

print("\n=== Structure ===")
ok("plant count", plants.nunique(), len(py), 0)
ok("plant-month rows in year", int(year_mask.sum()),
   plants.nunique() * num("Bulan_No")[year_mask].nunique(), 0)
ok("all-year rows", len(raw), len(prep_m), 0)

# --- Per-plant aggregation from scratch ----------------------------------
ind = (pd.DataFrame({"plant": plants, "prodv": prod, "nrw": nrw, "phys": phys})
       .groupby("plant", as_index=False).sum())
ind["pct"] = ind.nrw / ind.prodv * 100
merged = ind.merge(py[["plant", "nrw_m3", "nrw_pct", "physical_loss_m3"]], on="plant")

print("\n=== Per-plant agreement (74 plants) ===")
ok("max abs diff, plant NRW m3", (merged.nrw - merged.nrw_m3).abs().max(), 0.0, 1)
ok("max abs diff, plant NRW pct", (merged.pct - merged.nrw_pct).abs().max(), 0.0, 1e-6)
ok("max abs diff, plant physical m3",
   (merged.phys - merged.physical_loss_m3).abs().max(), 0.0, 1)

# --- The headline claim on the Rate vs Volume tab ------------------------
print("\n=== Rate-vs-volume claim ===")
ind["rate_rank"] = ind.pct.rank(ascending=False, method="min")
ind["vol_rank"] = ind.nrw.rank(ascending=False, method="min")
t_rate = ind.nsmallest(10, "rate_rank")
t_vol = ind.nsmallest(10, "vol_rank")
rho_ind = spearmanr(ind.pct, ind.nrw).statistic
rho_prep = spearmanr(py.nrw_pct, py.nrw_m3).statistic
ok("spearman rho (rate vs volume)", rho_ind, rho_prep, 1e-9)
ok("top-10 overlap between queues", len(set(t_rate.plant) & set(t_vol.plant)), 0, 0)
# Compared against the prepared artefact, not a frozen number: the true value
# legitimately changes when a new year arrives, and a literal here would fail
# every refresh for the wrong reason.
py_rate = py.nsmallest(10, "rate_rank")
py_vol = py.nsmallest(10, "volume_rank")
ok("water in volume queue / rate queue (x)",
   t_vol.nrw.sum() / t_rate.nrw.sum(),
   py_vol.nrw_m3.sum() / py_rate.nrw_m3.sum(), 0.02)
ok("top-10 share of all NRW (pct)",
   ind.nlargest(10, "nrw").nrw.sum() / ind.nrw.sum() * 100,
   py.nlargest(10, "nrw_m3").nrw_m3.sum() / py.nrw_m3.sum() * 100, 0.01)
checks += 1
_ratio = t_vol.nrw.sum() / t_rate.nrw.sum()
print(f"  {'PASS' if _ratio > 1 else 'FAIL'}  "
      f"{'volume queue holds more water than rate queue':52s} "
      f"{_ratio:>18,.2f}x")
if _ratio <= 1:
    fails.append("volume queue no longer dominates")
print(f"    rho = {rho_ind:.4f}  (negative => rate and volume rank plants oppositely)")

# --- LIPS behaviour -------------------------------------------------------
print("\n=== LIPS ===")
# LIPS is a four-component weighted percentile composite. These weights are the
# specification; prepare_data.py holds the implementation. They are repeated
# here on purpose — a verification that imported them could not catch the
# weights being changed in one place and not the other.
W = {
    "nrw_per_km_m3": 40,      # loss density
    "bursts_per_100km": 25,   # burst rate
    "plant_age_yr": 20,       # asset condition
    "account_density": 15,    # commercial exposure
}


def lips_of(df, w):
    tot = sum(w.values())
    s = pd.Series(0.0, index=df.index)
    for c, wt in w.items():
        s += df[c].rank(pct=True, method="average") * 100 * (wt / tot)
    return s


# Every input must exist and be usable. This is the check that would have
# caught account_density being added to the weights but never rebuilt into the
# artefacts — the app raised KeyError on load, and nothing here noticed.
for c in W:
    checks += 1
    present = c in py.columns
    nulls = int(py[c].isna().sum()) if present else -1
    good = present and nulls == 0
    print(f"  {'PASS' if good else 'FAIL'}  "
          f"{'LIPS input present and complete: ' + c:52s} "
          f"{'missing from artefacts' if not present else str(nulls) + ' nulls':>18s}")
    if not good:
        fails.append(f"LIPS input unusable: {c}")

ok("LIPS weights sum to 100", float(sum(W.values())), 100.0, 0)

recomputed = lips_of(py, W)
ok("max abs diff, LIPS score", (recomputed - py.lips).abs().max(), 0.0, 0.01)
ok("LIPS bounded 0-100", float(((py.lips < 0) | (py.lips > 100)).sum()), 0.0, 0)
ok("LIPS ranks are 1..n unique", py.lips_rank.nunique(), len(py), 0)

# A composite that reproduces one of its own inputs is not a composite. If any
# single component correlates near-perfectly with the result, the other three
# are decoration and the weighting is not doing what it claims.
for c in W:
    r = spearmanr(py.lips, py[c]).statistic
    checks += 1
    good = abs(r) < 0.97
    print(f"  {'PASS' if good else 'FAIL'}  "
          f"{'LIPS is not just ' + c:52s} {r:>18.3f} rho")
    if not good:
        fails.append(f"LIPS collapses to {c}")

# Reported, not asserted: how much water the queue actually reaches. A score
# built from densities and proxies is not optimising for volume, so this is
# context for the reader rather than a pass/fail condition.
_topl = py.nsmallest(10, "lips_rank").nrw_m3.sum() / py.nrw_m3.sum()
_topv = py.nsmallest(10, "volume_rank").nrw_m3.sum() / py.nrw_m3.sum()
_topr = py.nsmallest(10, "rate_rank").nrw_m3.sum() / py.nrw_m3.sum()
print(f"  ----  {'NRW reached by each top-10 (informational)':52s} "
      f"LIPS {_topl*100:.1f}%  volume {_topv*100:.1f}%  rate {_topr*100:.1f}%")

# Identity kept for the Loss Composition tab, which still displays the split.
ok("physical_loss == nrw x physical_share",
   (py.physical_loss_m3 - py.nrw_m3 * py.physical_share_pct / 100).abs().max(),
   0.0, 1.0)

# --- 2026 LIPS prediction -------------------------------------------------
_pred_p = _P(__file__).parent / "data" / "lips_2026_prediction.csv"
_bt_p = _P(__file__).parent / "data" / "forecast_backtest.csv"
if _pred_p.exists() and _bt_p.exists():
    print("\n=== 2026 LIPS prediction ===")
    _pr = pd.read_csv(_pred_p)
    _bt = pd.read_csv(_bt_p)

    ok("prediction covers every plant once", float(_pr.plant.nunique()),
       float(prep_y.plant.nunique()), 0)
    ok("prediction ranks are 1..n unique", float(_pr.lips_rank_2026.nunique()),
       float(len(_pr)), 0)

    # The projected burst rate is the multi-year mean, and that is the whole
    # forecasting content of the model. If this drifts back to a trend
    # extrapolation the ranking gets WORSE than doing nothing - measured.
    _mean_b = prep_y.groupby("plant").bursts_per_100km.mean()
    _j = _pr.set_index("plant")
    ok("projected burst rate is the multi-year mean",
       float((_j.bursts_per_100km - _mean_b.reindex(_j.index)).abs().max()),
       0.0, 1e-6)
    _now = prep_y[prep_y.year == YEAR].set_index("plant")
    ok("loss density carried forward unchanged",
       float((_j.nrw_per_km_m3 - _now.nrw_per_km_m3.reindex(_j.index)).abs().max()),
       0.0, 1e-6)
    ok("plant age advances exactly one year",
       float((_j.plant_age_yr - _now.plant_age_yr.reindex(_j.index) - 1).abs().max()),
       0.0, 1e-9)

    # The backtest must carry its control group, and the model must win it.
    _methods = set(_bt.method)
    checks += 1
    _has_base = any(m.startswith("persistence") for m in _methods)
    print(f"  {'PASS' if _has_base else 'FAIL'}  "
          f"{'backtest includes a persistence baseline':52s} "
          f"{len(_methods):>17d} methods")
    if not _has_base:
        fails.append("backtest has no baseline to compare against")

    _m = _bt[_bt.method == "projection"].iloc[0]
    _b = _bt[_bt.method.str.startswith("persistence")].iloc[0]
    checks += 1
    _win = _m.rank_mae_places < _b.rank_mae_places
    print(f"  {'PASS' if _win else 'FAIL'}  "
          f"{'projection beats doing nothing':52s} "
          f"{_m.rank_mae_places:>10.2f} vs {_b.rank_mae_places:.2f} places")
    if not _win:
        fails.append("the 2026 projection is worse than persistence")

# --- Financial identities -------------------------------------------------
print("\n=== Financials ===")
ok("NRW value == volume x tariff (per plant, max diff)",
   (py.nrw_value_rm - py.nrw_m3 * py.tariff_rm_m3).abs().max(), 0.0, 2500,
   )  # tariff is a 12-month mean, so a small residual is expected
sunk = py.nrw_sunk_cost_rm.sum()
ok("sunk cost < forgone revenue", float(sunk < py.nrw_value_rm.sum()), 1.0, 0)
# Recomputed from the monthly artefact rather than pinned to a literal.
_m = pd.read_csv("data/nrw_plant_month.csv")
_m = _m[_m.year == YEAR]
ok("sunk cost share of opex (pct)", sunk / py.opex_rm.sum() * 100,
   _m.nrw_sunk_cost_rm.sum() / _m.opex_rm.sum() * 100, 0.01)




# ==========================================================================
# Model checks — added with the Early Warning tab
# ==========================================================================
import json as _json
from pathlib import Path as _Path

if _Path("data/model_metrics.json").exists():
    import train_models as _tm
    from sklearn.model_selection import GroupKFold as _GKF

    met = _json.loads(_Path("data/model_metrics.json").read_text())
    mlp = pd.read_csv("data/ml_plant.csv")
    mlm = pd.read_csv("data/ml_monthly.csv", parse_dates=["date"])

    print("\n=== Model: target leakage ===")
    feats = set(_tm.NUMERIC_FEATURES) | set(_tm.CATEGORICAL_FEATURES)
    ok("no banned feature in the model", float(len(feats & _tm.BANNED)), 0.0, 0)

    # The two trap columns must be provably derived from BILLED volume, which is
    # production minus the target. Re-derive them here rather than trusting the
    # exclusion list.
    d = pd.read_csv("data/nrw_plant_month.csv", parse_dates=["date"])
    days = d.date.dt.days_in_month
    ok("consumption_per_capita IS billed-derived (would leak)",
       (d.consumption_per_capita_l_day - d.billed_m3 * 1000 / d.population_served / days).abs().max(),
       0.0, 0.06)
    ok("revenue_per_account IS billed-derived (would leak)",
       (d.revenue_per_account_rm - d.billed_revenue_rm / d.customer_accounts).abs().max(),
       0.0, 0.01)
    for c in ("consumption_per_capita_l_day", "revenue_per_account_rm"):
        checks += 1
        good = c not in feats
        print(f"  {'PASS' if good else 'FAIL'}  {('excluded: ' + c):52s} "
              f"{'absent from feature set':>18}")
        if not good:
            fails.append(f"{c} not excluded")

    # Features that divide by PRODUCTION are safe; assert they really do.
    ok("capacity_utilisation is production-derived (safe)",
       (d.capacity_utilisation_pct - d.production_m3 / (d.capacity_m3_day * days) * 100).abs().max(),
       0.0, 0.06)
    ok("energy_intensity is production-derived (safe)",
       (d.energy_intensity_kwh_m3 - d.energy_kwh / d.production_m3).abs().max(), 0.0, 0.001)

    print("\n=== Model: validation integrity ===")
    cv = met["cv_grouped_by_plant"]
    best = met["selected_model"]
    ok("selected model beats mean baseline", float(cv[best]["r2"] > cv["mean"]["r2"]), 1.0, 0)
    ok("selected model has best grouped-CV R2",
       float(cv[best]["r2"] >= max(cv[k]["r2"] for k in ("gb", "ridge"))), 1.0, 0)
    ok("mean baseline R2 is ~0 or negative", float(cv["mean"]["r2"] < 0.01), 1.0, 0)
    ok("selected MAE beats baseline MAE", float(cv[best]["mae"] < cv["mean"]["mae"]), 1.0, 0)

    # GroupKFold must place every plant entirely on one side of each split;
    # otherwise "unseen plants" is a false claim.
    m_eng = _tm.engineer(d)
    X_ = m_eng[_tm.NUMERIC_FEATURES + _tm.CATEGORICAL_FEATURES]
    g_ = m_eng.plant.values
    bleed = 0
    for tr_i, te_i in _GKF(n_splits=5).split(X_, m_eng.nrw_pct.values, g_):
        bleed += len(set(g_[tr_i]) & set(g_[te_i]))
    ok("no plant appears in both train and test", float(bleed), 0.0, 0)

    print("\n=== Model: outputs ===")
    ok("every plant scored", float(mlp.plant.nunique()),
       float(pd.read_csv("data/nrw_plant_year.csv")
             .query("year == @met['focus_year']").plant.nunique()), 0)
    ok("criticality bounded 0-100",
       float(((mlp.criticality < 0) | (mlp.criticality > 100)).sum()), 0.0, 0)
    ok("criticality ranks are 1..n unique", float(mlp.criticality_rank.nunique()),
       float(len(mlp)), 0)
    ok("residual == actual - expected",
       (mlp.unexplained_pp - (mlp.actual_nrw_pct - mlp.expected_nrw_pct)).abs().max(),
       0.0, 1e-6)
    ok("unexplained m3 == residual x production",
       (mlp.unexplained_m3 - mlp.unexplained_pp / 100 * mlp.production_m3).abs().max(),
       0.0, 1.0)
    # Residuals from an out-of-fold fit should be roughly centred, not biased.
    ok("mean residual is near zero (unbiased)", float(mlp.unexplained_pp.mean()), 0.0, 2.0)
    ok("monthly predictions cover every record", float(len(mlm)),
       float(len(pd.read_csv("data/nrw_plant_month.csv"))), 0)
    ok("no NaN predictions", float(mlm.predicted_nrw_pct.isna().sum()), 0.0, 0)

    print("\n=== Model: criticality is not a restatement of LIPS ===")
    yr = pd.read_csv("data/nrw_plant_year.csv")
    _fy = int(met["focus_year"])
    j = mlp[["plant", "criticality"]].merge(
        yr[yr.year == _fy][["plant", "lips", "nrw_m3", "nrw_pct"]], on="plant")
    r_lips = spearmanr(j.criticality, j.lips).statistic
    r_vol = spearmanr(j.criticality, j.nrw_m3).statistic
    checks += 1
    good = abs(r_lips) < 0.8 and abs(r_vol) < 0.8
    print(f"  {'PASS' if good else 'FAIL'}  "
          f"{'criticality is distinct from LIPS/volume':52s} "
          f"rho_lips={r_lips:6.3f}  rho_vol={r_vol:6.3f}")
    if not good:
        fails.append("criticality duplicates LIPS")

    print("\n=== Model: the null deterioration result ===")
    ds = met["deterioration_summary"]
    ok("worsening-trend count matches artefact",
       float(((mlp.trend_p < 0.10) & (mlp.trend_pp_yr > 0)).sum()),
       float(ds["plants_with_significant_worsening"]), 0)
    ok("step-increase count matches artefact",
       float(((mlp.step_p < 0.05) & (mlp.step_shift_pp > 0)).sum()),
       float(ds["plants_with_significant_step_increase"]), 0)
    ok("estate median trend is improving", float(ds["median_trend_pp_yr"] < 0), 1.0, 0)
    print(f"    {ds['plants_with_significant_worsening']} plants worsening, "
          f"{ds['total_anomaly_months']} anomaly months in 2,664 records — "
          f"reported as a null result, not concealed")


# ==========================================================================
# Burst-risk classifier checks
# ==========================================================================
if _P("data/burst_metrics.json").exists():
    from sklearn.metrics import roc_auc_score as _auc, brier_score_loss as _brier

    bm = _json.loads(_P("data/burst_metrics.json").read_text())
    bp = pd.read_csv("data/burst_predictions.csv", parse_dates=["date"])
    bh = pd.read_csv("data/burst_history.csv", parse_dates=["date"])
    _best = bm["selected_model"]
    _r = bm["results"]

    print("\n=== Burst model: target definition ===")
    _d = pd.read_csv("data/nrw_plant_month.csv", parse_dates=["date"]).sort_values(["plant","date"])
    _d["elev"] = (_d.pipe_bursts >= 2).astype(int)
    ok("'any burst' really is near-constant (why it is not the target)",
       float((_d.pipe_bursts >= 1).mean()), 0.981, 0.01)
    ok("elevated-month base rate matches artefact",
       float(_d.groupby("plant").elev.shift(-1).dropna().mean()),
       float(bm["positive_rate"]), 0.005)

    print("\n=== Burst model: leakage and temporal integrity ===")
    # Join each stored label back to the source month it should describe: the
    # elevated flag at (plant, date + 1 month). This is the check that the model
    # was trained to predict the FUTURE and not handed the present.
    _src = _d[["plant", "date", "elev"]].copy()
    _src["date"] = _src.date - pd.DateOffset(months=1)   # shift back one month
    _j = bh.merge(_src.rename(columns={"elev": "elev_next_from_source"}),
                  on=["plant", "date"], how="inner")
    ok("stored label equals next month's elevated flag in the source",
       float((_j.elevated_next != _j.elev_next_from_source).sum()), 0.0, 0)
    # And it must NOT simply echo the current month, or the task is trivial.
    _echo = float((_j.elevated_next == _j.elevated).mean())
    checks += 1
    print(f"  {'PASS' if _echo < 0.95 else 'FAIL'}  "
          f"{'label is not an echo of the current month':52s} "
          f"{_echo:>18.3f} agreement")
    if _echo >= 0.95:
        fails.append("target may be the current month, not next")

    _cut = pd.Timestamp(bm["train_end"])
    ok("no test row dated on or before the train cutoff",
       float((bh[bh.split == "test"].date <= _cut).sum()), 0.0, 0)
    ok("no train row dated after the cutoff",
       float((bh[bh.split == "train"].date > _cut).sum()), 0.0, 0)
    ok("train/test sizes match the artefact",
       float((bh.split == "train").sum()), float(bm["n_train"]), 0)

    print("\n=== Burst model: performance is real ===")
    ok("selected model beats the majority baseline",
       float(_r[_best]["test_roc_auc"] > _r["majority"]["test_roc_auc"]), 1.0, 0)
    ok("selected model beats the persistence rule",
       float(_r[_best]["test_roc_auc"] > _r["persistence"]["test_roc_auc"]), 1.0, 0)
    ok("majority baseline scores chance (0.5)",
       float(_r["majority"]["test_roc_auc"]), 0.5, 1e-9)
    ok("selected model has the best CV AUC of the real candidates",
       float(_r[_best]["cv_roc_auc"] >= max(_r[k]["cv_roc_auc"]
             for k in ("logistic", "forest", "boosting"))), 1.0, 0)
    # Recompute AUC independently from the stored per-row predictions.
    _te = bh[bh.split == "test"]
    ok("test AUC recomputed from stored predictions",
       float(_auc(_te.elevated_next, _te.predicted)),
       float(_r[_best]["test_roc_auc"]), 1e-6)
    ok("Brier score recomputed", float(_brier(_te.elevated_next, _te.predicted.clip(0, 1))),
       float(_r[_best]["brier"]), 1e-6)
    ok("selected model is better calibrated than the majority baseline",
       float(_r[_best]["brier"] < _r["majority"]["brier"]), 1.0, 0)

    print("\n=== Burst model: confusion matrix consistency ===")
    _c = bm["confusion_matrix"]
    ok("confusion matrix totals equal the test set",
       float(_c["tn"] + _c["fp"] + _c["fn"] + _c["tp"]), float(bm["n_test"]), 0)
    ok("positives in the matrix equal actual positives",
       float(_c["tp"] + _c["fn"]), float(_te.elevated_next.sum()), 0)
    _thr = bm["operating_threshold"]
    ok("matrix reproduces at the stated threshold",
       float(((_te.predicted >= _thr) & (_te.elevated_next == 1)).sum()),
       float(_c["tp"]), 0)

    print("\n=== Burst model: forward predictions ===")
    ok("every plant has a risk score", float(bp.plant.nunique()),
       float(_d.plant.nunique()), 0)
    ok("risk ranks are 1..n unique", float(bp.risk_rank.nunique()), float(len(bp)), 0)
    ok("probabilities lie in [0, 1]",
       float(((bp.risk < 0) | (bp.risk > 1)).sum()), 0.0, 0)
    ok("flagged count matches the artefact", float(bp.will_flag.sum()),
       float(bm["n_flagged"]), 0)
    ok("flag equals risk >= threshold",
       float((bp.will_flag != (bp.risk >= _thr).astype(int)).sum()), 0.0, 0)
    # Forward predictions must come from the final observed month of each plant.
    ok("predictions use each plant's latest month",
       float((bp.date != _d.date.max()).sum()), 0.0, 0)
    print(f"    AUC {_r[_best]['test_roc_auc']:.3f} vs persistence "
          f"{_r['persistence']['test_roc_auc']:.3f} — "
          f"{bm['n_flagged']} of {len(bp)} plants flagged for {bm['horizon']}")

print(f"\n{'='*88}")
if fails:
    print(f"{len(fails)} of {checks} CHECKS FAILED:")
    for f in fails:
        print("   -", f)
    raise SystemExit(1)
print(f"All {checks} checks passed.")
