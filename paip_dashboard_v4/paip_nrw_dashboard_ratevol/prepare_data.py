"""
PAIP Non-Revenue Water — data preparation
=========================================
Cleans the PAIP workbook export into a tidy long-format plant-month dataset and
writes the analysis-ready artefacts consumed by app.py.

Ingestion and validation live in dataloader.py; this module is only concerned
with derived measures and aggregation. Run it directly, or through refresh.py
which chains clean -> train -> verify.

    python prepare_data.py                    # reads data/raw/*.csv
    python prepare_data.py path/to/file.csv   # or an explicit file

Outputs (all in ./data):
    nrw_plant_month.csv   tidy plant-month records, English column names
    nrw_plant_year.csv    plant-year aggregates with LIPS inputs
    plant_crosswalk.csv   documented plant-name -> district/region/area crosswalk
    data_quality.csv      identity check results
    year_coverage.csv     months observed per year and annualisation factors
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

from dataloader import ingest, year_completeness, DataError

OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)

# --------------------------------------------------------------------------
# 1. Load and coerce
# --------------------------------------------------------------------------
# The published workbook stores numerics as strings with thousands separators
# and trailing percent signs, so every numeric column needs explicit coercion.

# --------------------------------------------------------------------------
# 2. Derived measures
# --------------------------------------------------------------------------

def derive(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    days = df["date"].dt.days_in_month

    # Value of water lost. Tariff is published per m3, so NRW volume can be
    # priced directly. This is forgone revenue, not avoidable cost.
    df["nrw_value_rm"] = df["nrw_m3"] * df["tariff_rm_m3"]
    df["physical_loss_value_rm"] = df["physical_loss_m3"] * df["tariff_rm_m3"]

    # Production cost already sunk into water that was never billed.
    df["nrw_sunk_cost_rm"] = df["nrw_m3"] * df["cost_per_m3_rm"]

    # Normalisers that make plants of very different size comparable.
    df["nrw_per_account_m3"] = df["nrw_m3"] / df["customer_accounts"]
    df["physical_loss_per_km_m3"] = df["physical_loss_m3"] / df["pipe_length_km"]
    df["bursts_per_100km"] = df["pipe_bursts"] / df["pipe_length_km"] * 100
    df["nrw_m3_per_day"] = df["nrw_m3"] / days
    df["production_m3_per_day"] = df["production_m3"] / days

    # Infrastructure Leakage proxy: litres lost per connection per day.
    df["loss_per_connection_l_day"] = (
        df["physical_loss_m3"] * 1000 / df["customer_accounts"] / days
    )
    df["commercial_share_pct"] = 100 - df["physical_share_pct"]
    return df


# --------------------------------------------------------------------------
# 3. Data quality audit
# --------------------------------------------------------------------------

def audit(df: pd.DataFrame) -> pd.DataFrame:
    """Identity checks against the published figures. Nothing is corrected here;
    discrepancies are reported so PAIP can adjudicate them."""
    checks = pd.DataFrame({
        "check": [
            "billed_m3 == domestic + commercial + industrial",
            "nrw_m3 == production_m3 - billed_m3",
            "nrw_pct == nrw_m3 / production_m3",
            "physical + commercial loss == nrw_m3",
            "billed_revenue_rm == billed_m3 * tariff",
            "opex_rm == energy + chemical + maintenance",
            "negative nrw_m3 records",
            "nrw_pct outside 0-100",
            "billed_m3 > production_m3",
        ],
        "max_abs_deviation": [
            (df.billed_m3 - (df.billed_domestic_m3 + df.billed_commercial_m3
                             + df.billed_industrial_m3)).abs().max(),
            (df.nrw_m3 - (df.production_m3 - df.billed_m3)).abs().max(),
            (df.nrw_pct - df.nrw_m3 / df.production_m3 * 100).abs().max(),
            (df.nrw_m3 - (df.physical_loss_m3 + df.commercial_loss_m3)).abs().max(),
            (df.billed_revenue_rm - df.billed_m3 * df.tariff_rm_m3).abs().max(),
            (df.opex_rm - (df.energy_cost_rm + df.chemical_cost_rm
                           + df.maintenance_cost_rm)).abs().max(),
            float((df.nrw_m3 < 0).sum()),
            float(((df.nrw_pct < 0) | (df.nrw_pct > 100)).sum()),
            float((df.billed_m3 > df.production_m3).sum()),
        ],
    })
    missing = (df.isna().sum().loc[lambda s: s > 0]
                 .rename("missing_values").reset_index()
                 .rename(columns={"index": "column"}))
    missing["pct_of_rows"] = (missing.missing_values / len(df) * 100).round(2)
    checks.attrs["missing"] = missing
    return checks, missing


# --------------------------------------------------------------------------
# 4. Plant-year aggregation (the LIPS unit of analysis)
# --------------------------------------------------------------------------

def plant_year(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["year", "plant"], as_index=False).agg(
        district=("district", "first"),
        region=("region", "first"),
        area_type=("area_type", "first"),
        months=("date", "nunique"),
        days_observed=("date", lambda s: s.dt.days_in_month.sum()),
        production_m3=("production_m3", "sum"),
        billed_m3=("billed_m3", "sum"),
        nrw_m3=("nrw_m3", "sum"),
        physical_loss_m3=("physical_loss_m3", "sum"),
        commercial_loss_m3=("commercial_loss_m3", "sum"),
        nrw_value_rm=("nrw_value_rm", "sum"),
        physical_loss_value_rm=("physical_loss_value_rm", "sum"),
        nrw_sunk_cost_rm=("nrw_sunk_cost_rm", "sum"),
        billed_revenue_rm=("billed_revenue_rm", "sum"),
        opex_rm=("opex_rm", "sum"),
        energy_cost_rm=("energy_cost_rm", "sum"),
        energy_kwh=("energy_kwh", "sum"),
        pipe_bursts=("pipe_bursts", "sum"),
        complaints=("complaints", "sum"),
        supply_interruption_hr=("supply_interruption_hr", "sum"),
        pipe_length_km=("pipe_length_km", "mean"),
        plant_age_yr=("plant_age_yr", "mean"),
        meter_age_yr=("meter_age_yr", "mean"),
        capacity_m3_day=("capacity_m3_day", "mean"),
        capacity_utilisation_pct=("capacity_utilisation_pct", "mean"),
        customer_accounts=("customer_accounts", "mean"),
        population_served=("population_served", "mean"),
        tariff_rm_m3=("tariff_rm_m3", "mean"),
        cost_per_m3_rm=("cost_per_m3_rm", "mean"),
        pressure_bar=("pressure_bar", "mean"),
        water_quality_compliance_pct=("water_quality_compliance_pct", "mean"),
        nrw_pct_monthly_sd=("nrw_pct", "std"),
    )
    
    # Rates recomputed from annual totals
    g["nrw_pct"] = g.nrw_m3 / g.production_m3 * 100
    g["physical_share_pct"] = g.physical_loss_m3 / g.nrw_m3 * 100
    g["nrw_per_km_m3"] = g.nrw_m3 / g.pipe_length_km
    g["physical_loss_per_km_m3"] = g.physical_loss_m3 / g.pipe_length_km
    g["bursts_per_100km"] = g.pipe_bursts / g.pipe_length_km * 100
    g["account_density"] = g.customer_accounts / g.pipe_length_km  # Added for 4-factor LIPS
    g["nrw_per_account_m3"] = g.nrw_m3 / g.customer_accounts
    
    g["loss_per_connection_l_day"] = (
        g.physical_loss_m3 * 1000 / g.customer_accounts / g.days_observed
    )
    g["operating_margin_pct"] = (
        (g.billed_revenue_rm - g.opex_rm) / g.billed_revenue_rm * 100
    )
    g["asset_age_index"] = (g.plant_age_yr / g.plant_age_yr.max() * 0.6
                            + g.meter_age_yr / g.meter_age_yr.max() * 0.4) * 100

    # ---- Partial-year handling ------------------------------------------
    g["complete_year"] = g.months >= 12
    g["annualise"] = 12 / g.months.clip(lower=1)
    for col in ["nrw_m3", "production_m3", "billed_m3", "physical_loss_m3",
                "commercial_loss_m3", "nrw_value_rm", "nrw_sunk_cost_rm",
                "billed_revenue_rm", "opex_rm", "pipe_bursts"]:
        g[f"{col}_annualised"] = g[col] * g.annualise
    return g


# --------------------------------------------------------------------------
# 5. Leakage Intervention Priority Score (Revised 4-Factor LIPS)
# --------------------------------------------------------------------------

# Updated weights configuration:
#   nrw_per_km_m3 (40%)    : Combined Loss Density (NRW volume concentration)
#   bursts_per_100km (25%) : Burst Rate (Proxy for physical pipe failure)
#   plant_age_yr (20%)     : Asset Condition / Deterioration risk
#   account_density (15%)  : Commercial & Metering risk exposure
DEFAULT_WEIGHTS = {
    "nrw_per_km_m3": 40,
    "bursts_per_100km": 25,
    "plant_age_yr": 20,
    "account_density": 15,
}


def percentile_rank(s: pd.Series) -> pd.Series:
    """0-100 percentile rank to maintain readable scaling without outlier distortion."""
    return s.rank(pct=True, method="average") * 100


def lips(g: pd.DataFrame, weights: dict = None) -> pd.DataFrame:
    weights = weights or DEFAULT_WEIGHTS
    total = sum(weights.values())
    out = g.copy()
    score = pd.Series(0.0, index=out.index)
    
    for col, w in weights.items():
        comp = percentile_rank(out[col])
        out[f"pr_{col}"] = comp
        score += comp * (w / total)
        
    out["lips"] = score.round(2)
    
    # Tie-breaking priority: LIPS Score -> NRW Loss Density -> Raw NRW Volume
    out = out.sort_values(["lips", "nrw_per_km_m3", "nrw_m3"], ascending=False)
    out["lips_rank"] = np.arange(1, len(out) + 1)
    out["volume_rank"] = out["nrw_m3"].rank(ascending=False, method="first").astype(int)
    out["rate_rank"] = out["nrw_pct"].rank(ascending=False, method="first").astype(int)
    out["rank_gap"] = out["rate_rank"] - out["volume_rank"]
    return out

# --------------------------------------------------------------------------
# 6. LIPS Prediction Model (for year 2026)
# --------------------------------------------------------------------------
def predict_2026_lips(scored_history: pd.DataFrame, weights: dict = None,
                       target_year: int = 2026) -> pd.DataFrame:
    """
    Projects each plant's 2026 LIPS score using the SAME 4-factor formula and
    weights as the historical score (see `lips()` / DEFAULT_WEIGHTS, mirrored
    in app.py's score_lips), so the projection stays on the same scale as the
    LIPS numbers analysts already trust instead of a separate ad-hoc formula.

    `scored_history` must contain one row per plant per year (i.e. the full
    output of `lips()` across all years, not just the latest year), because
    each component is extrapolated from that plant's OWN 2023-2025 trend:
        - nrw_per_km_m3, bursts_per_100km, account_density: per-plant linear
          regression against year, extrapolated to target_year.
        - plant_age_yr: advanced deterministically by the elapsed years.
    A plant with only one year of history (regression not possible) is held
    flat at its latest observed value.
    """
    weights = weights or DEFAULT_WEIGHTS
    total = sum(weights.values())

    hist = scored_history.copy()
    last_year = int(hist["year"].max())
    years_ahead = target_year - last_year
    latest = hist[hist.year == last_year].set_index("plant")

    def trend_project(col: str) -> pd.Series:
        """Per-plant linear trend in `col` across all historical years,
        extrapolated `years_ahead` beyond the latest observed value."""
        slopes = {}
        for plant, g in hist.groupby("plant"):
            sub = g[["year", col]].dropna()
            slopes[plant] = np.polyfit(sub.year, sub[col], 1)[0] if len(sub) >= 2 else 0.0
        slope = pd.Series(slopes).reindex(latest.index).fillna(0.0)
        return latest[col] + slope * years_ahead

    proj = pd.DataFrame(index=latest.index)
    proj["nrw_per_km_m3"] = trend_project("nrw_per_km_m3").clip(lower=0)
    proj["bursts_per_100km"] = trend_project("bursts_per_100km").clip(lower=0)
    proj["account_density"] = trend_project("account_density").clip(lower=0)
    proj["plant_age_yr"] = latest["plant_age_yr"] + years_ahead

    # Carried through for the schedule table / scatter plot, not part of the
    # score itself (nrw_per_km_m3 is the scored loss-density component).
    proj["predicted_nrw_2026"] = trend_project("nrw_pct").clip(lower=0, upper=100)
    proj["predicted_bursts_2026"] = proj["bursts_per_100km"]
    proj["pipe_age_2026"] = proj["plant_age_yr"]

    proj["district"] = latest["district"]
    proj["region"] = latest["region"]
    proj["area_type"] = latest["area_type"]
    proj["lips"] = latest["lips"]
    proj["lips_rank"] = latest["lips_rank"]

    score = pd.Series(0.0, index=proj.index)
    for col, w in weights.items():
        pr = percentile_rank(proj[col])
        proj[f"pr_{col}_2026"] = pr
        score += pr * (w / total)
    proj["lips_2026"] = score.round(1)

    # Same tie-breaking convention as lips(): score -> loss density -> age.
    proj = proj.sort_values(["lips_2026", "nrw_per_km_m3", "plant_age_yr"], ascending=False)
    proj["lips_rank_2026"] = np.arange(1, len(proj) + 1)
    proj["rank_change"] = proj["lips_rank"] - proj["lips_rank_2026"]  # +ve = escalating

    return proj.reset_index().rename(columns={"index": "plant"})


# --------------------------------------------------------------------------
# 7. Backtest the trend projection (real forecast-accuracy figure)
# --------------------------------------------------------------------------
def backtest_lips_forecast(scored_history: pd.DataFrame, weights: dict = None,
                            top_n: int = 10) -> pd.DataFrame:
    """
    Measures how good the trend projection in `predict_2026_lips` actually
    is, instead of asserting an accuracy figure. Trains on every year except
    the most recent, "predicts" that held-out year, and compares the
    prediction against what actually happened:
        - top_n_precision_pct : of the actual top-N priority plants in the
          held-out year, what % did the prediction also flag in its top N.
          This is the headline number — it answers the question a CAPEX
          planner actually cares about ("would I have dispatched crews to
          the right plants a year early?").
        - rank_spearman       : Spearman correlation between predicted and
          actual priority order across all plants.
        - score_mae           : mean absolute error between the predicted
          and actual LIPS score (0-100 scale).
    Requires at least 2 years of history (1 to train the trend, 1 to check
    it against); returns an empty frame if there isn't enough history yet.
    """
    weights = weights or DEFAULT_WEIGHTS
    years = sorted(scored_history.year.unique())
    if len(years) < 2:
        return pd.DataFrame()

    holdout_year = years[-1]
    train = scored_history[scored_history.year < holdout_year]

    pred = predict_2026_lips(train, weights=weights, target_year=holdout_year)
    actual = (scored_history[scored_history.year == holdout_year]
              [["plant", "lips", "lips_rank"]]
              .rename(columns={"lips": "lips_actual", "lips_rank": "lips_rank_actual"}))

    cmp = pred.merge(actual, on="plant")
    n = min(top_n, len(cmp))

    pred_top = set(cmp.nsmallest(n, "lips_rank_2026").plant)
    actual_top = set(cmp.nsmallest(n, "lips_rank_actual").plant)
    top_n_precision_pct = len(pred_top & actual_top) / n * 100

    return pd.DataFrame([{
        "train_years": f"{train.year.min()}-{train.year.max()}",
        "holdout_year": holdout_year,
        "n_plants": len(cmp),
        "top_n": n,
        "top_n_precision_pct": round(top_n_precision_pct, 1),
        "rank_spearman": round(cmp["lips_rank_2026"].corr(cmp["lips_rank_actual"], method="spearman"), 3),
        "score_mae": round((cmp["lips_2026"] - cmp["lips_actual"]).abs().mean(), 2),
    }])


# --------------------------------------------------------------------------

def build(source=None, strict: bool = True):
    """Full clean-and-aggregate pass. Returns (plant_month, plant_year, report, coverage, lips_2026)."""
    known = None
    prev = OUT / "plant_crosswalk.csv"
    if prev.exists():
        try:
            known = pd.read_csv(prev).plant.tolist()
        except Exception:
            known = None

    raw, report = ingest(source, known_plants=known, strict=strict)
    df = derive(raw)
    checks, missing = audit(df)

    py = plant_year(df)
    # LIPS is scored within each year: percentile ranks are only meaningful
    # against contemporaries, and a new year must not reshuffle history.
    scored = pd.concat([lips(sub) for _, sub in py.groupby("year")],
                       ignore_index=True)
    
    # 1. GENERATE THE PREDICTION DATAFRAME HERE
    # Pass the full multi-year `scored` history (not just the latest year) so
    # predict_2026_lips can fit each plant's own 2023-2025 trend per component.
    lips_2026_df = predict_2026_lips(scored)
    backtest = backtest_lips_forecast(scored)
    
    crosswalk = (df[["plant", "district", "region", "area_type"]]
                 .drop_duplicates()
                 .sort_values(["region", "district", "plant"])
                 .reset_index(drop=True))
    crosswalk.insert(0, "plant_id",
                     [f"P{i:03d}" for i in range(1, len(crosswalk) + 1)])

    coverage = year_completeness(df)

    # 2. EXPORT TO CSV NOW THAT IT IS DEFINED
    df.to_csv(OUT / "nrw_plant_month.csv", index=False)
    scored.to_csv(OUT / "nrw_plant_year.csv", index=False)
    lips_2026_df.to_csv(OUT / "lips_2026_prediction.csv", index=False)
    backtest.to_csv(OUT / "forecast_backtest.csv", index=False)
    crosswalk.to_csv(OUT / "plant_crosswalk.csv", index=False)
    checks.to_csv(OUT / "data_quality.csv", index=False)
    missing.to_csv(OUT / "missing_values.csv", index=False)
    coverage.to_csv(OUT / "year_coverage.csv", index=False)
    
    return df, scored, report, coverage, lips_2026_df, backtest


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else None
    try:
        df, scored, report, coverage, lips_2026_df, backtest = build(source)
    except DataError as exc:
        print(exc)
        raise SystemExit(1)

    print(report.text())
    print("\nYear coverage:")
    print(coverage.to_string(index=False))
    print(f"\nplant-month records    : {len(df):,}")
    print(f"plant-year rows        : {len(scored):,}")
    print(f"2026 predicted rows    : {len(lips_2026_df):,}")
    print("\nIdentity checks (max absolute deviation):")
    print(pd.read_csv(OUT / "data_quality.csv").to_string(index=False))
    if not backtest.empty:
        print("\nForecast backtest (trained on all but the latest year):")
        print(backtest.to_string(index=False))


if __name__ == "__main__":
    main()
