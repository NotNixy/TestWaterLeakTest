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

def build(source=None, strict: bool = True):
    """Full clean-and-aggregate pass. Returns (plant_month, plant_year, report)."""
    # Previously-known plants let the loader report additions and disappearances
    # across refreshes instead of silently absorbing a renamed plant.
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

    crosswalk = (df[["plant", "district", "region", "area_type"]]
                 .drop_duplicates()
                 .sort_values(["region", "district", "plant"])
                 .reset_index(drop=True))
    crosswalk.insert(0, "plant_id",
                     [f"P{i:03d}" for i in range(1, len(crosswalk) + 1)])

    coverage = year_completeness(df)

    df.to_csv(OUT / "nrw_plant_month.csv", index=False)
    scored.to_csv(OUT / "nrw_plant_year.csv", index=False)
    crosswalk.to_csv(OUT / "plant_crosswalk.csv", index=False)
    checks.to_csv(OUT / "data_quality.csv", index=False)
    missing.to_csv(OUT / "missing_values.csv", index=False)
    coverage.to_csv(OUT / "year_coverage.csv", index=False)
    return df, scored, report, coverage


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else None
    try:
        df, scored, report, coverage = build(source)
    except DataError as exc:
        print(exc)
        raise SystemExit(1)

    print(report.text())
    print("\nYear coverage:")
    print(coverage.to_string(index=False))
    print(f"\nplant-month records : {len(df):,}")
    print(f"plant-year rows     : {len(scored):,}")
    print("\nIdentity checks (max absolute deviation):")
    print(pd.read_csv(OUT / "data_quality.csv").to_string(index=False))


if __name__ == "__main__":
    main()
