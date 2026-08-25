"""
PAIP NRW — ingestion and validation
===================================

The single entry point for getting raw PAIP workbook exports into the pipeline.
Everything downstream (prepare_data, train_models, the dashboard) reads what this
module produces, so this is where new years arrive and where bad input is caught.

Two arrival patterns are supported and auto-detected:

  1. FULL REPLACEMENT   PAIP republishes one workbook covering every year.
                        Point it at the file; it replaces what came before.
  2. PER-YEAR APPEND    A new file is dropped into data/raw/ each year. All
                        files in the folder are concatenated.

Mixing the two is fine. Records are de-duplicated on (plant, date), keeping the
last occurrence, so a corrected re-issue of an old year supersedes the original
rather than double-counting it.

Validation is strict by design: structural problems raise, because a dashboard
built on a malformed extract is worse than no dashboard. Everything else is
reported as a warning and the build proceeds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RAW_DIR = HERE / "data" / "raw"

# Columns as published by PAIP, mapped to the analysis names used everywhere else.
RENAME = {
    "Record_ID": "record_id", "Tarikh": "date", "Tahun": "year",
    "Bulan": "month_name", "Bulan_No": "month", "Suku_Tahun": "quarter",
    "Daerah": "district", "Wilayah": "region", "Nama_Loji": "plant",
    "Jenis_Kawasan": "area_type", "Kapasiti_Loji_m3_hari": "capacity_m3_day",
    "Umur_Loji_Tahun": "plant_age_yr", "Populasi_Dilayan": "population_served",
    "Akaun_Pelanggan": "customer_accounts", "Pengeluaran_m3": "production_m3",
    "Bil_Domestik_m3": "billed_domestic_m3",
    "Bil_Komersial_m3": "billed_commercial_m3",
    "Bil_Industri_m3": "billed_industrial_m3",
    "Bahagian_Kehilangan_Fizikal": "physical_share_pct",
    "Tarif_Purata_RM_m3": "tariff_rm_m3", "Tenaga_kWh": "energy_kwh",
    "Tarif_Tenaga_RM_kWh": "energy_tariff_rm_kwh",
    "Kos_Kimia_RM": "chemical_cost_rm",
    "Kos_Penyelenggaraan_RM": "maintenance_cost_rm",
    "Bil_Kakitangan": "staff_count", "Panjang_Paip_km": "pipe_length_km",
    "Kejadian_Pecah_Paip": "pipe_bursts",
    "Umur_Meter_Purata_Tahun": "meter_age_yr", "Aduan_Pelanggan": "complaints",
    "Jam_Gangguan_Bekalan": "supply_interruption_hr",
    "Tekanan_Purata_bar": "pressure_bar", "Hujan_mm": "rainfall_mm",
    "Suhu_Purata_C": "temperature_c",
    "Kekeruhan_Air_Mentah_NTU": "raw_turbidity_ntu",
    "Pematuhan_Kualiti_Air_pct": "water_quality_compliance_pct",
    "Jumlah_Dibilkan_m3": "billed_m3", "NRW_m3": "nrw_m3",
    "NRW_Peratus": "nrw_pct", "Kehilangan_Fizikal_m3": "physical_loss_m3",
    "Kehilangan_Komersial_m3": "commercial_loss_m3",
    "Penggunaan_Kapasiti_pct": "capacity_utilisation_pct",
    "Hasil_Bil_RM": "billed_revenue_rm", "Kos_Tenaga_RM": "energy_cost_rm",
    "Jumlah_Kos_Operasi_RM": "opex_rm", "Kos_per_m3_RM": "cost_per_m3_rm",
    "Margin_Operasi_pct": "operating_margin_pct",
    "Hasil_per_Akaun_RM": "revenue_per_account_rm",
    "NRW_per_km_m3": "nrw_per_km_m3",
    "Penggunaan_per_Kapita_L_hari": "consumption_per_capita_l_day",
    "Kecekapan_Tenaga_kWh_m3": "energy_intensity_kwh_m3",
}

# Reverse map, so error messages can quote the header the user actually has.
ORIGINAL_NAME = {v: k for k, v in RENAME.items()}

TEXT_COLS = {"date", "month_name", "quarter", "district", "region", "plant",
             "area_type"}

# Without these the dashboard cannot be built at all.
REQUIRED = ["date", "plant", "district", "production_m3", "billed_m3"]

# Absent-but-derivable columns are recomputed rather than rejected; PAIP has
# changed its column set before and may again.
DERIVABLE = {
    "nrw_m3": lambda d: d.production_m3 - d.billed_m3,
    "nrw_pct": lambda d: d.nrw_m3 / d.production_m3 * 100,
    "billed_m3": lambda d: (d.billed_domestic_m3 + d.billed_commercial_m3
                            + d.billed_industrial_m3),
    "year": lambda d: d.date.dt.year,
    "month": lambda d: d.date.dt.month,
    "physical_loss_m3": lambda d: d.nrw_m3 * d.physical_share_pct / 100,
    "commercial_loss_m3": lambda d: d.nrw_m3 * (100 - d.physical_share_pct) / 100,
}

MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH_MAP = {"Jan": "Jan", "Feb": "Feb", "Mac": "Mar", "Mar": "Mar",
             "Apr": "Apr", "Mei": "May", "May": "May", "Jun": "Jun",
             "Jul": "Jul", "Ogos": "Aug", "Aug": "Aug", "Sep": "Sep",
             "Sept": "Sep", "Okt": "Oct", "Oct": "Oct", "Nov": "Nov",
             "Dis": "Dec", "Dec": "Dec"}

DATE_FORMATS = ["%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%Y/%m/%d"]


class DataError(Exception):
    """Raised on a structural problem that makes the extract unusable."""


@dataclass
class Report:
    """Human-readable outcome of an ingestion attempt."""
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def text(self) -> str:
        out = []
        if self.stats:
            out.append("SUMMARY")
            for k, v in self.stats.items():
                out.append(f"  {k:34s} {v}")
        for label, items in (("ERRORS", self.errors),
                             ("WARNINGS", self.warnings),
                             ("NOTES", self.notes)):
            if items:
                out.append(f"\n{label}")
                out += [f"  - {m}" for m in items]
        if self.ok and not self.warnings:
            out.append("\nAll checks passed.")
        return "\n".join(out)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def to_number(s: pd.Series) -> pd.Series:
    """PAIP publishes numerics as text with thousands separators and percent
    signs. Coerce explicitly rather than trusting pandas' inference, which
    silently yields object columns and breaks arithmetic downstream."""
    return pd.to_numeric(
        s.astype(str)
         .str.replace(",", "", regex=False)
         .str.replace("%", "", regex=False)
         .str.replace("RM", "", regex=False)
         .str.strip()
         .replace({"": None, "-": None, "nan": None, "NA": None, "N/A": None}),
        errors="coerce")


def parse_dates(s: pd.Series, rep: Report) -> pd.Series:
    """Try known formats in order. Day-first is tried before month-first because
    PAIP publishes dd/mm/yyyy; guessing wrong silently swaps months and days for
    the first twelve days of every month."""
    raw = s.astype(str).str.strip()
    for fmt in DATE_FORMATS:
        out = pd.to_datetime(raw, format=fmt, errors="coerce")
        if out.notna().mean() > 0.98:
            if fmt != DATE_FORMATS[0]:
                rep.notes.append(f"Dates parsed with format '{fmt}'.")
            return out
    out = pd.to_datetime(raw, errors="coerce", dayfirst=True)
    if out.isna().all():
        raise DataError("Could not parse the date column in any known format. "
                        f"First values seen: {raw.head(3).tolist()}")
    rep.warnings.append(
        "Dates fell back to flexible parsing; check that day/month are not "
        "swapped, especially for days 1-12.")
    return out


def discover(source) -> list:
    """Resolve the input into a list of CSV paths.

    Accepts a file, a directory, a glob, or a list. A directory is the
    per-year-append pattern; a single file is the full-replacement pattern.
    """
    if source is None:
        source = RAW_DIR
    if isinstance(source, (list, tuple)):
        paths = [Path(p) for p in source]
    else:
        p = Path(source)
        if p.is_dir():
            paths = sorted(list(p.glob("*.csv")) + list(p.glob("*.CSV")))
        elif any(ch in str(p) for ch in "*?["):
            paths = sorted(Path().glob(str(p)))
        else:
            paths = [p]
    paths = [p for p in paths if p.exists() and p.is_file()]
    if not paths:
        raise DataError(
            f"No CSV files found at '{source}'. Put the PAIP export at "
            f"data/raw/ or pass the path explicitly.")
    return paths


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def load_raw(source=None, rep: Report = None) -> tuple:
    """Read, rename, coerce and concatenate every input file."""
    rep = rep or Report()
    paths = discover(source)
    rep.stats["input files"] = len(paths)

    frames = []
    for path in paths:
        try:
            df = pd.read_csv(path, dtype=str)
        except Exception as exc:
            raise DataError(f"Could not read '{path.name}': {exc}") from exc
        if df.empty:
            rep.warnings.append(f"'{path.name}' is empty and was skipped.")
            continue

        df = df.rename(columns={c: c.strip() for c in df.columns})
        unknown = [c for c in df.columns if c not in RENAME]
        df = df.rename(columns=RENAME)
        if unknown:
            rep.notes.append(
                f"'{path.name}': {len(unknown)} unrecognised column(s) carried "
                f"through unchanged: {', '.join(unknown[:6])}"
                + (" ..." if len(unknown) > 6 else ""))

        missing_req = [c for c in REQUIRED if c not in df.columns]
        # billed_m3 can be rebuilt from its three streams.
        if "billed_m3" in missing_req and {
                "billed_domestic_m3", "billed_commercial_m3",
                "billed_industrial_m3"} <= set(df.columns):
            missing_req.remove("billed_m3")
        if missing_req:
            # Report the header as it appears in the user's file, not the
            # internal analysis name — otherwise the message names a column they
            # will not find anywhere in their spreadsheet.
            shown = [f"{ORIGINAL_NAME.get(c, c)} (-> {c})" for c in missing_req]
            raise DataError(
                f"'{path.name}' is missing required column(s): "
                f"{', '.join(shown)}. Expected the PAIP export layout — see "
                f"RENAME in dataloader.py for the full column map.")

        for col in df.columns:
            if col not in TEXT_COLS:
                df[col] = to_number(df[col])
        df["date"] = parse_dates(df["date"], rep)
        df["_source_file"] = path.name
        frames.append(df)

    if not frames:
        raise DataError("Every input file was empty.")

    data = pd.concat(frames, ignore_index=True)

    # Rebuild anything absent that can be derived. Order matters: billed_m3
    # feeds nrw_m3, which feeds nrw_pct and the loss split.
    for col in ["year", "month", "billed_m3", "nrw_m3", "nrw_pct",
                "physical_loss_m3", "commercial_loss_m3"]:
        if col not in data.columns or data[col].isna().all():
            try:
                data[col] = DERIVABLE[col](data)
                rep.notes.append(f"'{col}' was absent and has been derived.")
            except Exception:
                pass

    if "month_name" in data.columns:
        data["month_name"] = data["month_name"].map(MONTH_MAP).fillna(
            data["month_name"])
    else:
        data["month_name"] = data["date"].dt.strftime("%b")
    data["month_name"] = pd.Categorical(data["month_name"], MONTH_ORDER,
                                        ordered=True)
    return data, rep


def deduplicate(data: pd.DataFrame, rep: Report) -> pd.DataFrame:
    """One record per plant-month. A re-issued year supersedes the original."""
    before = len(data)
    dupes = data.duplicated(["plant", "date"], keep=False)
    if dupes.any():
        conflict = data[dupes].groupby(["plant", "date"])["_source_file"].nunique()
        cross_file = int((conflict > 1).sum())
        data = data.drop_duplicates(["plant", "date"], keep="last")
        msg = (f"{before - len(data)} duplicate plant-month record(s) removed, "
               f"keeping the last occurrence.")
        if cross_file:
            msg += (f" {cross_file} of these appeared in more than one file — "
                    f"the later file was treated as a correction.")
        rep.warnings.append(msg)
    return data


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(data: pd.DataFrame, rep: Report, known_plants=None) -> Report:
    """Structural problems become errors; everything else is a warning."""
    n = len(data)
    rep.stats["records"] = f"{n:,}"
    rep.stats["plants"] = data.plant.nunique()
    rep.stats["districts"] = data.district.nunique()

    if n == 0:
        rep.errors.append("No records after loading.")
        return rep

    # --- structural -------------------------------------------------------
    if data.date.isna().any():
        rep.errors.append(f"{int(data.date.isna().sum())} record(s) have an "
                          f"unparseable date.")
    if data.plant.isna().any():
        rep.errors.append(f"{int(data.plant.isna().sum())} record(s) have no "
                          f"plant name.")
    for col in ("production_m3", "billed_m3"):
        if data[col].isna().any():
            rep.errors.append(f"{int(data[col].isna().sum())} record(s) have a "
                              f"missing {col}.")
    if (data.production_m3 <= 0).any():
        rep.errors.append(f"{int((data.production_m3 <= 0).sum())} record(s) "
                          f"have zero or negative production, which makes the "
                          f"loss rate undefined.")

    # --- identity checks --------------------------------------------------
    nrw_calc = data.production_m3 - data.billed_m3
    drift = (data.nrw_m3 - nrw_calc).abs()
    bad = int((drift > 1).sum())
    if bad:
        rep.warnings.append(
            f"{bad} record(s) where published NRW differs from "
            f"production - billed by more than 1 m3 (max {drift.max():,.0f}). "
            f"The published figure has been kept, not overwritten.")

    neg = int((data.nrw_m3 < 0).sum())
    if neg:
        rep.warnings.append(
            f"{neg} record(s) have NEGATIVE NRW (billed exceeds production). "
            f"This indicates a metering or timing problem. They are retained "
            f"and flagged for PAIP rather than silently corrected.")

    over = int((data.nrw_pct > 100).sum())
    if over:
        rep.warnings.append(f"{over} record(s) have a loss rate above 100%.")

    # --- coverage ---------------------------------------------------------
    years = sorted(data.year.dropna().unique().astype(int))
    rep.stats["years"] = f"{years[0]}-{years[-1]}" if years else "none"
    gaps = [y for y in range(years[0], years[-1] + 1) if y not in years]
    if gaps:
        rep.warnings.append(f"Year gap in coverage: {gaps} missing. "
                            f"Year-on-year comparisons will skip these.")

    per_year = data.groupby("year").apply(
        lambda g: g.date.dt.month.nunique(), include_groups=False)
    partial = {int(y): int(m) for y, m in per_year.items() if m < 12}
    if partial:
        rep.notes.append(
            "Partial year(s): "
            + ", ".join(f"{y} has {m} of 12 months" for y, m in partial.items())
            + ". Volumes are annualised for comparison and labelled in the UI.")
    rep.stats["complete years"] = int((per_year == 12).sum())

    # Plants that skip months distort per-plant trends.
    counts = data.groupby(["plant", "year"]).size().reset_index(name="months")
    ragged = counts[(counts.months != 12)
                    & (~counts.year.astype(int).isin(partial))]
    if len(ragged):
        rep.warnings.append(
            f"{len(ragged)} plant-year(s) do not have 12 monthly records in an "
            f"otherwise complete year, e.g. "
            f"{ragged.iloc[0].plant} {int(ragged.iloc[0].year)} "
            f"({int(ragged.iloc[0].months)} months).")

    # --- estate continuity -------------------------------------------------
    if known_plants is not None:
        now = set(data.plant.unique())
        added, gone = sorted(now - set(known_plants)), sorted(set(known_plants) - now)
        if added:
            rep.notes.append(
                f"{len(added)} new plant(s) since the last build: "
                f"{', '.join(added[:5])}{' ...' if len(added) > 5 else ''}")
        if gone:
            rep.warnings.append(
                f"{len(gone)} plant(s) present before are absent now: "
                f"{', '.join(gone[:5])}{' ...' if len(gone) > 5 else ''}. "
                f"Check for a renaming rather than a closure.")

    # A plant name must map to exactly one district, or the crosswalk breaks.
    multi = data.groupby("plant").district.nunique()
    if (multi > 1).any():
        rep.errors.append(
            f"{int((multi > 1).sum())} plant name(s) appear in more than one "
            f"district: {', '.join(multi[multi > 1].index[:5])}. Plant names "
            f"must be unique across the estate.")

    return rep


def ingest(source=None, known_plants=None, strict: bool = True) -> tuple:
    """Load, de-duplicate and validate. Returns (DataFrame, Report).

    With strict=True a structural error raises DataError; the report is attached
    to the exception so the caller can show it.
    """
    rep = Report()
    data, rep = load_raw(source, rep)
    data = deduplicate(data, rep)
    rep = validate(data, rep, known_plants)

    if strict and not rep.ok:
        err = DataError("Input failed validation:\n\n" + rep.text())
        err.report = rep
        raise err

    data = data.sort_values(["plant", "date"]).reset_index(drop=True)
    return data, rep


def year_completeness(data: pd.DataFrame) -> pd.DataFrame:
    """Months observed per year, and the annualisation factor that puts a
    partial year on the same footing as a full one."""
    out = (data.groupby("year")
               .agg(months=("date", lambda s: s.dt.month.nunique()),
                    records=("date", "size"))
               .reset_index())
    out["complete"] = out.months == 12
    out["annualise"] = 12 / out.months.clip(lower=1)
    out["label"] = np.where(out.complete, out.year.astype(int).astype(str),
                            out.year.astype(int).astype(str) + " ("
                            + out.months.astype(str) + " of 12 months)")
    return out


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else None
    try:
        d, r = ingest(src)
        print(r.text())
        print("\n" + year_completeness(d).to_string(index=False))
    except DataError as e:
        print(e)
        raise SystemExit(1)
