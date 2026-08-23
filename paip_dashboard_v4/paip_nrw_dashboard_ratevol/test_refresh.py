"""
PAIP NRW — refresh regression test
=================================

Proves the pipeline is genuinely year-agnostic by fabricating next year's data
and running it end to end. Nothing here touches the real build: every scenario
runs in a temporary directory and the live data/ folder is restored afterwards.

    python test_refresh.py

Scenarios:
    1. full year appended        2026 complete, as a separate per-year file
    2. partial year appended     2026 with 6 months only
    3. full replacement workbook all years in one file
    4. corrected re-issue        a year supplied twice, later file wins
    5. missing required column   must be REJECTED
    6. unparseable dates         must be REJECTED
    7. zero production           must be REJECTED
    8. plant in two districts    must be REJECTED
    9. renamed plant             must be reported, not silently absorbed

Scenarios 5-8 are as important as the happy paths: a loader that accepts
malformed input is worse than one that refuses it.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from dataloader import ingest, year_completeness, DataError

HERE = Path(__file__).parent
SOURCE = HERE / "data" / "raw"
SEED = 7

passed, failed = 0, 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}" + (f" — {detail}" if detail else ""))
    else:
        failed += 1
        print(f"  FAIL  {name}" + (f" — {detail}" if detail else ""))


def base_frame() -> pd.DataFrame:
    """The most recent complete year from the real export, as raw text."""
    files = sorted(SOURCE.glob("*.csv"))
    if not files:
        raise SystemExit("No source CSV in data/raw/ — run prepare_data.py first.")
    df = pd.read_csv(files[0], dtype=str)
    latest = df["Tahun"].astype(int).max()
    return df[df["Tahun"].astype(int) == latest].copy(), latest


def synthesise(base: pd.DataFrame, old_year: int, new_year: int,
               months: int = 12, drift: float = 0.97) -> pd.DataFrame:
    """Fabricate a plausible next year by shifting dates and nudging volumes.

    Deliberately keeps the published columns internally consistent — NRW is
    recomputed from production and billed rather than scaled independently —
    because the point is to test the pipeline, not to smuggle in identity
    violations that would fail validation for the wrong reason.
    """
    rng = np.random.default_rng(SEED)
    d = base.copy()
    d["Tahun"] = str(new_year)
    d["Tarikh"] = d["Tarikh"].str.replace(f"/{old_year}", f"/{new_year}",
                                          regex=False)

    def num(col):
        return pd.to_numeric(d[col].astype(str).str.replace(",", "", regex=False)
                             .str.replace("%", "", regex=False), errors="coerce")

    noise = rng.normal(1.0, 0.02, len(d))
    production = (num("Pengeluaran_m3") * rng.normal(1.02, 0.01, len(d))).round()
    # Losses drift down slightly, matching the estate's observed trajectory.
    nrw = (num("NRW_m3") * drift * noise).round()
    nrw = np.minimum(nrw, production * 0.95)
    billed = production - nrw

    dom_share = num("Bil_Domestik_m3") / num("Jumlah_Dibilkan_m3")
    com_share = num("Bil_Komersial_m3") / num("Jumlah_Dibilkan_m3")
    phys_share = num("Bahagian_Kehilangan_Fizikal")

    d["Pengeluaran_m3"] = production.astype(int).astype(str)
    d["Jumlah_Dibilkan_m3"] = billed.round().astype(int).astype(str)
    d["NRW_m3"] = nrw.astype(int).astype(str)
    d["NRW_Peratus"] = (nrw / production * 100).round(2).astype(str)
    d["Bil_Domestik_m3"] = (billed * dom_share).round().astype(int).astype(str)
    d["Bil_Komersial_m3"] = (billed * com_share).round().astype(int).astype(str)
    d["Bil_Industri_m3"] = (billed - (billed * dom_share).round()
                            - (billed * com_share).round()).round().astype(int).astype(str)
    d["Kehilangan_Fizikal_m3"] = (nrw * phys_share / 100).round().astype(int).astype(str)
    d["Kehilangan_Komersial_m3"] = (nrw - (nrw * phys_share / 100).round()
                                    ).round().astype(int).astype(str)
    # Every column PAIP publishes as a derived quantity must be recomputed too.
    # Leaving them at last year's values produces a file that is internally
    # inconsistent in ways real data never is, and the verifier rightly fails on
    # it — which is how this omission was caught in the first place.
    days = pd.to_datetime(d["Tarikh"], format="%d/%m/%Y").dt.days_in_month
    pop = num("Populasi_Dilayan")
    acc = num("Akaun_Pelanggan")
    tariff = num("Tarif_Purata_RM_m3")
    cap = num("Kapasiti_Loji_m3_hari")
    pipe = num("Panjang_Paip_km")

    energy = (num("Tenaga_kWh") * rng.normal(1.02, 0.01, len(d))).round()
    chem = (num("Kos_Kimia_RM") * rng.normal(1.03, 0.01, len(d))).round()
    maint = (num("Kos_Penyelenggaraan_RM") * rng.normal(1.03, 0.02, len(d))).round()
    e_tariff = num("Tarif_Tenaga_RM_kWh")
    e_cost = (energy * e_tariff).round(2)
    opex = e_cost + chem + maint
    revenue = (billed * tariff).round(2)

    d["Tenaga_kWh"] = energy.astype(int).astype(str)
    d["Kos_Kimia_RM"] = chem.astype(int).astype(str)
    d["Kos_Penyelenggaraan_RM"] = maint.astype(int).astype(str)
    d["Kos_Tenaga_RM"] = e_cost.astype(str)
    d["Jumlah_Kos_Operasi_RM"] = opex.astype(str)
    d["Hasil_Bil_RM"] = revenue.astype(str)
    # Decimal places match what PAIP actually publishes per column. Rounding
    # more coarsely than the source makes the synthetic file fail the verifier's
    # identity tolerances for a reason that has nothing to do with the pipeline.
    d["Kos_per_m3_RM"] = (opex / production).round(3).astype(str)
    d["Margin_Operasi_pct"] = ((revenue - opex) / revenue * 100).round(2).astype(str)
    d["Hasil_per_Akaun_RM"] = (revenue / acc).round(2).astype(str)
    d["Penggunaan_Kapasiti_pct"] = (production / (cap * days) * 100).round(2).astype(str)
    d["Kecekapan_Tenaga_kWh_m3"] = (energy / production).round(3).astype(str)
    d["NRW_per_km_m3"] = (nrw / pipe).round(1).astype(str)
    d["Penggunaan_per_Kapita_L_hari"] = (billed * 1000 / pop / days).round(1).astype(str)

    if "Record_ID" in d.columns:
        d["Record_ID"] = [f"R{new_year}{i:05d}" for i in range(len(d))]

    if months < 12:
        d = d[pd.to_numeric(d["Bulan_No"], errors="coerce") <= months]
    return d


def scenario(title):
    print(f"\n--- {title} ---")


def main():
    base, latest = base_frame()
    nxt = latest + 1
    full = synthesise(base, latest, nxt, months=12)
    partial = synthesise(base, latest, nxt, months=6)
    original = pd.read_csv(sorted(SOURCE.glob("*.csv"))[0], dtype=str)

    tmp = Path(tempfile.mkdtemp(prefix="paip_test_"))
    try:
        # 1 -----------------------------------------------------------------
        scenario(f"1. Full {nxt} appended as a separate file")
        d1 = tmp / "s1"; d1.mkdir()
        original.to_csv(d1 / "paip_history.csv", index=False)
        full.to_csv(d1 / f"paip_{nxt}.csv", index=False)
        data, rep = ingest(d1)
        cov = year_completeness(data)
        check("accepted", rep.ok)
        check(f"{nxt} present", nxt in set(cov.year.astype(int)))
        check("all years complete", bool(cov.complete.all()),
              f"{len(cov)} years")
        check("record count grew by one year",
              len(data) == len(original) + len(full),
              f"{len(data):,} records")

        # 2 -----------------------------------------------------------------
        scenario(f"2. Partial {nxt} (6 months) appended")
        d2 = tmp / "s2"; d2.mkdir()
        original.to_csv(d2 / "paip_history.csv", index=False)
        partial.to_csv(d2 / f"paip_{nxt}.csv", index=False)
        data, rep = ingest(d2)
        cov = year_completeness(data)
        row = cov[cov.year.astype(int) == nxt].iloc[0]
        check("accepted", rep.ok)
        check("partial year flagged incomplete", not bool(row.complete))
        check("months counted correctly", int(row.months) == 6,
              f"{int(row.months)} months")
        check("annualisation factor is 2.0", abs(row.annualise - 2.0) < 1e-9)
        check("label states the shortfall", "6 of 12" in row.label, row.label)
        check("reported as a note, not an error",
              any("Partial year" in n for n in rep.notes))

        # 3 -----------------------------------------------------------------
        scenario("3. Full replacement workbook (all years in one file)")
        d3 = tmp / "s3"; d3.mkdir()
        pd.concat([original, full], ignore_index=True).to_csv(
            d3 / "paip_all_years.csv", index=False)
        data, rep = ingest(d3)
        check("accepted", rep.ok)
        check("same result as the append pattern",
              len(data) == len(original) + len(full),
              f"{len(data):,} records")

        # 4 -----------------------------------------------------------------
        scenario("4. Corrected re-issue — a year supplied twice")
        d4 = tmp / "s4"; d4.mkdir()
        original.to_csv(d4 / "a_history.csv", index=False)
        corrected = full.copy()
        corrected["Pengeluaran_m3"] = "999999"
        full.to_csv(d4 / f"b_{nxt}_original.csv", index=False)
        corrected.to_csv(d4 / f"c_{nxt}_corrected.csv", index=False)
        data, rep = ingest(d4, strict=False)
        got = data[data.year == nxt].production_m3.unique()
        check("no double counting",
              len(data) == len(original) + len(full),
              f"{len(data):,} records")
        check("later file wins", set(got) == {999999.0},
              f"production values: {got[:3]}")
        check("de-duplication was reported",
              any("duplicate" in w for w in rep.warnings))

        # 5 -----------------------------------------------------------------
        scenario("5. Missing a required column — must be rejected")
        d5 = tmp / "s5"; d5.mkdir()
        broken = original.drop(columns=["Pengeluaran_m3"])
        broken.to_csv(d5 / "broken.csv", index=False)
        try:
            ingest(d5)
            check("rejected", False, "it was accepted")
        except DataError as e:
            check("rejected", True)
            check("names the missing column", "Pengeluaran_m3" in str(e))

        # 6 -----------------------------------------------------------------
        scenario("6. Unparseable dates — must be rejected")
        d6 = tmp / "s6"; d6.mkdir()
        bad = original.copy()
        bad["Tarikh"] = "not-a-date"
        bad.to_csv(d6 / "broken.csv", index=False)
        try:
            ingest(d6)
            check("rejected", False, "it was accepted")
        except DataError as e:
            check("rejected", True, str(e).split("\n")[0][:60])

        # 7 -----------------------------------------------------------------
        scenario("7. Zero production — must be rejected")
        d7 = tmp / "s7"; d7.mkdir()
        bad = original.copy()
        bad.loc[bad.index[:5], "Pengeluaran_m3"] = "0"
        bad.to_csv(d7 / "broken.csv", index=False)
        try:
            ingest(d7)
            check("rejected", False, "it was accepted")
        except DataError as e:
            check("rejected", True)
            check("explains why the rate is undefined",
                  "undefined" in str(e) or "negative production" in str(e))

        # 8 -----------------------------------------------------------------
        scenario("8. One plant in two districts — must be rejected")
        d8 = tmp / "s8"; d8.mkdir()
        bad = original.copy()
        target = bad.Nama_Loji.iloc[0]
        mask = (bad.Nama_Loji == target)
        bad.loc[bad.index[mask][:3], "Daerah"] = "SOMEWHERE ELSE"
        bad.to_csv(d8 / "broken.csv", index=False)
        try:
            ingest(d8)
            check("rejected", False, "it was accepted")
        except DataError as e:
            check("rejected", True)
            check("names the offending plant", target in str(e))

        # 9 -----------------------------------------------------------------
        scenario("9. A plant renamed between years — must be reported")
        d9 = tmp / "s9"; d9.mkdir()
        renamed = full.copy()
        old_name = renamed.Nama_Loji.iloc[0]
        renamed.loc[renamed.Nama_Loji == old_name, "Nama_Loji"] = "BRAND NEW PLANT"
        original.to_csv(d9 / "a_history.csv", index=False)
        renamed.to_csv(d9 / f"b_{nxt}.csv", index=False)
        known = sorted(original.Nama_Loji.unique())
        data, rep = ingest(d9, known_plants=known, strict=False)
        check("accepted with a report", rep.ok)
        check("new plant reported",
              any("BRAND NEW PLANT" in n for n in rep.notes))
        # The old name still exists in history, so it is not "gone" — the real
        # signal is the appearance of an unfamiliar name.
        check("addition surfaced as a note", any("new plant" in n.lower()
                                                 for n in rep.notes))

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{'=' * 74}")
    print(f"{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
