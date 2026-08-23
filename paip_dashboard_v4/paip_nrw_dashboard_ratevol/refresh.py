"""
PAIP NRW — one-command refresh
==============================

Rebuilds every artefact the dashboard reads, in the order they depend on each
other, from whatever raw data is present.

    python refresh.py                       # rebuild from data/raw/*.csv
    python refresh.py path/to/2026.csv      # rebuild from one explicit file
    python refresh.py --add path/to/2026.csv    # copy into data/raw/, then rebuild
    python refresh.py --no-verify           # skip the check suite (faster)
    python refresh.py --dry-run             # validate only, write nothing

Stages:
    1. clean   dataloader + prepare_data  -> tidy dataset, LIPS, crosswalk
    2. train   train_models               -> criticality, archetypes, metrics
               train_burst_model          -> burst risk classifier
    3. verify  verify                     -> independent checks

A failure in stage 1 stops everything: training on a malformed extract would
produce confident nonsense. Stage 2 failing still leaves a working dashboard
minus the Early Warning tab, so it is reported but not fatal.

Existing artefacts are backed up to data/_backup/ before being overwritten, so a
bad refresh can be rolled back.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"
RAW_DIR = DATA / "raw"
BACKUP = DATA / "_backup"

ARTEFACTS = ["nrw_plant_month.csv", "nrw_plant_year.csv", "plant_crosswalk.csv",
             "data_quality.csv", "missing_values.csv", "year_coverage.csv",
             "ml_plant.csv", "ml_monthly.csv", "model_metrics.json",
             "burst_predictions.csv", "burst_history.csv", "burst_metrics.json"]


def rule(title: str):
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def backup():
    """Snapshot current artefacts so a bad refresh is reversible."""
    existing = [p for p in (DATA / a for a in ARTEFACTS) if p.exists()]
    if not existing:
        return None
    BACKUP.mkdir(parents=True, exist_ok=True)
    for p in existing:
        shutil.copy2(p, BACKUP / p.name)
    print(f"  Backed up {len(existing)} artefact(s) to data/_backup/")
    return BACKUP


def restore():
    if not BACKUP.exists():
        print("  No backup to restore from.")
        return
    n = 0
    for p in BACKUP.glob("*"):
        shutil.copy2(p, DATA / p.name)
        n += 1
    print(f"  Restored {n} artefact(s) from data/_backup/")


def add_file(path: Path) -> Path:
    """Copy a new export into data/raw/ so future refreshes pick it up."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / path.name
    if dest.exists() and dest.resolve() == path.resolve():
        return dest
    shutil.copy2(path, dest)
    print(f"  Copied '{path.name}' into data/raw/")
    return dest


def run_step(name: str, argv: list) -> tuple:
    """Run a pipeline stage as a subprocess so a crash cannot corrupt this one."""
    t0 = time.time()
    proc = subprocess.run([sys.executable] + argv, cwd=HERE,
                          capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    print(out.rstrip())
    print(f"  [{name}] {'OK' if proc.returncode == 0 else 'FAILED'} "
          f"in {time.time() - t0:.1f}s")
    return proc.returncode == 0, out


def main():
    ap = argparse.ArgumentParser(description="Rebuild all PAIP NRW artefacts.")
    ap.add_argument("source", nargs="?", default=None,
                    help="CSV file, directory or glob. Default: data/raw/")
    ap.add_argument("--add", metavar="FILE",
                    help="Copy FILE into data/raw/ before rebuilding.")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--no-train", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="Validate the input and stop; write nothing.")
    args = ap.parse_args()

    source = args.source
    if args.add:
        p = Path(args.add)
        if not p.exists():
            print(f"File not found: {p}")
            return 1
        add_file(p)
        source = None  # rebuild from the whole raw folder

    # ---- dry run: validate only -----------------------------------------
    if args.dry_run:
        rule("VALIDATE ONLY (--dry-run)")
        from dataloader import ingest, year_completeness, DataError
        try:
            d, rep = ingest(source, strict=False)
        except DataError as exc:
            print(exc)
            return 1
        print(rep.text())
        print("\nYear coverage:")
        print(year_completeness(d).to_string(index=False))
        print("\nNothing was written." if rep.ok else
              "\nNothing was written. Fix the errors above before refreshing.")
        return 0 if rep.ok else 1

    rule("PAIP NRW — full refresh")
    backup()

    # ---- 1. clean --------------------------------------------------------
    rule("1/3  CLEAN")
    argv = ["prepare_data.py"] + ([str(source)] if source else [])
    ok, out = run_step("clean", argv)
    if not ok:
        print("\nClean failed — the raw input did not pass validation.\n"
              "Nothing downstream was rebuilt; restoring previous artefacts.")
        restore()
        return 1

    # ---- 2. train --------------------------------------------------------
    if args.no_train:
        print("\n  [train] skipped (--no-train)")
    else:
        rule("2/3  TRAIN")
        ok, _ = run_step("expected-loss model", ["train_models.py"])
        if not ok:
            print("\nExpected-loss training failed. The dashboard still runs, "
                  "but model-derived columns will be unavailable.")
        okb, _ = run_step("burst-risk model", ["train_burst_model.py"])
        if not okb:
            print("\nBurst-risk training failed. The dashboard still runs, but "
                  "the Burst Risk tab will be unavailable until this succeeds.")

    # ---- 3. verify -------------------------------------------------------
    if args.no_verify:
        print("\n  [verify] skipped (--no-verify)")
    else:
        rule("3/3  VERIFY")
        ok, out = run_step("verify", ["verify.py"])
        if not ok:
            print("\nVerification FAILED. The artefacts were rebuilt but do not "
                  "agree with the raw data — do not publish this build.\n"
                  "Run `python refresh.py` again after fixing, or restore the "
                  "previous build with the backup in data/_backup/.")
            return 1

    rule("DONE")
    print("All artefacts rebuilt. Start the dashboard with:\n"
          "    streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
