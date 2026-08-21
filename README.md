# PAIP Non-Revenue Water — Intervention Priority Dashboard

A Streamlit dashboard that turns Pengurusan Air Pahang Berhad's published monthly
production and billing figures into a ranked, volume-weighted repair schedule.

The operational question is not *how large are the losses* — PAIP already knows —
but *which plants to fix first*. NRW is published as a percentage because a
percentage makes plants comparable; but a percentage measures efficiency, not
recoverable water. This dashboard shows both, and quantifies how far apart they
are.

---

## Running it

```bash
pip install -r requirements.txt
# put the PAIP export in data/raw/ (any filename)
python refresh.py           # clean -> train -> verify, one command
streamlit run app.py        # opens on http://localhost:8501
```

Verify or test at any time:

```bash
python refresh.py --dry-run # validate the input, write nothing
python verify.py            # 81 independent checks against the raw CSV
python test_refresh.py      # 25 checks that next year's data will load
python shoot.py             # headless render test, both colour modes
python fitcheck.py          # asserts no view scrolls at 1920x980
```

**Layout.** Dense by design and never scrolling. The **Command centre** view is
a single-screen grid — a five-tile KPI rail over nine compact cards — that carries
the whole argument at a glance; the four sections behind it hold the same
material at full size with the reasoning attached. All fifteen views fit a
1920×980 viewport, the tallest at 945px.

**Controls.** Navigation, year, scope and appearance all live in the left
sidebar, and one
filter row scopes every view — no chart carries its own controls. The scope
selectors treat empty as "all", so the common case shows three tidy boxes rather
than a wall of chips. LIPS weights are deliberately *not* exposed: a priority
score that changes when the viewer drags a slider is not a queue.

**Light and dark mode.** **Light is the default**, whatever the operating system
is set to — a dashboard shown to an examiner or printed into a report should not
change appearance with the viewing machine. Dark and Auto are both offered in
the sidebar. The two palettes are separately validated against their own
surfaces rather than one being an inversion of the other: the categorical set
clears the adjacent-pair CVD floor (ΔE ≥ 8) and the normal-vision floor
(ΔE ≥ 15) in both modes.

**Water, lightly.** Surfaces carry a faint cool tint instead of neutral grey, a
droplet glyph marks each KPI label, and a thin wave rule divides the sidebar.
That is the whole of it — nothing animated, nothing competing with the data.

---

## The headline finding

Ranking the same 74 plants by loss **rate** and by loss **volume** produces two
almost unrelated queues:

| Measure | Value |
|---|---|
| Spearman ρ between rate and volume | **−0.54** (negative — they rank plants *oppositely*) |
| Kendall τ between the two rank orders | −0.35 |
| Plants shared by the two top-10 queues | **0 of 10** |
| Water in the top-10-by-volume queue | 140.2M m³ |
| Water in the top-10-by-rate queue | 8.9M m³ |
| Ratio | **15.8× more water for the same ten crew deployments** |

Losses are also highly concentrated: **65% of all non-revenue water sits in 10 of
74 plants**. Monthly loss rates move within a 0.9 pp band across 2025 with no
wet/dry-season signal, which is the signature of continuous physical leakage
rather than seasonal demand or intermittent billing error — and continuous
leakage is what repair work recovers.

System figures for 2025: **32.1%** loss rate, **216.5M m³** lost, **RM 268.5M**
in forgone revenue at the published average tariff — **68%** of it physical on
PAIP's published apportionment, which this dashboard reports but cannot verify.

---

## The LIPS score

The Leakage Intervention Priority Score ranks plants for intervention by
combining four operational indicators into a balanced 0–100 score:

| Component | Weight | Reads as |
|---|---|---|
| `nrw_per_km_m3` | 40% | **Loss density** — NRW volume per km of main |
| `bursts_per_100km` | 25% | **Burst rate** — a proxy for physical pipe failure |
| `plant_age_yr` | 20% | **Plant age** — deterioration and asset-condition risk |
| `account_density` | 15% | **Account density** — commercial and metering exposure |

Each component becomes a 0–100 percentile rank within the plants currently on
screen, and the four are combined at those weights. Percentile rank rather than
min–max because plant size is heavily right-skewed: min–max would let the single
largest plant compress everything else into the bottom decile.

Scoring happens **within each year**. Ranks are only meaningful against
contemporaries, and adding a new year must not reshuffle history.

### What the verification asserts

`verify.py` checks the score rather than trusting it:

- every component is present and non-null in the artefacts — the check that
  catches a weight being added without a rebuild, which is exactly how the
  dashboard once shipped a `KeyError` on load;
- the weights sum to 100;
- the score recomputes from the four columns to within 0.01;
- ranks form a strict `1..n` order, so two plants can never both be "priority
  31" — ties break deterministically;
- **no single component correlates above ρ 0.97 with the result.** A composite
  that reproduces one of its own inputs is not a composite; the other three
  would be decoration.

### What it reports without asserting

The same block prints how much of the estate's NRW each of the three possible
queues reaches in its top ten — the LIPS order, a pure volume order, and a pure
rate order. It is reported, not enforced, because a score built from densities
and proxies is not optimising for volume, and the size of that trade belongs in
front of the reader rather than hidden behind a passing test.

This matters given the headline finding above: loss density is a *rate per km*,
not a quantity of water. The four-factor score is an operational-risk ranking,
and it is not the same thing as ranking by recoverable volume.

### The Criticality Index is not folded in

It measures how *abnormal* a plant looks, not how urgent it is. Blending the two
would make each harder to interpret, so criticality stays a separate signal in
the Plant Profile, and the Command screen plots the two against each other so
their disagreement is visible.

---

## The early-warning models

LIPS asks *where is the operational risk concentrated*. These models ask a
different question — *where is something wrong* — by finding the plants that
lose more than their physical characteristics can account for. Their outputs
surface in the Plant Profile and on the Command screen, not in the ranking.

**There are no ground-truth labels in this dataset** — no repair records, no
inspections, no confirmed defects. Nothing here is a trained failure classifier
and it is not presented as one. What is modelled is *expected loss given plant
characteristics*; a large positive residual means "this plant is unusual for its
type", which is a lead to investigate, not a diagnosis.

### 1. Unexplained loss

A regression model predicts monthly NRW% from 26 asset, network, operational and
environmental features. Three candidates compete; the winner on grouped CV is
used.

| Model | R² (unseen plants) | MAE pp | R² (forward in time) |
|---|---|---|---|
| Ridge regression | **0.451** ✓ selected | **5.16** | 0.613 |
| Gradient boosting | 0.384 | 5.35 | 0.652 |
| Mean baseline | −0.036 | 7.31 | −0.040 |

Two design decisions carry this result:

**Validation is grouped by plant.** Static characteristics (pipe length, age,
capacity) are near-constant within a plant, so a random split would let the model
memorise each plant's own loss level and the residual would collapse toward zero.
`GroupKFold` forces every prediction to come from a model that has never seen the
plant it is scoring. `verify.py` asserts zero plant overlap between train and
test folds — the claim is tested, not asserted.

**The simpler model won, and it was selected on that basis.** An unconstrained
booster (depth 6, 400 iterations) scored just 0.21 on unseen plants: with only 74
plant groups it spends its capacity on plant-specific structure that does not
transfer. Heavy regularisation lifted it to 0.38, still short of ridge. Note the
reversal in the last column — gradient boosting looks *better* forward in time
(0.652) precisely because the same plants appear on both sides of that split and
it can memorise them. That number is optimistic and is **not** what the residual
rests on.

**Leakage control is the central safeguard.** NRW is production minus billed
volume, so any feature carrying billed volume encodes the target algebraically.
Two source columns are exactly this trap and are excluded:

| Column | Actually equals | Verdict |
|---|---|---|
| `consumption_per_capita_l_day` | billed ÷ population ÷ days | leaks — excluded |
| `revenue_per_account_rm` | billed × tariff ÷ accounts | leaks — excluded |
| `capacity_utilisation_pct` | production ÷ (capacity × days) | safe — retained |
| `energy_intensity_kwh_m3` | kWh ÷ production | safe — retained |

Each was confirmed numerically, not assumed. The exclusion list is asserted at
runtime and re-derived independently in `verify.py`.

### 2. Sudden deterioration — a null result

Robust per-plant z-scores (median/MAD), a global Isolation Forest, and a Welch
step-change test comparing the last 6 months against prior history.

**These detectors found nothing, and the dashboard says so.** Across 74 plants
and 36 months: **0** plants with a statistically significant worsening trend,
**0** with a significant step increase, and **1** month-level anomaly in 2,664
records. The estate is improving uniformly at a median **1.06 pp/year**.

That is a genuine finding, not a failure. It says PAIP's losses are *chronic and
structural*, not the result of sudden failures — which is why criticality here is
driven almost entirely by unexplained loss. Because every plant is improving, the
trend component measures *improving more slowly than peers* rather than outright
worsening; an absolute test would flag nobody and the component would be a
constant.

### 3. Criticality Index

Percentile-rank blend: unexplained loss 40%, sudden deterioration 30%, relative
trend 30%. Ties break on unexplained volume.

Criticality is **not** a restatement of LIPS — `verify.py` checks this. Spearman
ρ against LIPS is 0.43 and against NRW volume 0.15, so it carries genuinely
different information. Top of the 2025 ranking:

| # | Plant | District | Criticality | Unexplained |
|---|---|---|---|---|
| 1 | SG BILUT | RAUB | 84.3 | +2.4 pp |
| 2 | BERA KOMPLEKS | BERA | 82.6 | +9.6 pp |
| 3 | KECHAU | LIPIS | 80.3 | +8.2 pp (534k m³) |
| 4 | SG BERA (KEPAYANG) | BERA | 80.0 | +14.2 pp |
| 5 | PADANG PIOL | JERANTUT | 79.7 | +6.9 pp (562k m³) |

Across the estate, **7.4M m³** — about 3% of all NRW — sits above what the model
predicts from plant characteristics.

### 4. Failure archetypes

KMeans over the loss signature, named from the two features whose centroid
deviates most from the estate, each mapped to the intervention it implies.

**Separation is weak** (silhouette 0.20 at the best k). The estate varies
continuously rather than falling into discrete failure types. The tab shows the
silhouette curve for k = 2…6 and lets you change k, precisely so this is visible
rather than hidden behind a confident-looking label. Treat the archetypes as a
communication aid, not evidence of real categories.

### Kept separate from LIPS

Criticality is not blended into the priority score. The two answer different
questions — *where is the most operational risk* versus *where is something
behaving abnormally* — and a reader can compare them directly on the Command
screen, where every plant is plotted as criticality against LIPS.

---

## The burst-risk classifier

The dashboard's genuinely *predictive* model. Unlike the expected-loss model,
this one has a **real ground-truth label** already in the data — the burst count
PAIP records — so it predicts a future event and can be scored against whether
that event actually happened.

### The target, and why it is not "will there be a burst"

**98.1%** of plant-months already record at least one burst. "Any burst" is a
constant, not a prediction. The target is an **elevated month — 2 or more
bursts** — which occurs in **39.8%** of plant-months and is the event a
maintenance planner can actually act on.

### Results

| Model | CV AUC | Test AUC | PR-AUC | F1 | Brier |
|---|---|---|---|---|---|
| Majority class | 0.500 | 0.500 | 0.401 | 0.00 | 0.240 |
| Persistence rule | 0.700 | 0.710 | 0.563 | 0.66 | 0.282 |
| Logistic regression | 0.843 | 0.867 | 0.836 | 0.70 | 0.144 |
| **Random forest** ✓ selected | **0.846** | **0.866** | **0.828** | 0.72 | 0.145 |
| Gradient boosting | 0.832 | 0.845 | 0.800 | 0.69 | 0.158 |

At the operating threshold: **precision 66%, recall 84%** (TN 190, FP 76,
FN 28, TP 150 over 444 held-out plant-months).

**Two baselines, both beaten.** Majority class is the floor at 0.500.
*Persistence* — "next month looks like this month" — is the rule a planner would
use with no model at all, and it reaches 0.710. Beating chance proves nothing;
beating persistence by 16 AUC points is the real result.

### Validation — temporal, deliberately unlike the other model

Splits are **chronological**. Every fold trains strictly on months preceding the
ones it scores, and the final 6 months were never touched until the model was
chosen. The expected-loss model groups by *plant* because its question is "does
this generalise to an unseen plant"; here the plants are fixed and the question
is "does this generalise to a month that has not happened yet" — so time is what
must be held out.

**No leakage.** Features come from month *t*, the label from *t+1*; rolling
windows look only backwards. `verify.py` joins every stored label back to the
source month to confirm it really is next month's outcome, and separately
confirms the label is not an echo of the current month (71.6% agreement, not
~100%).

**The threshold was tuned on training data** (0.326, maximising F1). Tuning it
on the test window would make the reported precision and recall flattering.

### Calibration

Predicted 0.95 → observed 0.95; predicted 0.21 → observed 0.21. Brier 0.145.
The risk percentages are usable as probabilities, not merely as a ranking.

### What it relies on — and the trap in measuring that

Single-feature permutation importance is **unreliable here**: seven
burst-history features carry nearly the same information and substitute for one
another when any single one is shuffled, which is how *calendar month* ended up
ranked first despite the elevated-burst rate varying only between 35% and 47%
across the year. Permuting whole **families** together gives the honest picture:

| Group | AUC drop |
|---|---|
| Asset condition | +0.050 |
| Network shape | +0.035 |
| Burst history | +0.007 |
| Loss state | +0.007 |
| Everything else | ≈ 0 |

**Burst risk on this estate is structural.** Recent bursts do predict — the
persistence rule reaches 0.71 alone — but once the model knows a plant is large,
old and long-networked, last month's count adds little.

### Limitations

- Predicts **one month ahead, at plant level** — not where on the network, and
  not when in the month.
- **Burst counts are partly a reporting artefact.** A plant with more staff
  records more bursts; no field here separates reporting behaviour from physical
  failure.
- **The dataset appears synthetic** (every published identity holds to the
  rounding digit), so the relationships may be partly manufactured. The method
  transfers to real PAIP data; the exact AUC may not.

---

## Adding next year's data

**Nothing is hard-coded to a year.** Every year label, heading, split and model
focus is derived from what is in `data/raw/`. Dropping 2026 in and running
`python refresh.py` is the whole procedure — no code change.

### Two arrival patterns, auto-detected

| Pattern | What you do |
|---|---|
| **Full replacement** | PAIP republishes one workbook covering every year — replace the file in `data/raw/` |
| **Per-year append** | Drop `paip_2026.csv` alongside the existing file; everything in the folder is concatenated |

Records are de-duplicated on plant and month keeping the **last** occurrence, so
a corrected re-issue of an old year supersedes the original instead of
double-counting it.

### The refresh command

```bash
python refresh.py --dry-run              # validate only, write nothing
python refresh.py                        # rebuild from data/raw/
python refresh.py --add ~/paip_2026.csv  # copy the file in, then rebuild
```

Stages run in dependency order: **clean → train → verify**. Clean failing stops
everything, because training on a malformed extract produces confident nonsense.
Verify failing means the rebuilt artefacts disagree with the raw data — do not
publish that build. Existing artefacts are backed up to `data/_backup/` first,
and restored automatically if the clean stage fails.

`python refresh.py --dry-run` does the same validation without writing
anything, so a new export can be checked before committing to a rebuild.

### Partial years are kept, flagged and annualised

A year still in progress is not hidden and not silently mixed with complete
years. It is labelled everywhere (`2026 · 7 of 12 months`), drawn in the amber
status colour on year-comparison charts with a footnote, and its volumes are
annualised wherever years are compared. Rates are ratios and need no adjustment.
Both raw and annualised volumes are stored, so the UI shows actuals while
charting comparables.

### What the loader rejects

Structural problems raise and stop the build; everything else is a warning and
the build proceeds.

| Check | Severity |
|---|---|
| Required columns present (named as they appear in *your* file) | Error |
| Dates parse unambiguously — day-first tried first, matching PAIP | Error |
| Production is positive | Error |
| Each plant maps to exactly one district | Error |
| NRW = production − billed | Warning — published figure kept |
| Negative NRW | Warning — retained and flagged, never corrected |
| Duplicate plant-months | Warning — last wins, treated as a correction |
| Year gaps | Warning |
| New or vanished plants | Note — a vanished plant is usually a rename |
| Incomplete years | Note |

### Proof it works

`test_refresh.py` fabricates next year's data and runs nine scenarios end to
end — full year appended, partial year, full replacement, corrected re-issue,
and five deliberately malformed files that **must** be rejected. All 25 checks
pass. The synthetic year is built to be internally consistent, matching PAIP's
own per-column decimal precision, so it exercises the pipeline rather than
tripping identity checks for the wrong reason.

---

## Navigation

There is no tab strip. Every view is a row in the sidebar, grouped by section,
so any of the fifteen is one click from any other rather than two.

| Section | Views | What it answers |
|---|---|---|
| **Overview** | Command centre | The whole argument on one screen — priority queue, what drives each score, the rate-vs-volume divergence, monthly loss rate, water by district, loss concentration, burst-risk bands, loss composition, criticality vs LIPS. Everything below is this material at full size with the reasoning attached. |
| **Priority** | Ranking · Full schedule · Recovery curve | The four-factor LIPS ranking and its component profile, the downloadable schedule, and how far a crew programme gets under each queue ordering |
| **Loss Dynamic** | Rate vs volume · Loss composition | The two measures rank plants oppositely, quantified per plant; the physical/commercial split and whether leakage tracks bursts, age and pressure |
| **Burst Risk** | Risk ranking · Performance · Model evidence · Validation · Register · Limitations | Next-month burst prediction, ROC and PR curves, confusion matrix, calibration, grouped permutation importance, and what the model relies on |
| **Plant Profile** | Summary · What the model sees · History | Per-plant drill-down: 36-month history, peer comparison, burst record, and the model's view of that plant |

The sidebar opens expanded on first load. Collapsing it leaves a **»** control
pinned top-left to bring it back — Streamlit keeps that control inside the page
header, so the header is flattened to zero height rather than hidden outright,
because `display:none` on it would zero the button too and strand the sidebar
closed.

---

## Files

```
app.py                 the dashboard
theme.py               light + dark palettes, chart template, CSS
dataloader.py          ingestion, schema validation, de-duplication
prepare_data.py        cleaning and feature engineering
train_models.py        expected-loss model, anomaly detection, clustering
train_burst_model.py   supervised burst-risk classifier
refresh.py             one-command rebuild: clean -> train -> verify
verify.py              81 independent checks against the raw CSV
test_refresh.py        25 checks that next year's data will load correctly
shoot.py               headless render test — every tab, both colour modes
fitcheck.py            no-scroll acceptance test — all 15 views, both modes
requirements.txt
assets/
  logo.png             brand mark, kept with the code rather than in data/
data/
  raw/                 PUT THE PAIP EXPORT HERE
  year_coverage.csv    months observed per year, annualisation factors
  nrw_plant_month.csv  2,664 tidy plant-month records
  nrw_plant_year.csv   222 plant-year rows with LIPS and its components
  plant_crosswalk.csv  74 plants → district / region / area type
  data_quality.csv     identity check results
  missing_values.csv
  ml_plant.csv         per-plant criticality, residuals, archetypes
  ml_monthly.csv       per-plant-month predictions and anomaly flags
  model_metrics.json   validation table, importances, cluster profiles
  burst_predictions.csv  next-month risk per plant, ranked
  burst_history.csv    per plant-month predicted probability vs outcome
  burst_metrics.json   ROC/PR curves, confusion matrix, calibration
```

---

## Data quality

Every published figure was recomputed from its components. Nothing was silently
corrected.

| Check | Max deviation |
|---|---|
| `billed = domestic + commercial + industrial` | 0 |
| `nrw = production − billed` | 0 |
| `nrw_pct = nrw / production` | 0.05 pp (source rounding) |
| `physical + commercial loss = nrw` | 1 m³ (source rounding) |
| `billed_revenue = billed × tariff` | RM 0.005 |
| `opex = energy + chemical + maintenance` | ~0 |
| Negative NRW records | **0** |
| Months where billed > production | **0** |

Missing values: `pressure_bar` (58 rows, 2.18%) and `raw_turbidity_ntu` (89 rows,
3.34%). Both are sensor-derived, neither is a LIPS input, and both are excluded
pairwise from the correlations that use them. No imputation was performed.

---

## Deviation from the original proposal

The dataset supplied differs materially from the one the proposal described. The
dashboard was extended accordingly, and this README is where that is stated.

| Proposal assumed | Dataset actually contains |
|---|---|
| 888 plant-month records, 2025 only | **2,664 records, 2023–2025** |
| No cost data | Chemical, maintenance, energy, total opex, cost per m³ |
| No tariff → no financial estimates | `Tarif_Purata_RM_m3`, RM 0.93–1.37 |
| Real vs apparent losses inseparable | Explicit physical / commercial split per plant |
| Negative NRW records to be flagged | **None** — the anomaly does not occur |
| Median NRW 48.9–55.6% per month | Median plant rate **37.2%**; system rate 32.1% |

Consequences: financial and multi-year analysis is in scope where the proposal
had ruled it out, and the negative-record limitation does not arise. The
proposal's central argument — that rate-based and volume-based prioritisation
produce materially different intervention orders — is **confirmed, and more
strongly than anticipated**: the two measures are negatively correlated.

---

## Limitations

- **LIPS ranks operational risk, not recoverable water.** Three of its four
  components are rates, counts or proxies rather than quantities — loss density
  is m³ per km, not m³. A plant can rank high on density while holding modest
  absolute volume. `verify.py` prints how much of the estate's NRW the queue
  actually reaches so the trade is visible; read that number before treating
  the ranking as a recovery plan.
- **Repair cost is not in the dataset.** LIPS is a priority score, not a
  benefit–cost ratio. Adding a cost term would mean inventing one.
- **The physical/commercial split is a published apportionment**, used as given
  and shown only as descriptive context. It is not derived from a formal IWA
  water-balance audit, and in this data it is a per-plant constant — which is
  why it is labelled "assumed" wherever it appears on screen.
- **Correlation across plants is not causal.** Plants differ in network length,
  terrain, age and pressure simultaneously.
- **Three years is short.** Trend and year-specific event cannot be fully
  separated; per-plant anomaly detection rests on 36 monthly observations.
- **Financial figures are tariff-based, not full-cost.** Operating cost covers
  energy, chemicals and maintenance only — staff, capital charges and
  depreciation are absent, so operating margins are overstated relative to full
  cost recovery.
- **The model has no ground truth**, so a large residual is a lead, not a
  diagnosis. It is also only as good as the feature set: terrain, soil and
  historic construction quality are not in the data, so the residual bounds
  *where to look* without identifying the cause.
- **Permutation importance is unreliable under collinearity.** Capacity, staff
  count, population and connections all scale with plant size, and the method
  splits credit unpredictably among correlated features. The safe reading is
  that plant size and area type dominate, not that any single column is
  decisive.

### Two known cosmetic limitations

Streamlit's slider track paints its fill with an inline `linear-gradient`
carrying the default red accent. Because the colour is baked into an inline
style, it cannot be overridden from CSS without also destroying the fill
proportion indicator, so the slider track stays red in both modes. Every other
accent (chips, tabs, radio, checkbox) is re-pointed at the palette blue, and no
*data* mark is affected — the chart palettes are fully under the design system's
control.

The brand mark is a raster PNG on a light ground, so it carries a faint halo
against the dark surface. An SVG or a transparent-background export would fix
it; nothing else about dark mode depends on it.

---

## Deploying

The dashboard runs entirely from this folder — `streamlit run app.py` is all it
needs, on any machine with the requirements installed.

If you later want it hosted, [share.streamlit.io](https://share.streamlit.io)
serves it from a repository pointed at `app.py`. Keep the `data/` folder with
it: the host has no build step that would run `refresh.py`, so the app needs the
prepared CSVs present or it will start with nothing to show.

**Before you publish anything, decide what may be public.** `data/raw/` holds
PAIP's source workbook and the prepared CSVs carry plant-level production and
billing figures. In a public repository all of it is downloadable by anyone,
with no authentication, and removing a file in a later commit does not remove it
from git history. If those figures are not meant to be public, the repository
has to be private — or the data has to live somewhere the repository does not.
