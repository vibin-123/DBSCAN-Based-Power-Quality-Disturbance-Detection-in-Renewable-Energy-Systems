# DBSCAN-Based Detection of Power Quality Disturbance Clusters Linked to High Renewable-Energy Injection

An unsupervised machine-learning study on one year of real power-quality
measurements. DBSCAN is the only clustering model. No labels, no train/test
split, no classifier.

## 1. Introduction

Solar and wind plants connect through power-electronic inverters and their
output changes with the weather. Both can affect power quality: voltage level,
voltage fluctuation, harmonic distortion, unbalance and frequency. This project
asks a narrow, testable question on real data: do the power-quality patterns
that DBSCAN finds on its own occur more often when renewable penetration is
high?

## 2. Problem statement

Given 10-minute power-quality features measured at an industrial facility's
grid connection point in 2024, find dense groups of similar operating
conditions and the windows that fall outside them, without any predefined
classes. Then test whether those groups are distributed differently across
low, medium and high renewable penetration.

## 3. Objectives

1. Acquire and verify real datasets that contain both power quality and renewable generation.
2. Engineer electrical features that map to EN 50160 / IEC 61000-4-30 phenomena.
3. Cluster with DBSCAN, with parameters derived from the data.
4. Treat DBSCAN noise as unusual observations to investigate, not as confirmed disturbances.
5. Relate clusters to regional solar and wind penetration and to the site's own PV.
6. Compare a PQ-only experiment with a PQ + renewables experiment.
7. Present everything in an interactive dashboard.

## 4. Datasets

All data are real measurements. Nothing is simulated.

| Dataset | Used for | Resolution | Period | Licence |
|---|---|---|---|---|
| SPARK, `EPI_TotalLoad` meter (KIT) | Power quality at the facility connection point: three-phase voltage, current, frequency, THD, harmonics, P, Q | 5 s | 2024 | CC BY 4.0 |
| SPARK, `EPI_PV` | AC power of the facility's 10.72 kWp rooftop PV | 5 s | 1 Jan - 9 Apr 2024 | CC BY 4.0 |
| SPARK exogenous data (from SMARD) | TransnetBW generation by technology, including solar and onshore wind | 15 min | 2024 | CC BY 4.0 |
| SMARD, Bundesnetzagentur | TransnetBW grid load (penetration denominator) | 15 min | 2024 | CC BY 4.0 |

After preprocessing: **47,083 ten-minute windows** (about 89 % of the year;
December has almost no data). Local PV overlaps 11,110 of them.

The four sources named in the brief (IEEE PES Open Data, UK Power Networks,
Mendeley PQ Disturbances, SEED) were downloaded or queried and rejected: none
has power quality and renewable generation on a shared timeline. Details and
the compatibility argument for SPARK + SMARD are in
[`docs/DATASET_VERIFICATION.md`](docs/DATASET_VERIFICATION.md).

### References

- Sievers, J., Bischof, S., Simon, F., Blank, T. (2026). *SPARK: High-Resolution Energy Data from a Sustainable Industrial Production Area in Karlsruhe.* Karlsruhe Institute of Technology. DOI 10.35097/bjdg3m3rg5jv3skk. Data paper: *Scientific Data* 13, 310, DOI 10.1038/s41597-026-06955-4.
- Bundesnetzagentur, SMARD.de electricity market data, https://www.smard.de (CC BY 4.0).

## 5. System architecture

```
 A  data_acquisition.py   SPARK by HTTP byte range (76 GB tar, ~290 MB used), SMARD API,
        |                 MD5 / CRC / SHA-256 checks  ->  data/raw
 B,C preprocessing.py     duplicates, 5 s grid, physical range checks, units, UTC,
        |                 15-min to 10-min alignment  ->  data/interim + report
 D  feature_engineering.py  10-minute windows, measured and calculated PQ features
        |
 E  renewable_analysis.py   penetration metrics, low / medium / high levels
        |                                  ->  data/processed
 F  dbscan_model.py       scaling, weighting, k-distance knee, two-stage DBSCAN
        |
 G,H experiments.py       Experiment 1 and 2, cluster and noise profiles,
        |                 association tests, sensitivity, stability
    experiment_comparison.py   Exp 1 vs Exp 2, weight sweep
        |
 I  visualization.py      Plotly figures (dashboard) and PNG / HTML exports
 J  evaluation.py, reporting.py   metrics, results report  ->  outputs/
        |
    dashboard/app.py      Streamlit, 13 sections
```

## 6. Methodology

**Windows.** 5 s samples are aggregated to 10-minute windows, the
IEC 61000-4-30 interval. The meter reports about every 6.5 s, so a full window
has about 92 of 120 slots filled; a window is kept with at least 72. Missing
samples are never interpolated.

**Features used for clustering (10).**

| Feature | Kind | Meaning |
|---|---|---|
| `v_dev_pct` | calculated | deviation of mean phase voltage from 230 V |
| `v_std_pct` | calculated | within-window voltage fluctuation |
| `rvc_max_pct` | calculated | largest 5 s voltage step (rapid voltage change) |
| `vuf_pct` | calculated | voltage unbalance factor from line-line magnitudes (IEC formula) |
| `f_dev_abs_hz` | calculated | mean absolute deviation from 50 Hz |
| `thd_u_pct` | measured | voltage THD, three-phase mean |
| `u_h5_pct` | measured | 5th voltage harmonic |
| `thd_i_pct` | measured | current THD |
| `i_rms_a` | measured | RMS phase current |
| `i_n_a` | measured | neutral current |

Timestamps, calendar fields and all renewable variables are kept outside the
Experiment 1 matrix. The full catalogue of 27 features is in
`data/processed/feature_catalogue.csv`.

**Renewable penetration.**

```
VRE penetration (%) = (solar + onshore wind generation) / grid load x 100     (TransnetBW)
```

Grid load is the denominator because TransnetBW is a net importer; share of
generation is reported as a sensitivity. Levels are terciles of the observed
values: low <= 6.9 %, medium <= 21.5 %, high above. Local PV is expressed as
% of its 10.72 kWp rating: "none" below 1 %, then terciles of generating
windows (4.5 % and 14.6 %).

## 7. DBSCAN

DBSCAN groups points that have at least `MinPts` neighbours within distance
`eps` and labels the rest as noise. It needs no cluster count, finds
arbitrarily shaped clusters and has an explicit noise class, which suits
measurements that are mostly normal with rare, irregular departures.

- **Scaling:** RobustScaler (median and IQR), so unusual windows do not set the scale.
- **MinPts:** 2 x number of dimensions.
- **eps:** knee of the sorted k-distance curve.
- **Two stages.** At these settings the data form one dominant cluster plus
  about 2 % noise. One global `eps` cannot resolve structure among sparse
  points, so DBSCAN is run again on the stage-1 noise only, with
  `MinPts` = dimensions + 1 and that subset's own knee `eps`. Stage-2 clusters
  are groups of mutually similar unusual windows; what remains is isolated noise.
- **Evaluation:** silhouette and Davies-Bouldin on clustered points only,
  noise percentage, parameter grids for both stages, three distance metrics,
  two scalers, and ARI stability over 20 subsamples.
- K-Means and Ward clustering are run once as a reference (silhouette about
  0.2 for K-Means at every k): the data have no convex partition, which
  supports a density-based reading.

## 8. Experimental design

| | Experiment 1 | Experiment 2 |
|---|---|---|
| Inputs | 10 PQ features | 10 PQ features + VRE penetration, regional solar MW, regional wind MW |
| Weighting | equal | renewable block = 20 % of squared distance (swept 10-50 %) |
| Role of renewables | compared after clustering | part of the distance |
| Evidence value | clusters cannot be shaped by renewables | exploratory; partly circular |

Association is tested by shuffling whole-day renewable profiles between days
of the same type (weekday or weekend), 500 times. This keeps the daily rhythm
shared by solar output, factory load and power quality, so a small p-value
means more than "both follow the clock".

## 9. Results

Full tables: [`outputs/reports/results_report.md`](outputs/reports/results_report.md).
Academic write-up: [`outputs/reports/academic_report.md`](outputs/reports/academic_report.md).

**Power quality was compliant all year.** Largest 10-minute values: voltage
deviation +4.7 % (limit 10 %), voltage THD 5.0 % (limit 8 %), unbalance 0.48 %
(limit 2 %). No 5 s sample was below 0.90 pu or above 1.10 pu. The clusters
below are therefore unusual operating patterns, not standard violations.

**Experiment 1 (PQ only).**

| Label | Windows | Share | What distinguishes it (median vs dominant cluster) |
|---|---|---|---|
| C0 dominant | 46,125 | 97.97 % | normal operation |
| U1 | 815 | 1.73 % | voltage fluctuation 3x higher (0.21 % vs 0.07 %), voltage steps 0.38 % vs 0.14 %, higher load; 255 days |
| U2 | 128 | 0.27 % | voltage steps of 1.25 % and fluctuation 0.47 %; weekdays only, 71 days, high load |
| Isolated noise | 15 | 0.03 % | single windows unlike anything else |

- Stage 1: MinPts 20, eps 1.515. Stage 2: MinPts 11, eps 3.205, silhouette 0.59, Davies-Bouldin 0.56.
- Stability: mean ARI 0.97 over 20 subsamples; three clusters every time.

**Unusual windows are about twice as frequent at high regional renewable penetration.**

| Regional level | Unusual windows | After standardising for hour and weekend |
|---|---|---|
| Low | 1.36 % | 2.36 % |
| Medium | 1.94 % | 1.87 % |
| High | 2.80 % | 3.29 % |

- Permutation p = 0.002; effect size is small (Cramér's V = 0.04).
- The contrast holds in all nine parameter settings tested (high-to-low ratio 1.8 to 3.4).
- It comes from solar (p = 0.002), not wind (p = 0.27).
- U2 is the most concentrated: 84 of its 128 windows fall in the high level.
- For the site's own PV the unusual rate rises from 1.1 % (not generating) to
  2.6 % (high output), but among generating windows the trend is not
  significant (p = 0.25; 3,197 windows over 14 winter weeks).

**Experiment 2 (PQ + renewables).** One unusual cluster (832 windows) and 9
isolated windows; the U1/U2 split disappears. Agreement with Experiment 1:
ARI 0.80, 725 unusual windows in common. Unusual rate by level: 0.81 %, 1.45 %,
3.10 %. This steeper gradient is partly built in, because renewable values are
now part of the distance.

**What this does and does not show.** The association is statistical. High
solar output coincides with weekday working hours and high facility load, and
the standardisation removes only part of that. A 10.72 kWp array and a
regional mix cannot be shown to cause voltage steps at this meter from these
data.

## 10. Installation

Python 3.11 or newer.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

On Windows (PowerShell) activate the environment with `.venv\Scripts\Activate.ps1`
instead of `source .venv/bin/activate`.

## 11. Execution

```bash
python run_pipeline.py                  # download (if needed), preprocess, cluster, report
```

```bash
python run_pipeline.py --skip-download  # reuse data/raw
```

```bash
python run_pipeline.py --quick          # 99 permutations, 5 stability rounds
```

```bash
python -m pytest tests -q               # 54 tests
```

The full run takes about 70 minutes on 4 cores, almost all of it in the
sensitivity grids and stability rounds. The download needs access to
`radar.kit.edu` and `www.smard.de`.

Individual stages: `python -m src.data_acquisition`, `src.preprocessing`,
`src.feature_engineering`, `src.renewable_analysis`, `src.experiments`,
`src.experiment_comparison`, `src.visualization`, `src.reporting`.

## 12. Dashboard

```bash
streamlit run dashboard/app.py
```

Run it from the project root, so that Streamlit picks up the theme in
`.streamlit/config.toml`. The dashboard follows the system light / dark setting.

Thirteen sections, grouped in the sidebar:

- **Overview**
- **Data:** dataset, power quality, renewable generation, renewable penetration
- **Experiments:** Experiment 1, Experiment 2 (each with tabs for clusters,
  renewable levels and robustness)
- **Analysis:** cluster map, unusual windows, experiment comparison, correlation
- **Tools:** live parameter explorer, downloads (file preview, everything as a zip)

Each section has its own address, for example
`http://localhost:8501/?page=experiment-1`, so a reload or a shared link opens
the same section. The dashboard reads the result files again whenever the
pipeline has rewritten them; no restart is needed.

## 13. Project structure

```
project/
  data/raw/           SPARK subset, SMARD, rejected reference sources, MANIFEST.csv
  data/interim/       cleaned 5 s and 15 min tables (parquet)
  data/processed/     10-minute features, analysis dataset, feature catalogue
  src/                config, remote_tar, data_acquisition, preprocessing,
                      feature_engineering, renewable_analysis, dbscan_model,
                      evaluation, experiments, experiment_comparison,
                      visualization, reporting
  dashboard/app.py
  .streamlit/config.toml   dashboard theme (light and dark)
  notebooks/exploratory_analysis.ipynb
  outputs/plots/      PNG and interactive HTML figures
  outputs/reports/    preprocessing, results and academic reports
  outputs/results/    every result table (CSV), labels (parquet), summary.json
  docs/               dataset verification record, decision log
  tests/              54 pytest tests
  run_pipeline.py, requirements.txt
```

## 14. Limitations

- One site, one year, with December almost absent.
- 5 s RMS values cannot show sub-cycle dips, swells or transients.
- Local PV covers 14 winter and early-spring weeks; there is no local wind.
- Regional penetration is the same for every site in the control area.
- Whether the PV feeds in behind the facility meter is not documented.
- The share of windows called unusual depends on `eps` (0.7 % to 6.5 % across
  the nine settings of the robustness check); it is a density threshold, not a natural boundary. The U1/U2
  split appears in five of nine stage-2 settings.
- Load is a confounder that the hour and weekend standardisation only partly removes.

## 15. Future scope

- Multi-year analysis (SPARK has 2018-2024 for the load meter).
- Waveform-level data to detect true dips, swells and transients.
- A site with larger PV relative to load and with documented topology.
- OPTICS or HDBSCAN as variable-density extensions.
- Load-matched comparisons (same load band, different solar output).
- Tap-changer or switching logs to explain the U2 voltage steps.
