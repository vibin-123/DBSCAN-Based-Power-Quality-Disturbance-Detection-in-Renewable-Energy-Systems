# Technical decision log

| # | Decision | Reason |
|---|---|---|
| 1 | Use SPARK `EPI_TotalLoad` as the power-quality source and 2024 as the study year | Only real open dataset found with PQ channels and PV generation on one site and clock; PV exists only in 2024 |
| 2 | Fetch SPARK by HTTP byte range instead of downloading 76 GB | The archive is an uncompressed tar of zips; 26 files (about 290 MB) are needed. `src/remote_tar.py` indexes the tar and reads zip members remotely |
| 3 | Add SMARD grid load | The SPARK exogenous file has generation but no load; load is the denominator for penetration |
| 4 | 10-minute windows | IEC 61000-4-30 / EN 50160 aggregation interval; 47,083 windows is tractable for DBSCAN |
| 5 | Window accepted at >= 60 % filled 5 s slots | The meter reports every ~6.5 s, so 76.7 % is the structural maximum; 60 % is about 78 % of what the meter can deliver |
| 6 | No interpolation of missing samples | Interpolated values would create artificial dense points |
| 7 | Ten PQ clustering features (voltage deviation, fluctuation, rapid change, unbalance, frequency deviation, THD-U, 5th harmonic, THD-I, RMS current, neutral current) | Each maps to an EN 50160 / IEEE 519 phenomenon; redundant phase-wise copies were averaged |
| 8 | RobustScaler (median / IQR) | Unusual windows must not set the scale used to find them; StandardScaler is run as a comparison |
| 9 | Cluster in the full scaled space; PCA only for plots | Six components are needed for 90 % of the variance, so PCA would discard information without simplifying |
| 10 | MinPts = 2 x dimensions; epsilon = knee of the sorted k-distance curve | Standard heuristics (Sander et al. 1998; Ester et al. 1996); nothing is tuned towards a cluster count |
| 11 | Two-stage DBSCAN | Stage 1 returned one dominant cluster plus noise. A single epsilon cannot resolve structure among sparse points, so DBSCAN is re-run on the stage-1 noise with its own knee epsilon and MinPts = dimensions + 1 |
| 12 | Report "unusual windows", not "disturbances" | Every window is inside EN 50160 limits; DBSCAN flags rarity, not non-compliance |
| 13 | Primary penetration metric = (solar + wind) / grid load | TransnetBW is a net importer, so a generation denominator overstates penetration when local generation is low. Share of generation is reported as a sensitivity |
| 14 | Low / medium / high = terciles of the observed penetration | Data-driven, equal group sizes; cut-offs 6.9 % and 21.5 % |
| 15 | Local PV levels: "none" below 1 % of rating, then terciles of generating windows | Night must not be mixed with low daytime output |
| 16 | Whole-day permutation test, within weekday / weekend | 10-minute windows are autocorrelated and both PQ and solar follow the clock; ordinary chi-square p-values are far too small |
| 17 | Experiment 2 renewable block weight = 20 % of squared distance | Three renewable features against ten PQ features; weight swept from 10 % to 50 % |
| 18 | Local PV not used as an Experiment 2 input | It exists for 24 % of the windows; used for post-clustering comparison instead |
| 19 | K-Means and Ward clustering run only as a reference | Justifies the algorithm choice; neither has a noise concept |
