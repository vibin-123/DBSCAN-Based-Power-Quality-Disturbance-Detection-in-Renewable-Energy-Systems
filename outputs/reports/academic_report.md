# DBSCAN-Based Detection of Power Quality Disturbance Clusters Linked to High Renewable-Energy Injection

**Academic project report**

## Abstract

This project applies DBSCAN, an unsupervised density-based clustering
algorithm, to one year (2024) of real power-quality measurements from the grid
connection point of an industrial facility in Karlsruhe, Germany (SPARK
dataset, 5-second resolution). Ten electrical features were computed on
47,083 ten-minute windows. With parameters taken from the k-distance curve,
DBSCAN placed 97.97 % of the windows in one dense cluster. A second DBSCAN
pass on the remaining 958 windows found two clusters of unusual voltage
behaviour (815 and 128 windows) and 15 isolated windows. All 10-minute values
were inside EN 50160 limits, so these are unusual operating patterns, not
violations. Unusual windows occurred in 1.36 %, 1.94 % and 2.80 % of the
windows at low, medium and high regional solar-plus-wind penetration. The
association is statistically significant under a whole-day permutation test
(p = 0.002) but small (Cramér's V = 0.04); it is carried by solar, not wind,
and it holds across all nine parameter settings tested. Adding renewable
variables to the clustering input (Experiment 2) merged the two unusual
clusters into one and reproduced 725 of the unusual windows. The findings are
associations; load is a confounder that could not be fully removed.

## 1. Introduction

Inverter-connected solar and wind generation changes how distribution networks
behave. Reverse power flow raises voltage, fast changes in irradiance cause
voltage fluctuation, inverters inject harmonic currents, and single-phase
units add unbalance. Network operators monitor these effects with power-quality
indices defined in EN 50160 and measured according to IEC 61000-4-30.

Most machine-learning work on power quality is supervised: synthetic waveforms
with known labels are classified. Field data rarely have labels, and the
patterns of interest are not known in advance. This project therefore uses an
unsupervised method and asks whether the patterns it finds are related to
renewable injection.

### 1.1 Problem statement

Discover groups of similar power-quality conditions, and the observations that
belong to no group, in real measurements without predefined classes. Then
determine whether these groups are distributed differently across levels of
renewable penetration.

### 1.2 Objectives

1. Identify and verify real datasets with both power-quality and renewable-generation measurements.
2. Derive features with clear electrical meaning.
3. Apply DBSCAN with data-driven parameters and evaluate it without labels.
4. Analyse clusters and noise against renewable penetration.
5. Compare clustering with and without renewable variables as inputs.
6. Deliver a reproducible pipeline, tests and an interactive dashboard.

## 2. Background and related work

**Power-quality standards.** EN 50160 sets limits for supply voltage at the
point of connection: voltage within +/-10 % of nominal, voltage THD up to 8 %,
unbalance up to 2 %, frequency within +/-1 % (0.5 Hz), each assessed on
10-minute means. IEC 61000-4-30 defines the measurement and aggregation
methods. IEEE 519 gives harmonic limits and IEEE 1159 classifies disturbance
types.

**Renewables and power quality.** The effects of distributed generation on
voltage level, flicker, harmonics and unbalance are described by Bollen and
Hassan (2011). Field studies report site-specific outcomes that depend on
network strength and on the size of generation relative to load.

**Density-based clustering.** DBSCAN (Ester et al., 1996) defines clusters as
connected regions where every core point has at least MinPts neighbours within
distance eps; points in no such region are noise. Sander et al. (1998)
recommend MinPts = 2 x dimensions and the k-distance plot for eps. Schubert et
al. (2017) confirm these heuristics. DBSCAN's known limitation is a single
global density; OPTICS and HDBSCAN (Campello et al., 2013) address it.

**Internal validation.** The silhouette coefficient (Rousseeuw, 1987) and the
Davies-Bouldin index (1979) reward compact, well-separated, convex clusters.
They ignore noise and are undefined for a single cluster, so they must be read
together with the noise share. The adjusted Rand index (Hubert and Arabie,
1985) measures agreement between two partitions and is used here for stability.

## 3. Data

### 3.1 Dataset selection

Six public sources were inspected by downloading from the original hosts
(`docs/DATASET_VERIFICATION.md`). The Mendeley and SEED datasets are synthetic
waveforms without timestamps. The UK Power Networks dataset contains weekly
harmonic percentiles without generation. The IEEE PES entry is a catalogue of
unrelated sets. Only SPARK contains power-quality channels and PV generation
from one site on one clock.

### 3.2 Data used

| Data | Source | Resolution | Coverage in 2024 |
|---|---|---|---|
| 26 PQ channels, facility connection meter `EPI_TotalLoad` | SPARK (KIT), CC BY 4.0 | 5 s | 68.3 % of 5 s slots |
| PV AC power, 10.72 kWp rooftop array | SPARK | 5 s | 1 Jan to 9 Apr |
| Generation by technology, TransnetBW | SPARK exogenous file (SMARD) | 15 min | complete |
| Grid load, TransnetBW | SMARD, CC BY 4.0 | 15 min | complete |

The SPARK archive is a single 76.2 GB tar file. The 26 required files (about
290 MB) were read by HTTP byte range; integrity was checked against the
archive's MD5 manifest and zip CRC values.

### 3.3 Compatibility of the combined sources

Both sources use UTC; the local PV series correlates best with regional solar
generation at zero lag (r = 0.76), which confirms it. The regional solar series
downloaded from SMARD is identical to the copy inside SPARK. Karlsruhe lies in
the TransnetBW control area. The regional series describe the surrounding
grid, not the facility.

## 4. Methodology

### 4.1 Preprocessing

Duplicate timestamps are removed (none were found), samples are placed on a
regular 5 s grid, values outside physical limits are set to missing (none were
found in this cleaned release), and power is converted to kW and kvar. Missing
samples are not interpolated. The meter reports about every 6.5 s, which leaves
every fourth 5 s slot empty; a 10-minute window is accepted with at least 60 %
of its slots filled in every channel. This yields 47,083 windows. Each window
takes the regional values of the 15-minute interval that contains its start.

### 4.2 Features

Measured: mean RMS voltage, frequency, voltage THD, 3rd, 5th and 7th voltage
harmonics, current THD, RMS and neutral current, active and reactive power,
power factor.

Calculated: voltage deviation from 230 V; within-window voltage standard
deviation; largest 5 s voltage step; voltage unbalance factor from the three
line-to-line magnitudes,

    beta = (U12^4 + U23^4 + U31^4) / (U12^2 + U23^2 + U31^2)^2
    VUF  = sqrt((1 - sqrt(3 - 6 beta)) / (1 + sqrt(3 - 6 beta))) x 100 %;

absolute frequency deviation; counts of 5 s samples below 0.90 pu or above
1.10 pu; a severity index equal to the largest fraction of an EN 50160 limit
in use; one-hour rolling means and standard deviations; and the duration of
runs of consecutive unusual windows.

Ten features enter the clustering: `v_dev_pct`, `v_std_pct`, `rvc_max_pct`,
`vuf_pct`, `f_dev_abs_hz`, `thd_u_pct`, `u_h5_pct`, `thd_i_pct`, `i_rms_a`,
`i_n_a`.

### 4.3 Renewable penetration

    VRE penetration (%) = (P_solar + P_wind,onshore) / P_grid load x 100

computed for the TransnetBW area. Load is used as the denominator because the
area is a net importer; with a generation denominator the ratio rises whenever
conventional local generation is low. The share of generation is reported as
a sensitivity. Observed penetration ranges from 0 % to 143 % (median 12.9 %).
Levels are terciles: low up to 6.9 %, medium up to 21.5 %, high above. Local PV
output is expressed in percent of the 10.72 kWp rating, with "none" below 1 %
and terciles of the generating windows at 4.5 % and 14.6 %.

### 4.4 DBSCAN procedure

1. Scale each feature with the median and interquartile range.
2. Stage 1: MinPts = 2d; eps = knee of the sorted distance to the MinPts-th neighbour.
3. Stage 2: on the stage-1 noise only, MinPts = d + 1 and the knee eps of that subset.
4. Labels: 0 for the dominant cluster, U1, U2, ... for stage-2 clusters, -1 for isolated noise.

The second stage was added after stage 1 returned a single cluster. One global
eps cannot separate structure among sparse points; repeating DBSCAN at the
density of those points keeps the method purely DBSCAN while addressing its
single-density limitation. The number of clusters is never specified.

### 4.5 Evaluation

- Silhouette and Davies-Bouldin on clustered points only, reported with the noise share.
- Parameter grids: MinPts in {d, 2d, 4d} by eps in {0.6 to 1.5} x knee for stage 1; three by three for stage 2.
- Euclidean, Manhattan and Chebyshev distance; robust and standard scaling.
- Stability: 20 random 80 % subsamples, ARI against the full-data labels.
- K-Means and Ward clustering as a reference only.

### 4.6 Association analysis

Chi-square statistics on 10-minute windows overstate significance because
neighbouring windows are correlated and because solar output, factory load and
power quality all follow the time of day. Significance is therefore assessed
by permutation: whole-day renewable profiles are exchanged between days of the
same type (weekday or weekend) 500 times, and the observed bias-corrected
Cramér's V is compared with the permuted values. Rates are also standardised
directly over hour-of-day by weekend strata.

## 5. Results

### 5.1 Power quality at the site

| Indicator (10-minute) | Median | Maximum | EN 50160 limit |
|---|---|---|---|
| Voltage deviation | +2.76 % | +4.68 % | +/-10 % |
| Voltage THD | 3.39 % | 5.03 % | 8 % |
| Voltage unbalance | 0.25 % | 0.48 % | 2 % |
| Largest 5 s frequency deviation | - | 0.19 Hz | 0.5 Hz |

No 5 s sample was below 0.90 pu or above 1.10 pu. The site was compliant
throughout.

### 5.2 Experiment 1: power quality only

Stage 1 (MinPts 20, eps 1.515): one cluster, 2.03 % noise. Stage 2 (MinPts 11,
eps 3.205): two clusters, silhouette 0.59, Davies-Bouldin 0.56.

| Label | Windows | Days | Voltage fluctuation | Largest voltage step | Current | Weekend share |
|---|---|---|---|---|---|---|
| C0 dominant | 46,125 | 330 | 0.07 % | 0.14 % | 21 A | 29 % |
| U1 | 815 | 255 | 0.21 % | 0.38 % | 31 A | 15 % |
| U2 | 128 | 71 | 0.47 % | 1.25 % | 43 A | 0 % |
| Isolated | 15 | 15 | 0.19 % | 0.95 % | 41 A | 0 % |

Values are cluster medians. THD, unbalance and frequency deviation are nearly
identical across labels; what separates the unusual windows is voltage
dynamics. U2 is characterised by voltage steps of about 1.25 % on working
days. Steps of this size are consistent with transformer tap changes or the
switching of large loads; the data cannot distinguish the two.

Unusual windows form 865 episodes with a median length of 10 minutes and a
maximum of 60 minutes.

### 5.3 Robustness

- Subsample stability: mean ARI 0.966 (minimum 0.954); three clusters in all 20 rounds.
- Distance metric: ARI against Euclidean 0.89 (Manhattan) and 0.87 (Chebyshev).
- Scaler: StandardScaler gives 2.17 % unusual and merges U1 and U2; ARI 0.87.
- eps: the noise share is sensitive to eps (0.9 % at 1.25 x knee, 5.4 % at 0.8 x knee). The unusual set is a density threshold, not a sharply separated group.
- Stage 2: two or more clusters in five of nine settings; one cluster at larger eps.
- K-Means reaches a silhouette of about 0.2 for k = 2 to 6, so the data have no convex partition.

### 5.4 Clusters and renewable penetration

| Regional level | Windows | Unusual, raw | Unusual, standardised | U1 share | U2 share |
|---|---|---|---|---|---|
| Low (<= 6.9 %) | 15,695 | 1.36 % | 2.36 % | 1.21 % | 0.14 % |
| Medium | 15,693 | 1.94 % | 1.87 % | 1.77 % | 0.14 % |
| High (> 21.5 %) | 15,694 | 2.80 % | 3.29 % | 2.21 % | 0.54 % |

| Renewable variable | Cramér's V (unusual vs level) | Permuted 95th percentile | p |
|---|---|---|---|
| Solar + wind, share of load | 0.042 | 0.034 | 0.002 |
| Solar + wind, share of generation | 0.046 | 0.034 | 0.002 |
| Solar, share of load | 0.047 | 0.044 | 0.002 |
| Wind, share of load | 0.016 | 0.023 | 0.27 |
| Local PV output, generating windows | 0.025 | 0.045 | 0.25 |

The unusual rate at high penetration is about twice the rate at low
penetration. After standardising for hour and weekend the ratio falls to 1.4
and the trend is no longer monotonic, which shows that much of the raw contrast
is shared daily rhythm. The permutation test still rejects the hypothesis that
the rhythm explains all of it. The contrast is present in all nine stage-1
parameter settings, with high-to-low ratios from 1.8 to 3.4.

Wind penetration shows no association. For the facility's own PV the unusual
rate rises from 1.14 % (not generating) to 1.41 %, 1.97 % and 2.63 % at low,
medium and high output, but among generating windows this is not significant.
The sample is 3,197 windows from 14 winter and early-spring weeks.

By month, the unusual rate is highest from April to September (2.3 % to 2.9 %,
except August at 1.8 %), the months with the highest mean penetration (24 % to
29 %).

Continuous indicators: at high penetration the median 5th voltage harmonic is
1.50 % against 1.24 % at low, and the median voltage deviation is 2.29 %
against 3.05 %. After removing the hour-by-weekend mean, the correlation of
the 5th harmonic with solar penetration falls from 0.49 to 0.15 and the
correlation of voltage deviation falls from -0.45 to -0.02. Most of the raw
correlation is therefore time-of-day structure.

### 5.5 Experiment 2: power quality plus renewables

Stage 1 (MinPts 26, eps 1.885): one cluster, 1.79 % noise. Stage 2 (MinPts 14,
eps 3.943): one cluster of 832 windows and 9 isolated windows. Stage-2
silhouette is undefined with a single cluster. Stability: mean ARI 0.961.

Unusual rate by regional level: 0.81 %, 1.45 %, 3.10 % (standardised 1.92 %,
1.36 %, 3.36 %). Cramér's V for the unusual flag is 0.073 (p = 0.002).

### 5.6 Comparison of the experiments

| | Experiment 1 | Experiment 2 |
|---|---|---|
| Dimensions | 10 | 13 |
| Clusters | 3 | 2 |
| Unusual windows | 2.03 % | 1.79 % |
| Isolated noise | 0.03 % | 0.02 % |
| Silhouette, all clustered points | 0.53 | 0.59 |
| Davies-Bouldin | 1.07 | 1.21 |
| Stability (mean ARI) | 0.966 | 0.961 |

ARI between the two labelings is 0.80; 725 windows are unusual in both, 233
only in Experiment 1 and 116 only in Experiment 2. Raising the renewable weight
from 10 % to 50 % lowers the agreement steadily (ARI 0.85 to 0.72) without
creating new clusters.

Renewable-aware clustering did not reveal additional patterns. It removed the
U1/U2 distinction and moved the unusual set towards windows with extreme
renewable values. Its stronger gradient across levels is partly circular,
since the level variable is part of the distance. Experiment 1 is the
appropriate basis for conclusions; Experiment 2 shows that they do not depend
on excluding the renewable variables.

## 6. Discussion

**What DBSCAN contributed.** The algorithm separated a continuous mass of
normal operation from sparse departures without being told how many groups to
find, and its noise class identified the windows worth inspecting. The
second-stage clusters have a consistent physical reading: elevated voltage
fluctuation (U1) and discrete voltage steps (U2).

**Association, not causation.** High solar output coincides with weekday
working hours and with high facility load, and U2 occurs only on weekdays at
high load. Standardising for hour and weekend reduces the contrast
substantially. Load switching inside the facility could produce the same
voltage steps as network regulation responding to solar infeed, and the data
cannot separate them. The 10.72 kWp array is small, and the regional
penetration is identical for every site in the control area.

**Meaning of "disturbance".** No EN 50160 limit was approached. The patterns
are departures from the site's own normal behaviour. With 5 s RMS values,
short dips, swells and transients are invisible.

**Metric limitations.** Silhouette and Davies-Bouldin are computed on a
labelling where one cluster holds 98 % of the points, so they mostly describe
how far the small clusters are from the bulk. They cannot confirm that the
clusters are meaningful; the stability results and the physical profiles carry
more weight.

## 7. Limitations

- One facility and one year; December is almost missing.
- Local PV limited to 14 low-irradiance weeks; no local wind.
- Metering topology of the PV relative to the facility meter is undocumented.
- The published dataset was already cleaned of implausible samples by its authors.
- The unusual share depends on eps; the U1/U2 split is not present at every setting.
- Confounding by load is reduced, not eliminated.

## 8. Conclusion

On real 2024 measurements from a compliant industrial connection point, DBSCAN
found one dominant operating cluster and two small clusters of unusual voltage
dynamics. Unusual windows were about twice as frequent at high regional solar
and wind penetration as at low penetration. The association is statistically
robust, small in size, driven by solar rather than wind, and partly explained
by the shared daily rhythm of solar output and facility load. Including
renewable variables in the clustering did not uncover further structure. The
project delivers a reproducible, tested pipeline and dashboard that can be
applied to longer periods and other sites.

## 9. Future work

Multi-year analysis with the 2018-2024 SPARK load-meter data; waveform-level
measurements; sites with higher local PV share and documented topology;
load-matched comparisons; HDBSCAN or OPTICS; tap-changer and switching logs to
explain voltage steps.

## References

1. Ester, M., Kriegel, H.-P., Sander, J., Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. *Proc. KDD-96*, 226-231.
2. Sander, J., Ester, M., Kriegel, H.-P., Xu, X. (1998). Density-based clustering in spatial databases: the algorithm GDBSCAN and its applications. *Data Mining and Knowledge Discovery*, 2(2), 169-194.
3. Schubert, E., Sander, J., Ester, M., Kriegel, H.-P., Xu, X. (2017). DBSCAN revisited, revisited: why and how you should (still) use DBSCAN. *ACM Transactions on Database Systems*, 42(3), 19.
4. Campello, R. J. G. B., Moulavi, D., Sander, J. (2013). Density-based clustering based on hierarchical density estimates. *PAKDD 2013*, LNCS 7819, 160-172.
5. Satopää, V., Albrecht, J., Irwin, D., Raghavan, B. (2011). Finding a "kneedle" in a haystack: detecting knee points in system behavior. *ICDCS Workshops*, 166-171.
6. Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, 20, 53-65.
7. Davies, D. L., Bouldin, D. W. (1979). A cluster separation measure. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 1(2), 224-227.
8. Hubert, L., Arabie, P. (1985). Comparing partitions. *Journal of Classification*, 2, 193-218.
9. Bergsma, W. (2013). A bias-correction for Cramér's V and Tschuprow's T. *Journal of the Korean Statistical Society*, 42(3), 323-328.
10. Bollen, M. H. J., Hassan, F. (2011). *Integration of Distributed Generation in the Power System*. Wiley-IEEE Press.
11. EN 50160. Voltage characteristics of electricity supplied by public electricity networks. CENELEC.
12. IEC 61000-4-30. Electromagnetic compatibility: testing and measurement techniques - power quality measurement methods.
13. IEEE Std 519. IEEE standard for harmonic control in electric power systems.
14. IEEE Std 1159. IEEE recommended practice for monitoring electric power quality.
15. Pedregosa, F. et al. (2011). Scikit-learn: machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
16. Sievers, J., Bischof, S., Simon, F., Blank, T. (2026). SPARK: High-Resolution Energy Data from a Sustainable Industrial Production Area in Karlsruhe. Karlsruhe Institute of Technology. DOI 10.35097/bjdg3m3rg5jv3skk.
17. Sievers, J., Bischof, S., Blank, T., Simon, F. (2026). High-resolution energy data from a sustainable industrial production area in Karlsruhe. *Scientific Data*, 13, 310. DOI 10.1038/s41597-026-06955-4.
18. Bundesnetzagentur. SMARD electricity market data. https://www.smard.de (CC BY 4.0).
