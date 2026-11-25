"""Stages E + H - renewable penetration metrics and association analysis.

Two geographic levels are kept strictly apart:

* **Regional** (TransnetBW control area, 15 min, whole year): solar PV and
  onshore wind generation, total generation and grid load.
* **Local** (the facility's own 10.72 kWp rooftop PV, 5 s, 1 Jan - 9 Apr 2024).

Association tests are permutation-based on whole days, because consecutive
10-minute windows are strongly autocorrelated and both power quality and
solar output follow the time of day.

Run:  python -m src.renewable_analysis
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config as C
from .preprocessing import align_to_windows

RENEWABLE_COLS = ["solar_pv_mw", "wind_onshore_mw"]
ALL_RENEWABLE_COLS = RENEWABLE_COLS + ["hydro_mw", "biomass_mw", "other_renewables_mw"]
LEVELS = ["low", "medium", "high"]


def penetration_pct(renewable, denominator):
    """Renewable share in percent; undefined (NaN) where denominator <= 0."""
    renewable = np.asarray(renewable, dtype="float64")
    denominator = np.asarray(denominator, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = renewable / denominator * 100.0
    out[~(denominator > 0)] = np.nan
    return out


def regional_metrics(regional: pd.DataFrame) -> pd.DataFrame:
    """Add generation totals and penetration metrics to the 15-min table."""
    r = regional.copy()
    gen_cols = [c for c in r.columns if c.endswith("_mw") and c != "grid_load_mw"]
    r["total_generation_mw"] = r[gen_cols].sum(axis=1, min_count=len(gen_cols))
    r["vre_generation_mw"] = r[RENEWABLE_COLS].sum(axis=1, min_count=2)
    r["renewable_generation_mw"] = r[ALL_RENEWABLE_COLS].sum(
        axis=1, min_count=len(ALL_RENEWABLE_COLS))
    # Primary metric: variable renewables (solar + wind) relative to grid load.
    r["vre_penetration_pct"] = penetration_pct(r["vre_generation_mw"], r["grid_load_mw"])
    # Alternative denominators, reported for sensitivity.
    r["vre_share_of_generation_pct"] = penetration_pct(
        r["vre_generation_mw"], r["total_generation_mw"])
    r["renewable_share_of_generation_pct"] = penetration_pct(
        r["renewable_generation_mw"], r["total_generation_mw"])
    r["solar_penetration_pct"] = penetration_pct(r["solar_pv_mw"], r["grid_load_mw"])
    r["wind_penetration_pct"] = penetration_pct(r["wind_onshore_mw"], r["grid_load_mw"])
    return r


def local_pv_windows(pv_5s: pd.Series, window: str = C.WINDOW,
                     min_coverage: float = C.MIN_COVERAGE) -> pd.DataFrame:
    """10-minute mean PV power (kW) and output as % of the 10.72 kWp rating."""
    g = pv_5s.resample(window)
    expected = pd.Timedelta(window) / pd.Timedelta(seconds=C.SAMPLE_SECONDS)
    out = pd.DataFrame({"pv_power_kw": g.mean(), "pv_coverage": g.count() / expected})
    out = out[out["pv_coverage"] >= min_coverage]
    out["pv_output_pct_of_rated"] = out["pv_power_kw"] / (C.PV_RATED_W / 1000.0) * 100.0
    return out


def tercile_thresholds(x: pd.Series) -> tuple[float, float]:
    q = x.dropna().quantile([1 / 3, 2 / 3])
    return float(q.iloc[0]), float(q.iloc[1])


def assign_levels(x: pd.Series, thresholds: tuple[float, float]) -> pd.Series:
    lo, hi = thresholds
    lev = pd.Series(pd.NA, index=x.index, dtype="object")
    lev[x <= lo] = "low"
    lev[(x > lo) & (x <= hi)] = "medium"
    lev[x > hi] = "high"
    return pd.Categorical(lev, categories=LEVELS, ordered=True)


def assign_local_pv_levels(pct_of_rated: pd.Series, zero_below: float = 1.0
                           ) -> tuple[pd.Series, tuple[float, float]]:
    """'none' when the array is not generating; terciles of generating windows."""
    gen = pct_of_rated[pct_of_rated >= zero_below]
    thr = tercile_thresholds(gen)
    lev = pd.Series(pd.NA, index=pct_of_rated.index, dtype="object")
    lev[pct_of_rated < zero_below] = "none"
    lev[(pct_of_rated >= zero_below) & (pct_of_rated <= thr[0])] = "low"
    lev[(pct_of_rated > thr[0]) & (pct_of_rated <= thr[1])] = "medium"
    lev[pct_of_rated > thr[1]] = "high"
    cat = pd.Categorical(lev, categories=["none"] + LEVELS, ordered=True)
    return pd.Series(cat, index=pct_of_rated.index), thr


def build_analysis_dataset(features: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Join PQ feature windows with regional and local renewable metrics."""
    regional = regional_metrics(pd.read_parquet(C.INTERIM / "regional_15min.parquet"))
    reg = align_to_windows(regional, features.index)
    df = features.join(reg)
    pv = pd.read_parquet(C.INTERIM / "pv_5s.parquet")["pv_power_kw"]
    df = df.join(local_pv_windows(pv))
    # ratio of two measured powers; the metering topology (PV in front of or
    # behind the facility meter) is not documented, so this is not called
    # "self-supply share".
    df["pv_to_facility_load_pct"] = penetration_pct(df["pv_power_kw"], df["p_kw"])
    thr = tercile_thresholds(df["vre_penetration_pct"])
    df["re_level"] = assign_levels(df["vre_penetration_pct"], thr)
    df["local_pv_level"], pv_thr = assign_local_pv_levels(
        df["pv_output_pct_of_rated"].dropna())
    info = {"regional_vre_penetration_terciles_pct": thr,
            "local_pv_output_terciles_pct_of_rated": pv_thr,
            "local_pv_zero_below_pct": 1.0}
    return df, info


# ----------------------------------------------------------------------------
# Association statistics
# ----------------------------------------------------------------------------
def cramers_v(table: pd.DataFrame) -> tuple[float, float, float, int]:
    """Chi-square, naive p, bias-corrected Cramer's V and dof."""
    t = table.loc[table.sum(axis=1) > 0, table.sum(axis=0) > 0]
    if t.shape[0] < 2 or t.shape[1] < 2:
        return np.nan, np.nan, np.nan, 0
    chi2, p, dof, _ = stats.chi2_contingency(t.values)
    n = t.values.sum()
    r, k = t.shape
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    v = np.sqrt(phi2 / max(min(kc - 1, rc - 1), 1e-12))
    return float(chi2), float(p), float(v), int(dof)


def contingency(labels: pd.Series, levels: pd.Series) -> pd.DataFrame:
    ok = labels.notna() & pd.Series(levels, index=labels.index).notna()
    return pd.crosstab(labels[ok], pd.Series(levels, index=labels.index)[ok])


def day_block_permutation_test(labels: pd.Series, value: pd.Series,
                               thresholds: tuple[float, float] | None = None,
                               n_perm: int = 500, seed: int = C.RANDOM_STATE
                               ) -> dict:
    """Permutation test of cluster-label vs renewable-level association.

    Whole-day renewable profiles are shuffled between days of the same type
    (weekday / weekend). This keeps (a) the time-of-day shape of renewable
    output, (b) the time-of-day shape of power quality and (c) the serial
    correlation inside a day, so a small p-value means the association is
    stronger than the shared daily rhythm alone would produce.
    """
    ok = labels.notna() & value.notna()
    labels, value = labels[ok], value[ok]
    thr = thresholds or tercile_thresholds(value)
    idx = labels.index
    day = idx.normalize()
    slot = idx.hour * 6 + idx.minute // 10
    wide = pd.Series(value.values, index=pd.MultiIndex.from_arrays([day, slot])
                     ).unstack()                       # days x 144 slots
    observed = cramers_v(contingency(labels, assign_levels(value, thr)))[2]
    days = wide.index
    weekend = (days.weekday >= 5)
    rng = np.random.default_rng(seed)
    row_of_day = pd.Series(np.arange(len(days)), index=days)
    rows = row_of_day.reindex(day).values
    cols = wide.columns.get_indexer(slot)
    null = np.empty(n_perm)
    arr = wide.values
    for b in range(n_perm):
        perm = np.arange(len(days))
        for grp in (np.where(weekend)[0], np.where(~weekend)[0]):
            perm[grp] = rng.permutation(grp)
        shuffled = pd.Series(arr[perm[rows], cols], index=idx)
        m = shuffled.notna()
        null[b] = cramers_v(contingency(labels[m], assign_levels(shuffled[m], thr)))[2]
    p = (1 + np.sum(null >= observed)) / (n_perm + 1)
    return {"cramers_v": float(observed), "null_mean": float(np.nanmean(null)),
            "null_p95": float(np.nanpercentile(null, 95)), "p_value": float(p),
            "n_perm": n_perm, "n": int(len(labels))}


def level_summary(df: pd.DataFrame, level_col: str, value_cols: list[str]
                  ) -> pd.DataFrame:
    """Median / mean / p95 of PQ indicators per renewable level."""
    g = df.dropna(subset=[level_col]).groupby(level_col, observed=True)
    out = g[value_cols].agg(["median", "mean", lambda s: s.quantile(0.95)])
    out.columns = [f"{c}_{'p95' if 'lambda' in a else a}" for c, a in out.columns]
    out.insert(0, "n_windows", g.size())
    return out


def kruskal_by_level(df: pd.DataFrame, level_col: str, value_cols: list[str]
                     ) -> pd.DataFrame:
    """Kruskal-Wallis H with epsilon-squared effect size per indicator."""
    rows = []
    d = df.dropna(subset=[level_col])
    for c in value_cols:
        groups = [g[c].dropna().values for _, g in d.groupby(level_col, observed=True)]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) < 2:
            continue
        h, p = stats.kruskal(*groups)
        n = sum(len(g) for g in groups)
        rows.append({"indicator": c, "H": h, "p_naive": p,
                     "epsilon_sq": (h - len(groups) + 1) / (n - len(groups))})
    return pd.DataFrame(rows)


def stratified_rate(df: pd.DataFrame, flag_col: str, level_col: str,
                    strata: list[str]) -> pd.DataFrame:
    """Rate of ``flag_col`` per level, raw and standardised over strata.

    Direct standardisation: each stratum (e.g. hour x weekend) contributes
    with the same weight at every level, removing the shared daily rhythm.
    Only strata observed at every level are used.
    """
    d = df.dropna(subset=[level_col, flag_col])
    raw = d.groupby(level_col, observed=True)[flag_col].mean()
    cell = d.groupby(strata + [level_col], observed=True)[flag_col].agg(["mean", "size"])
    wide = cell["mean"].unstack(level_col)
    n = cell["size"].unstack(level_col)
    full = wide.dropna()
    weights = n.loc[full.index].sum(axis=1)
    std = (full.mul(weights, axis=0)).sum() / weights.sum()
    return pd.DataFrame({"raw_rate": raw, "standardised_rate": std,
                         "n_windows": d.groupby(level_col, observed=True).size()})


def main() -> None:
    feats = pd.read_parquet(C.PROCESSED / "pq_features_10min.parquet")
    df, info = build_analysis_dataset(feats)
    df.to_parquet(C.PROCESSED / "analysis_dataset_10min.parquet")
    pd.Series({k: str(v) for k, v in info.items()}).to_csv(
        C.PROCESSED / "renewable_level_thresholds.csv", header=["value"])
    print(df.shape, info)
    print(df[["vre_penetration_pct", "vre_share_of_generation_pct",
              "renewable_share_of_generation_pct", "solar_penetration_pct",
              "wind_penetration_pct", "pv_output_pct_of_rated",
              "pv_to_facility_load_pct", "p_kw"]].describe().T.round(2).to_string())
    print(df.re_level.value_counts().to_dict(), df.local_pv_level.value_counts().to_dict())


if __name__ == "__main__":
    main()
