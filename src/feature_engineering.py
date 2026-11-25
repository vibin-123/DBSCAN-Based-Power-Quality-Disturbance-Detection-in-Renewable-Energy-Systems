"""Stage D - feature engineering on 10-minute windows.

Every feature is either a direct aggregate of a *measured* channel or a
documented *calculation* from measured channels (see ``FEATURE_CATALOGUE``).
Timestamps and metadata stay in the index / separate columns and never enter
the clustering matrix.

Run:  python -m src.feature_engineering
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

# name -> (kind, unit, engineering meaning)
FEATURE_CATALOGUE = {
    "v_rms": ("measured", "V", "Mean phase-to-neutral RMS voltage (3-phase average)."),
    "v_dev_pct": ("calculated", "%", "Deviation of v_rms from 230 V nominal; EN 50160 allows +/-10 %."),
    "v_std_pct": ("calculated", "% of Un", "Within-window std of 5 s voltage: slow voltage fluctuation."),
    "rvc_max_pct": ("calculated", "% of Un", "Largest 5 s step change of voltage in the window (rapid voltage change proxy)."),
    "vuf_pct": ("calculated", "%", "Voltage unbalance factor (negative/positive sequence) from the three line-line magnitudes, IEC 61000-4-30 formula."),
    "freq_hz": ("measured", "Hz", "Mean system frequency."),
    "f_dev_abs_hz": ("calculated", "Hz", "Mean absolute deviation from 50 Hz: active-power balance of the synchronous area."),
    "f_dev_max_hz": ("calculated", "Hz", "Largest absolute frequency deviation in the window."),
    "thd_u_pct": ("measured", "%", "Voltage THD (3-phase mean); EN 50160 limit 8 %."),
    "thd_u_max_pct": ("measured", "%", "Maximum 5 s voltage THD of any phase in the window."),
    "thd_i_pct": ("measured", "%", "Current THD (3-phase mean): non-linearity of the facility load."),
    "u_h3_pct": ("measured", "%", "3rd voltage harmonic, phase 1 (zero-sequence, single-phase loads)."),
    "u_h5_pct": ("measured", "%", "5th voltage harmonic (3-phase mean): six-pulse converters / inverters."),
    "u_h7_pct": ("measured", "%", "7th voltage harmonic, phase 1."),
    "i_rms_a": ("measured", "A", "Mean phase RMS current (loading of the connection point)."),
    "i_n_a": ("measured", "A", "Neutral conductor current (unbalance + triplen harmonics)."),
    "i_unb_pct": ("calculated", "%", "Current unbalance: max deviation of a phase current from the mean."),
    "p_kw": ("measured", "kW", "Total active power at the metering point."),
    "q_kvar": ("measured", "kvar", "Total reactive power."),
    "pf": ("measured", "-", "Total power factor."),
    "sag_samples": ("calculated", "count", "5 s RMS samples below 0.90 pu (EN 50160 dip threshold) on any phase."),
    "swell_samples": ("calculated", "count", "5 s RMS samples above 1.10 pu on any phase."),
    "severity_index": ("calculated", "-", "Largest fraction of an EN 50160 limit in use: max(|dV|/10 %, THDu/8 %, VUF/2 %, |df|/0.5 Hz)."),
    "v_dev_roll_mean_1h": ("calculated", "%", "1-hour rolling mean of v_dev_pct (trend)."),
    "v_dev_roll_std_1h": ("calculated", "%", "1-hour rolling std of v_dev_pct (volatility)."),
    "thd_u_roll_mean_1h": ("calculated", "%", "1-hour rolling mean of thd_u_pct."),
    "thd_u_roll_std_1h": ("calculated", "%", "1-hour rolling std of thd_u_pct."),
}

# Experiment-1 clustering inputs: power-quality quantities only.
PQ_CLUSTER_FEATURES = [
    "v_dev_pct", "v_std_pct", "rvc_max_pct", "vuf_pct", "f_dev_abs_hz",
    "thd_u_pct", "u_h5_pct", "thd_i_pct", "i_rms_a", "i_n_a",
]


def voltage_unbalance_factor(u12, u23, u31):
    """IEC 61000-4-30 VUF (%) from line-to-line RMS magnitudes.

    beta = (U12^4+U23^4+U31^4) / (U12^2+U23^2+U31^2)^2
    VUF  = sqrt((1 - sqrt(3 - 6 beta)) / (1 + sqrt(3 - 6 beta))) * 100
    """
    u12, u23, u31 = (np.asarray(x, dtype="float64") for x in (u12, u23, u31))
    num = u12 ** 4 + u23 ** 4 + u31 ** 4
    den = (u12 ** 2 + u23 ** 2 + u31 ** 2) ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        beta = num / den
        root = np.sqrt(np.clip(3.0 - 6.0 * beta, 0.0, None))
        return np.sqrt(np.clip((1.0 - root) / (1.0 + root), 0.0, None)) * 100.0


def severity_index(v_dev_pct, thd_u_pct, vuf_pct, f_dev_abs_hz):
    """Fraction of the tightest EN 50160 limit in use (1.0 = at the limit)."""
    parts = np.vstack([np.abs(v_dev_pct) / 10.0, np.asarray(thd_u_pct) / 8.0,
                       np.asarray(vuf_pct) / 2.0,
                       np.abs(f_dev_abs_hz) / 0.5])
    return np.nanmax(parts, axis=0)


def episode_durations(flag: pd.Series, step_minutes: int = 10) -> pd.DataFrame:
    """Start, end and duration of consecutive runs where ``flag`` is True.

    A run is broken by a False value *or* by a gap in the time index.
    """
    flag = flag.fillna(False).astype(bool)
    if not flag.any():
        return pd.DataFrame(columns=["start", "end", "duration_min"])
    step = pd.Timedelta(minutes=step_minutes)
    gap = pd.Series(flag.index, index=flag.index).diff() != step
    run_id = ((flag != flag.shift()) | gap).cumsum()
    rows = []
    for _, grp in flag[flag].groupby(run_id[flag]):
        rows.append({"start": grp.index[0], "end": grp.index[-1] + step,
                     "duration_min": len(grp) * step_minutes})
    return pd.DataFrame(rows)


def compute_5s_derived(pq: pd.DataFrame) -> pd.DataFrame:
    """Per-sample derived quantities needed before window aggregation."""
    d = pd.DataFrame(index=pq.index)
    d["v_rms"] = pq[["U1", "U2", "U3"]].mean(axis=1)
    d["v_pu_min"] = pq[["U1", "U2", "U3"]].min(axis=1) / C.U_NOMINAL_LN
    d["v_pu_max"] = pq[["U1", "U2", "U3"]].max(axis=1) / C.U_NOMINAL_LN
    # step to the previous valid sample (at most one empty 5 s slot between)
    prev = d["v_rms"].ffill(limit=1).shift(1)
    d["v_step_pct"] = (d["v_rms"] - prev).abs() / C.U_NOMINAL_LN * 100.0
    d["vuf_pct"] = voltage_unbalance_factor(pq["U12"], pq["U23"], pq["U31"])
    d["f_dev_abs_hz"] = (pq["Freq"] - C.F_NOMINAL).abs()
    d["freq_hz"] = pq["Freq"]
    d["thd_u_pct"] = pq[["THD_U1", "THD_U2", "THD_U3"]].mean(axis=1)
    d["thd_u_phase_max"] = pq[["THD_U1", "THD_U2", "THD_U3"]].max(axis=1)
    d["thd_i_pct"] = pq[["THD_I1", "THD_I2", "THD_I3"]].mean(axis=1)
    d["u_h3_pct"] = pq["U1_h3"]
    d["u_h5_pct"] = pq[["U1_h5", "U2_h5", "U3_h5"]].mean(axis=1)
    d["u_h7_pct"] = pq["U1_h7"]
    i = pq[["I1", "I2", "I3"]]
    d["i_rms_a"] = i.mean(axis=1)
    d["i_unb_pct"] = (i.sub(d["i_rms_a"], axis=0).abs().max(axis=1)
                      / d["i_rms_a"].where(d["i_rms_a"] > 0) * 100.0)
    d["i_n_a"] = pq["IN"]
    d["p_kw"] = pq["P_total_kw"]
    d["q_kvar"] = pq["Q_total_kvar"]
    d["pf"] = pq["PF_total"]
    d["valid"] = pq.notna().all(axis=1)
    for c in d.columns.drop("valid"):
        d.loc[~d["valid"], c] = np.nan
    return d


def aggregate_windows(d: pd.DataFrame, window: str = C.WINDOW,
                      min_coverage: float = C.MIN_COVERAGE) -> pd.DataFrame:
    """Aggregate 5 s derived data to windows; drop windows with low coverage."""
    g = d.resample(window)
    expected = pd.Timedelta(window) / pd.Timedelta(seconds=C.SAMPLE_SECONDS)
    f = pd.DataFrame({
        "coverage": g["valid"].sum() / expected,
        "v_rms": g["v_rms"].mean(),
        "v_std_pct": g["v_rms"].std() / C.U_NOMINAL_LN * 100.0,
        "rvc_max_pct": g["v_step_pct"].max(),
        "vuf_pct": g["vuf_pct"].mean(),
        "freq_hz": g["freq_hz"].mean(),
        "f_dev_abs_hz": g["f_dev_abs_hz"].mean(),
        "f_dev_max_hz": g["f_dev_abs_hz"].max(),
        "thd_u_pct": g["thd_u_pct"].mean(),
        "thd_u_max_pct": g["thd_u_phase_max"].max(),
        "thd_i_pct": g["thd_i_pct"].mean(),
        "u_h3_pct": g["u_h3_pct"].mean(),
        "u_h5_pct": g["u_h5_pct"].mean(),
        "u_h7_pct": g["u_h7_pct"].mean(),
        "i_rms_a": g["i_rms_a"].mean(),
        "i_n_a": g["i_n_a"].mean(),
        "i_unb_pct": g["i_unb_pct"].mean(),
        "p_kw": g["p_kw"].mean(),
        "q_kvar": g["q_kvar"].mean(),
        "pf": g["pf"].mean(),
        "sag_samples": (d["v_pu_min"] < C.SAG_PU).resample(window).sum(),
        "swell_samples": (d["v_pu_max"] > C.SWELL_PU).resample(window).sum(),
    })
    f["v_dev_pct"] = (f["v_rms"] - C.U_NOMINAL_LN) / C.U_NOMINAL_LN * 100.0
    f = f[f["coverage"] >= min_coverage].copy()
    f["severity_index"] = severity_index(f["v_dev_pct"], f["thd_u_pct"],
                                         f["vuf_pct"], f["f_dev_max_hz"])
    return f


def add_rolling(f: pd.DataFrame, span: str = "1h", min_periods: int = 4
                ) -> pd.DataFrame:
    """Time-based rolling statistics (gaps shorten the window, never bridge it)."""
    f = f.copy()
    for src, stem in (("v_dev_pct", "v_dev"), ("thd_u_pct", "thd_u")):
        r = f[src].rolling(span, min_periods=min_periods)
        f[f"{stem}_roll_mean_1h"] = r.mean()
        f[f"{stem}_roll_std_1h"] = r.std()
    return f


def add_time_metadata(f: pd.DataFrame) -> pd.DataFrame:
    """Calendar context for interpretation only (never a clustering input)."""
    f = f.copy()
    local = f.index.tz_localize("UTC").tz_convert("Europe/Berlin")
    f["hour_local"] = local.hour
    f["weekday"] = local.weekday
    f["is_weekend"] = local.weekday >= 5
    f["month"] = local.month
    return f


def build_feature_table() -> pd.DataFrame:
    pq = pd.read_parquet(C.INTERIM / "pq_5s.parquet")
    feats = aggregate_windows(compute_5s_derived(pq))
    feats = add_time_metadata(add_rolling(feats))
    feats.index.name = "timestamp_utc"
    return feats


def main() -> None:
    f = build_feature_table()
    f.to_parquet(C.PROCESSED / "pq_features_10min.parquet")
    cat = pd.DataFrame([{"feature": k, "kind": v[0], "unit": v[1], "meaning": v[2]}
                        for k, v in FEATURE_CATALOGUE.items()])
    cat.to_csv(C.PROCESSED / "feature_catalogue.csv", index=False)
    print(f.shape)
    print(f[PQ_CLUSTER_FEATURES + ["severity_index", "sag_samples", "swell_samples"]]
          .describe().T.round(3).to_string())


if __name__ == "__main__":
    main()
