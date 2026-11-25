"""Stages B + C - validation and preprocessing.

Raw files are never modified. This module

1. loads the 5-second SPARK measurement files and merges them on timestamp,
2. removes duplicate timestamps, enforces a regular 5 s grid, range-checks
   every channel against physical limits (values outside become NaN and are
   counted - they are *not* silently interpolated),
3. standardises units (W -> kW, var -> kvar) and time zone (UTC),
4. aligns PV power and the 15-minute regional generation/load series,
5. writes ``data/interim`` parquet files and a preprocessing report.

Run:  python -m src.preprocessing
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

# Physical plausibility limits (LV 230/400 V, 50 Hz system). Samples outside
# are measurement artefacts, not disturbances, and are set to NaN.
PHYSICAL_LIMITS = {
    "U": (0.0, 300.0), "U_LL": (0.0, 520.0), "Freq": (45.0, 55.0),
    "I": (0.0, 5000.0), "THD": (0.0, 100.0), "H": (0.0, 100.0),
    "PF": (-1.0, 1.0),
}


def _limit_key(name: str) -> str | None:
    if name in ("U12", "U23", "U31"):
        return "U_LL"
    if name in ("U1", "U2", "U3"):
        return "U"
    if name == "Freq":
        return "Freq"
    if name in ("I1", "I2", "I3", "IN"):
        return "I"
    if name.startswith("THD"):
        return "THD"
    if "_h" in name:
        return "H"
    if name.startswith("PF"):
        return "PF"
    return None


def load_measurement(path: Path, name: str) -> pd.Series:
    """Read one ``WsDateTime,<name>`` file as a float32 series (UTC-naive)."""
    df = pd.read_csv(path, parse_dates=["WsDateTime"])
    s = df.set_index("WsDateTime")[name].astype("float32")
    s.index.name = "timestamp_utc"
    return s


def clean_series(s: pd.Series, name: str, freq: str = "5s"
                 ) -> tuple[pd.Series, dict]:
    """Deduplicate, sort, regularise to ``freq`` and range-check one channel."""
    stats = {"rows_raw": int(len(s))}
    dup = s.index.duplicated(keep="first")
    stats["duplicate_timestamps"] = int(dup.sum())
    s = s[~dup].sort_index()
    off_grid = s.index != s.index.floor(freq)
    stats["off_grid_timestamps"] = int(off_grid.sum())
    s = s[~off_grid]
    if len(s):
        s = s.reindex(pd.date_range(s.index[0], s.index[-1], freq=freq,
                                    name=s.index.name))
    stats["missing_raw"] = int(s.isna().sum())
    key = _limit_key(name)
    if key:
        lo, hi = PHYSICAL_LIMITS[key]
        bad = (s < lo) | (s > hi)
        stats["out_of_physical_range"] = int(bad.sum())
        s = s.mask(bad)
    else:
        stats["out_of_physical_range"] = 0
    stats["missing_final"] = int(s.isna().sum())
    stats["rows_final"] = int(len(s))
    return s, stats


def build_pq_frame(year: int = C.YEAR) -> tuple[pd.DataFrame, dict]:
    """Merge all selected PQ channels of the total-load meter (5 s grid)."""
    cols, report = {}, {}
    for m in C.PQ_MEASUREMENTS:
        path = C.SPARK_DIR / C.PQ_METER / m / f"{year}_{m}.csv.xz"
        s, st = clean_series(load_measurement(path, m), m)
        cols[m], report[m] = s, st
    df = pd.DataFrame(cols)
    # unit standardisation: W / var / VA -> kW / kvar / kVA
    for c in ("P_total", "Q_total", "S_total"):
        df[c] = df[c] / 1000.0
    df = df.rename(columns={"P_total": "P_total_kw", "Q_total": "Q_total_kvar",
                            "S_total": "S_total_kva"})
    return df, report


def build_pv_frame(year: int = C.YEAR) -> tuple[pd.Series, dict]:
    path = C.SPARK_DIR / C.PV_METER / "AC_ActivePower" / f"{year}_AC_ActivePower.csv.xz"
    s, st = clean_series(load_measurement(path, "AC_ActivePower"), "AC_ActivePower")
    bad = (s < 0) | (s > 1.2 * C.PV_RATED_W)   # inverter cannot exceed rating
    st["out_of_physical_range"] = int(bad.sum())
    s = s.mask(bad)
    return (s / 1000.0).rename("pv_power_kw"), st


def build_regional_frame(year: int = C.YEAR) -> tuple[pd.DataFrame, dict]:
    """15-min TransnetBW generation mix (SPARK exogenous) + grid load (SMARD)."""
    mix = pd.read_csv(C.SPARK_DIR / "01exogenous_data" /
                      "electricity_emissions_15min_transnetbw_2018_2024.csv",
                      parse_dates=["utc_datetime"]).set_index("utc_datetime")
    mix.index = mix.index.tz_localize(None)
    mix = mix.loc[f"{year}-01-01":f"{year}-12-31 23:59:59"]
    mix = mix.drop(columns=["emission_intensity_g_per_kwh"])
    rep = {"rows": int(len(mix)),
           "duplicates": int(mix.index.duplicated().sum()),
           "missing_share": mix.isna().mean().round(4).to_dict()}
    # Nuclear is reported as missing after the German phase-out (15 Apr 2023);
    # generation is physically zero, so this is a documented fill, not a guess.
    mix["nuclear"] = mix["nuclear"].fillna(0.0)
    mix = mix[~mix.index.duplicated()]
    mix.columns = [f"{c}_mw" for c in mix.columns]
    load = pd.read_csv(C.SMARD_DIR / f"smard_grid_load_{C.SMARD_REGION}_{year}_15min.csv",
                       parse_dates=["utc_datetime"]).set_index("utc_datetime")
    load.index = load.index.tz_localize(None)
    # cross-check: SMARD solar downloaded directly vs the copy shipped in SPARK
    sm = pd.read_csv(C.SMARD_DIR / f"smard_solar_pv_{C.SMARD_REGION}_{year}_15min.csv",
                     parse_dates=["utc_datetime"]).set_index("utc_datetime")
    sm.index = sm.index.tz_localize(None)
    both = pd.concat([sm["solar_pv_mw"], mix["solar_pv_mw"]], axis=1).dropna()
    rep["smard_vs_spark_solar_max_abs_diff_mw"] = float(
        (both.iloc[:, 0] - both.iloc[:, 1]).abs().max())
    out = mix.join(load[["grid_load_mw"]], how="left")
    out.index.name = "timestamp_utc"
    rep["grid_load_missing"] = int(out["grid_load_mw"].isna().sum())
    return out, rep


def align_to_windows(regional: pd.DataFrame, window_index: pd.DatetimeIndex
                     ) -> pd.DataFrame:
    """Map each 10-min window start to the 15-min interval that contains it.

    A 15-min value is an interval average stamped at the interval start, so a
    backward as-of join (tolerance < 15 min) is the correct alignment; no
    values are interpolated or carried across gaps.
    """
    left = pd.DataFrame({"timestamp_utc": window_index})
    right = regional.reset_index().sort_values("timestamp_utc")
    left["timestamp_utc"] = left["timestamp_utc"].astype(right["timestamp_utc"].dtype)
    out = pd.merge_asof(left, right, on="timestamp_utc", direction="backward",
                        tolerance=pd.Timedelta("14min59s"))
    return out.set_index("timestamp_utc")


def write_report(pq_rep: dict, pv_rep: dict, reg_rep: dict, pq: pd.DataFrame,
                 pv: pd.Series) -> Path:
    lines = ["# Preprocessing report", "",
             f"Year: {C.YEAR}. PQ meter: `{C.PQ_METER}` (SPARK). "
             "All timestamps are UTC.", "",
             "## Power-quality channels (5 s)", "",
             "| channel | rows | duplicates | off-grid | missing (raw) | out of physical range | missing (final) | availability |",
             "|---|---|---|---|---|---|---|---|"]
    for m, st in pq_rep.items():
        avail = 1 - st["missing_final"] / max(st["rows_final"], 1)
        lines.append(f"| {m} | {st['rows_final']:,} | {st['duplicate_timestamps']} | "
                     f"{st['off_grid_timestamps']} | {st['missing_raw']:,} | "
                     f"{st['out_of_physical_range']} | {st['missing_final']:,} | {avail:.1%} |")
    avail = pq.notna().all(axis=1)
    month = avail.groupby(avail.index.month).mean()
    lines += ["", "Share of 5 s samples with all channels present, by month: " +
              ", ".join(f"{m}: {v:.0%}" for m, v in month.items()), "",
              "## PV active power (5 s)", "",
              f"Rows {pv_rep['rows_final']:,}; duplicates {pv_rep['duplicate_timestamps']}; "
              f"missing {pv_rep['missing_final']:,}; out of range {pv_rep['out_of_physical_range']}. ",
              f"Valid PV data span: {pv.dropna().index.min()} to {pv.dropna().index.max()}.", "",
              "## Regional generation mix and load (15 min)", "",
              "```json", json.dumps(reg_rep, indent=1), "```", "",
              "## Handling rules", "",
              "- Duplicated timestamps: first occurrence kept.",
              "- Missing 5 s samples: left missing, never interpolated. The meter reports "
              "about every 6.5 s, so on the 5 s grid every fourth slot is structurally "
              "empty and a complete 10-minute window has ~76.7 % of slots filled. "
              f"A window is used only if at least {C.MIN_COVERAGE:.0%} of its slots are "
              "valid in every channel (about 78 % of the achievable samples).",
              "- Out-of-physical-range samples: set to missing and counted above.",
              "- Units: P, Q, S converted from W/var/VA to kW/kvar/kVA; PV from W to kW.",
              "- Nuclear generation missing after April 2023 set to 0 MW (plant shutdown).",
              "- 15-minute regional values are assigned to 10-minute windows by a backward "
              "as-of join (the 15-min interval that contains the window start)."]
    dest = C.REPORTS / "preprocessing_report.md"
    dest.write_text("\n".join(lines))
    return dest


def main() -> None:
    pq, pq_rep = build_pq_frame()
    pv, pv_rep = build_pv_frame()
    regional, reg_rep = build_regional_frame()
    pq.to_parquet(C.INTERIM / "pq_5s.parquet")
    pv.to_frame().to_parquet(C.INTERIM / "pv_5s.parquet")
    regional.to_parquet(C.INTERIM / "regional_15min.parquet")
    print("report:", write_report(pq_rep, pv_rep, reg_rep, pq, pv))
    print(pq.shape, pv.shape, regional.shape)


if __name__ == "__main__":
    main()
