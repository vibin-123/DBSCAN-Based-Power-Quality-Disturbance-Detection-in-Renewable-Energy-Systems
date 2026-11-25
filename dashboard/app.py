"""Streamlit dashboard.   Run from the project root:  streamlit run dashboard/app.py"""
from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as C                                   # noqa: E402
from src import visualization as V                            # noqa: E402
from src.dbscan_model import (block_weights, k_distance, knee_point,  # noqa: E402
                              pca_projection, prepare_matrix, relabel_by_size,
                              run_dbscan)
from src.evaluation import clustering_metrics                 # noqa: E402
from src.feature_engineering import PQ_CLUSTER_FEATURES       # noqa: E402

st.set_page_config(page_title="Power quality x renewables | DBSCAN",
                   page_icon="⚡", layout="wide",
                   initial_sidebar_state="expanded")

# (url slug, sidebar label). The order is fixed: sections are addressed by position.
NAV = [
    ("overview", ":material/dashboard: Overview"),
    ("dataset", ":material/database: Dataset"),
    ("power-quality", ":material/electric_bolt: Power quality"),
    ("generation", ":material/solar_power: Renewable generation"),
    ("penetration", ":material/percent: Renewable penetration"),
    ("experiment-1", ":material/counter_1: Experiment 1"),
    ("experiment-2", ":material/counter_2: Experiment 2"),
    ("clusters", ":material/scatter_plot: Cluster map"),
    ("unusual", ":material/troubleshoot: Unusual windows"),
    ("comparison", ":material/compare_arrows: Experiment comparison"),
    ("correlation", ":material/grid_on: Correlation"),
    ("explorer", ":material/tune: Parameter explorer"),
    ("downloads", ":material/download: Downloads"),
]
SLUGS = [s for s, _ in NAV]
PAGES = [label for _, label in NAV]
# sidebar group headings: position of the first section in each group (1-based)
NAV_GROUPS = {2: "Data", 6: "Experiments", 8: "Analysis", 12: "Tools"}

RE_FEATURES = ["vre_penetration_pct", "solar_pv_mw", "wind_onshore_mw"]
EXPERIMENTS = {"exp1": "Experiment 1 · PQ only",
               "exp2": "Experiment 2 · PQ + renewables"}
COLUMN_LABELS = {
    "name": "Label", "cluster": "Id", "n_windows": "Windows", "share_pct": "Share (%)",
    "n_days": "Days", "weekend_share_pct": "Weekend share (%)",
    "re_level": "Level", "local_pv_level": "Level",
    "metric": "Distance metric", "eps": "Epsilon", "min_samples": "MinPts",
    "n_clusters": "Clusters", "n_noise": "Noise windows", "noise_pct": "Noise (%)",
    "largest_cluster_pct": "Largest cluster (%)", "silhouette": "Silhouette",
    "davies_bouldin": "Davies-Bouldin", "ari_vs_euclidean": "ARI vs euclidean",
    "noise_jaccard_vs_euclidean": "Noise Jaccard vs euclidean",
    "ari_labels": "ARI (all labels)", "ari_unusual_vs_normal": "ARI (unusual vs normal)",
    "unusual_jaccard": "Unusual-set Jaccard", "unusual_pct": "Unusual (%)",
    "cramers_v": "Cramér's V", "null_mean": "Null mean", "null_p95": "Null 95th pct",
    "p_value": "p-value", "n_perm": "Permutations", "n": "Windows",
    "indicator": "Indicator", "H": "Kruskal-Wallis H", "p_naive": "p (naive)",
    "epsilon_sq": "Effect size (epsilon²)", "re_share": "Renewable weight share",
    "per_feature_multiplier": "Per-feature multiplier",
    "isolated_noise_pct": "Isolated noise (%)", "ari_vs_exp1": "ARI vs Experiment 1",
    "unusual_jaccard_vs_exp1": "Unusual-set Jaccard vs Experiment 1",
    "algorithm": "Algorithm", "k": "k", "smallest_cluster_pct": "Smallest cluster (%)",
}
COMPARISON_ROWS = {
    "dimensions": "Clustering dimensions", "eps_stage1": "Epsilon, stage 1",
    "eps_stage2": "Epsilon, stage 2", "n_clusters": "Clusters found",
    "dominant_cluster_pct": "Dominant cluster (%)", "unusual_pct": "Unusual windows (%)",
    "isolated_noise_pct": "Isolated noise (%)", "silhouette": "Silhouette",
    "davies_bouldin": "Davies-Bouldin", "stage2_silhouette": "Stage-2 silhouette",
    "stability_ari_mean": "Stability (mean ARI)",
    "cramers_v_vs_re_level": "Cramér's V vs renewable level",
}


# ------------------------------- data --------------------------------------
@st.cache_data(show_spinner=False)
def _read_parquet(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def _read_csv(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def _read_json(path: str, mtime: float) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _cached(reader, path: Path):
    """Cache on the file's modification time, so a pipeline re-run is picked
    up without restarting the dashboard."""
    return reader(str(path), path.stat().st_mtime)


def load_dataset() -> pd.DataFrame:
    return _cached(_read_parquet, C.PROCESSED / "analysis_dataset_10min.parquet")


def load_labels(exp: str) -> pd.DataFrame:
    return _cached(_read_parquet, C.RESULTS / f"{exp}_labels.parquet")


def load_csv(name: str) -> pd.DataFrame:
    return _cached(_read_csv, C.RESULTS / f"{name}.csv")


def load_summary() -> dict:
    return _cached(_read_json, C.RESULTS / "summary.json")


@st.cache_data(show_spinner="Running DBSCAN on the subsample...")
def explore(features: tuple, re_share: float | None, eps: float, min_samples: int,
            n: int, metric: str, mtime: float) -> tuple[pd.DataFrame, dict, float]:
    df = load_dataset()
    feats = list(features)
    n_re = len([f for f in feats if f in RE_FEATURES])
    w = block_weights(len(feats) - n_re, n_re, re_share) if n_re else None
    pm = prepare_matrix(df, feats, w)
    rng = np.random.default_rng(C.RANDOM_STATE)
    idx = np.sort(rng.choice(len(pm.X), size=min(n, len(pm.X)), replace=False))
    X = pm.X[idx]
    _, knee = knee_point(k_distance(X, min_samples, metric))
    labels = relabel_by_size(run_dbscan(X, eps, min_samples, metric))
    Z, _, _ = pca_projection(X, 2)
    out = pd.DataFrame({"pc1": Z[:, 0], "pc2": Z[:, 1], "label": labels},
                       index=pm.index[idx])
    return out, clustering_metrics(X, labels, sample_size=3000), knee


def results_ready() -> bool:
    return (C.RESULTS / "summary.json").exists() and \
        (C.PROCESSED / "analysis_dataset_10min.parquet").exists()


# ------------------------------- look --------------------------------------
def is_dark() -> bool:
    try:
        return st.context.theme.type == "dark"
    except Exception:                       # no browser context (tests)
        return False


# Hero decoration: a voltage waveform with a short sag. Percent-encoded, because
# Streamlit's HTML sanitiser drops any <style> block that contains markup.
_WAVE = "data:image/svg+xml," + quote(
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 640 200' fill='none' "
    "stroke='#9ec5f4' stroke-width='2' stroke-linecap='round'>"
    "<path opacity='.55' d='M0 100 C 20 20, 40 20, 60 100 S 100 180, 120 100 "
    "S 160 20, 180 100 S 220 180, 240 100 S 270 62, 285 100 S 310 138, 325 100 "
    "S 360 20, 380 100 S 420 180, 440 100 S 480 20, 500 100 S 540 180, 560 100 "
    "S 600 20, 620 100'/><path opacity='.22' d='M0 100 H640'/></svg>")


def inject_css(dark: bool) -> None:
    accent = "#86b6ef" if dark else "#1c5cab"
    tile = "#141a24" if dark else "#f8f9fb"
    groups = "".join(
        f'[data-testid="stSidebar"] [role="radiogroup"] > label:nth-of-type({i})'
        f'{{margin-top:1.9rem}}'
        f'[data-testid="stSidebar"] [role="radiogroup"] > label:nth-of-type({i})::before'
        f'{{content:"{name}"}}' for i, name in NAV_GROUPS.items())
    st.html(f"""<style>
[data-testid="stMainBlockContainer"] {{ max-width: 1400px; padding-top: 4.5rem; }}

/* ---- sidebar: brand + navigation list ---- */
[data-testid="stSidebar"] {{ background: linear-gradient(180deg, #10203a 0%, #0b1526 100%); }}
.pq-brand {{ display: flex; align-items: center; gap: .7rem; margin: .1rem 0 .9rem; }}
.pq-brand-mark {{ width: 2.4rem; height: 2.4rem; border-radius: .7rem; flex: none;
  display: grid; place-items: center; font-size: 1.2rem;
  background: linear-gradient(135deg, #2a78d6, #1baf7a); }}
.pq-brand-name {{ color: #fff; font-weight: 700; font-size: 1.02rem; line-height: 1.2; }}
.pq-brand-sub {{ color: rgba(230,235,242,.6); font-size: .78rem; }}
.pq-side-note {{ color: rgba(230,235,242,.5); font-size: .74rem; line-height: 1.5;
  border-top: 1px solid rgba(255,255,255,.1); padding-top: .8rem; margin-top: .6rem; }}
[data-testid="stSidebar"] [data-testid="stElementContainer"]:has([data-testid="stRadio"]),
[data-testid="stSidebar"] [data-testid="stRadio"] {{ width: 100% !important; }}
[data-testid="stSidebar"] [role="radiogroup"] {{ gap: 2px; width: 100%; align-items: stretch; }}
[data-testid="stSidebar"] [role="radiogroup"] > label {{
  position: relative; width: 100%; margin: 0; padding: .38rem .75rem;
  border-radius: .5rem; color: rgba(230,235,242,.8); cursor: pointer;
  transition: background .12s ease, color .12s ease; }}
/* hide the radio circle: every box in the option that is not (around) its text */
[data-testid="stSidebar"] [role="radiogroup"] > label div:not([data-testid="stMarkdownContainer"]):not(:has([data-testid="stMarkdownContainer"])):not([data-testid="stMarkdownContainer"] *) {{ display: none; }}
[data-testid="stSidebar"] [role="radiogroup"] > label * {{ color: inherit; }}
[data-testid="stSidebar"] [role="radiogroup"] > label p {{ font-size: .92rem; }}
[data-testid="stSidebar"] [role="radiogroup"] > label:hover {{
  background: rgba(255,255,255,.07); color: #fff; }}
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {{
  background: rgba(57,135,229,.24); color: #fff; box-shadow: inset 3px 0 0 #3987e5; }}
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) p {{ font-weight: 600; }}
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:focus-visible) {{
  outline: 2px solid #86b6ef; outline-offset: 1px; }}
[data-testid="stSidebar"] [role="radiogroup"] > label::before {{
  position: absolute; left: .75rem; top: -1.4rem; font-size: .68rem; font-weight: 600;
  letter-spacing: .1em; text-transform: uppercase; color: rgba(230,235,242,.42);
  pointer-events: none; }}
{groups}

/* ---- page header and hero ---- */
.pq-eyebrow {{ font-size: .72rem; font-weight: 600; letter-spacing: .12em;
  text-transform: uppercase; color: {accent}; }}
.pq-head h1 {{ font-size: 1.9rem; font-weight: 700; line-height: 1.2;
  margin: .2rem 0 .35rem; padding: 0; }}
.pq-head p {{ margin: 0; max-width: 78ch; opacity: .74; font-size: 1.02rem; }}
.pq-hero {{ position: relative; overflow: hidden; border-radius: 1rem;
  padding: 2.1rem 2.3rem 2rem; color: #fff;
  background: linear-gradient(120deg, #0f1b2d 0%, #15335d 58%, #1c5cab 100%); }}
.pq-hero::after {{ content: ""; position: absolute; right: -1.5rem; bottom: -.5rem;
  width: 46%; height: 100%; background: url("{_WAVE}") right center / contain no-repeat;
  pointer-events: none; }}
.pq-hero .pq-eyebrow {{ color: #9ec5f4; }}
.pq-hero h1 {{ position: relative; z-index: 1; color: #fff; font-size: 2.05rem;
  font-weight: 700; line-height: 1.18; margin: .45rem 0 .7rem; padding: 0;
  max-width: 24ch; }}
.pq-hero p {{ position: relative; z-index: 1; margin: 0; max-width: 60ch;
  color: rgba(255,255,255,.84); font-size: 1.03rem; line-height: 1.55; }}
.pq-chips {{ position: relative; z-index: 1; display: flex; flex-wrap: wrap;
  gap: .45rem; margin-top: 1.1rem; }}
.pq-chip {{ font-size: .78rem; padding: .22rem .65rem; border-radius: 999px;
  background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.2); }}
@media (max-width: 900px) {{ .pq-hero::after {{ display: none; }} }}

/* ---- stat tiles ---- */
[data-testid="stMetric"] {{ background: {tile}; }}
.st-key-plain_metrics [data-testid="stMetric"] {{ background: transparent; }}
@media (prefers-reduced-motion: reduce) {{
  [data-testid="stSidebar"] [role="radiogroup"] > label {{ transition: none; }} }}
</style>""")


def page_header(group: str, title: str, lead: str) -> None:
    st.html(f'<div class="pq-head"><div class="pq-eyebrow">{group}</div>'
            f'<h1>{title}</h1><p>{lead}</p></div>')


def chart(fig, key: str | None = None) -> None:
    st.plotly_chart(V.themed(fig, is_dark()), width="stretch", key=key,
                    config={"displaylogo": False,
                            "modeBarButtonsToRemove": ["lasso2d", "select2d"]})


def table(df: pd.DataFrame, digits: int = 3, labels: dict | None = None,
          formats: dict | None = None, pin: str | None = None, **kwargs) -> None:
    """Dataframe with readable column names and fixed number formats.

    ``pin`` keeps one column in view while a wide table scrolls sideways.
    """
    cfg = {}
    for c in df.columns:
        label = (labels or {}).get(c) or COLUMN_LABELS.get(c) or V.label_for(c)
        if pd.api.types.is_float_dtype(df[c]):
            fmt = (formats or {}).get(c, f"%.{digits}f")
            cfg[c] = st.column_config.NumberColumn(label, format=fmt, pinned=c == pin)
        else:
            cfg[c] = st.column_config.Column(label, pinned=c == pin)
    st.dataframe(df, width="stretch", hide_index=True, column_config=cfg, **kwargs)


def num(x, spec: str = ".2f") -> str:
    """Format a number; undefined metrics (NaN / missing) read 'n/a'."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return format(x, spec)


def card(title: str | None = None):
    box = st.container(border=True)
    if title:
        box.markdown(f"**{title}**")
    return box


# ------------------------------- pages -------------------------------------
def page_overview(df: pd.DataFrame, summ: dict) -> None:
    e1 = summ["exp1"]
    rate = load_csv("exp1_unusual_rate_by_re_level").set_index("re_level")
    low, high = rate.raw_rate["low"] * 100, rate.raw_rate["high"] * 100
    test = e1["association"]["permutation_tests"]["vre_penetration_pct__unusual_flag"]
    st.html(
        '<div class="pq-hero"><div class="pq-eyebrow">Unsupervised study · SPARK (KIT) '
        '+ SMARD · 2024</div>'
        '<h1>Do unusual power-quality patterns cluster around high renewable '
        'injection?</h1>'
        '<p>One year of real power-quality measurements from an industrial facility in '
        'Karlsruhe, clustered with DBSCAN and compared with regional solar and wind '
        'generation and the facility\'s own rooftop PV.</p>'
        '<div class="pq-chips"><span class="pq-chip">DBSCAN only, two density levels'
        '</span><span class="pq-chip">No labels, no train/test split</span>'
        '<span class="pq-chip">10-minute windows (IEC 61000-4-30)</span></div></div>')

    c = st.columns(4)
    c[0].metric("10-minute windows analysed", f"{len(df):,}", border=True)
    c[1].metric("Unusual windows (Exp. 1)", f"{e1['final']['unusual_pct']:.2f}%",
                border=True,
                help="Windows DBSCAN did not place in the dominant dense cluster.")
    c[2].metric("Unusual rate · low renewables", f"{low:.2f}%", border=True)
    c[3].metric("Unusual rate · high renewables", f"{high:.2f}%", border=True)

    left, right = st.columns([3, 2])
    with left.container(border=True):
        chart(V.fig_rate_by_level(
            rate.reset_index(), "Unusual-window rate by regional renewable level"))
        st.caption(
            f"Unusual windows are {high / low:.1f}x as frequent at high renewable "
            f"penetration as at low. Whole-day permutation test: p = "
            f"{test['p_value']:.3f}, Cramér's V = {test['cramers_v']:.2f}: statistically "
            "clear, but a small effect.")
    with right.container(border=True):
        st.markdown("**What DBSCAN found (Experiment 1)**")
        counts = pd.DataFrame({
            "Label": [e1["label_names"][k] for k in e1["label_counts"]],
            "Windows": list(e1["label_counts"].values())})
        counts["Share (%)"] = counts["Windows"] / counts["Windows"].sum() * 100
        counts = counts.sort_values("Windows", ascending=False)
        st.dataframe(counts, hide_index=True, width="stretch", column_config={
            "Windows": st.column_config.NumberColumn(format="localized"),
            "Share (%)": st.column_config.NumberColumn(format="%.2f")})
        st.caption("C0 is normal operation. U clusters are groups of mutually similar "
                   "unusual windows found in a second, sparser DBSCAN pass. Isolated "
                   "windows resemble nothing else.")

    st.subheader("Study design")
    c = st.columns(3)
    with c[0].container(border=True):
        st.markdown("**:material/hub: Algorithm**")
        st.markdown("DBSCAN only, applied at two density levels. Parameters come from "
                    "the data: MinPts from the dimensionality, epsilon from the knee of "
                    "the k-distance curve.")
    with c[1].container(border=True):
        st.markdown("**:material/counter_1: Experiment 1**")
        st.markdown("Ten power-quality features only. Renewable levels are compared "
                    "*after* clustering, so they cannot shape the clusters.")
    with c[2].container(border=True):
        st.markdown("**:material/counter_2: Experiment 2**")
        st.markdown("The same features plus renewable penetration, regional solar and "
                    "wind generation. Exploratory, and partly circular by design.")
    st.info(
        "**Reading the results.** DBSCAN noise and small clusters are *unusual* windows, "
        "not confirmed disturbances: every 10-minute mean of voltage, THD, unbalance and "
        "frequency stayed inside EN 50160 limits. Associations with renewable levels are "
        "statistical, not causal.", icon=":material/info:")


def page_dataset(df: pd.DataFrame, summ: dict) -> None:
    page_header("Data", "Dataset",
                "Real measurements only: the SPARK facility meter (KIT) joined with "
                "regional generation and load from SMARD. Nothing is simulated.")
    possible = pd.Series({m: pd.Period(f"{C.YEAR}-{m:02d}").days_in_month * 144
                          for m in range(1, 13)})
    c = st.columns(4)
    c[0].metric("Analysis windows (10 min)", f"{len(df):,}", border=True)
    c[1].metric("Share of the year covered", f"{len(df) / possible.sum():.0%}",
                border=True)
    c[2].metric("Days with data", f"{df.index.normalize().nunique():,}", border=True)
    c[3].metric("Windows with rooftop-PV data", f"{int(df.pv_power_kw.notna().sum()):,}",
                border=True, help="The PV meter covers 1 January to 9 April 2024.")

    with card("Sources"):
        st.markdown(
            "| Source | Content | Resolution | Period used | Licence |\n|---|---|---|---|---|\n"
            "| SPARK `EPI_TotalLoad` (KIT, DOI 10.35097/bjdg3m3rg5jv3skk) | Facility "
            "connection-point meter: voltages, currents, frequency, THD, harmonics, P, Q | "
            "5 s | 2024 | CC BY 4.0 |\n"
            "| SPARK `EPI_PV` | AC power of the 10.72 kWp rooftop PV | 5 s | "
            "1 Jan - 9 Apr 2024 | CC BY 4.0 |\n"
            "| SPARK exogenous / SMARD | TransnetBW generation by technology | 15 min | "
            "2024 | CC BY 4.0 |\n"
            "| SMARD (Bundesnetzagentur) | TransnetBW grid load | 15 min | 2024 | "
            "CC BY 4.0 |")

    with st.container(border=True):
        cov = (df.groupby(df.index.month).size() / possible).fillna(0) * 100
        chart(V.fig_bars(V.MONTHS, cov.values,
                         "Coverage per month (share of possible 10-minute windows)",
                         "month (2024)", "coverage (%)",
                         text=[f"{v:.0f}%" for v in cov.values]))
        st.caption("December has almost no data, so seasonal statements stop at "
                   "November.")

    st.subheader("Feature catalogue")
    cat = pd.read_csv(C.PROCESSED / "feature_catalogue.csv")
    cat.insert(1, "clustering input", cat.feature.isin(PQ_CLUSTER_FEATURES))
    st.dataframe(cat, hide_index=True, width="stretch", column_config={
        "feature": "Feature", "kind": "Kind", "unit": "Unit",
        "clustering input": st.column_config.CheckboxColumn(
            "Clustering input", help="One of the 10 features in the Experiment 1 matrix."),
        "meaning": st.column_config.TextColumn("Meaning", width="large")})
    rep = C.REPORTS / "preprocessing_report.md"
    if rep.exists():
        with st.expander("Preprocessing report", icon=":material/description:"):
            st.markdown(rep.read_text(encoding="utf-8", errors="replace"))


def page_power_quality(df: pd.DataFrame, summ: dict) -> None:
    page_header("Data", "Power quality",
                "Distribution, daily rhythm and year-long trace of each power-quality "
                "feature, with the EN 50160 limits for context.")
    col = st.selectbox("Feature", PQ_CLUSTER_FEATURES + ["severity_index", "p_kw"],
                       format_func=V.label_for, key="pq_feature")
    name = V.label_for(col, unit=False)
    a, b = st.columns(2)
    with a.container(border=True):
        chart(V.fig_hist(df[col], f"Distribution of {name.lower()}", V.label_for(col)))
    with b.container(border=True):
        hourly = df.groupby("hour_local")[col].median().rename(col)
        chart(V.fig_lines(hourly, f"Median {name.lower()} by local hour",
                          V.label_for(col), "local hour", markers=True, height=360))
    with st.container(border=True):
        chart(V.fig_lines(df[col].resample("1h").mean(), f"{name}, hourly mean",
                          V.label_for(col), "time (UTC)", height=400, width=1))

    st.subheader("EN 50160 context")
    lim = pd.DataFrame({
        "Indicator": ["Voltage deviation (%)", "Voltage THD (%)", "Voltage unbalance (%)",
                      "Frequency deviation (Hz)"],
        "Limit": [10.0, 8.0, 2.0, 0.5],
        "Largest 10-minute value": [df.v_dev_pct.abs().max(), df.thd_u_pct.max(),
                                    df.vuf_pct.max(), df.f_dev_max_hz.max()]})
    lim["Share of limit used"] = lim["Largest 10-minute value"] / lim["Limit"] * 100
    st.dataframe(lim, hide_index=True, width="stretch", column_config={
        "Limit": st.column_config.NumberColumn(format="%.1f"),
        "Largest 10-minute value": st.column_config.NumberColumn(format="%.3f"),
        "Share of limit used": st.column_config.ProgressColumn(
            format="%.0f%%", min_value=0, max_value=100)})
    st.caption(f"5-second samples below 0.90 pu: {int(df.sag_samples.sum()):,}; "
               f"above 1.10 pu: {int(df.swell_samples.sum()):,}. Power quality was "
               "compliant all year, so the clusters describe unusual operating "
               "patterns, not standard violations.")


def page_generation(df: pd.DataFrame, summ: dict) -> None:
    page_header("Data", "Renewable generation",
                "Regional generation and load in the TransnetBW control area, and the "
                "facility's own 10.72 kWp rooftop PV.")
    cols = ["solar_pv_mw", "wind_onshore_mw", "total_generation_mw", "grid_load_mw"]
    with st.container(border=True):
        chart(V.fig_lines(df[cols].resample("1D").mean(),
                          "TransnetBW daily mean generation and load", "MW",
                          colours=V.SERIES_COLOURS, height=420))
    a, b = st.columns(2)
    with a.container(border=True):
        prof = df.groupby("hour_local")[["solar_pv_mw", "wind_onshore_mw"]].mean()
        chart(V.fig_lines(prof, "Mean regional generation by local hour", "MW",
                          "local hour", markers=True, colours=V.SERIES_COLOURS))
    with b.container(border=True):
        pv = df["pv_power_kw"].dropna().resample("1h").mean()
        chart(V.fig_lines(pv, "Facility rooftop PV, hourly mean", "kW", "time (UTC)",
                          colours={"pv_power_kw": V.ORANGE}, width=1))
        st.caption("Rooftop PV data exist only from 1 January to 9 April 2024.")


def page_penetration(df: pd.DataFrame, summ: dict) -> None:
    page_header("Data", "Renewable penetration",
                "Variable renewable (solar + onshore wind) generation as a share of "
                "grid load, split into terciles that define the low / medium / high "
                "levels used everywhere else.")
    lo, hi = V.level_thresholds(df)
    a, b = st.columns([2, 3])
    with a.container(border=True):
        st.latex(r"\text{VRE penetration}=\frac{P_{solar}+P_{wind}}{P_{grid\ load}}"
                 r"\times 100")
        st.caption("Grid load is the denominator because TransnetBW is a net importer.")
        m = st.container(key="plain_metrics").columns(3)
        m[0].metric("Low", f"≤ {lo:.1f}%")
        m[1].metric("Medium", f"≤ {hi:.1f}%")
        m[2].metric("High", f"> {hi:.1f}%")
    with b.container(border=True):
        chart(V.fig_penetration_hist(df, (lo, hi)))

    st.subheader("Power quality by renewable level")
    col = st.selectbox("Indicator", ["thd_u_pct", "v_dev_pct", "v_std_pct",
                                     "rvc_max_pct", "vuf_pct", "severity_index"],
                       format_func=V.label_for, key="pen_feature")
    a, b = st.columns(2)
    with a.container(border=True):
        chart(V.fig_box_by_level(df, col, "re_level"))
    with b.container(border=True):
        chart(V.fig_box_by_level(df, col, "local_pv_level"))
    st.caption("Boxes show the quartiles; whiskers reach the most extreme window "
               "within 1.5 IQR.")

    t1, t2 = st.tabs(["Indicators by regional level", "Kruskal-Wallis test"])
    with t1:
        by = load_csv("exp1_pq_by_re_level").set_index("re_level")
        stat = st.radio("Statistic", ["median", "mean", "p95"], horizontal=True,
                        key="pen_stat",
                        format_func={"median": "Median", "mean": "Mean",
                                     "p95": "95th percentile"}.get)
        wide = by[[c for c in by.columns if c.endswith(f"_{stat}")]].T
        wide.index = [V.label_for(i[: -len(stat) - 1]) for i in wide.index]
        wide = wide.rename_axis("Indicator").reset_index()
        table(wide, labels={"low": "Low", "medium": "Medium", "high": "High"})
    with t2:
        kw = load_csv("exp1_kruskal_re_level")
        kw["indicator"] = kw["indicator"].map(V.label_for)
        table(kw, formats={"H": "%.1f", "p_naive": "%.1e", "epsilon_sq": "%.4f"})
        st.caption("Naive p-values ignore autocorrelation between consecutive windows; "
                   "read the effect size, not the p-value.")


def page_experiment(exp: str, title: str, lead: str) -> None:
    s = load_summary()[exp]
    lab = load_labels(exp)
    f, s1, s2 = s["final"], s["stage1"], s["stage2"]
    page_header("Experiments", title, lead)
    c = st.columns(5)
    c[0].metric("Windows clustered", f"{s['n_windows']:,}", border=True)
    c[1].metric("Clusters found", f["n_clusters"], border=True)
    c[2].metric("Dominant cluster", f"{f['dominant_cluster_pct']:.1f}%", border=True)
    c[3].metric("Unusual windows", f"{f['unusual_pct']:.2f}%", border=True)
    c[4].metric("Isolated noise", f"{f['isolated_noise_pct']:.2f}%", border=True)
    quality = (f"silhouette {num(s2['silhouette'])}, Davies-Bouldin "
               f"{num(s2['davies_bouldin'])} (clustered points only)"
               if s2["n_clusters"] >= 2 else
               "silhouette and Davies-Bouldin are undefined because stage 2 found a "
               "single cluster")
    st.markdown(
        f"**Stage 1** · MinPts = {s1['min_samples']}, eps = {s1['eps']:.3f} "
        f"(k-distance knee).  \n**Stage 2** · on the {s2['n_input_windows']:,} stage-1 "
        f"noise windows: MinPts = {s2['min_samples']}, eps = {s2['eps']:.3f}; {quality}.")
    st.caption("Features: " + ", ".join(V.label_for(x, unit=False) for x in s["features"]))

    t_clusters, t_levels, t_robust = st.tabs(
        [":material/bubble_chart: Clusters", ":material/solar_power: Renewable levels",
         ":material/verified: Robustness"])

    with t_clusters:
        a, b = st.columns(2)
        with a.container(border=True):
            chart(V.fig_kdistance(np.load(C.RESULTS / f"{exp}_kdist_stage1.npy"),
                                  s1["eps"], "Stage 1 k-distance"))
        with b.container(border=True):
            chart(V.fig_kdistance(np.load(C.RESULTS / f"{exp}_kdist_stage2.npy"),
                                  s2["eps"], "Stage 2 k-distance"))
        st.subheader("Cluster characteristics (medians)")
        table(load_csv(f"{exp}_cluster_profile").drop(columns="cluster"), pin="name",
              formats={"share_pct": "%.2f", "weekend_share_pct": "%.1f",
                       "hour_local": "%.0f"})
        with st.container(border=True):
            chart(V.fig_profile_heatmap(load_csv(f"{exp}_cluster_profile_robust_z"),
                                        s["label_names"]))

    with t_levels:
        a, b = st.columns(2)
        with a.container(border=True):
            chart(V.fig_cluster_share_by_level(
                load_csv(f"{exp}_cluster_by_re_level_colpct"), s["label_names"]))
        with b.container(border=True):
            chart(V.fig_rate_by_level(load_csv(f"{exp}_unusual_rate_by_re_level"),
                                      "Unusual-window rate by regional level"))
        st.subheader("Association tests (whole-day permutation)")
        perm = load_csv(f"{exp}_permutation_tests")
        perm = perm.rename(columns={perm.columns[0]: "test"})
        flag = perm.test.str.endswith("__unusual_flag")
        var = perm.test.str.replace("__unusual_flag", "", regex=False)
        perm.insert(0, "Renewable variable", var.map(
            lambda v: "Rooftop PV output (generating windows)"
            if v == "local_pv_output_generating_windows" else V.label_for(v, unit=False)))
        perm.insert(1, "Compared with", np.where(flag, "unusual vs normal",
                                                 "all cluster labels"))
        table(perm.drop(columns="test"), digits=4,
              formats={"n_perm": "%d", "n": "%d"})
        st.caption("Cramér's V measures strength of association (0 = none, 1 = perfect). "
                   "The p-value compares it with shuffled whole-day renewable profiles. "
                   "An association is not evidence of causation.")
        with st.container(border=True):
            chart(V.fig_monthly(load_dataset(), lab))

    with t_robust:
        g1 = load_csv(f"{exp}_sensitivity_stage1")
        a, b = st.columns(2)
        with a.container(border=True):
            chart(V.fig_sensitivity(g1, "noise_pct", "Stage 1: noise %"))
        with b.container(border=True):
            chart(V.fig_sensitivity(g1, "n_clusters", "Stage 1: number of clusters"))
        a, b = st.columns(2)
        with a:
            st.subheader("Stability over subsamples")
            stab = load_csv(f"{exp}_stability").drop(columns="round")
            agg = stab.agg(["mean", "min", "max"]).rename_axis("Statistic").reset_index()
            table(agg, formats={"n_clusters": "%.1f"})
            st.caption(f"Two-stage DBSCAN refitted on {len(stab)} random 80 % subsamples "
                       "and compared with the full-data labels.")
        with b:
            st.subheader("Distance metrics (stage 1)")
            table(load_csv(f"{exp}_metric_comparison"))
            st.caption("Silhouette and Davies-Bouldin are empty when a run finds a "
                       "single cluster.")
        with st.expander("K-Means and Ward reference (8,000-window subsample)"):
            table(load_csv(f"{exp}_baseline_kmeans_ward"))
            st.caption("Context only: both need k in advance and have no noise class.")


def page_clusters(df: pd.DataFrame, summ: dict) -> None:
    page_header("Analysis", "Cluster map",
                "Where the unusual windows sit: projected to two principal components, "
                "and along the year for any feature.")
    exp = st.radio("Experiment", list(EXPERIMENTS), horizontal=True, key="viz_exp",
                   format_func=EXPERIMENTS.get)
    lab = load_labels(exp)
    evr = summ[exp]["pca_explained_variance_3"]
    with st.container(border=True):
        chart(V.fig_pca_scatter(lab))
        st.caption(f"PC1 and PC2 carry {evr[0] + evr[1]:.0%} of the variance. "
                   "Clustering is done in the full feature space; the dominant cluster "
                   "is subsampled to 6,000 points for display.")
    col = st.selectbox("Timeline feature", PQ_CLUSTER_FEATURES, index=2,
                       format_func=V.label_for, key="viz_feature")
    with st.container(border=True):
        chart(V.fig_timeline(df, lab, col))


def page_unusual(df: pd.DataFrame, summ: dict) -> None:
    page_header("Analysis", "Unusual windows",
                "Everything DBSCAN left outside the dominant cluster: when it happens, "
                "how long it lasts and which windows are fully isolated.")
    exp = st.radio("Experiment", list(EXPERIMENTS), horizontal=True, key="noise_exp",
                   format_func=EXPERIMENTS.get)
    lab = load_labels(exp)
    un = df.join(lab[["label", "label_name"]], how="inner")
    un = un[un.label != 0]
    ep = load_csv(f"{exp}_unusual_episodes")
    c = st.columns(4)
    c[0].metric("Unusual windows", f"{len(un):,}", border=True,
                help="Windows not in the dominant cluster.")
    c[1].metric("Consecutive episodes", f"{len(ep):,}", border=True)
    c[2].metric("Median episode", f"{ep.duration_min.median():.0f} min", border=True)
    c[3].metric("Longest episode", f"{ep.duration_min.max():.0f} min", border=True)

    a, b = st.columns(2)
    with a.container(border=True):
        by_hour = pd.crosstab(un.hour_local, un.label_name).reindex(
            range(24), fill_value=0)
        order = sorted(by_hour.columns, key=lambda n: -by_hour[n].sum())
        chart(V.fig_stacked_counts(by_hour[order], "Unusual windows by local hour",
                                   "local hour", "windows"))
    with b.container(border=True):
        chart(V.fig_rate_by_level(load_csv(f"{exp}_unusual_rate_by_local_pv_level"),
                                  "Unusual rate by local PV level (Jan-Apr)"))
    a, b = st.columns([2, 3])
    with a.container(border=True):
        dur = ep.duration_min.value_counts().sort_index()
        chart(V.fig_bars([f"{int(d)} min" for d in dur.index], dur.values,
                         "Duration of consecutive unusual episodes", "episode duration",
                         "episodes", text=[f"{v:,}" for v in dur.values], height=400))
    with b:
        st.subheader("Isolated noise windows (label -1)")
        iso = un[un.label == -1][["hour_local", "vre_penetration_pct"]
                                 + PQ_CLUSTER_FEATURES]
        iso = iso.rename_axis("Timestamp (UTC)").reset_index()
        table(iso, formats={"hour_local": "%d"}, pin="Timestamp (UTC)", height=350,
              labels={"Timestamp (UTC)": "Timestamp (UTC)"})


def page_comparison(df: pd.DataFrame, summ: dict) -> None:
    page_header("Analysis", "Experiment comparison",
                "How much the picture changes when renewable variables become part of "
                "the distance (Experiment 2) instead of being compared afterwards "
                "(Experiment 1).")
    cmp_ = _cached(_read_json, C.RESULTS / "comparison.json")
    c = st.columns(4)
    c[0].metric("Adjusted Rand index", f"{cmp_['ari']:.3f}", border=True)
    c[1].metric("Unusual-set Jaccard", f"{cmp_['unusual_jaccard']:.3f}", border=True)
    c[2].metric("Unusual in both", f"{cmp_['unusual_both']:,}", border=True)
    c[3].metric("Unusual in one only",
                f"{cmp_['unusual_only_a'] + cmp_['unusual_only_b']:,}", border=True,
                help=f"{cmp_['unusual_only_a']:,} only in Experiment 1, "
                     f"{cmp_['unusual_only_b']:,} only in Experiment 2.")
    a, b = st.columns([2, 3])
    with a:
        st.subheader("Side by side")
        side = load_csv("comparison_summary").set_index("experiment").T
        whole = ["dimensions", "n_clusters"]
        side = pd.DataFrame(
            {EXPERIMENTS[e].split(" · ")[0]: [
                num(v, ".0f" if i in whole else ".3f") for i, v in side[e].items()]
             for e in side.columns},
            index=[COMPARISON_ROWS.get(i, i) for i in side.index])
        st.dataframe(side.rename_axis("Measure").reset_index(), hide_index=True,
                     width="stretch", height=35 * (len(side) + 1) + 3)
    with b.container(border=True):
        chart(V.fig_migration(load_csv("comparison_migration_exp1_vs_exp2"),
                              summ["exp1"]["label_names"], summ["exp2"]["label_names"]))
        st.caption("Cell shading uses a log scale so the small off-diagonal groups "
                   "stay visible next to the dominant cluster.")
    st.subheader("Effect of the renewable block weight")
    table(load_csv("comparison_weight_sweep"), formats={"re_share": "%.1f"})
    st.caption("Share of the squared distance carried by the three renewable features "
               "in Experiment 2. The study uses 0.2.")


def page_correlation(df: pd.DataFrame, summ: dict) -> None:
    page_header("Analysis", "Correlation",
                "Rank correlation between power-quality indicators and renewable "
                "variables, before and after removing the shared daily rhythm.")
    corr = load_csv("correlation_pq_vs_renewables")
    a, b = st.columns(2)
    with a.container(border=True):
        chart(V.fig_correlation(corr, "spearman"))
    with b.container(border=True):
        chart(V.fig_correlation(corr, "spearman_within_hour"))
    st.caption("Right: both variables have their hour-of-day x weekend mean removed, "
               "which takes out the shared daily rhythm.")
    with st.container(border=True):
        chart(V.fig_feature_correlation(df, PQ_CLUSTER_FEATURES))


def page_explorer(df: pd.DataFrame, summ: dict) -> None:
    page_header("Tools", "Parameter explorer",
                "Run a single-stage DBSCAN live on a random subsample and see how MinPts "
                "and epsilon move the boundary between cluster and noise.")
    with st.container(border=True):
        c = st.columns([1.3, 1, 1, 1.2])
        use_re = c[0].toggle("Include renewable features", key="px_re")
        n = c[1].select_slider("Sample size", [3000, 6000, 12000], 6000, key="px_n")
        metric = c[2].selectbox("Distance metric",
                                ["euclidean", "manhattan", "chebyshev"], key="px_metric")
        share = c[3].slider("Renewable weight share", 0.1, 0.5, 0.2, 0.1, key="px_share",
                            disabled=not use_re,
                            help="Only used when renewable features are included.")
        feats = PQ_CLUSTER_FEATURES + (RE_FEATURES if use_re else [])
        c = st.columns(2)
        # the default follows the dimensionality, so each feature set has its own slider
        ms = c[0].slider("MinPts", 3, 60, 2 * len(feats), key=f"px_ms_{len(feats)}",
                         help="The study uses 2 x the number of dimensions.")
        eps = c[1].slider("Epsilon", 0.2, 5.0, 1.5, 0.05, key="px_eps")
    path = C.PROCESSED / "analysis_dataset_10min.parquet"
    out, m, knee = explore(tuple(feats), share if use_re else None, eps, ms, n, metric,
                           path.stat().st_mtime)
    k = st.columns(4)
    k[0].metric("Clusters", m["n_clusters"], border=True)
    k[1].metric("Noise", f"{m['noise_pct']:.2f}%", border=True)
    k[2].metric("Silhouette", num(m["silhouette"], ".3f"), border=True,
                help="Needs at least two clusters; computed on clustered points only.")
    k[3].metric("Knee epsilon for this MinPts", f"{knee:.3f}", border=True,
                help="The data-driven epsilon the pipeline would choose.")
    with st.container(border=True):
        chart(V.fig_explore_scatter(out))


def _group_of(path: Path) -> str:
    if path.suffix == ".md":
        return "Reports"
    return {"exp1": "Experiment 1", "exp2": "Experiment 2"}.get(
        path.name.split("_")[0], "Comparison and correlation")


def _zip_results() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(C.RESULTS.glob("*.csv")) + sorted(C.REPORTS.glob("*.md")):
            z.write(f, f"{f.parent.name}/{f.name}")
    return buf.getvalue()


def _dataset_with_labels() -> bytes:
    full = load_dataset()
    for exp in EXPERIMENTS:
        full = full.join(load_labels(exp)[["label"]].rename(
            columns={"label": f"{exp}_label"}))
    return full.to_csv().encode("utf-8")


def page_downloads(df: pd.DataFrame, summ: dict) -> None:
    page_header("Tools", "Downloads",
                "Every result table and report the pipeline produced. Preview a file "
                "before downloading it, or take everything at once.")
    a, b = st.columns(2)
    with a.container(border=True):
        st.markdown("**:material/folder_zip: Everything**")
        st.caption("All result tables (CSV) and reports (Markdown) in one archive.")
        st.download_button("Download results.zip", _zip_results,
                           file_name="pq_dbscan_results.zip", mime="application/zip",
                           icon=":material/download:", type="primary", key="dl_zip",
                           on_click="ignore")
    with b.container(border=True):
        st.markdown("**:material/table: Analysis dataset with cluster labels**")
        st.caption(f"{len(df):,} windows x {df.shape[1] + 2} columns, with the labels of "
                   "both experiments attached.")
        st.download_button("Download dataset.csv", _dataset_with_labels,
                           file_name="analysis_dataset_with_labels.csv", mime="text/csv",
                           icon=":material/download:", key="dl_full", on_click="ignore")

    st.subheader("Browse files")
    files = sorted(C.RESULTS.glob("*.csv")) + sorted(C.REPORTS.glob("*.md"))
    groups = ["Experiment 1", "Experiment 2", "Comparison and correlation", "Reports"]
    for group, tab in zip(groups, st.tabs(groups)):
        members = [f for f in files if _group_of(f) == group]
        if not members:
            tab.caption("No files yet.")
            continue
        with tab:
            a, b = st.columns([3, 1], vertical_alignment="bottom")
            f = a.selectbox("File", members, format_func=lambda p: p.name,
                            key=f"dl_pick_{group}")
            b.download_button("Download", f.read_bytes(), file_name=f.name,
                              icon=":material/download:", key=f"dl_{group}",
                              width="stretch", on_click="ignore")
            if f.suffix == ".csv":
                data = _cached(_read_csv, f)
                st.caption(f"{len(data):,} rows x {data.shape[1]} columns · "
                           f"{f.stat().st_size / 1024:.1f} KB")
                st.dataframe(data.head(200), hide_index=True, width="stretch")
            else:
                st.caption(f"{f.stat().st_size / 1024:.1f} KB")
                with st.container(border=True, height=420):
                    st.markdown(f.read_text(encoding="utf-8", errors="replace"))


RENDER = [
    page_overview, page_dataset, page_power_quality, page_generation, page_penetration,
    lambda df, summ: page_experiment(
        "exp1", "Experiment 1: power quality only",
        "DBSCAN on ten power-quality features. Renewable variables are joined after "
        "clustering, so they cannot shape the clusters."),
    lambda df, summ: page_experiment(
        "exp2", "Experiment 2: power quality + renewables",
        "The same ten features plus renewable penetration, regional solar and wind, "
        "weighted to 20 % of the squared distance. Exploratory and partly circular."),
    page_clusters, page_unusual, page_comparison, page_correlation, page_explorer,
    page_downloads,
]


def main() -> None:
    inject_css(is_dark())
    if "page" not in st.session_state:          # deep link: ?page=<slug>
        wanted = st.query_params.get("page")
        st.session_state["page"] = PAGES[SLUGS.index(wanted)] if wanted in SLUGS \
            else PAGES[0]
    with st.sidebar:
        st.html('<div class="pq-brand"><div class="pq-brand-mark">⚡</div><div>'
                '<div class="pq-brand-name">PQ x Renewables</div>'
                '<div class="pq-brand-sub">DBSCAN cluster study · 2024</div></div></div>')
        page = st.radio("Section", PAGES, key="page", label_visibility="collapsed")
        st.html('<div class="pq-side-note">Data: SPARK (KIT) and SMARD '
                '(Bundesnetzagentur), CC BY 4.0.<br>Unusual does not mean '
                'non-compliant.</div>')
    st.query_params["page"] = SLUGS[PAGES.index(page)]

    if not results_ready():
        st.error("**Results not found.** Run `python run_pipeline.py` from the project "
                 "root first, then reload this page.", icon=":material/error:")
        return
    RENDER[PAGES.index(page)](load_dataset(), load_summary())


main()
