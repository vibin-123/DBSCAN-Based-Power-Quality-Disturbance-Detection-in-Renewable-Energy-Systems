"""Stage I - figures.

Plotly builders (``fig_*``) are shared by the Streamlit dashboard; ``main``
also writes static PNG copies with matplotlib for the written report.

Colour is assigned by job: fixed categorical hues for cluster labels, one-hue
ramps for ordered levels and magnitudes, and a blue-grey-red diverging scale
for signed quantities. ``themed`` swaps every colour for its dark-surface step.

Run:  python -m src.visualization   (after src.experiments)
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import config as C

LEVEL_ORDER = ["low", "medium", "high"]

# ------------------------------- palette -----------------------------------
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MAGENTA, GREEN, VIOLET, RED = "#e87ba4", "#008300", "#6250d6", "#e34948"
NEUTRAL = "#8f959e"          # de-emphasised series (dominant cluster, reference)
REFERENCE = "#898781"        # threshold / reference lines

# fixed colour per label so every chart uses the same identity
LABEL_COLOURS = {"C0 dominant": NEUTRAL, "Isolated (noise)": AQUA,
                 "U1 unusual pattern": BLUE, "U2 unusual pattern": ORANGE,
                 "U3 unusual pattern": YELLOW, "U4 unusual pattern": MAGENTA,
                 "U5 unusual pattern": RED, "C1 dense": GREEN,
                 "C2 dense": VIOLET, "C3 dense": "#52514e"}
# ordered levels: one hue, light -> dark; "none" is neutral
LEVEL_COLOURS = {"none": "#c3c2b7", "low": "#86b6ef", "medium": "#3987e5",
                 "high": "#184f95"}
SERIES_COLOURS = {"solar_pv_mw": ORANGE, "wind_onshore_mw": BLUE,
                  "total_generation_mw": AQUA, "grid_load_mw": NEUTRAL}

# Heatmap scales stop at mid tones so one ink colour stays readable on every
# cell: dark ink on the light-surface scales, light ink on the dark ones.
INK, _INK_DARK = "#0b0b0b", "#f0efec"
SEQUENTIAL = [[0.0, "#eaf2fd"], [1.0, "#5598e7"]]
DIVERGING = [[0.0, "#5598e7"], [0.5, "#f0efec"], [1.0, "#e9807f"]]
_SEQUENTIAL_DARK = [[0.0, "#16263f"], [1.0, "#256abf"]]
_DIVERGING_DARK = [[0.0, "#256abf"], [0.5, "#2c2c2a"], [1.0, "#c23a3a"]]
# light step -> dark step of the same hue
_DARK = {BLUE: "#3987e5", ORANGE: "#d95926", AQUA: "#199e70", YELLOW: "#c98500",
         MAGENTA: "#d55181", GREEN: "#0ca30c", VIOLET: "#9085e9", RED: "#e66767",
         "#52514e": "#c3c2b7", "#c3c2b7": "#52514e"}

FEATURE_LABELS = {
    "v_dev_pct": ("Voltage deviation", "%"),
    "v_std_pct": ("Voltage fluctuation", "% of Un"),
    "rvc_max_pct": ("Largest voltage step", "% of Un"),
    "vuf_pct": ("Voltage unbalance", "%"),
    "f_dev_abs_hz": ("Frequency deviation", "Hz"),
    "thd_u_pct": ("Voltage THD", "%"),
    "u_h5_pct": ("5th voltage harmonic", "%"),
    "thd_i_pct": ("Current THD", "%"),
    "i_rms_a": ("Phase current", "A"),
    "i_n_a": ("Neutral current", "A"),
    "severity_index": ("Severity index", "share of EN 50160 limit"),
    "p_kw": ("Active power", "kW"),
    "pf": ("Power factor", ""),
    "vre_penetration_pct": ("Solar + wind penetration", "%"),
    "solar_penetration_pct": ("Solar penetration", "%"),
    "wind_penetration_pct": ("Wind penetration", "%"),
    "vre_share_of_generation_pct": ("Solar + wind share of generation", "%"),
    "pv_output_pct_of_rated": ("Rooftop PV output", "% of rated"),
    "pv_power_kw": ("Rooftop PV power", "kW"),
    "solar_pv_mw": ("Regional solar", "MW"),
    "wind_onshore_mw": ("Regional onshore wind", "MW"),
    "grid_load_mw": ("Grid load", "MW"),
    "total_generation_mw": ("Total generation", "MW"),
    "hour_local": ("Local hour", ""),
}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
          "Nov", "Dec"]


def colour_for(name: str) -> str:
    return LABEL_COLOURS.get(name, "#7f7f7f")


def label_for(column: str, unit: bool = True) -> str:
    """Readable name of a dataset column, e.g. 'Voltage THD (%)'."""
    name, u = FEATURE_LABELS.get(column, (column.replace("_", " "), ""))
    return f"{name} ({u})" if unit and u else name


# ------------------------------- styling -----------------------------------
_FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
_AXIS = dict(gridcolor="#e1e0d9", linecolor="#c3c2b7", zeroline=False,
             ticks="outside", tickcolor="#c3c2b7", ticklen=4)
TEMPLATE = go.layout.Template(layout=dict(
    font=dict(family=_FONT, size=13, color="#0b0b0b"),
    paper_bgcolor="#fcfcfb", plot_bgcolor="#fcfcfb",
    colorway=[BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED],
    xaxis=_AXIS, yaxis=_AXIS))


def _style(fig: go.Figure, title: str, height: int = 380,
           legend: bool = True) -> go.Figure:
    """Shared chrome: left-aligned title, legend above the plot, thin marks.

    A single series needs no legend (the title names it): ``legend=False``.
    """
    fig.update_layout(
        template=TEMPLATE, height=height, showlegend=legend,
        title=dict(text=title, x=0, xanchor="left", y=0.97, yanchor="top",
                   font=dict(size=15)),
        margin=dict(l=8, r=8, t=78 if legend else 52, b=8),
        legend=dict(orientation="h", x=0, xanchor="left", y=1.0,
                    yanchor="bottom", title_text=""),
        hoverlabel=dict(font_size=12), bargap=0.35, bargroupgap=0.08)
    fig.update_traces(selector=dict(type="bar"), marker_cornerradius=4)
    fig.update_traces(selector=dict(type="heatmap"), textfont_color=INK)
    return fig


def themed(fig: go.Figure, dark: bool = False) -> go.Figure:
    """Return the figure with every colour stepped for a dark surface."""
    if not dark:
        return fig
    fig = go.Figure(fig)
    for tr in fig.data:
        if tr.type == "heatmap":
            tr.colorscale = (_DIVERGING_DARK if tr.meta == "diverging"
                             else _SEQUENTIAL_DARK)
            tr.textfont.color = _INK_DARK
            continue
        for part in ("marker", "line"):
            obj = getattr(tr, part, None)
            col = getattr(obj, "color", None)
            if isinstance(col, str) and col in _DARK:
                obj.color = _DARK[col]
    return fig


def _reference_line(fig: go.Figure, axis: str, value: float, text: str,
                    position: str = "top left") -> None:
    add = fig.add_hline if axis == "y" else fig.add_vline
    add(**{axis: value}, line_dash="dash", line_color=REFERENCE, line_width=1.5,
        annotation_text=text, annotation_position=position)


# ------------------------------- plotly ------------------------------------
def fig_kdistance(kd: np.ndarray, eps: float, title: str) -> go.Figure:
    x = np.linspace(0, 100, len(kd))
    fig = go.Figure(go.Scatter(x=x, y=kd, mode="lines", name="k-distance",
                               line=dict(color=BLUE, width=2),
                               hovertemplate="%{x:.1f}% of windows<br>"
                                             "k-distance %{y:.3f}<extra></extra>"))
    _reference_line(fig, "y", eps, f"selected eps = {eps:.3f}")
    fig.update_layout(xaxis_title="windows sorted by k-distance (%)",
                      yaxis_title="distance to k-th neighbour")
    return _style(fig, title, legend=False)


def fig_pca_scatter(lab: pd.DataFrame, max_normal: int = 6000,
                    seed: int = C.RANDOM_STATE) -> go.Figure:
    """PC1 / PC2 scatter; the dominant cluster is subsampled for speed."""
    normal = lab[lab.label == 0]
    if len(normal) > max_normal:
        normal = normal.sample(max_normal, random_state=seed)
    d = pd.concat([normal, lab[lab.label != 0]])
    order = sorted(d.label_name.unique(), key=lambda n: (n != "C0 dominant", n))
    counts = lab.label_name.value_counts()
    fig = go.Figure()
    for name in order:
        g = d[d.label_name == name]
        dominant, noise = name == "C0 dominant", name == "Isolated (noise)"
        fig.add_trace((go.Scattergl if dominant else go.Scatter)(
            x=g.pc1, y=g.pc2, mode="markers",
            name=f"{name} ({counts[name]:,})", customdata=g.index,
            marker=dict(color=colour_for(name),
                        size=4 if dominant else 9 if noise else 6,
                        symbol="diamond" if noise else "circle",
                        opacity=0.4 if dominant else 0.85),
            hovertemplate=f"{name}<br>%{{customdata|%d %b %Y %H:%M}} UTC"
                          "<br>PC1 %{x:.2f}, PC2 %{y:.2f}<extra></extra>"))
    fig.update_layout(xaxis_title="PC1", yaxis_title="PC2")
    return _style(fig, "DBSCAN labels in PCA space (visualisation only)", 520)


def fig_rate_by_level(rate: pd.DataFrame, title: str) -> go.Figure:
    r = rate.copy()
    lev = r.columns[0]
    fig = go.Figure()
    for col, name, colour in (
            ("raw_rate", "raw", BLUE),
            ("standardised_rate", "standardised for hour x weekend", NEUTRAL)):
        fig.add_bar(x=r[lev], y=r[col] * 100, name=name, marker_color=colour,
                    text=[f"{v:.2f}%" for v in r[col] * 100],
                    textposition="outside", cliponaxis=False,
                    hovertemplate=f"%{{x}} · {name}<br>%{{y:.2f}}% of windows"
                                  "<extra></extra>")
    fig.update_layout(barmode="group", yaxis_title="% of windows",
                      xaxis_title="renewable level")
    return _style(fig, title)


def fig_cluster_share_by_level(colpct: pd.DataFrame, names: dict) -> go.Figure:
    """Share of each non-dominant label within each renewable level."""
    d = colpct.set_index(colpct.columns[0])
    fig = go.Figure()
    for label in d.index:
        if int(label) == 0:
            continue
        n = names.get(str(int(label)), str(label))
        fig.add_bar(x=list(d.columns), y=d.loc[label].values, name=n,
                    marker_color=colour_for(n),
                    hovertemplate=f"{n}<br>%{{x}}: %{{y:.2f}}% of windows"
                                  "<extra></extra>")
    fig.update_layout(barmode="stack", yaxis_title="% of windows in level",
                      xaxis_title="regional renewable level")
    return _style(fig, "Unusual-pattern share by renewable level")


def fig_profile_heatmap(z: pd.DataFrame, names: dict) -> go.Figure:
    d = z.set_index(z.columns[0])
    y = [names.get(str(int(i)), str(i)) for i in d.index]
    fig = go.Figure(go.Heatmap(
        z=d.values, x=[label_for(c, unit=False) for c in d.columns], y=y,
        colorscale=DIVERGING, meta="diverging", zmid=0, zmin=-6, zmax=6,
        text=np.round(d.values, 1), texttemplate="%{text}", xgap=2, ygap=2,
        colorbar=dict(title="robust z", thickness=12),
        hovertemplate="%{y}<br>%{x}: robust z %{z:.2f}<extra></extra>"))
    return _style(fig, "Cluster medians relative to all windows "
                       "((median - overall median) / IQR)", 360, legend=False)


def fig_sensitivity(grid: pd.DataFrame, value: str, title: str) -> go.Figure:
    p = grid.pivot(index="min_samples", columns="eps_factor", values=value)
    fig = go.Figure(go.Heatmap(
        z=p.values, x=[f"{c:g} x knee" for c in p.columns],
        y=[str(i) for i in p.index], colorscale=SEQUENTIAL, meta="sequential",
        text=np.round(p.values, 2), texttemplate="%{text}", xgap=2, ygap=2,
        colorbar=dict(title=value.replace("_", " "), thickness=12),
        hovertemplate="MinPts %{y}, eps %{x}<br>%{z:.2f}<extra></extra>"))
    fig.update_layout(xaxis_title="epsilon", yaxis_title="MinPts",
                      yaxis_type="category")
    return _style(fig, title, 320, legend=False)


def fig_timeline(df: pd.DataFrame, lab: pd.DataFrame, column: str) -> go.Figure:
    d = df[[column]].join(lab[["label", "label_name"]], how="inner")
    hourly = d[column].resample("1h").mean()
    name = label_for(column)
    fig = go.Figure(go.Scatter(x=hourly.index, y=hourly.values, mode="lines",
                               name="hourly mean",
                               line=dict(color=NEUTRAL, width=1)))
    for n, g in d[d.label != 0].groupby("label_name"):
        fig.add_scatter(x=g.index, y=g[column], mode="markers", name=n,
                        marker=dict(color=colour_for(n), size=6,
                                    symbol="diamond" if n == "Isolated (noise)"
                                    else "circle"))
    fig.update_layout(yaxis_title=name, xaxis_title="time (UTC)")
    return _style(fig, f"{label_for(column, unit=False)} over time, unusual "
                       "windows marked", 420)


def fig_monthly(df: pd.DataFrame, lab: pd.DataFrame) -> go.Figure:
    """Two panels on a shared month axis (not two y-scales on one plot)."""
    d = df[["vre_penetration_pct"]].join(lab["label"], how="inner")
    m = d.groupby(d.index.month).agg(
        unusual_pct=("label", lambda s: (s != 0).mean() * 100),
        vre=("vre_penetration_pct", "mean"))
    x = [MONTHS[i - 1] for i in m.index]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12)
    fig.add_bar(x=x, y=m.unusual_pct, name="unusual windows (%)",
                marker_color=ORANGE, row=1, col=1,
                hovertemplate="%{x}: %{y:.2f}% unusual<extra></extra>")
    fig.add_scatter(x=x, y=m.vre, name="mean regional solar + wind penetration (%)",
                    mode="lines+markers", line=dict(color=BLUE, width=2),
                    marker=dict(size=8), row=2, col=1,
                    hovertemplate="%{x}: %{y:.1f}% penetration<extra></extra>")
    fig.update_yaxes(title_text="unusual (%)", rangemode="tozero", row=1, col=1)
    fig.update_yaxes(title_text="penetration (%)", rangemode="tozero", row=2, col=1)
    fig.update_xaxes(title_text="month (2024)", row=2, col=1)
    return _style(fig, "Monthly unusual-window rate and renewable penetration", 460)


def fig_hist(values: pd.Series, title: str, xlabel: str, bins: int = 80,
             colour: str = BLUE) -> go.Figure:
    """Histogram binned here, so only the bin counts go to the browser."""
    v = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna()
    counts, edges = np.histogram(v, bins=bins)
    fig = go.Figure(go.Bar(
        x=(edges[:-1] + edges[1:]) / 2, y=counts, width=np.diff(edges),
        marker_color=colour, name="windows",
        customdata=np.column_stack([edges[:-1], edges[1:]]),
        hovertemplate="%{customdata[0]:.3g} to %{customdata[1]:.3g}<br>"
                      "%{y:,} windows<extra></extra>"))
    fig.update_layout(xaxis_title=xlabel, yaxis_title="windows")
    fig = _style(fig, title, 360, legend=False)
    fig.update_traces(marker_cornerradius=0)
    return fig.update_layout(bargap=0.04)


def fig_penetration_hist(df: pd.DataFrame, thresholds: tuple[float, float]
                         ) -> go.Figure:
    fig = fig_hist(df["vre_penetration_pct"],
                   "Regional solar + wind penetration (share of grid load)",
                   "solar + wind penetration (%)")
    lo, hi = thresholds
    # the two terciles sit close together on this axis: one shared label
    _reference_line(fig, "x", lo, "")
    _reference_line(fig, "x", hi, f"terciles: low up to {lo:.1f}%, medium up to "
                                  f"{hi:.1f}%, high above", "top right")
    return fig


def fig_box_by_level(df: pd.DataFrame, column: str, level: str = "re_level"
                     ) -> go.Figure:
    """Box per level, always in the order none / low / medium / high.

    Quartiles are computed here (Tukey whiskers), so the browser receives five
    numbers per box instead of every window.
    """
    d = df.dropna(subset=[level, column])
    groups = d[column].groupby(d[level].astype(str))
    order = [l for l in ["none"] + LEVEL_ORDER if l in groups.groups]
    fig = go.Figure()
    for lev in order:
        v = groups.get_group(lev)
        q1, med, q3 = v.quantile([0.25, 0.5, 0.75])
        lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        fig.add_trace(go.Box(
            x=[lev], q1=[q1], median=[med], q3=[q3],
            lowerfence=[v[v >= lo].min()], upperfence=[v[v <= hi].max()],
            name=f"{lev} ({len(v):,})", marker_color=LEVEL_COLOURS[lev],
            line=dict(color=LEVEL_COLOURS[lev], width=2), boxpoints=False))
    what = ("local PV level (Jan-Apr)" if level == "local_pv_level"
            else "regional renewable level")
    fig.update_layout(xaxis_title=what, yaxis_title=label_for(column),
                      xaxis=dict(categoryorder="array", categoryarray=order))
    return _style(fig, f"{label_for(column, unit=False)} by {what}", 360,
                  legend=False)


def fig_correlation(corr: pd.DataFrame, value: str = "spearman") -> go.Figure:
    p = corr.pivot(index="pq_indicator", columns="renewable_variable", values=value)
    title = {"spearman": "Spearman correlation, raw",
             "spearman_within_hour": "Spearman correlation, hour-of-day x weekend "
                                     "mean removed"}.get(value, value)
    fig = go.Figure(go.Heatmap(
        z=p.values, x=[label_for(c, unit=False) for c in p.columns],
        y=[label_for(i, unit=False) for i in p.index],
        colorscale=DIVERGING, meta="diverging", zmid=0, zmin=-0.6, zmax=0.6,
        text=np.round(p.values, 2), texttemplate="%{text}", xgap=2, ygap=2,
        colorbar=dict(title="rho", thickness=12),
        hovertemplate="%{y} vs %{x}<br>rho = %{z:.3f}<extra></extra>"))
    return _style(fig, title, 420, legend=False)


def fig_feature_correlation(df: pd.DataFrame, columns: list[str]) -> go.Figure:
    c = df[columns].corr(method="spearman")
    names = [label_for(x, unit=False) for x in columns]
    fig = go.Figure(go.Heatmap(
        z=c.values, x=names, y=names, colorscale=DIVERGING, meta="diverging",
        zmid=0, zmin=-1, zmax=1, text=np.round(c.values, 2),
        texttemplate="%{text}", xgap=2, ygap=2,
        colorbar=dict(title="rho", thickness=12),
        hovertemplate="%{y} vs %{x}<br>rho = %{z:.2f}<extra></extra>"))
    fig.update_layout(yaxis_autorange="reversed")
    return _style(fig, "Spearman correlation between power-quality features", 520,
                  legend=False)


def fig_migration(mig: pd.DataFrame, names_a: dict | None = None,
                  names_b: dict | None = None) -> go.Figure:
    d = mig.set_index(mig.columns[0])
    rows = [(names_a or {}).get(str(i), str(i)) for i in d.index]
    cols = [(names_b or {}).get(str(c), str(c)) for c in d.columns]
    fig = go.Figure(go.Heatmap(
        z=np.log10(d.values + 1), x=cols, y=rows, colorscale=SEQUENTIAL,
        meta="sequential", text=d.values, texttemplate="%{text:,}",
        showscale=False, xgap=2, ygap=2,
        hovertemplate="Exp 1: %{y}<br>Exp 2: %{x}<br>%{text:,} windows"
                      "<extra></extra>"))
    fig.update_layout(xaxis_title="Experiment 2 label",
                      yaxis_title="Experiment 1 label",
                      xaxis_type="category", yaxis_type="category")
    return _style(fig, "Label migration: Experiment 1 (rows) to Experiment 2 "
                       "(columns), window counts", 360, legend=False)


def fig_lines(data: pd.DataFrame | pd.Series, title: str, ytitle: str,
              xtitle: str = "", markers: bool = False,
              colours: dict | None = None, names: dict | None = None,
              height: int = 380, width: float = 2) -> go.Figure:
    """One line per column, coloured by the entity it shows.

    Use ``width=1`` for dense series (thousands of points), where a 2 px
    stroke would fill the plot.
    """
    d = data.to_frame() if isinstance(data, pd.Series) else data
    palette = TEMPLATE.layout.colorway
    fig = go.Figure()
    for i, col in enumerate(d.columns):
        fig.add_scatter(
            x=d.index, y=d[col], name=(names or {}).get(col, label_for(col)),
            mode="lines+markers" if markers else "lines",
            line=dict(color=(colours or {}).get(col, palette[i % len(palette)]),
                      width=width), marker=dict(size=8))
    fig.update_layout(yaxis_title=ytitle, xaxis_title=xtitle, hovermode="x unified")
    return _style(fig, title, height, legend=len(d.columns) > 1)


def fig_bars(x, y, title: str, xtitle: str, ytitle: str, colour: str = BLUE,
             text: list[str] | None = None, height: int = 360) -> go.Figure:
    fig = go.Figure(go.Bar(x=list(x), y=list(y), marker_color=colour, text=text,
                           textposition="outside", cliponaxis=False,
                           hovertemplate="%{x}: %{y:,.4g}<extra></extra>"))
    fig.update_layout(xaxis_title=xtitle, yaxis_title=ytitle, xaxis_type="category")
    return _style(fig, title, height, legend=False)


def fig_stacked_counts(table: pd.DataFrame, title: str, xtitle: str,
                       ytitle: str) -> go.Figure:
    """Stacked bars from a table with one column per cluster label."""
    fig = go.Figure()
    for name in table.columns:
        fig.add_bar(x=list(table.index), y=table[name], name=name,
                    marker_color=colour_for(name),
                    hovertemplate=f"{name}<br>%{{x}}: %{{y:,}}<extra></extra>")
    fig.update_layout(barmode="stack", xaxis_title=xtitle, yaxis_title=ytitle,
                      xaxis=dict(dtick=2), bargap=0.2)
    fig = _style(fig, title, 380)
    return fig.update_layout(bargap=0.2)


def fig_explore_scatter(points: pd.DataFrame) -> go.Figure:
    """PCA scatter of a live single-stage DBSCAN run.

    ``points.label``: -1 = noise, 0 = largest cluster, 1.. = smaller clusters.
    Clusters beyond the palette are folded into one 'other clusters' group.
    """
    slots = [BLUE, ORANGE, YELLOW, MAGENTA, GREEN, VIOLET, RED]
    fig = go.Figure()
    groups = [(0, "Largest cluster", NEUTRAL)] + [
        (i, f"Cluster {i}", slots[i - 1]) for i in range(1, len(slots) + 1)]
    for lab, name, colour in groups:
        g = points[points.label == lab]
        if len(g):
            fig.add_trace(go.Scattergl(
                x=g.pc1, y=g.pc2, mode="markers", name=f"{name} ({len(g):,})",
                marker=dict(color=colour, size=4 if lab == 0 else 6,
                            opacity=0.4 if lab == 0 else 0.85)))
    rest = points[points.label > len(slots)]
    if len(rest):
        fig.add_trace(go.Scattergl(
            x=rest.pc1, y=rest.pc2, mode="markers",
            name=f"Other clusters ({len(rest):,})",
            marker=dict(color="#52514e", size=6, opacity=0.85)))
    noise = points[points.label == -1]
    if len(noise):
        fig.add_trace(go.Scattergl(
            x=noise.pc1, y=noise.pc2, mode="markers",
            name=f"Noise ({len(noise):,})",
            marker=dict(color=AQUA, size=7, symbol="diamond", opacity=0.9)))
    fig.update_layout(xaxis_title="PC1", yaxis_title="PC2")
    return _style(fig, "Labels in PCA space (subsample)", 520)


# ----------------------------- static PNGs ---------------------------------
def _png_kdist(name: str, summ: dict) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for a, stage in zip(ax, ("stage1", "stage2")):
        kd = np.load(C.RESULTS / f"{name}_kdist_{stage}.npy")
        a.plot(np.linspace(0, 100, len(kd)), kd, color=BLUE)
        eps = summ[name][stage]["eps"]
        a.axhline(eps, ls="--", color=REFERENCE)
        a.set_title(f"{name} {stage}: k = {summ[name][stage]['min_samples']}, "
                    f"knee eps = {eps:.3f}")
        a.set_xlabel("windows sorted by k-distance (%)")
        a.set_ylabel("k-distance")
        a.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(C.PLOTS / f"{name}_kdistance.png", dpi=150)
    plt.close(fig)


def _png_pca(name: str) -> None:
    lab = pd.read_parquet(C.RESULTS / f"{name}_labels.parquet")
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    n = lab[lab.label == 0]
    ax.scatter(n.pc1, n.pc2, s=2, c=colour_for("C0 dominant"), alpha=.25,
               label=f"C0 dominant ({len(n):,})", rasterized=True)
    for nm, g in lab[lab.label != 0].groupby("label_name"):
        ax.scatter(g.pc1, g.pc2, s=10, c=colour_for(nm), label=f"{nm} ({len(g):,})",
                   marker="D" if nm == "Isolated (noise)" else "o")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title(f"{name}: DBSCAN labels in PCA space")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(C.PLOTS / f"{name}_pca_scatter.png", dpi=150)
    plt.close(fig)


def _png_rates(name: str) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for a, lev in zip(ax, ("re_level", "local_pv_level")):
        r = pd.read_csv(C.RESULTS / f"{name}_unusual_rate_by_{lev}.csv")
        x = np.arange(len(r))
        a.bar(x - .2, r.raw_rate * 100, .4, label="raw", color=BLUE)
        a.bar(x + .2, r.standardised_rate * 100, .4,
              label="standardised (hour x weekend)", color=NEUTRAL)
        a.set_xticks(x, r.iloc[:, 0]); a.set_ylabel("unusual windows (%)")
        a.set_title(f"{name}: unusual-window rate by {lev}")
        a.legend(fontsize=8); a.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(C.PLOTS / f"{name}_unusual_rate_by_level.png", dpi=150)
    plt.close(fig)


def level_thresholds(df: pd.DataFrame) -> tuple[float, float]:
    """Upper bounds of the 'low' and 'medium' regional renewable levels."""
    top = df.groupby("re_level", observed=True)["vre_penetration_pct"].max()
    return float(top["low"]), float(top["medium"])


def main() -> None:
    summ = json.loads((C.RESULTS / "summary.json").read_text(encoding="utf-8"))
    df = pd.read_parquet(C.PROCESSED / "analysis_dataset_10min.parquet")
    for name in ("exp1", "exp2"):
        _png_kdist(name, summ)
        _png_pca(name)
        _png_rates(name)
        lab = pd.read_parquet(C.RESULTS / f"{name}_labels.parquet")
        names = summ[name]["label_names"]
        figs = {
            "profile_heatmap": fig_profile_heatmap(
                pd.read_csv(C.RESULTS / f"{name}_cluster_profile_robust_z.csv"), names),
            "share_by_level": fig_cluster_share_by_level(
                pd.read_csv(C.RESULTS / f"{name}_cluster_by_re_level_colpct.csv"), names),
            "sensitivity_stage1_noise": fig_sensitivity(
                pd.read_csv(C.RESULTS / f"{name}_sensitivity_stage1.csv"),
                "noise_pct", f"{name} stage 1: noise % by parameters"),
            "monthly": fig_monthly(df, lab),
        }
        for k, f in figs.items():
            f.write_html(C.PLOTS / f"{name}_{k}.html", include_plotlyjs="cdn")
    fig_penetration_hist(df, level_thresholds(df)).write_html(
        C.PLOTS / "penetration_hist.html", include_plotlyjs="cdn")
    fig_correlation(pd.read_csv(C.RESULTS / "correlation_pq_vs_renewables.csv")
                    ).write_html(C.PLOTS / "correlation.html", include_plotlyjs="cdn")
    print("plots written to", C.PLOTS)


if __name__ == "__main__":
    main()
