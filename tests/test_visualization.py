import numpy as np
import pandas as pd

from src import visualization as V


def _levels_frame():
    """Synthetic windows whose first rows are 'medium', then 'high', then 'low'."""
    rng = np.random.default_rng(0)
    level = np.repeat(["medium", "high", "low"], 50)
    return pd.DataFrame({
        "thd_u_pct": rng.normal(3.4, 0.3, len(level)),
        "re_level": pd.Categorical(level, categories=V.LEVEL_ORDER, ordered=True)})


def test_box_plot_levels_follow_low_medium_high():
    fig = V.fig_box_by_level(_levels_frame(), "thd_u_pct", "re_level")
    assert [t.x[0] for t in fig.data] == ["low", "medium", "high"]
    assert list(fig.layout.xaxis.categoryarray) == ["low", "medium", "high"]


def test_box_plot_sends_quartiles_not_raw_points():
    df = _levels_frame()
    fig = V.fig_box_by_level(df, "thd_u_pct", "re_level")
    low = df.loc[df.re_level == "low", "thd_u_pct"]
    assert fig.data[0].y is None
    assert fig.data[0].median[0] == low.median()
    assert low.min() <= fig.data[0].lowerfence[0] <= fig.data[0].q1[0]


def test_monthly_figure_has_no_second_y_scale_on_one_plot():
    idx = pd.date_range("2024-01-01", periods=6 * 24 * 60, freq="10min")
    df = pd.DataFrame({"vre_penetration_pct": np.linspace(5, 30, len(idx))}, index=idx)
    lab = pd.DataFrame({"label": (np.arange(len(idx)) % 50 == 0).astype(int)}, index=idx)
    fig = V.fig_monthly(df, lab)
    assert fig.layout.yaxis2.overlaying is None          # stacked panels instead
    assert list(fig.data[0].x) == ["Jan", "Feb"]


def test_themed_steps_colours_for_dark_and_leaves_the_original_alone():
    rate = pd.DataFrame({"re_level": V.LEVEL_ORDER, "raw_rate": [0.01, 0.02, 0.03],
                         "standardised_rate": [0.02, 0.02, 0.03]})
    fig = V.fig_rate_by_level(rate, "rate")
    dark = V.themed(fig, dark=True)
    assert fig.data[0].marker.color == V.BLUE
    assert dark.data[0].marker.color != V.BLUE
    assert V.themed(fig, dark=False) is fig


def test_feature_labels_are_readable():
    assert V.label_for("thd_u_pct") == "Voltage THD (%)"
    assert V.label_for("thd_u_pct", unit=False) == "Voltage THD"
    assert V.label_for("some_new_column") == "some new column"
