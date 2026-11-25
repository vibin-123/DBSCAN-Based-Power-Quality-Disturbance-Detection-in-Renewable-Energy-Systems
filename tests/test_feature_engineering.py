import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.feature_engineering import (FEATURE_CATALOGUE, PQ_CLUSTER_FEATURES,
                                     add_rolling, aggregate_windows,
                                     compute_5s_derived, episode_durations,
                                     severity_index, voltage_unbalance_factor)


def test_vuf_zero_for_balanced_and_matches_reference():
    assert voltage_unbalance_factor(400, 400, 400) == pytest.approx(0.0, abs=1e-6)
    # symmetrical components check: positive 230 V, negative 2.3 V (1 %)
    a = np.exp(2j * np.pi / 3)
    v1, v2 = 230.0, 2.3
    va, vb, vc = v1 + v2, a ** 2 * v1 + a * v2, a * v1 + a ** 2 * v2
    vuf = voltage_unbalance_factor(abs(va - vb), abs(vb - vc), abs(vc - va))
    assert vuf == pytest.approx(1.0, rel=1e-3)


def test_window_features(pq_5s):
    f = aggregate_windows(compute_5s_derived(pq_5s), min_coverage=0.6)
    assert len(f) == 3
    assert f.v_dev_pct.iloc[0] == pytest.approx(1.0, abs=0.05)   # 232.3 V vs 230 V
    assert f.f_dev_abs_hz.iloc[0] == pytest.approx(0.02, abs=1e-4)
    assert f.thd_u_pct.iloc[0] == pytest.approx(3.0)
    assert f.p_kw.iloc[0] == pytest.approx(12.0)
    assert f.sag_samples.sum() == 0 and f.swell_samples.sum() == 0


def test_low_coverage_window_dropped_and_sag_counted(pq_5s):
    d = pq_5s.copy()
    d.iloc[120:200, d.columns.get_loc("U1")] = np.nan      # 2nd window: 33 % kept
    d.iloc[250:253, d.columns.get_loc("U2")] = 200.0       # 0.87 pu in 3rd window
    f = aggregate_windows(compute_5s_derived(d), min_coverage=0.6)
    assert len(f) == 2
    assert f.sag_samples.iloc[-1] == 3


def test_severity_index_is_fraction_of_limit():
    assert severity_index([5.0], [4.0], [0.5], [0.1])[0] == pytest.approx(0.5)
    assert severity_index([1.0], [8.0], [0.5], [0.1])[0] == pytest.approx(1.0)


def test_rolling_does_not_bridge_gaps():
    idx = pd.date_range("2024-01-01", periods=6, freq="10min").append(
        pd.date_range("2024-01-02", periods=2, freq="10min"))
    f = pd.DataFrame({"v_dev_pct": np.arange(8.0), "thd_u_pct": 3.0}, index=idx)
    out = add_rolling(f)
    assert np.isnan(out.v_dev_roll_mean_1h.iloc[6])        # first point after the gap
    assert out.v_dev_roll_mean_1h.iloc[5] == pytest.approx(2.5)


def test_episode_durations():
    idx = pd.date_range("2024-01-01", periods=6, freq="10min")
    ep = episode_durations(pd.Series([True, True, False, True, False, False], index=idx))
    assert ep.duration_min.tolist() == [20, 10]


def test_cluster_features_documented_and_exclude_renewables():
    assert set(PQ_CLUSTER_FEATURES) <= set(FEATURE_CATALOGUE)
    banned = ("solar", "wind", "pv", "penetration", "re_level", "hour", "label")
    assert not any(b in f for f in PQ_CLUSTER_FEATURES for b in banned)
