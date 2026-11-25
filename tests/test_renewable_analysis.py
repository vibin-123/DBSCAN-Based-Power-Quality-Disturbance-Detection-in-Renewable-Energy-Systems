import numpy as np
import pandas as pd
import pytest

from src import config as C
from src import renewable_analysis as R


def test_penetration_formula_and_zero_denominator():
    out = R.penetration_pct([50, 10, 5], [200, 0, -1])
    assert out[0] == pytest.approx(25.0)
    assert np.isnan(out[1]) and np.isnan(out[2])


def test_regional_metrics():
    reg = pd.DataFrame({"solar_pv_mw": [100.0], "wind_onshore_mw": [50.0],
                        "hydro_mw": [50.0], "biomass_mw": [0.0],
                        "other_renewables_mw": [0.0], "natural_gas_mw": [300.0],
                        "grid_load_mw": [600.0]})
    r = R.regional_metrics(reg)
    assert r.total_generation_mw[0] == 500
    assert r.vre_penetration_pct[0] == pytest.approx(25.0)
    assert r.vre_share_of_generation_pct[0] == pytest.approx(30.0)
    assert r.renewable_share_of_generation_pct[0] == pytest.approx(40.0)


def test_levels_are_terciles():
    x = pd.Series(np.arange(300.0))
    lev = pd.Series(R.assign_levels(x, R.tercile_thresholds(x)))
    assert lev.value_counts().to_dict() == {"low": 100, "medium": 100, "high": 100}


def test_local_pv_levels_and_rating():
    pv = pd.Series([0.0] * 60 + [5.36] * 120,
                   index=pd.date_range("2024-03-01", periods=180, freq="5s"))
    w = R.local_pv_windows(pv, min_coverage=0.4)
    assert w.pv_output_pct_of_rated.iloc[-1] == pytest.approx(50.0)
    lev, _ = R.assign_local_pv_levels(pd.Series([0.0, 0.5, 5, 20, 60, 80]))
    assert list(lev[:2]) == ["none", "none"] and lev.iloc[-1] == "high"


def test_cramers_v_extremes():
    perfect = pd.DataFrame([[50, 0], [0, 50]])
    assert R.cramers_v(perfect)[2] > 0.95
    none = pd.DataFrame([[25, 25], [25, 25]])
    assert R.cramers_v(none)[2] == pytest.approx(0.0, abs=1e-9)


def _daily_frame(days=40, seed=0):
    idx = pd.date_range("2024-01-01", periods=days * 144, freq="10min")
    rng = np.random.default_rng(seed)
    day_strength = np.repeat(rng.uniform(0, 1, days), 144)
    shape = np.clip(np.sin((idx.hour + idx.minute / 60 - 6) / 12 * np.pi), 0, None)
    return idx, rng, day_strength, pd.Series(day_strength * shape * 100, index=idx)


def test_permutation_test_detects_real_association():
    idx, rng, strength, value = _daily_frame()
    labels = pd.Series((rng.uniform(size=len(idx)) < 0.02 + 0.3 * (value / 100)).astype(int),
                       index=idx)
    res = R.day_block_permutation_test(labels, value, n_perm=99)
    assert res["p_value"] <= 0.05


def test_permutation_test_ignores_shared_daily_rhythm():
    """Labels follow only the hour of day; the test must not flag them."""
    idx, rng, strength, value = _daily_frame(seed=3)
    midday = (idx.hour >= 10) & (idx.hour < 14)
    labels = pd.Series((rng.uniform(size=len(idx)) < np.where(midday, 0.3, 0.02)).astype(int),
                       index=idx)
    res = R.day_block_permutation_test(labels, value, n_perm=99)
    assert res["p_value"] > 0.05


def test_stratified_rate_removes_confounding():
    # flag depends only on hour; level is correlated with hour
    d = pd.DataFrame({"hour_local": [0] * 100 + [12] * 100,
                      "lev": pd.Categorical(["low"] * 80 + ["high"] * 20 +
                                            ["low"] * 20 + ["high"] * 80),
                      "flag": [0.0] * 100 + [1.0] * 100})
    r = R.stratified_rate(d, "flag", "lev", ["hour_local"])
    assert r.raw_rate["high"] > r.raw_rate["low"]
    assert r.standardised_rate["high"] == pytest.approx(r.standardised_rate["low"])
