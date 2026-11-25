import inspect

import numpy as np
import pandas as pd
import pytest

from src import dbscan_model as M
from src import evaluation as E
from src import experiment_comparison as X
from tests.conftest import real_data, results
from src import config as C


def test_block_weights_share():
    w = M.block_weights(10, 3, 0.2)
    pq, re = (w[:10] ** 2).sum(), (w[10:] ** 2).sum()
    assert re / (pq + re) == pytest.approx(0.2)
    assert np.allclose(M.block_weights(4, 0, 0.2), 1.0)


def test_prepare_matrix_keeps_timestamps_outside():
    idx = pd.date_range("2024-01-01", periods=5, freq="10min")
    df = pd.DataFrame({"a": [1, 2, 3, np.nan, 5.0], "b": [5, 4, 3, 2, 1.0],
                       "meta": list("vwxyz")}, index=idx)
    pm = M.prepare_matrix(df, ["a", "b"])
    assert pm.X.shape == (4, 2) and len(pm.index) == 4
    assert np.allclose(np.median(pm.X, axis=0), 0)


def test_knee_and_dbscan_recover_blobs(blobs):
    kd = M.k_distance(blobs, 6)
    assert np.all(np.diff(kd) >= 0)
    _, eps = M.knee_point(kd)
    labels = M.relabel_by_size(M.run_dbscan(blobs, eps, 6))
    assert len(set(labels) - {-1}) == 2
    assert (labels == -1).sum() >= 8             # the scattered outliers
    a, b = set(labels[:400]) - {-1}, set(labels[400:800]) - {-1}
    assert len(a) == 1 and len(b) == 1 and a != b


def test_two_stage_finds_pattern_among_noise():
    rng = np.random.default_rng(2)
    main = rng.normal(0, 1, (3000, 4))
    pattern = rng.normal(9, 0.4, (40, 4))         # compact group of unusual points
    lone = rng.uniform(-40, 40, (8, 4))
    res = M.two_stage_dbscan(np.vstack([main, pattern, lone]))
    assert (res.labels[:3000] == 0).mean() > 0.9
    assert len(set(res.labels[3000:3040]) - {-1, 0}) >= 1
    assert res.unusual[3000:].mean() > 0.9


def test_metrics_and_noise_handling(blobs):
    labels = M.run_dbscan(blobs, 1.0, 6)
    m = E.clustering_metrics(blobs, labels, sample_size=None)
    assert m["n_clusters"] == 2 and m["silhouette"] > 0.8
    assert m["davies_bouldin"] < 0.5 and 0 < m["noise_pct"] < 5
    one = E.clustering_metrics(blobs, np.zeros(len(blobs), int))
    assert np.isnan(one["silhouette"]) and np.isnan(one["davies_bouldin"])


def test_parameter_grid_and_stability(blobs):
    ref = M.two_stage_dbscan(blobs, 6, 4)
    grid = E.parameter_grid(blobs, [4, 6], factors=(0.8, 1.0, 1.2), reference=ref.stage1)
    assert len(grid) == 6 and {"noise_pct", "ari_vs_selected"} <= set(grid.columns)
    stab = E.subsample_stability(blobs, ref, n_rounds=3)
    assert stab.ari_labels.mean() > 0.8
    mc = E.metric_comparison(blobs, 6, reference=ref.stage1)
    assert set(mc.metric) == {"euclidean", "manhattan", "chebyshev"}


def test_experiment_comparison_functions():
    idx = pd.RangeIndex(6)
    a = pd.Series([0, 0, 0, 1, 1, -1], index=idx)
    b = pd.Series([0, 0, 1, 1, 1, -1], index=idx)
    c = X.compare_labelings(a, b)
    assert c["unusual_both"] == 3 and c["unusual_only_b"] == 1
    assert c["unusual_jaccard"] == pytest.approx(0.75)
    assert X.compare_labelings(a, a)["ari"] == pytest.approx(1.0)
    assert X.migration_table(a, b).values.sum() == 6


def test_no_supervised_learning_in_source():
    """Guard the project rule: no classifiers, no train/test split."""
    import pathlib
    banned = ("train_test_split", "Classifier", "LogisticRegression", "accuracy_score",
              "f1_score", "precision_score", "recall_score", ".fit(X, y")
    for p in pathlib.Path(C.ROOT / "src").glob("*.py"):
        text = p.read_text()
        for b in banned:
            assert b not in text, f"{b} found in {p.name}"


@results
def test_experiment_outputs_consistent():
    import json
    s = json.loads((C.RESULTS / "summary.json").read_text())
    from src.feature_engineering import PQ_CLUSTER_FEATURES
    assert s["exp1"]["features"] == PQ_CLUSTER_FEATURES
    assert set(s["exp2"]["features"]) > set(PQ_CLUSTER_FEATURES)
    for e in ("exp1", "exp2"):
        lab = pd.read_parquet(C.RESULTS / f"{e}_labels.parquet")
        assert len(lab) == s[e]["n_windows"]
        assert (lab.label != 0).mean() * 100 == pytest.approx(s[e]["final"]["unusual_pct"])
        assert s[e]["stage1"]["min_samples"] == 2 * s[e]["dimensions"]
