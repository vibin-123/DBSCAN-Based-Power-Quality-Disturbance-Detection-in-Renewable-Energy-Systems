"""Stage J - internal validation, parameter sensitivity and stability.

Only internal (label-free) criteria are used. Their limits:

* Silhouette / Davies-Bouldin assume compact, convex clusters and ignore
  noise, so they are computed on clustered points only and always reported
  next to the noise percentage (a run can 'improve' them by discarding points).
* Both are undefined with fewer than two clusters.
* Stability (ARI between a subsample clustering and the reference) measures
  reproducibility, not correctness.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import (adjusted_rand_score, davies_bouldin_score,
                             silhouette_score)

from . import config as C
from .dbscan_model import k_distance, knee_point, run_dbscan


def clustering_metrics(X: np.ndarray, labels: np.ndarray,
                       sample_size: int | None = 10_000,
                       seed: int = C.RANDOM_STATE) -> dict:
    labels = np.asarray(labels)
    mask = labels >= 0
    n_clusters = int(len(set(labels[mask])))
    out = {"n_clusters": n_clusters, "n_noise": int((~mask).sum()),
           "noise_pct": float((~mask).mean() * 100.0),
           "largest_cluster_pct": float(
               pd.Series(labels[mask]).value_counts(normalize=True).iloc[0] * 100.0)
           if mask.any() else np.nan,
           "silhouette": np.nan, "davies_bouldin": np.nan}
    if n_clusters >= 2:
        Xc, lc = X[mask], labels[mask]
        ss = None if sample_size is None or len(Xc) <= sample_size else sample_size
        if ss is not None:
            # stratified subsample so that small clusters are not lost
            rng = np.random.default_rng(seed)
            keep = np.concatenate([
                rng.choice(np.where(lc == c)[0],
                           size=max(2, min((lc == c).sum(),
                                           int(round(ss * (lc == c).mean())))),
                           replace=False) for c in np.unique(lc)])
            out["silhouette"] = float(silhouette_score(Xc[keep], lc[keep]))
        else:
            out["silhouette"] = float(silhouette_score(Xc, lc))
        out["davies_bouldin"] = float(davies_bouldin_score(Xc, lc))
    return out


def eps_candidates(X: np.ndarray, min_samples: int, metric: str = "euclidean",
                   factors=(0.6, 0.8, 1.0, 1.25, 1.5)) -> tuple[float, list[float]]:
    """Knee epsilon for this MinPts and a bracket of multiples around it."""
    _, eps = knee_point(k_distance(X, min_samples, metric))
    return eps, [eps * f for f in factors]


def parameter_grid(X: np.ndarray, min_samples_list: list[int],
                   metric: str = "euclidean",
                   factors=(0.6, 0.8, 1.0, 1.25, 1.5),
                   reference: np.ndarray | None = None) -> pd.DataFrame:
    """DBSCAN over MinPts x (multiples of the knee epsilon for that MinPts).

    With ``reference`` labels, also reports the ARI of every setting against
    them (agreement of neighbouring parameter choices with the selected one).
    """
    rows = []
    for ms in min_samples_list:
        knee, eps_list = eps_candidates(X, ms, metric, factors)
        for f, eps in zip(factors, eps_list):
            labels = run_dbscan(X, eps, ms, metric)
            row = {"min_samples": ms, "eps_factor": f, "eps": eps,
                   "knee_eps": knee, "metric": metric,
                   **clustering_metrics(X, labels)}
            if reference is not None:
                row["ari_vs_selected"] = adjusted_rand_score(reference, labels)
                row["noise_jaccard_vs_selected"] = jaccard(reference < 0, labels < 0)
            rows.append(row)
    return pd.DataFrame(rows)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 1.0


def subsample_stability(X: np.ndarray, result, n_rounds: int = 20,
                        fraction: float = 0.8, seed: int = C.RANDOM_STATE
                        ) -> pd.DataFrame:
    """Refit the two-stage DBSCAN on random subsamples; compare with the
    full-data result on the shared points.

    Epsilons are kept (they are distances); MinPts values are scaled by
    ``fraction`` because removing points lowers density by that factor.
    """
    from .dbscan_model import two_stage_dbscan
    rng = np.random.default_rng(seed)
    n = len(X)
    ms1 = max(2, int(round(result.min_samples1 * fraction)))
    ms2 = max(2, int(round(result.min_samples2 * fraction)))
    ref = result.labels
    rows = []
    for r in range(n_rounds):
        idx = np.sort(rng.choice(n, size=int(n * fraction), replace=False))
        sub = two_stage_dbscan(X[idx], ms1, ms2, result.eps1, result.eps2,
                               result.metric)
        rows.append({
            "round": r,
            "ari_labels": adjusted_rand_score(ref[idx], sub.labels),
            "ari_unusual_vs_normal": adjusted_rand_score(ref[idx] != 0,
                                                         sub.labels != 0),
            "unusual_jaccard": jaccard(ref[idx] != 0, sub.labels != 0),
            "n_clusters": int(len(set(sub.labels) - {-1})),
            "unusual_pct": float((sub.labels != 0).mean() * 100.0)})
    return pd.DataFrame(rows)


def metric_comparison(X: np.ndarray, min_samples: int,
                      metrics=("euclidean", "manhattan", "chebyshev"),
                      reference: np.ndarray | None = None) -> pd.DataFrame:
    """Same procedure (knee epsilon) under different distance metrics."""
    rows = []
    for m in metrics:
        _, eps = knee_point(k_distance(X, min_samples, m))
        lab = run_dbscan(X, eps, min_samples, m)
        row = {"metric": m, "eps": eps, "min_samples": min_samples,
               **clustering_metrics(X, lab)}
        if reference is not None:
            row["ari_vs_euclidean"] = adjusted_rand_score(reference, lab)
            row["noise_jaccard_vs_euclidean"] = jaccard(reference < 0, lab < 0)
        rows.append(row)
    return pd.DataFrame(rows)


def baseline_comparison(X: np.ndarray, k_values=(2, 3, 4, 5, 6),
                        sample: int = 8000, seed: int = C.RANDOM_STATE
                        ) -> pd.DataFrame:
    """K-Means and Ward hierarchical clustering on the same matrix.

    Context for the algorithm choice only: both need k in advance and assign
    every window to a cluster (no noise concept). Ward linkage needs O(n^2)
    memory, so both are scored on the same random subsample.
    """
    from sklearn.cluster import AgglomerativeClustering, KMeans
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(sample, len(X)), replace=False)
    Xs = X[idx]
    rows = []
    for k in k_values:
        for name, model in (("k-means", KMeans(k, n_init=10, random_state=seed)),
                            ("ward", AgglomerativeClustering(k))):
            lab = model.fit_predict(Xs)
            rows.append({"algorithm": name, "k": k,
                         "silhouette": silhouette_score(Xs, lab),
                         "davies_bouldin": davies_bouldin_score(Xs, lab),
                         "smallest_cluster_pct": pd.Series(lab).value_counts(
                             normalize=True).min() * 100.0})
    return pd.DataFrame(rows)
