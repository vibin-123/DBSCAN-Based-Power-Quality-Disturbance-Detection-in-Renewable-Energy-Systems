"""Stage F - DBSCAN clustering.

Everything here is unsupervised: the only inputs are a numeric feature matrix
and DBSCAN's two density parameters. No labels, no train/test split.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler, StandardScaler

from . import config as C


@dataclass
class PreparedMatrix:
    X: np.ndarray                     # scaled (and weighted) clustering matrix
    index: pd.Index                   # timestamps, kept outside the matrix
    features: list[str]
    weights: np.ndarray
    scaler: object = field(repr=False, default=None)


def block_weights(n_pq: int, n_re: int, re_share: float) -> np.ndarray:
    """Per-feature multipliers so the renewable block carries ``re_share`` of
    the total squared-distance weight, whatever the number of features.

    PQ features keep multiplier 1. With ``re_share = s`` each renewable
    feature gets ``sqrt(s / (1 - s) * n_pq / n_re)``.
    """
    if n_re == 0:
        return np.ones(n_pq)
    if not 0 < re_share < 1:
        raise ValueError("re_share must be in (0, 1)")
    m = np.sqrt(re_share / (1.0 - re_share) * n_pq / n_re)
    return np.concatenate([np.ones(n_pq), np.full(n_re, m)])


def prepare_matrix(df: pd.DataFrame, features: list[str],
                   weights: np.ndarray | None = None,
                   scaler: str = "robust") -> PreparedMatrix:
    """Drop incomplete rows, scale each feature, apply block weights.

    RobustScaler (median / IQR) is the default: the unusual observations are
    the objects of interest, so they must not define the scale themselves.
    """
    data = df[features].replace([np.inf, -np.inf], np.nan).dropna()
    sc = RobustScaler() if scaler == "robust" else StandardScaler()
    X = sc.fit_transform(data.values.astype("float64"))
    w = np.ones(len(features)) if weights is None else np.asarray(weights, float)
    if len(w) != len(features):
        raise ValueError("one weight per feature required")
    return PreparedMatrix(X * w, data.index, list(features), w, sc)


def k_distance(X: np.ndarray, k: int, metric: str = "euclidean") -> np.ndarray:
    """Sorted distance of every point to its k-th nearest neighbour."""
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(X)
    dist, _ = nn.kneighbors(X)
    return np.sort(dist[:, k])


def knee_point(sorted_kdist: np.ndarray) -> tuple[int, float]:
    """Knee of the k-distance curve: point farthest below the chord joining
    its ends after normalising both axes (the 'Kneedle' construction).

    Points left of the knee sit in dense regions; points right of it are the
    sparse tail, so the k-distance at the knee is a data-driven epsilon.
    """
    y = np.asarray(sorted_kdist, dtype="float64")
    n = len(y)
    x = np.linspace(0.0, 1.0, n)
    span = y[-1] - y[0]
    yn = (y - y[0]) / span if span > 0 else np.zeros(n)
    i = int(np.argmax(x - yn))
    return i, float(y[i])


def run_dbscan(X: np.ndarray, eps: float, min_samples: int,
               metric: str = "euclidean") -> np.ndarray:
    return DBSCAN(eps=eps, min_samples=min_samples, metric=metric,
                  n_jobs=-1).fit_predict(X)


def relabel_by_size(labels: np.ndarray) -> np.ndarray:
    """Rename clusters 0, 1, 2 ... in order of decreasing size; noise stays -1."""
    s = pd.Series(labels)
    order = s[s >= 0].value_counts().index
    mapping = {old: new for new, old in enumerate(order)}
    mapping[-1] = -1
    return s.map(mapping).values


def pca_projection(X: np.ndarray, n_components: int = 2
                   ) -> tuple[np.ndarray, np.ndarray, PCA]:
    """PCA for visualisation only; clustering is done in the full space."""
    p = PCA(n_components=n_components, random_state=C.RANDOM_STATE).fit(X)
    return p.transform(X), p.explained_variance_ratio_, p


def cluster_profile(df: pd.DataFrame, labels: pd.Series, cols: list[str]
                    ) -> pd.DataFrame:
    """Per-cluster size, share and median of each column (noise = -1)."""
    d = df.loc[labels.index, cols].copy()
    d["cluster"] = labels.values
    g = d.groupby("cluster")
    out = g[cols].median()
    out.insert(0, "share_pct", g.size() / len(d) * 100.0)
    out.insert(0, "n_windows", g.size())
    return out


# ----------------------------------------------------------------------------
# Two-stage DBSCAN
# ----------------------------------------------------------------------------
NORMAL, RESIDUAL = 0, -1


@dataclass
class TwoStageResult:
    labels: np.ndarray        # 0 = dominant cluster, 1..k = pattern clusters, -1 = residual noise
    stage1: np.ndarray        # raw stage-1 DBSCAN labels (relabelled by size)
    eps1: float
    min_samples1: int
    eps2: float
    min_samples2: int
    metric: str = "euclidean"

    @property
    def unusual(self) -> np.ndarray:
        """Everything DBSCAN did not place in the dominant dense cluster."""
        return self.labels != NORMAL


def two_stage_dbscan(X: np.ndarray, min_samples1: int | None = None,
                     min_samples2: int | None = None,
                     eps1: float | None = None, eps2: float | None = None,
                     metric: str = "euclidean") -> TwoStageResult:
    """DBSCAN at two density levels.

    Stage 1 (global density): MinPts = 2 x dimensions, epsilon = knee of the
    k-distance curve. Stage-1 clusters keep their identity (0 = largest).

    Stage 2 (local density): DBSCAN's single global epsilon cannot resolve
    structure among sparse points - they are all 'noise' at the stage-1
    density. Stage 2 therefore re-runs DBSCAN on the stage-1 noise points
    only, with MinPts = dimensions + 1 and the knee epsilon of *their own*
    k-distance curve. Stage-2 clusters are groups of mutually similar unusual
    windows; what is still noise is isolated.

    Nothing fixes the number of clusters at either stage.
    """
    d = X.shape[1]
    ms1 = min_samples1 or 2 * d
    ms2 = min_samples2 or d + 1
    if eps1 is None:
        _, eps1 = knee_point(k_distance(X, ms1, metric))
    s1 = relabel_by_size(run_dbscan(X, eps1, ms1, metric))
    labels = s1.copy()
    noise = s1 < 0
    if noise.sum() > ms2:
        Xn = X[noise]
        if eps2 is None:
            _, eps2 = knee_point(k_distance(Xn, ms2, metric))
        s2 = relabel_by_size(run_dbscan(Xn, eps2, ms2, metric))
        offset = s1.max() + 1
        labels[noise] = np.where(s2 >= 0, s2 + offset, RESIDUAL)
    else:
        eps2 = float("nan") if eps2 is None else eps2
    return TwoStageResult(labels, s1, float(eps1), ms1, float(eps2), ms2, metric)


def label_names(labels: np.ndarray, n_stage1_clusters: int) -> dict[int, str]:
    """Readable names: stage-1 clusters 'C0..', stage-2 clusters 'U1..'."""
    names = {-1: "Isolated (noise)"}
    for l in sorted(set(labels) - {-1}):
        if l < n_stage1_clusters:
            names[l] = "C0 dominant" if l == 0 else f"C{l} dense"
        else:
            names[l] = f"U{l - n_stage1_clusters + 1} unusual pattern"
    return names
