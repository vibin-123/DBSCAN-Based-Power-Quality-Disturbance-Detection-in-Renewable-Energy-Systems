"""Comparison of Experiment 1 (PQ only) and Experiment 2 (PQ + renewables).

Run:  python -m src.experiment_comparison   (after src.experiments)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from . import config as C
from .dbscan_model import block_weights, prepare_matrix, two_stage_dbscan
from .evaluation import jaccard
from .feature_engineering import PQ_CLUSTER_FEATURES


def compare_labelings(a: pd.Series, b: pd.Series) -> dict:
    """Agreement of two labelings on their shared timestamps."""
    idx = a.index.intersection(b.index)
    a, b = a.loc[idx], b.loc[idx]
    ua, ub = a != 0, b != 0
    return {"n_shared": int(len(idx)),
            "ari": float(adjusted_rand_score(a, b)),
            "unusual_jaccard": jaccard(ua.values, ub.values),
            "unusual_only_a": int((ua & ~ub).sum()),
            "unusual_only_b": int((~ua & ub).sum()),
            "unusual_both": int((ua & ub).sum())}


def migration_table(a: pd.Series, b: pd.Series, name_a: str = "exp1",
                    name_b: str = "exp2") -> pd.DataFrame:
    idx = a.index.intersection(b.index)
    return pd.crosstab(a.loc[idx].rename(name_a), b.loc[idx].rename(name_b))


def weight_sweep(df: pd.DataFrame, re_features: list[str],
                 reference: pd.Series, shares=(0.1, 0.2, 0.3, 0.5)
                 ) -> pd.DataFrame:
    """How the renewable block weight changes the Experiment-2 structure."""
    rows = []
    for s in shares:
        w = block_weights(len(PQ_CLUSTER_FEATURES), len(re_features), s)
        pm = prepare_matrix(df, PQ_CLUSTER_FEATURES + re_features, w)
        res = two_stage_dbscan(pm.X)
        lab = pd.Series(res.labels, index=pm.index)
        cmp_ = compare_labelings(reference, lab)
        rows.append({"re_share": s, "per_feature_multiplier": float(w[-1]),
                     "n_clusters": int(len(set(res.labels) - {-1})),
                     "unusual_pct": float(res.unusual.mean() * 100),
                     "isolated_noise_pct": float((res.labels == -1).mean() * 100),
                     "ari_vs_exp1": cmp_["ari"],
                     "unusual_jaccard_vs_exp1": cmp_["unusual_jaccard"]})
    return pd.DataFrame(rows)


def main() -> None:
    from .experiments import RE_CLUSTER_FEATURES
    df = pd.read_parquet(C.PROCESSED / "analysis_dataset_10min.parquet")
    l1 = pd.read_parquet(C.RESULTS / "exp1_labels.parquet")["label"]
    l2 = pd.read_parquet(C.RESULTS / "exp2_labels.parquet")["label"]
    cmp_ = compare_labelings(l1, l2)
    mig = migration_table(l1, l2)
    mig.to_csv(C.RESULTS / "comparison_migration_exp1_vs_exp2.csv")
    sweep = weight_sweep(df, RE_CLUSTER_FEATURES, l1)
    sweep.to_csv(C.RESULTS / "comparison_weight_sweep.csv", index=False)
    summ = json.loads((C.RESULTS / "summary.json").read_text())
    rows = []
    for e in ("exp1", "exp2"):
        s = summ[e]
        rows.append({"experiment": e, "dimensions": s["dimensions"],
                     "eps_stage1": s["stage1"]["eps"], "eps_stage2": s["stage2"]["eps"],
                     "n_clusters": s["final"]["n_clusters"],
                     "dominant_cluster_pct": s["final"]["dominant_cluster_pct"],
                     "unusual_pct": s["final"]["unusual_pct"],
                     "isolated_noise_pct": s["final"]["isolated_noise_pct"],
                     "silhouette": s["final"]["silhouette"],
                     "davies_bouldin": s["final"]["davies_bouldin"],
                     "stage2_silhouette": s["stage2"]["silhouette"],
                     "stability_ari_mean": s.get("stability", {}).get(
                         "ari_labels", {}).get("mean"),
                     "cramers_v_vs_re_level": s["association"]["cluster_vs_re_level"]["cramers_v"]})
    pd.DataFrame(rows).to_csv(C.RESULTS / "comparison_summary.csv", index=False)
    (C.RESULTS / "comparison.json").write_text(json.dumps(cmp_, indent=1))
    print(pd.DataFrame(rows).T.to_string())
    print(cmp_)
    print(mig)
    print(sweep.round(3).to_string())


if __name__ == "__main__":
    main()
