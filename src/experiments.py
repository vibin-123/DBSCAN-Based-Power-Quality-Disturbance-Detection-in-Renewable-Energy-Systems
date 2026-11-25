"""Runs the two DBSCAN experiments end to end and stores every result table.

Experiment 1: power-quality features only. Renewable variables are joined
              *after* clustering for interpretation.
Experiment 2: the same PQ features plus regional renewable penetration,
              solar and wind generation (block-weighted).

Run:  python -m src.experiments
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from . import config as C
from . import evaluation as E
from . import renewable_analysis as R
from .dbscan_model import (block_weights, cluster_profile, k_distance,
                           knee_point, label_names, pca_projection,
                           prepare_matrix, two_stage_dbscan)
from .feature_engineering import PQ_CLUSTER_FEATURES, episode_durations

RE_CLUSTER_FEATURES = ["vre_penetration_pct", "solar_pv_mw", "wind_onshore_mw"]
RE_SHARE = 0.2            # renewable block share of squared-distance weight
PQ_INDICATORS = ["v_dev_pct", "v_std_pct", "rvc_max_pct", "vuf_pct",
                 "f_dev_abs_hz", "thd_u_pct", "u_h5_pct", "thd_i_pct",
                 "severity_index"]


def _save(df: pd.DataFrame, name: str, index: bool = True) -> None:
    df.to_csv(C.RESULTS / f"{name}.csv", index=index)


def association_block(df: pd.DataFrame, labels: pd.Series, prefix: str,
                      n_perm: int = 500) -> dict:
    """All post-clustering renewable analyses for one labelling."""
    d = df.loc[labels.index].copy()
    d["label"] = labels.values
    d["unusual"] = (d["label"] != 0).astype(float)
    d["isolated_noise"] = (d["label"] == -1).astype(float)
    out = {}

    # 1. cluster x regional level
    ct = R.contingency(d["label"], d["re_level"])
    _save(ct, f"{prefix}_cluster_by_re_level_counts")
    _save(ct.div(ct.sum(axis=0), axis=1) * 100, f"{prefix}_cluster_by_re_level_colpct")
    chi2, p, v, dof = R.cramers_v(ct)
    out["cluster_vs_re_level"] = {"chi2": chi2, "p_naive": p, "cramers_v": v, "dof": dof}

    # 2. permutation tests for different renewable variables
    perm = {}
    for var in ("vre_penetration_pct", "solar_penetration_pct",
                "wind_penetration_pct", "vre_share_of_generation_pct"):
        perm[var] = R.day_block_permutation_test(d["label"], d[var], n_perm=n_perm)
        perm[var + "__unusual_flag"] = R.day_block_permutation_test(
            d["unusual"].astype(int), d[var], n_perm=n_perm)
    gen = d[d["pv_output_pct_of_rated"] >= 1.0]
    if len(gen) > 300:
        perm["local_pv_output_generating_windows"] = R.day_block_permutation_test(
            gen["label"], gen["pv_output_pct_of_rated"], n_perm=n_perm)
        perm["local_pv_output_generating_windows__unusual_flag"] = \
            R.day_block_permutation_test(gen["unusual"].astype(int),
                                         gen["pv_output_pct_of_rated"], n_perm=n_perm)
    out["permutation_tests"] = perm
    _save(pd.DataFrame(perm).T, f"{prefix}_permutation_tests")

    # 3. unusual-window rate per level, raw and standardised for hour x weekend
    for flag in ("unusual", "isolated_noise"):
        for lev in ("re_level", "local_pv_level"):
            _save(R.stratified_rate(d, flag, lev, ["hour_local", "is_weekend"]) ,
                  f"{prefix}_{flag}_rate_by_{lev}")

    # 4. PQ indicators per level
    _save(R.level_summary(d, "re_level", PQ_INDICATORS), f"{prefix}_pq_by_re_level")
    _save(R.level_summary(d, "local_pv_level", PQ_INDICATORS),
          f"{prefix}_pq_by_local_pv_level")
    _save(R.kruskal_by_level(d, "re_level", PQ_INDICATORS),
          f"{prefix}_kruskal_re_level", index=False)

    # 5. local PV cross-tab (Jan - Apr subset)
    ctl = R.contingency(d["label"], d["local_pv_level"])
    _save(ctl, f"{prefix}_cluster_by_local_pv_level_counts")
    chi2, p, v, dof = R.cramers_v(ctl)
    out["cluster_vs_local_pv_level"] = {"chi2": chi2, "p_naive": p,
                                        "cramers_v": v, "dof": dof,
                                        "n": int(ctl.values.sum())}
    return out


def correlation_tables(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation of PQ indicators with renewable variables, raw and
    after removing the hour-of-day x weekend mean from both sides."""
    re_vars = ["vre_penetration_pct", "solar_penetration_pct",
               "wind_penetration_pct", "pv_output_pct_of_rated"]
    strata = [df["hour_local"], df["is_weekend"]]
    rows = []
    for rv in re_vars:
        for pq in PQ_INDICATORS:
            d = df[[rv, pq]].dropna()
            if len(d) < 100:
                continue
            rho = stats.spearmanr(d[rv], d[pq]).statistic
            g = [s.loc[d.index] for s in strata]
            res_r = d[rv] - d[rv].groupby(g).transform("mean")
            res_p = d[pq] - d[pq].groupby(g).transform("mean")
            rows.append({"renewable_variable": rv, "pq_indicator": pq,
                         "spearman": rho,
                         "spearman_within_hour": stats.spearmanr(res_r, res_p).statistic,
                         "n": len(d)})
    return pd.DataFrame(rows)


def run_experiment(df: pd.DataFrame, name: str, features: list[str],
                   weights: np.ndarray | None, n_perm: int = 500,
                   stability_rounds: int = 20, full: bool = True) -> dict:
    pm = prepare_matrix(df, features, weights)
    X = pm.X
    d = X.shape[1]
    res = two_stage_dbscan(X)
    labels = pd.Series(res.labels, index=pm.index, name="label")
    n_s1 = int(res.stage1.max() + 1)
    names = label_names(res.labels, n_s1)

    # --- k-distance curves (for the plots) ---
    kd1 = k_distance(X, res.min_samples1)
    noise1 = res.stage1 < 0
    kd2 = k_distance(X[noise1], res.min_samples2)
    np.save(C.RESULTS / f"{name}_kdist_stage1.npy", kd1[:: max(1, len(kd1) // 4000)])
    np.save(C.RESULTS / f"{name}_kdist_stage2.npy", kd2)

    # --- label table with PCA coordinates for visualisation ---
    Z, evr, _ = pca_projection(X, 3)
    lab = pd.DataFrame({"label": res.labels, "stage1_label": res.stage1,
                        "label_name": [names[l] for l in res.labels],
                        "pc1": Z[:, 0], "pc2": Z[:, 1], "pc3": Z[:, 2]},
                       index=pm.index)
    lab.to_parquet(C.RESULTS / f"{name}_labels.parquet")

    # --- metrics ---
    m_all = E.clustering_metrics(X, res.labels)
    m_s1 = E.clustering_metrics(X, res.stage1)
    m_s2 = E.clustering_metrics(X[noise1], res.labels[noise1] - n_s1 *
                                (res.labels[noise1] >= 0), sample_size=None)
    summary = {
        "experiment": name, "features": features,
        "weights": [float(w) for w in pm.weights], "n_windows": int(len(X)),
        "dimensions": d, "pca_explained_variance_3": [float(v) for v in evr],
        "stage1": {"min_samples": res.min_samples1, "eps": res.eps1,
                   "n_clusters": m_s1["n_clusters"], "noise_pct": m_s1["noise_pct"],
                   "silhouette": m_s1["silhouette"],
                   "davies_bouldin": m_s1["davies_bouldin"]},
        "stage2": {"min_samples": res.min_samples2, "eps": res.eps2,
                   "n_input_windows": int(noise1.sum()),
                   "n_clusters": m_s2["n_clusters"],
                   "residual_noise_pct_of_stage2_input": m_s2["noise_pct"],
                   "silhouette": m_s2["silhouette"],
                   "davies_bouldin": m_s2["davies_bouldin"]},
        "final": {"n_clusters": m_all["n_clusters"],
                  "isolated_noise_pct": m_all["noise_pct"],
                  "unusual_pct": float(res.unusual.mean() * 100),
                  "dominant_cluster_pct": float((res.labels == 0).mean() * 100),
                  "silhouette": m_all["silhouette"],
                  "davies_bouldin": m_all["davies_bouldin"]},
        "label_names": {str(k): v for k, v in names.items()},
        "label_counts": {str(k): int(v) for k, v in
                         pd.Series(res.labels).value_counts().sort_index().items()},
    }

    # --- cluster profiles ---
    prof_cols = sorted(set(PQ_CLUSTER_FEATURES + ["severity_index", "p_kw", "pf",
                                                  "hour_local", "vre_penetration_pct",
                                                  "solar_penetration_pct",
                                                  "wind_penetration_pct"]),
                       key=lambda c: (PQ_CLUSTER_FEATURES + ["severity_index"]).index(c)
                       if c in PQ_CLUSTER_FEATURES + ["severity_index"] else 99)
    prof = cluster_profile(df, labels, prof_cols)
    prof.insert(0, "name", [names[i] for i in prof.index])
    days = labels.groupby(labels).apply(lambda s: s.index.normalize().nunique())
    prof.insert(3, "n_days", days)
    prof["weekend_share_pct"] = df.loc[labels.index, "is_weekend"].groupby(
        labels.values).mean() * 100
    _save(prof, f"{name}_cluster_profile")
    # robust z-score of each cluster median vs the whole data set
    med, iqr = df[PQ_CLUSTER_FEATURES].median(), (df[PQ_CLUSTER_FEATURES].quantile(.75)
                                                  - df[PQ_CLUSTER_FEATURES].quantile(.25))
    _save((prof[PQ_CLUSTER_FEATURES] - med) / iqr, f"{name}_cluster_profile_robust_z")

    # --- episodes of consecutive unusual windows ---
    full_idx = pd.Series(False, index=df.index)
    full_idx.loc[labels.index] = res.unusual
    ep = episode_durations(full_idx)
    _save(ep, f"{name}_unusual_episodes", index=False)
    summary["unusual_episodes"] = {
        "count": int(len(ep)),
        "median_duration_min": float(ep.duration_min.median()) if len(ep) else 0.0,
        "max_duration_min": float(ep.duration_min.max()) if len(ep) else 0.0}

    # --- sensitivity, stability, metric / scaler comparison ---
    if full:
        grid1 = E.parameter_grid(X, [d, 2 * d, 4 * d], reference=res.stage1)
        _save(grid1, f"{name}_sensitivity_stage1", index=False)
        ref2 = res.labels[noise1]
        grid2 = E.parameter_grid(X[noise1], sorted({max(3, d // 2), d + 1, 2 * d}),
                                 factors=(0.75, 1.0, 1.25), reference=ref2)
        _save(grid2, f"{name}_sensitivity_stage2", index=False)
        stab = E.subsample_stability(X, res, n_rounds=stability_rounds)
        _save(stab, f"{name}_stability", index=False)
        summary["stability"] = {c: {"mean": float(stab[c].mean()),
                                    "min": float(stab[c].min())}
                                for c in ("ari_labels", "ari_unusual_vs_normal",
                                          "unusual_jaccard")}
        _save(E.metric_comparison(X, res.min_samples1, reference=res.stage1),
              f"{name}_metric_comparison", index=False)
        pm_std = prepare_matrix(df, features, weights, scaler="standard")
        res_std = two_stage_dbscan(pm_std.X)
        from sklearn.metrics import adjusted_rand_score
        summary["scaler_comparison"] = {
            "standard_unusual_pct": float(res_std.unusual.mean() * 100),
            "standard_n_clusters": int(len(set(res_std.labels) - {-1})),
            "unusual_jaccard_robust_vs_standard": E.jaccard(res.unusual, res_std.unusual),
            "ari_robust_vs_standard": float(adjusted_rand_score(res.labels, res_std.labels))}
        _save(E.baseline_comparison(X), f"{name}_baseline_kmeans_ward", index=False)

    # --- renewable association (post-clustering) ---
    summary["association"] = association_block(df, labels, name, n_perm=n_perm)
    return summary


def association_robustness(df: pd.DataFrame, name: str, features: list[str],
                           weights: np.ndarray | None) -> pd.DataFrame:
    """Does the renewable-level contrast survive other parameter choices?

    For MinPts in {d, 2d, 4d} and epsilon in {0.8, 1.0, 1.25} x knee, the
    stage-1 noise rate is recomputed per regional renewable level.
    """
    from .dbscan_model import run_dbscan
    pm = prepare_matrix(df, features, weights)
    d = pm.X.shape[1]
    lev = df.loc[pm.index, "re_level"]
    rows = []
    for ms in (d, 2 * d, 4 * d):
        _, knee = knee_point(k_distance(pm.X, ms))
        for f in (0.8, 1.0, 1.25):
            noise = pd.Series(run_dbscan(pm.X, knee * f, ms) < 0, index=pm.index)
            rate = noise.groupby(lev, observed=True).mean() * 100
            v = R.cramers_v(R.contingency(noise.astype(int), lev))[2]
            rows.append({"min_samples": ms, "eps_factor": f, "eps": knee * f,
                         "noise_pct": noise.mean() * 100,
                         "noise_pct_low": rate["low"], "noise_pct_medium": rate["medium"],
                         "noise_pct_high": rate["high"],
                         "high_to_low_ratio": rate["high"] / rate["low"],
                         "cramers_v": v})
    out = pd.DataFrame(rows)
    _save(out, f"{name}_association_robustness", index=False)
    return out


def main(n_perm: int = 500, stability_rounds: int = 20) -> None:
    df = pd.read_parquet(C.PROCESSED / "analysis_dataset_10min.parquet")
    s1 = run_experiment(df, "exp1", PQ_CLUSTER_FEATURES, None, n_perm,
                        stability_rounds)
    feats2 = PQ_CLUSTER_FEATURES + RE_CLUSTER_FEATURES
    w2 = block_weights(len(PQ_CLUSTER_FEATURES), len(RE_CLUSTER_FEATURES), RE_SHARE)
    s2 = run_experiment(df, "exp2", feats2, w2, n_perm, stability_rounds)
    association_robustness(df, "exp1", PQ_CLUSTER_FEATURES, None)
    _save(correlation_tables(df), "correlation_pq_vs_renewables", index=False)
    (C.RESULTS / "summary.json").write_text(
        json.dumps({"exp1": s1, "exp2": s2}, indent=1, default=float))
    print(json.dumps({k: {kk: v[kk] for kk in ("stage1", "stage2", "final")}
                      for k, v in (("exp1", s1), ("exp2", s2))}, indent=1))


if __name__ == "__main__":
    main()
