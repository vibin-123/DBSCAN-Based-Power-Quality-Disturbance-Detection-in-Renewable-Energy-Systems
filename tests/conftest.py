import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as C  # noqa: E402


@pytest.fixture
def blobs():
    """Two dense blobs plus sparse outliers (synthetic, tests only)."""
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.3, (400, 3))
    b = rng.normal(6, 0.3, (400, 3))
    out = rng.uniform(-20, 20, (12, 3))
    return np.vstack([a, b, out])


@pytest.fixture
def pq_5s():
    """Ten minutes x 3 of synthetic 5 s meter data on the project schema."""
    idx = pd.date_range("2024-03-01 10:00", periods=360, freq="5s")
    rng = np.random.default_rng(1)
    df = pd.DataFrame(index=idx)
    for c in ("U1", "U2", "U3"):
        df[c] = 232.3 + rng.normal(0, 0.1, len(idx))
    for c in ("U12", "U23", "U31"):
        df[c] = 402.0
    df["Freq"] = 50.02
    for c in ("I1", "I2", "I3"):
        df[c] = 20.0
    df["IN"] = 5.0
    for c in ("THD_U1", "THD_U2", "THD_U3"):
        df[c] = 3.0
    for c in ("THD_I1", "THD_I2", "THD_I3"):
        df[c] = 1.5
    df["P_total_kw"], df["Q_total_kvar"], df["S_total_kva"] = 12.0, 3.0, 12.4
    df["PF_total"] = 0.97
    for c in ("U1_h3", "U1_h5", "U1_h7", "U2_h5", "U3_h5"):
        df[c] = 1.0
    return df


def _have(path):
    return Path(path).exists()


real_data = pytest.mark.skipif(
    not _have(C.PROCESSED / "analysis_dataset_10min.parquet"),
    reason="processed dataset not built (run run_pipeline.py)")
results = pytest.mark.skipif(not _have(C.RESULTS / "summary.json"),
                             reason="experiment results not built")
