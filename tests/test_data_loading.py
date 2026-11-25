import io
import tarfile
import zipfile

import numpy as np
import pandas as pd
import pytest

from src import config as C
from src import remote_tar
from src.preprocessing import clean_series, load_measurement
from tests.conftest import real_data


def test_load_measurement_schema(tmp_path):
    p = tmp_path / "2024_U1.csv.xz"
    pd.DataFrame({"WsDateTime": ["2024-01-01 00:00:00", "2024-01-01 00:00:05"],
                  "U1": [230.1, None]}).to_csv(p, index=False)
    s = load_measurement(p, "U1")
    assert s.index.name == "timestamp_utc" and len(s) == 2
    assert s.dtype == "float32" and s.isna().sum() == 1


def test_remote_tar_and_zip_range_reader(tmp_path, monkeypatch):
    """Index a tar and read a zip member through fake HTTP range requests."""
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("M/U1/2024_U1.csv", "WsDateTime,U1\n2024-01-01 00:00:00,230\n")
    tbuf = io.BytesIO()
    with tarfile.open(fileobj=tbuf, mode="w") as t:
        for name, data in (("bag/readme.txt", b"hello"), ("bag/M.zip", zbuf.getvalue())):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    blob = tbuf.getvalue()
    monkeypatch.setattr(remote_tar, "_ranged_get",
                        lambda session, url, start, length, retries=8:
                        blob[start:start + length])
    members = list(remote_tar.iter_members("http://x"))
    assert [m[0] for m in members] == ["bag/readme.txt", "bag/M.zip"]
    name, off, size, _ = members[1]
    assert blob[off:off + size] == zbuf.getvalue()
    f = remote_tar.HttpRangeFile("http://x", off, size, block=64)
    with zipfile.ZipFile(f) as z:
        assert z.read("M/U1/2024_U1.csv").startswith(b"WsDateTime,U1")


@pytest.mark.skipif(not (C.SPARK_DIR / C.PQ_METER).exists(),
                    reason="raw SPARK files not present (compact copy)")
def test_raw_files_present_and_hashed():
    man = pd.read_csv(C.RAW / "MANIFEST.csv")
    assert man.sha256.str.len().eq(64).all()
    for m in C.PQ_MEASUREMENTS:
        assert (C.SPARK_DIR / C.PQ_METER / m / f"{C.YEAR}_{m}.csv.xz").exists()


@real_data
def test_processed_dataset_integrity():
    df = pd.read_parquet(C.PROCESSED / "analysis_dataset_10min.parquet")
    assert df.index.is_monotonic_increasing and df.index.is_unique
    assert len(df) > 40_000
    assert df.v_rms.between(200, 260).all()
    assert df.freq_hz.between(49, 51).all()
    assert (df.coverage >= C.MIN_COVERAGE).all()
