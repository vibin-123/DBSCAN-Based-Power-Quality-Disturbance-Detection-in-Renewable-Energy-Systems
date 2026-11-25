"""Stage A - data acquisition.

Downloads only what the project needs:

* SPARK (KIT, CC BY 4.0): PV zip, metadata zip, exogenous-data zip, and the
  selected 2024 measurement files of the facility total-load meter. The
  archive is a single 76 GB tar, so members are fetched by HTTP byte range.
* SMARD (Bundesnetzagentur, CC BY 4.0): quarter-hourly solar, onshore wind and
  grid load for the TransnetBW control area, straight from the original source.

Every file is recorded with its SHA-256 in ``data/raw/MANIFEST.csv``.
Run:  python -m src.data_acquisition
"""
from __future__ import annotations

import csv
import hashlib
import json
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

from . import config as C
from .remote_tar import HttpRangeFile, build_index, fetch_member


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_tar_index() -> dict[str, dict]:
    idx_file = C.SPARK_DIR / "tar_index.csv"
    if not idx_file.exists():
        C.SPARK_DIR.mkdir(parents=True, exist_ok=True)
        build_index(C.SPARK_TAR_URL, idx_file)
    with idx_file.open() as fh:
        return {r["name"].split("/")[-1]: r for r in csv.DictReader(fh)}


def download_spark_small(session: requests.Session | None = None) -> list[Path]:
    """PV, metadata, exogenous zips + BagIt manifest; verify MD5; extract."""
    idx = load_tar_index()
    out = []
    for name in C.SPARK_SMALL_ZIPS + ["manifest-md5.txt"]:
        r = idx[name]
        out.append(fetch_member(C.SPARK_TAR_URL, int(r["offset"]),
                                int(r["size"]), C.SPARK_DIR / name,
                                session=session))
    expected = {}
    for line in (C.SPARK_DIR / "manifest-md5.txt").read_text().splitlines():
        digest, path = line.split(maxsplit=1)
        expected[path.split("/")[-1]] = digest
    for name in C.SPARK_SMALL_ZIPS:
        got = md5(C.SPARK_DIR / name)
        if got != expected[name]:
            raise IOError(f"MD5 mismatch for {name}: {got} != {expected[name]}")
        with zipfile.ZipFile(C.SPARK_DIR / name) as z:
            z.extractall(C.SPARK_DIR)
    return out


def download_spark_pq(measurements=C.PQ_MEASUREMENTS, year: int = C.YEAR,
                      meter: str = C.PQ_METER) -> list[Path]:
    """Fetch selected ``<meter>/<measurement>/<year>_<measurement>.csv.xz``."""
    idx = load_tar_index()
    r = idx[f"{meter}.zip"]
    remote = HttpRangeFile(C.SPARK_TAR_URL, int(r["offset"]), int(r["size"]))
    out = []
    with zipfile.ZipFile(remote) as z:
        for m in measurements:
            member = f"{meter}/{m}/{year}_{m}.csv.xz"
            dest = C.SPARK_DIR / member
            info = z.getinfo(member)
            if not (dest.exists() and dest.stat().st_size == info.file_size):
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_suffix(".part")
                with z.open(info) as src, tmp.open("wb") as dst:
                    while chunk := src.read(1 << 20):
                        dst.write(chunk)
                tmp.rename(dest)   # zipfile has verified the member CRC-32
                print("downloaded", member, info.file_size, flush=True)
            out.append(dest)
    return out


def download_smard(year: int = C.YEAR) -> dict[str, Path]:
    """Quarter-hourly series from the SMARD chart-data API (weekly chunks)."""
    C.SMARD_DIR.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    t0 = pd.Timestamp(f"{year}-01-01", tz="UTC") - pd.Timedelta(days=8)
    t1 = pd.Timestamp(f"{year + 1}-01-01", tz="UTC") + pd.Timedelta(days=1)
    out = {}
    for name, fid in C.SMARD_FILTERS.items():
        dest = C.SMARD_DIR / f"smard_{name}_{C.SMARD_REGION}_{year}_15min.csv"
        if not dest.exists():
            base = f"{C.SMARD_BASE}/{fid}/{C.SMARD_REGION}"
            stamps = s.get(f"{base}/index_quarterhour.json",
                           timeout=60).json()["timestamps"]
            rows = []
            for ts in stamps:
                t = pd.Timestamp(ts, unit="ms", tz="UTC")
                if not (t0 <= t <= t1):
                    continue
                url = f"{base}/{fid}_{C.SMARD_REGION}_quarterhour_{ts}.json"
                rows += s.get(url, timeout=60).json()["series"]
                time.sleep(0.3)
            df = pd.DataFrame(rows, columns=["ts_ms", f"{name}_mw"])
            df["utc_datetime"] = pd.to_datetime(df.ts_ms, unit="ms", utc=True)
            df = df[(df.utc_datetime >= f"{year}-01-01") &
                    (df.utc_datetime < f"{year + 1}-01-01")]
            df[["utc_datetime", f"{name}_mw"]].to_csv(dest, index=False)
            print("downloaded", dest.name, len(df), flush=True)
        out[name] = dest
    return out


def write_manifest() -> Path:
    rows = []
    for p in sorted(C.RAW.rglob("*")):
        if p.is_file() and p.name != "MANIFEST.csv" and p.suffix != ".part":
            rows.append({"path": str(p.relative_to(C.RAW)),
                         "bytes": p.stat().st_size, "sha256": sha256(p)})
    dest = C.RAW / "MANIFEST.csv"
    pd.DataFrame(rows).to_csv(dest, index=False)
    return dest


def main() -> None:
    download_spark_small()
    download_smard()
    download_spark_pq()
    print("manifest:", write_manifest())


if __name__ == "__main__":
    main()
