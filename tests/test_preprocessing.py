import numpy as np
import pandas as pd

from src.preprocessing import align_to_windows, clean_series


def _series(times, values):
    return pd.Series(values, index=pd.to_datetime(times), dtype="float32",
                     name="U1").rename_axis("timestamp_utc")


def test_duplicates_removed_and_grid_regularised():
    s = _series(["2024-01-01 00:00:00", "2024-01-01 00:00:00",
                 "2024-01-01 00:00:15"], [230, 999, 231])
    out, st = clean_series(s, "U1")
    assert st["duplicate_timestamps"] == 1
    assert len(out) == 4                       # 00:00, :05, :10, :15
    assert out.iloc[0] == 230                  # first occurrence kept
    assert out.isna().sum() == 2               # gaps stay missing


def test_missing_values_are_not_interpolated():
    s = _series(["2024-01-01 00:00:00", "2024-01-01 00:00:20"], [230, 232])
    out, st = clean_series(s, "U1")
    assert out.isna().sum() == 3 and st["missing_final"] == 3


def test_out_of_range_set_to_nan():
    s = _series(["2024-01-01 00:00:00", "2024-01-01 00:00:05"], [230, 900])
    out, st = clean_series(s, "U1")
    assert st["out_of_physical_range"] == 1 and np.isnan(out.iloc[1])


def test_off_grid_timestamps_dropped():
    s = _series(["2024-01-01 00:00:00", "2024-01-01 00:00:07",
                 "2024-01-01 00:00:10"], [230, 231, 232])
    out, st = clean_series(s, "U1")
    assert st["off_grid_timestamps"] == 1 and len(out) == 3


def test_timestamp_alignment_15min_to_10min():
    reg = pd.DataFrame({"x": [1.0, 2.0, 3.0]},
                       index=pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:15",
                                             "2024-01-01 00:30"]))
    reg.index.name = "timestamp_utc"
    win = pd.date_range("2024-01-01 00:00", periods=6, freq="10min")
    out = align_to_windows(reg, win)
    # 00:00->00:00, 00:10->00:00, 00:20->00:15, 00:30->00:30, 00:40->00:30,
    # 00:50 is 20 min after the last interval start -> no match
    assert out.x.tolist()[:5] == [1.0, 1.0, 2.0, 3.0, 3.0]
    assert np.isnan(out.x.iloc[5])
    assert list(out.index) == list(win)
