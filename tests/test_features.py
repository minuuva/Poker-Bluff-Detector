"""Per-player z-scoring and window aggregation."""

import numpy as np
import pandas as pd

from pokertell.behavior.features import summarize_window, zscore_per_player


def test_zscore_is_per_player():
    df = pd.DataFrame(
        {
            "player": ["a"] * 4 + ["b"] * 4,
            "blink_rate": [1.0, 2.0, 3.0, 4.0, 100.0, 200.0, 300.0, 400.0],
        }
    )
    out = zscore_per_player(df, ["blink_rate"])
    for player in ("a", "b"):
        vals = out.loc[out.player == player, "blink_rate"]
        assert abs(vals.mean()) < 1e-9
        assert abs(vals.std() - 1.0) < 1e-9


def test_zscore_zero_variance_gives_nan():
    df = pd.DataFrame({"player": ["a", "a"], "x": [2.0, 2.0]})
    out = zscore_per_player(df, ["x"])
    assert out["x"].isna().all()


def test_summarize_window_handles_nan_frames():
    stats = summarize_window({"gaze": np.array([1.0, np.nan, 3.0])})
    assert stats["gaze_mean"] == 2.0
    assert stats["gaze_max"] == 3.0


def test_summarize_window_all_nan():
    stats = summarize_window({"gaze": np.array([np.nan, np.nan])})
    assert np.isnan(stats["gaze_mean"])
