"""Grouped splits must never leak groups across the boundary."""

import pandas as pd
import pytest

from pokertell.eval.splits import player_holdout, session_holdout


def _df():
    return pd.DataFrame(
        {
            "session_id": ["s1"] * 3 + ["s2"] * 3 + ["s3"] * 3 + ["s4"] * 3,
            "player": ["a", "b", "c"] * 4,
            "x": range(12),
        }
    )


def test_session_holdout_no_leakage():
    df = _df()
    train_idx, test_idx = session_holdout(df, test_frac=0.25, seed=3)
    train_sessions = set(df.loc[train_idx, "session_id"])
    test_sessions = set(df.loc[test_idx, "session_id"])
    assert train_sessions.isdisjoint(test_sessions)
    assert len(train_idx) + len(test_idx) == len(df)


def test_session_holdout_deterministic():
    df = _df()
    a = session_holdout(df, seed=3)
    b = session_holdout(df, seed=3)
    assert list(a[0]) == list(b[0]) and list(a[1]) == list(b[1])


def test_player_holdout():
    df = _df()
    train_idx, test_idx = player_holdout(df, ["c"])
    assert set(df.loc[test_idx, "player"]) == {"c"}
    assert "c" not in set(df.loc[train_idx, "player"])


def test_player_holdout_unknown_player_raises():
    with pytest.raises(ValueError):
        player_holdout(_df(), ["nobody"])
