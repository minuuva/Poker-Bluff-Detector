"""Grouped train/test splits.

Rules learned from the deception-detection literature the hard way (one 2026
study reported 99.94% accuracy with overlapping-window splits and AUC 0.58
once sessions were held out properly):

1. A session's decisions are heavily autocorrelated; whole sessions go to one
   side of the split, never individual decisions or frames.
2. Player-held-out is the stricter test and the more honest headline for
   "does this generalize", since tells are player-specific. Expect the
   per-player-baseline features to carry it and raw features to collapse.
"""

import numpy as np
import pandas as pd


def session_holdout(
    df: pd.DataFrame,
    session_col: str = "session_id",
    test_frac: float = 0.25,
    seed: int = 7,
) -> tuple[pd.Index, pd.Index]:
    """Assign whole sessions to train/test. Returns (train_index, test_index)."""
    sessions = np.array(sorted(df[session_col].unique()))
    if len(sessions) < 2:
        raise ValueError("need at least 2 sessions for a session-held-out split")
    rng = np.random.default_rng(seed)
    rng.shuffle(sessions)
    n_test = max(1, round(len(sessions) * test_frac))
    test_sessions = set(sessions[:n_test])
    mask = df[session_col].isin(test_sessions)
    return df.index[~mask], df.index[mask]


def player_holdout(
    df: pd.DataFrame,
    test_players: list[str],
    player_col: str = "player",
) -> tuple[pd.Index, pd.Index]:
    """Hold out all decisions by the named players. Returns (train, test) indices."""
    missing = set(test_players) - set(df[player_col].unique())
    if missing:
        raise ValueError(f"players not in data: {sorted(missing)}")
    mask = df[player_col].isin(set(test_players))
    return df.index[~mask], df.index[mask]
