"""Assemble the modeling dataset: decisions + behavior, z-scored per player."""

import pandas as pd

from pokertell.behavior.face import FACE_FEATURES
from pokertell.behavior.features import zscore_per_player
from pokertell.behavior.pose import POSE_FEATURES
from pokertell.config import Paths

BEHAVIOR_FEATURES = FACE_FEATURES + POSE_FEATURES


def load_dataset(paths: Paths) -> pd.DataFrame:
    """All sessions' decision tables, left-joined with behavior features.

    Behavioral columns are z-scored within each player's own distribution
    (tells are player-specific; raw values mostly encode who the player
    is). Normalization uses the full dataset rather than train folds only:
    per-player centering leaks no label information, and the alternative
    breaks players who straddle folds.
    """
    dec_files = sorted(paths.features.glob("*.decisions.csv"))
    if not dec_files:
        raise FileNotFoundError("no decision tables in data/features; run label first")
    df = pd.concat([pd.read_csv(f) for f in dec_files], ignore_index=True)

    beh_files = sorted(paths.features.glob("*.behavior.csv"))
    if beh_files:
        beh = pd.concat([pd.read_csv(f) for f in beh_files], ignore_index=True)
        df = df.merge(
            beh.drop(columns=["t_start", "window_s", "n_frames"], errors="ignore"),
            on=["hand_id", "player", "t_end"],
            how="left",
        )
        df = zscore_per_player(df, [c for c in BEHAVIOR_FEATURES if c in df.columns])
    return df
