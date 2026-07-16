"""Per-decision behavioral feature assembly.

The unit of analysis is one player decision: the window from when action is on
the player until the action is committed. Frame-level signals get aggregated
into one row per decision here, then z-scored per player.

Why per-player z-scores: tells are player-specific. The literature's practical
consensus (Elwood) is that behavioral signal is only usable against a player's
own baseline, not a population mean. Modeling raw feature values would mostly
learn who the player is, not what their behavior says.
"""

import numpy as np
import pandas as pd


def summarize_window(samples: dict[str, np.ndarray]) -> dict[str, float]:
    """Aggregate frame-level channels into per-decision statistics.

    For each named channel, emits mean, std, and max. NaN-safe: windows with
    missed detections keep whatever frames were tracked.
    """
    out: dict[str, float] = {}
    for name, values in samples.items():
        values = np.asarray(values, dtype=float)
        valid = values[~np.isnan(values)]
        if len(valid) == 0:
            out[f"{name}_mean"] = float("nan")
            out[f"{name}_std"] = float("nan")
            out[f"{name}_max"] = float("nan")
            continue
        out[f"{name}_mean"] = float(np.mean(valid))
        out[f"{name}_std"] = float(np.std(valid))
        out[f"{name}_max"] = float(np.max(valid))
    return out


def zscore_per_player(
    df: pd.DataFrame,
    feature_cols: list[str],
    player_col: str = "player",
) -> pd.DataFrame:
    """Z-score each feature within each player's own distribution.

    Returns a copy. Players with fewer than 2 decisions or zero variance in a
    feature get NaN for that feature (not enough baseline to normalize
    against), which downstream models treat as missing.
    """
    out = df.copy()
    grouped = out.groupby(player_col)
    for col in feature_cols:
        mean = grouped[col].transform("mean")
        std = grouped[col].transform("std")
        out[col] = (out[col] - mean) / std.where(std > 0)
    return out
