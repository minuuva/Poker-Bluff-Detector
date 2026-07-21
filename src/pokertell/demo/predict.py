"""LOSO-consistent probabilities for the demo clip.

The clip must show the same kind of number the report is built on: models
fit only on the sessions the clip is NOT from. One betting-only model and
one baseline + behavior model, both predicting the clip's decision.
"""

import pandas as pd

from pokertell.config import Paths
from pokertell.dataset import BEHAVIOR_FEATURES, load_dataset
from pokertell.models.baseline import BETTING_FEATURES
from pokertell.models.train import _make_model


def demo_probabilities(
    paths: Paths,
    session_id: str,
    hand_id: str,
    player: str,
    t_end: float,
    target: str = "is_bluff",
    min_coverage: float = 0.5,
    model_kind: str = "logreg",
) -> dict:
    """P(target) for one decision from held-out-session models.

    Returns {"base": float, "full": float, "row": dict} where row is the
    decision's joined feature row (betting + z-scored behavior + labels).
    """
    df = load_dataset(paths)
    match = df[
        (df["hand_id"] == hand_id)
        & (df["player"] == player)
        & (df["t_end"].round(2) == round(t_end, 2))
    ]
    if match.empty:
        raise ValueError(f"decision not found: {hand_id} {player} t_end={t_end}")
    row = match.iloc[[0]]

    train = df[(df["session_id"] != session_id) & df[target].notna()]
    if train.empty:
        raise ValueError("no held-out training rows; need a second session")
    base = _make_model(model_kind).fit(train[BETTING_FEATURES], train[target])

    behavior_cols = [c for c in BEHAVIOR_FEATURES if c in train.columns]
    covered = train[
        (train.get("face_coverage", pd.Series(0, index=train.index)).fillna(0) >= min_coverage)
        | (train.get("pose_coverage", pd.Series(0, index=train.index)).fillna(0) >= min_coverage)
    ]
    full_cols = BETTING_FEATURES + behavior_cols
    full = _make_model(model_kind).fit(covered[full_cols], covered[target])

    strength = {}
    strength_train = df[
        (df["session_id"] != session_id) & df["strength_class"].notna()
    ]
    if len(strength_train) and strength_train["strength_class"].nunique() > 1:
        clf = _make_model(model_kind).fit(
            strength_train[full_cols], strength_train["strength_class"]
        )
        p = clf.predict_proba(row[full_cols])[0]
        strength = {cls: float(v) for cls, v in zip(clf.classes_, p)}

    return {
        "base": float(base.predict_proba(row[BETTING_FEATURES])[0, 1]),
        "full": float(full.predict_proba(row[full_cols])[0, 1]),
        "strength": strength,
        "row": row.iloc[0].to_dict(),
    }
