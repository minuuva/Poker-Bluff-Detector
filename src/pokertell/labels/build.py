"""Build the per-decision table: identifiers, betting features, and labels.

Labels come from the broadcast's exposed hole cards. Hand strength is Monte
Carlo equity versus a random hand at the decision's street (the vacuum
strength of the holding, matching the original system's class taxonomy).
The broadcast's own win percentage is carried along as equity_broadcast: it
is equity versus the opponents' ACTUAL hands, so it is not a fair label
(it leaks opponent information) but it is a useful cross-check.

Two binary targets:
- is_bluff: aggressive action taken with a weak, low-potential hand.
  Defined only for aggressive actions (bet/raise/all-in); NaN elsewhere.
- is_weak: equity below the medium threshold, defined for all carded
  decisions. More rows, softer question; useful at small sample sizes.

Draw potential is not yet modeled (draw_equity=0): a flush draw labels as
weak even though it plays closer to medium. Revisit before scaling up.
"""

import math
import zlib

import pandas as pd

from pokertell.labels.equity import (
    MEDIUM_EQ,
    classify_strength,
    equity_vs_random,
    is_bluff,
)
from pokertell.models.baseline import AGGRESSIVE, betting_features

MC_TRIALS = 400


def label_decision(decision: dict) -> dict:
    cards = decision.get("hole_cards")
    aggressive = decision.get("action") in AGGRESSIVE
    out = {
        "equity_mc": math.nan,
        "strength_class": None,
        "is_bluff": math.nan,
        "is_weak": math.nan,
        "equity_broadcast": decision.get("equity_pct"),
    }
    board = decision.get("board") or []
    if not cards or len(cards) != 2:
        return out
    if len(set(cards + board)) != len(cards) + len(board):
        # A misread somewhere: the same card cannot be in a hand and on the
        # board. The assembler flags these hands; leave the label as NaN.
        return out
    seed = zlib.crc32(f"{decision['hand_id']}|{decision['player']}".encode()) & 0x7FFFFFFF
    eq = equity_vs_random(cards, board, n_trials=MC_TRIALS, seed=seed)
    cls = classify_strength(eq, draw_equity=0.0, is_aggressive_action=aggressive)
    out["equity_mc"] = round(eq, 4)
    out["strength_class"] = cls.value
    out["is_weak"] = float(eq < MEDIUM_EQ)
    if aggressive:
        out["is_bluff"] = float(is_bluff(cls))
    return out


def build_decision_table(hands: list[dict]) -> pd.DataFrame:
    """One labeled, feature-bearing row per Decision across hands."""
    rows = []
    for hand in hands:
        for decision in hand["decisions"]:
            rows.append(
                {
                    "hand_id": decision["hand_id"],
                    "session_id": decision["hand_id"].split("#")[0],
                    "player": decision["player"],
                    "t_start": decision["t_start"],
                    "t_end": decision["t_end"],
                    "street": decision["street"],
                    "action": decision["action"],
                    "window_source": decision["window_source"],
                    **betting_features(decision, hand),
                    **label_decision(decision),
                }
            )
    return pd.DataFrame(rows)
