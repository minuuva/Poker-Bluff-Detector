"""Betting features and label construction."""

import math

from pokertell.labels.build import build_decision_table, label_decision
from pokertell.models.baseline import betting_features

HAND = {
    "hand_id": "s#0001",
    "players": ["A", "B", "C"],
    "actions": [
        {"player": "A", "action": "raise", "amount": 300, "street": "preflop", "t": 10.0},
        {"player": "C", "action": "fold", "amount": None, "street": "preflop", "t": 12.0},
        {"player": "B", "action": "call", "amount": 300, "street": "preflop", "t": 15.0},
        {"player": "A", "action": "bet", "amount": 500, "street": "flop", "t": 30.0},
    ],
    "decisions": [
        {
            "hand_id": "s#0001", "player": "B", "street": "preflop", "action": "call",
            "amount": 300, "t_start": 11.0, "t_end": 15.0, "window_source": "to_act",
            "pot_before": 450, "to_call": 200, "stack_before": 8000, "position": "BB",
            "equity_pct": 40.0, "board": [], "hole_cards": ["9h", "7h"],
        },
        {
            "hand_id": "s#0001", "player": "A", "street": "flop", "action": "bet",
            "amount": 500, "t_start": 28.0, "t_end": 30.0, "window_source": "to_act",
            "pot_before": 650, "to_call": None, "stack_before": 9700, "position": "SB",
            "equity_pct": 60.0, "board": ["As", "Kd", "9h"], "hole_cards": ["7c", "5d"],
        },
    ],
}


def test_betting_features_call_decision():
    f = betting_features(HAND["decisions"][0], HAND)
    assert f["street_idx"] == 0
    assert f["bet_to_pot"] == 0.0          # passive action
    assert abs(f["pot_odds"] - 200 / 650) < 1e-9
    assert f["is_aggressive"] == 0.0
    assert f["n_prior_raises"] == 1        # A's preflop raise
    assert f["players_in_hand"] == 3       # C folds after window start
    assert f["position_idx"] == 1
    assert f["time_to_act_s"] == 4.0


def test_betting_features_bet_decision():
    f = betting_features(HAND["decisions"][1], HAND)
    assert f["street_idx"] == 1
    assert abs(f["bet_to_pot"] - math.log1p(500 / 650)) < 1e-9
    assert f["is_aggressive"] == 1.0
    assert f["n_prior_raises"] == 1        # only the preflop raise precedes t=30
    assert f["players_in_hand"] == 2       # C folded at t=12 <= 28
    assert math.isnan(f["pot_odds"])       # nothing to call


def test_label_weak_bet_is_bluff():
    lab = label_decision(HAND["decisions"][1])  # 7c2d on AK2 rainbow
    assert lab["equity_mc"] < 0.45
    assert lab["is_bluff"] == 1.0
    assert lab["is_weak"] == 1.0
    assert lab["strength_class"] == "bluff"


def test_label_passive_action_has_no_bluff_target():
    lab = label_decision(HAND["decisions"][0])
    assert math.isnan(lab["is_bluff"])
    assert lab["equity_mc"] > 0


def test_labels_are_deterministic():
    a = label_decision(HAND["decisions"][1])
    b = label_decision(HAND["decisions"][1])
    assert a["equity_mc"] == b["equity_mc"]


def test_build_decision_table_shape():
    df = build_decision_table([HAND])
    assert len(df) == 2
    assert df["session_id"].unique().tolist() == ["s"]
    assert "bet_to_pot" in df.columns
    assert "equity_mc" in df.columns
