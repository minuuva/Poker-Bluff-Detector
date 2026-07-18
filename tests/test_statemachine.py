"""Hand assembly from synthetic snapshot streams.

The synthetic session below encodes the HUD rendering rules the assembler
relies on: persistent action text, the to-act facing marker, panels that
exist only for players in the hand, and noisy reads (name jitter, partial
boards) that voting must absorb.
"""

import pytest

from pokertell.gamestate.names import NameResolver, canonical
from pokertell.gamestate.statemachine import (
    ActionType,
    HudSnapshot,
    Street,
    assemble_session,
    parse_action_text,
    street_from_board,
)


def test_street_from_board():
    assert street_from_board([]) == Street.PREFLOP
    assert street_from_board(["As", "Kd", "2c"]) == Street.FLOP
    assert street_from_board(["As", "Kd", "2c", "7h"]) == Street.TURN
    assert street_from_board(["As", "Kd", "2c", "7h", "9s"]) == Street.RIVER


def test_impossible_board_raises():
    with pytest.raises(ValueError):
        street_from_board(["As", "Kd"])


def test_parse_action_text():
    assert parse_action_text("BET $8,100") == (ActionType.BET, 8100)
    assert parse_action_text("RAISE TO $1,200") == (ActionType.RAISE, 1200)
    assert parse_action_text("CALL $100") == (ActionType.CALL, 100)
    assert parse_action_text("ALL IN $10,500") == (ActionType.ALL_IN, 10500)
    assert parse_action_text("CHECK") == (ActionType.CHECK, None)
    assert parse_action_text("$8,100 TO CALL") is None


def test_name_canonical_and_merge():
    assert canonical("JACK C") == canonical("JACKC")
    resolver = NameResolver(["ALICE"] * 10 + ["AL1CE"] + ["BOB"] * 10)
    assert resolver.resolve("AL1CE") == "ALICE"
    assert resolver.resolve("ALICE") == "ALICE"
    assert resolver.display("ALICE") == "ALICE"


def _snap(t, **kw):
    return HudSnapshot(t=t, **kw)


def _synthetic_session():
    """Two hands separated by dead air, with jitter and partial reads."""
    alice = {"positions": {"ALICE": "SB"}, "stacks": {"ALICE": 10000.0}}
    flop = ["Ah", "7d", "2c"]
    snaps = [
        # Hand 1: preflop, ALICE raises, BOB faces it and calls.
        _snap(1, pot=150, positions={"ALICE": "SB", "BOB": "BB"},
              stacks={"ALICE": 10000, "BOB": 8000},
              hole_cards={"ALICE": ["Kd", "9d"], "BOB": ["9h", "7h"]},
              blinds=[50, 100]),
        _snap(2, pot=150, positions={"ALICE": "SB", "BOB": "BB"},
              stacks={"ALICE": 9700, "BOB": 8000},
              actions={"ALICE": "RAISE TO $300"},
              hole_cards={"ALICE": ["Kd", "9d"], "BOB": ["9h", "7h"]},
              to_act="BOB", to_call=200, blinds=[50, 100]),
        _snap(3, pot=150, positions={"ALICE": "SB", "BOB": "BB"},
              stacks={"ALICE": 9700, "BOB": 8000},
              actions={"ALICE": "RAISE TO $300"},
              to_act="BOB", to_call=200, blinds=[50, 100]),
        _snap(4, pot=600, positions={"ALICE": "SB", "BOB": "BB"},
              stacks={"ALICE": 9700, "BOB": 7700},
              actions={"ALICE": "RAISE TO $300", "BOB": "CALL $300"},
              hole_cards={"ALICE": ["Kd", "9d"], "BOB": ["9h", "7h"]}),
        # Flop: one partial board read, then stable; ALICE bets, BOB folds
        # (panel disappears).
        _snap(5, pot=600, board=flop[:2],  # dealing animation, partial
              positions={"ALICE": "SB", "BOB": "BB"},
              stacks={"ALICE": 9700, "BOB": 7700},
              actions={"ALICE": "RAISE TO $300", "BOB": "CALL $300"}),
        _snap(6, pot=600, board=flop,
              positions={"ALICE": "SB", "BOB": "BB"},
              stacks={"ALICE": 9700, "BOB": 7700},
              hole_cards={"ALICE": ["Kd", "9d"], "BOB": ["9h", "7h"]},
              actions={"ALICE": "RAISE TO $300", "BOB": "CALL $300"},
              to_act="ALICE", to_call=None),
        _snap(7, pot=600, board=flop,
              positions={"ALICE": "SB", "BOB": "BB"},
              stacks={"ALICE": 9200, "BOB": 7700},
              actions={"ALICE": "BET $500", "BOB": "CALL $300"}),
        _snap(8, pot=600, board=flop,
              positions={"ALICE": "SB"}, stacks={"ALICE": 9200},
              actions={"ALICE": "BET $500"}),
        _snap(9, pot=1100, board=flop,
              positions={"ALICE": "SB"}, stacks={"ALICE": 9200},
              actions={"ALICE": "BET $500"}),
        _snap(10, pot=1100, board=flop, **alice),
        # Dead air between hands.
        _snap(11), _snap(12), _snap(13),
        # Hand 2: positions rotated, one jittered name read.
        _snap(14, pot=150, positions={"ALICE": "BB", "BOB": "SB"},
              stacks={"ALICE": 10500, "BOB": 7300},
              hole_cards={"ALICE": ["2c", "2d"]}, blinds=[50, 100]),
        _snap(15, pot=150, positions={"AL1CE": "BB", "BOB": "SB"},
              stacks={"AL1CE": 10500, "BOB": 7300},
              actions={"BOB": "CALL $100"}, blinds=[50, 100]),
        _snap(16, pot=200, positions={"ALICE": "BB", "BOB": "SB"},
              stacks={"ALICE": 10500, "BOB": 7200},
              actions={"BOB": "CALL $100", "ALICE": "CHECK"}, blinds=[50, 100]),
        _snap(17, pot=200, positions={"ALICE": "BB", "BOB": "SB"},
              stacks={"ALICE": 10500, "BOB": 7200},
              actions={"BOB": "CALL $100", "ALICE": "CHECK"}, blinds=[50, 100]),
    ]
    return snaps


def test_two_hands_segmented():
    hands, report = assemble_session(_synthetic_session(), "sess")
    assert report["hands"] == 2
    assert hands[0].hand_id == "sess#0001"
    assert hands[0].players == ["ALICE", "BOB"]
    assert hands[1].players == ["ALICE", "BOB"]  # AL1CE merged


def test_hand1_board_voted_over_partial_reads():
    hands, _ = assemble_session(_synthetic_session(), "sess")
    assert hands[0].board == ["Ah", "7d", "2c"]


def test_hand1_hole_cards_voted():
    hands, _ = assemble_session(_synthetic_session(), "sess")
    assert hands[0].hole_cards == {"ALICE": ["Kd", "9d"], "BOB": ["9h", "7h"]}


def test_hand1_actions_include_fold_by_disappearance():
    hands, _ = assemble_session(_synthetic_session(), "sess")
    acts = [(a.player, a.action) for a in hands[0].actions]
    assert ("ALICE", ActionType.RAISE) in acts
    assert ("BOB", ActionType.CALL) in acts
    assert ("ALICE", ActionType.BET) in acts
    assert ("BOB", ActionType.FOLD) in acts
    fold = next(a for a in hands[0].actions if a.action == ActionType.FOLD)
    assert fold.street == Street.FLOP


def test_hand1_to_act_decision_window():
    hands, _ = assemble_session(_synthetic_session(), "sess")
    bob_call = next(
        d for d in hands[0].decisions
        if d.player == "BOB" and d.action == ActionType.CALL
    )
    assert bob_call.window_source == "to_act"
    assert bob_call.t_start == 2
    assert bob_call.t_end == 4
    assert bob_call.street == Street.PREFLOP
    assert bob_call.to_call == 200
    assert bob_call.pot_before == 150
    assert bob_call.hole_cards == ["9h", "7h"]

    alice_bet = next(
        d for d in hands[0].decisions
        if d.player == "ALICE" and d.action == ActionType.BET
    )
    assert alice_bet.window_source == "to_act"
    assert alice_bet.street == Street.FLOP
    assert alice_bet.amount == 500


def test_hand2_rotation_and_check_decision():
    hands, _ = assemble_session(_synthetic_session(), "sess")
    assert hands[1].positions == {"ALICE": "BB", "BOB": "SB"}
    check = next(
        d for d in hands[1].decisions
        if d.player == "ALICE" and d.action == ActionType.CHECK
    )
    assert check.window_source == "action_only"


def test_positions_conflict_without_dead_gap_splits_hands():
    snaps = []
    for t in range(1, 5):
        snaps.append(_snap(t, pot=100, positions={"A": "SB", "B": "BB"},
                           stacks={"A": 1000, "B": 1000}))
    for t in range(5, 9):
        snaps.append(_snap(t, pot=100, positions={"A": "BB", "B": "SB"},
                           stacks={"A": 1000, "B": 1000}))
    hands, report = assemble_session(snaps, "sess")
    assert report["hands"] == 2
