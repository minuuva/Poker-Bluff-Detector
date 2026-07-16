"""Hand assembly from HUD snapshots."""

import pytest

from pokertell.gamestate.statemachine import (
    ActionType,
    HandAssembler,
    HudSnapshot,
    Street,
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


def _hand_snapshots(offset=0.0):
    return [
        HudSnapshot(t=offset + 0.0, pot=1.5, stacks={"hero": 100, "villain": 100}),
        HudSnapshot(
            t=offset + 1.0,
            pot=1.5,
            stacks={"hero": 100, "villain": 100},
            bets={"hero": 3.0},
            hole_cards={"hero": ["As", "Kd"]},
        ),
        HudSnapshot(
            t=offset + 2.0,
            pot=6.5,
            board=["Qs", "Js", "2c"],
            stacks={"hero": 97, "villain": 97},
        ),
    ]


def test_hand_boundary_on_board_clear():
    assembler = HandAssembler("sess1")
    hands = []
    for snap in _hand_snapshots() + _hand_snapshots(offset=10.0):
        done = assembler.push(snap)
        if done:
            hands.append(done)
    hands.append(assembler.flush())

    assert len(hands) == 2
    assert hands[0].hand_id == "sess1#0001"
    assert hands[0].hole_cards["hero"] == ["As", "Kd"]
    assert hands[0].final_board == ["Qs", "Js", "2c"]
    assert hands[0].players == ["hero", "villain"]


def test_first_bet_is_inferred_as_bet():
    assembler = HandAssembler("sess1")
    for snap in _hand_snapshots():
        assembler.push(snap)
    hand = assembler.flush()
    bets = [a for a in hand.actions if a.action == ActionType.BET]
    assert len(bets) == 1
    assert bets[0].player == "hero"
    assert bets[0].amount == 3.0
