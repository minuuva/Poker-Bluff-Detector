"""Hand strength labeling."""

from pokertell.labels.equity import (
    StrengthClass,
    classify_strength,
    equity_vs_random,
    is_bluff,
)


def test_aces_have_high_equity_preflop():
    eq = equity_vs_random(["As", "Ah"], [], n_trials=400, seed=1)
    assert eq > 0.75


def test_seven_deuce_is_weak():
    eq = equity_vs_random(["7s", "2h"], [], n_trials=400, seed=1)
    assert eq < 0.45


def test_nuts_on_board_is_monster():
    eq = equity_vs_random(["As", "Ks"], ["Qs", "Js", "Ts"], n_trials=300, seed=1)
    assert classify_strength(eq) == StrengthClass.MONSTER


def test_weak_aggressive_is_bluff():
    label = classify_strength(0.2, draw_equity=0.0, is_aggressive_action=True)
    assert label == StrengthClass.BLUFF
    assert is_bluff(label)


def test_weak_passive_is_not_bluff():
    label = classify_strength(0.2, draw_equity=0.0, is_aggressive_action=False)
    assert not is_bluff(label)


def test_draw_classes():
    assert classify_strength(0.35, draw_equity=0.4) == StrengthClass.STRONG_DRAW
    assert classify_strength(0.2, draw_equity=0.2) == StrengthClass.WEAK_DRAW
