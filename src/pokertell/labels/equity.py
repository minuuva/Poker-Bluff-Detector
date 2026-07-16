"""Hand strength labeling from exposed hole cards.

Labels follow the class taxonomy of the original system (more informative than
binary bluff/no-bluff): bluff, weak draw, medium, strong draw, strong, monster.

Equity here is Monte Carlo equity versus a uniform random opponent hand. That
is a deliberate simplification for v1: range-aware equity (eval7 hand-vs-range)
is a fast follow once the betting baseline defines villain ranges. The class
thresholds below are provisional and must be sanity-checked against real
labeled hands before the modeling milestone.

Card strings use treys format: rank then suit, e.g. "As", "Td", "9c".
"""

import random
from enum import Enum

from treys import Card, Evaluator


class StrengthClass(str, Enum):
    BLUFF = "bluff"
    WEAK_DRAW = "weak_draw"
    MEDIUM = "medium"
    STRONG_DRAW = "strong_draw"
    STRONG = "strong"
    MONSTER = "monster"


# Provisional equity thresholds for made-hand classes.
MONSTER_EQ = 0.85
STRONG_EQ = 0.70
MEDIUM_EQ = 0.45
# Draws: weak made-hand equity now but meaningful improvement potential.
DRAW_IMPROVE_EQ = 0.15


def equity_vs_random(
    hole: list[str],
    board: list[str],
    n_trials: int = 500,
    seed: int | None = None,
) -> float:
    """Monte Carlo equity of a hole-card pair versus one random opponent.

    Splits count as half a win. Accurate to roughly +/- 0.02 at the default
    trial count, which is enough for class labeling.
    """
    rng = random.Random(seed)
    evaluator = Evaluator()
    hero = [Card.new(c) for c in hole]
    board_cards = [Card.new(c) for c in board]

    deck = [
        Card.new(r + s)
        for r in "23456789TJQKA"
        for s in "shdc"
        if Card.new(r + s) not in set(hero) | set(board_cards)
    ]

    wins = 0.0
    for _ in range(n_trials):
        drawn = rng.sample(deck, 2 + (5 - len(board_cards)))
        villain = drawn[:2]
        runout = board_cards + drawn[2:]
        hero_rank = evaluator.evaluate(runout, hero)
        villain_rank = evaluator.evaluate(runout, villain)
        if hero_rank < villain_rank:  # treys: lower rank is better
            wins += 1.0
        elif hero_rank == villain_rank:
            wins += 0.5
    return wins / n_trials


def classify_strength(
    equity: float,
    draw_equity: float = 0.0,
    is_aggressive_action: bool = False,
) -> StrengthClass:
    """Map equity (and drawing potential) to a strength class.

    A "bluff" is defined as an aggressive action (bet or raise) taken with a
    weak hand and little to improve to. Weak hands without aggression are
    labeled by strength alone; only aggressive actions can be bluffs.

    Args:
        equity: made-hand equity versus random at the decision point.
        draw_equity: additional equity from likely improvement (0 preflop).
        is_aggressive_action: whether the decision was a bet or raise.
    """
    if equity >= MONSTER_EQ:
        return StrengthClass.MONSTER
    if equity >= STRONG_EQ:
        return StrengthClass.STRONG
    if equity >= MEDIUM_EQ:
        return StrengthClass.MEDIUM
    if draw_equity >= DRAW_IMPROVE_EQ:
        if equity + draw_equity >= STRONG_EQ:
            return StrengthClass.STRONG_DRAW
        return StrengthClass.WEAK_DRAW
    if is_aggressive_action:
        return StrengthClass.BLUFF
    return StrengthClass.WEAK_DRAW


def is_bluff(label: StrengthClass) -> bool:
    """Binary target for the headline ablation metric."""
    return label == StrengthClass.BLUFF
