"""Card cell finding and matching.

Fixture cells under tests/fixtures/cells are real normalized sprites from
the two hand-verified HCL frames; the template library in configs/templates
was harvested from the same sessions (partially overlapping exemplars, so
these tests validate wiring and obvious regressions, not generalization;
generalization was measured leave-one-out during day 2 at 18/18 suits,
12/13 ranks).
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from pokertell.gamestate.cards import (
    CELL,
    CardMatcher,
    find_card_cells,
    normalize_cell,
    suit_color,
)

FIXTURES = Path(__file__).parent / "fixtures" / "cells"
TEMPLATES = Path(__file__).parent.parent / "configs" / "templates" / "hcl"


def _synthetic_region(n_cards=2, card_w=55, card_h=50, gap=3):
    """Purple background with adjacent white card sprites and dark glyphs.

    Real sprites meet at a darker seam; cv2.rectangle corners are inclusive
    so the gap leaves (gap - 1) dark columns between cards.
    """
    region = np.full((70, 300, 3), (120, 40, 90), np.uint8)
    for i in range(n_cards):
        x0 = 30 + i * (card_w + gap)
        cv2.rectangle(region, (x0, 10), (x0 + card_w, 10 + card_h), (235, 235, 235), -1)
        cv2.putText(region, "9", (x0 + 6, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (30, 30, 30), 3)
        cv2.circle(region, (x0 + 40, 35), 9, (30, 30, 30), -1)
    return region


def test_find_cells_splits_adjacent_cards():
    region = _synthetic_region(n_cards=3)
    cells = find_card_cells(region)
    assert len(cells) == 3
    xs = sorted(c.x for c in cells)
    assert xs[1] - xs[0] == pytest.approx(58, abs=6)


def test_find_cells_empty_background():
    region = np.full((70, 300, 3), (120, 40, 90), np.uint8)
    assert find_card_cells(region) == []


def test_find_cells_ignores_white_text():
    region = np.full((70, 300, 3), (120, 40, 90), np.uint8)
    cv2.putText(region, "$8,100 TO CALL", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (255, 255, 255), 2)
    assert find_card_cells(region) == []


def test_normalize_cell_shape():
    region = _synthetic_region(n_cards=1)
    cells = find_card_cells(region)
    assert len(cells) == 1
    assert normalize_cell(region, cells[0]).shape == (CELL, CELL, 3)


def test_suit_color_on_fixtures():
    red = cv2.imread(str(FIXTURES / "Kd.png"))
    black = cv2.imread(str(FIXTURES / "Ac.png"))
    assert suit_color(red) == "red"
    assert suit_color(black) == "black"


def test_matcher_identifies_fixture_cells():
    matcher = CardMatcher(TEMPLATES)
    for path in sorted(FIXTURES.glob("*.png")):
        cell = cv2.imread(str(path))
        m = matcher.match_cell(cell)
        assert m.card == path.stem, f"{path.stem} matched {m.card} ({m.score:.2f})"
        assert m.score >= 0.75


def test_matcher_low_confidence_returns_none():
    matcher = CardMatcher(TEMPLATES)
    noise = np.random.default_rng(0).integers(0, 255, (CELL, CELL, 3), dtype=np.uint8)
    assert matcher.match_cell(noise).card is None
