"""Card recognition from HUD sprites.

Card sprites are white rounded cells with a rank glyph on the left (red or
black) and a suit glyph on the right. OCR reads ranks unreliably and suits
not at all, so cards are template-matched.

Finding cells: adjacent sprites touch, so contour detection merges them.
Instead, within a search region: (1) find the horizontal band of rows whose
white fraction says "card row", (2) inside it find runs of columns that are
mostly white, (3) split each run into equal cells by the known sprite width.
Solid white sprites pass the column-fraction test; white TEXT on purple
(names, equity, actions) does not. Callers must still restrict the search
region to where cards can appear (board ROI, panel row-1 bands).

Matching: cells are normalized to CELL x CELL, the rank (left) and suit
(right) glyphs are binarized and matched with TM_CCOEFF_NORMED against a
small template library (13 ranks + 4 suits) harvested from labeled frames.
Card strings use treys format: rank then suit, e.g. "As", "Td", "9c".
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

CELL = 52
WHITE_THRESH = 150
ROW_FRAC = 0.12
COL_FRAC = 0.7
COL_FRAC_RELAXED = 0.4
RUN_EXTEND = 8
MIN_CELL_H = 34
MAX_CELL_H = 66
SPRITE_W = 55
SEAM_SEARCH = 7
GLYPH_THRESH = 150
MAX_WHITE_SAT = 90
MIN_GLYPH_FRAC = 0.04
MAX_GLYPH_FRAC = 0.45

RANKS = "23456789TJQKA"
SUITS = "shdc"


@dataclass(frozen=True)
class CellBox:
    x: int
    y: int
    w: int
    h: int


def _card_white_mask(bgr: np.ndarray) -> np.ndarray:
    """Card-body pixels: bright AND desaturated.

    Brightness alone is not enough: some panel backgrounds (the gold side
    panel in early-2026 sessions) are brighter than WHITE_THRESH. Card white
    has saturation ~25; gold ~180; purple panels are dark anyway.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    white = (hsv[..., 2] > WHITE_THRESH) & (hsv[..., 1] < MAX_WHITE_SAT)
    return (white * 255).astype(np.uint8)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill enclosed holes (the dark glyphs inside card sprites)."""
    padded = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), np.uint8)
    padded[1:-1, 1:-1] = mask
    ff = padded.copy()
    fill_mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(ff, fill_mask, (0, 0), 255)
    holes = (ff == 0)[1:-1, 1:-1]
    return mask | (holes.astype(np.uint8) * 255)


def find_card_cells(region: np.ndarray, sprite_w: int = SPRITE_W) -> list[CellBox]:
    """Locate card sprite cells in a BGR search region.

    The white threshold sits between card white (~180-235) and the purple
    panel / board backgrounds (~50-90); the sprites' interior glyphs are
    dark, so the mask is hole-filled before profiling.
    """
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    white = _fill_holes(_card_white_mask(region)) > 0

    row_frac = white.mean(axis=1)
    bands = _runs(row_frac > ROW_FRAC, min_len=MIN_CELL_H)
    cells: list[CellBox] = []
    for y0, y1 in bands:
        if y1 - y0 > MAX_CELL_H:
            # Trim to the densest window of card height.
            best_y0, best = y0, -1.0
            for yy in range(y0, y1 - MAX_CELL_H + 1):
                s = row_frac[yy : yy + MAX_CELL_H].sum()
                if s > best:
                    best_y0, best = yy, s
            y0, y1 = best_y0, best_y0 + MAX_CELL_H
        band = white[y0:y1]
        col_frac = band.mean(axis=0)
        for x0, x1 in _runs(col_frac > COL_FRAC, min_len=int(sprite_w * 0.6)):
            # Hysteresis: runs are found with a strict threshold, then edges
            # extend under a relaxed one. Sprite borders are slightly darker
            # and a strict cut there can slice off an edge glyph.
            for _ in range(RUN_EXTEND):
                if x0 > 0 and col_frac[x0 - 1] > COL_FRAC_RELAXED:
                    x0 -= 1
            for _ in range(RUN_EXTEND):
                if x1 < len(col_frac) and col_frac[x1] > COL_FRAC_RELAXED:
                    x1 += 1
            n = max(1, round((x1 - x0) / sprite_w))
            edges = _seam_split(col_frac, x0, x1, n)
            for a, b in zip(edges, edges[1:]):
                box = CellBox(x=int(a), y=int(y0), w=int(b - a), h=int(y1 - y0))
                if _looks_like_card(gray, box):
                    cells.append(box)
    return cells


def _seam_split(col_frac: np.ndarray, x0: int, x1: int, n: int) -> list[int]:
    """Boundaries between n touching sprites in [x0, x1].

    An even split can land a few px off and cut a suit glyph. Sprites meet
    at a slightly darker seam, so each boundary snaps to the column-fraction
    minimum near its expected position.
    """
    edges = [int(x0)]
    for k in range(1, n):
        target = round(x0 + k * (x1 - x0) / n)
        lo = max(x0 + 8, target - SEAM_SEARCH)
        hi = min(x1 - 8, target + SEAM_SEARCH + 1)
        edges.append(int(lo + np.argmin(col_frac[lo:hi])) if hi > lo else int(target))
    edges.append(int(x1))
    return edges


def _looks_like_card(gray: np.ndarray, box: CellBox) -> bool:
    """A card cell must contain a dark glyph of plausible size."""
    crop = gray[box.y : box.y + box.h, box.x : box.x + box.w]
    glyph_frac = (crop < GLYPH_THRESH).mean()
    return MIN_GLYPH_FRAC <= glyph_frac <= MAX_GLYPH_FRAC


def _runs(flags: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """Contiguous True runs of at least min_len, as (start, end) pairs."""
    out = []
    start = None
    for i, v in enumerate(flags):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                out.append((start, i))
            start = None
    if start is not None and len(flags) - start >= min_len:
        out.append((start, len(flags)))
    return out


def normalize_cell(region: np.ndarray, box: CellBox) -> np.ndarray:
    """Crop a cell, re-localize to the sprite's white blob, resize to CELL.

    Re-localization standardizes geometry: split boundaries on touching
    sprite pairs can be a few px off, and matching needs the glyphs at
    consistent positions.
    """
    crop = region[box.y : box.y + box.h, box.x : box.x + box.w]
    mask = _fill_holes(_card_white_mask(crop))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        if w >= 28 and h >= 28:
            crop = crop[y : y + h, x : x + w]
    return cv2.resize(crop, (CELL, CELL), interpolation=cv2.INTER_AREA)


def suit_color(cell: np.ndarray) -> str:
    """'red' or 'black' from the suit glyph pixels of a normalized cell."""
    right = cell[:, 22:CELL]
    gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    glyph = gray < GLYPH_THRESH
    if glyph.sum() < 10:
        return "black"
    b, g, r = right[..., 0].astype(int), right[..., 1].astype(int), right[..., 2].astype(int)
    redness = float((r - np.maximum(b, g))[glyph].mean())
    return "red" if redness > 18 else "black"


RED_SUITS = {"h", "d"}
BLACK_SUITS = {"s", "c"}






@dataclass
class CardMatch:
    card: str | None
    rank_score: float
    suit_score: float

    @property
    def score(self) -> float:
        return min(self.rank_score, self.suit_score)


RANK_WINDOW = slice(0, 30)
SUIT_WINDOW = slice(22, CELL)
TEMPLATE_TRIM = 4


def glyph_window(cell: np.ndarray, window: slice) -> np.ndarray:
    """Equalized grayscale glyph window of a normalized cell.

    Histogram equalization normalizes brightness and contrast across
    overlay eras (the Feb-2026 sprites are warmer and dimmer than Jul).
    """
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)[:, window]
    return cv2.equalizeHist(gray)


class CardMatcher:
    """Rank/suit classification by correlating equalized glyph windows.

    Validated design (day 2): whole-cell correlation is dominated by the
    card background and generalizes badly across contexts; binary glyph
    templates and component segmentation are brittle on compressed sprites.
    Correlating equalized grayscale windows (rank: left, suit: right of the
    normalized cell) against labeled exemplars separates cleanly: correct
    suits score 0.93+, correct ranks 0.79+, and corrupted cells announce
    themselves with low scores instead of confident mismatches. Suits only
    compete within their color group (red/black from glyph pixels).

    Template dir layout: <dir>/ranks/<R>.png, <dir>/suits/<s>.png, extra
    exemplar variants as <label>_1.png etc. Templates are full windows;
    matching center-trims them by TEMPLATE_TRIM for shift tolerance.
    Card strings use treys format ('As', 'Td', '9c').
    """

    def __init__(self, template_dir: Path, min_score: float = 0.75) -> None:
        self.min_score = min_score
        self.ranks: dict[str, list[np.ndarray]] = {}
        self.suits: dict[str, list[np.ndarray]] = {}
        template_dir = Path(template_dir)
        for p in sorted((template_dir / "ranks").glob("*.png")):
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                self.ranks.setdefault(p.stem.split("_")[0], []).append(img)
        for p in sorted((template_dir / "suits").glob("*.png")):
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                self.suits.setdefault(p.stem.split("_")[0], []).append(img)
        if not self.ranks or not self.suits:
            raise FileNotFoundError(f"no templates under {template_dir}")

    @staticmethod
    def _best(
        search: np.ndarray, templates: dict[str, list[np.ndarray]]
    ) -> tuple[str | None, float]:
        t = TEMPLATE_TRIM
        best_key, best_score = None, -1.0
        for key, variants in templates.items():
            for tmpl in variants:
                trimmed = tmpl[t:-t, t:-t]
                if trimmed.shape[0] > search.shape[0] or trimmed.shape[1] > search.shape[1]:
                    continue
                score = float(cv2.matchTemplate(search, trimmed, cv2.TM_CCOEFF_NORMED).max())
                if score > best_score:
                    best_key, best_score = key, score
        return best_key, best_score

    def match_cell(self, cell: np.ndarray) -> CardMatch:
        """Identify a normalized (CELL x CELL BGR) sprite cell."""
        rank, r_score = self._best(glyph_window(cell, RANK_WINDOW), self.ranks)
        allowed = RED_SUITS if suit_color(cell) == "red" else BLACK_SUITS
        suit, s_score = self._best(
            glyph_window(cell, SUIT_WINDOW),
            {k: v for k, v in self.suits.items() if k in allowed},
        )
        card = None
        if rank and suit and min(r_score, s_score) >= self.min_score:
            card = rank + suit
        return CardMatch(card=card, rank_score=r_score, suit_score=s_score)

    def read_region(self, region: np.ndarray, sprite_w: int = SPRITE_W) -> list[CardMatch]:
        """Find and identify every card sprite in a search region."""
        return [
            self.match_cell(normalize_cell(region, box))
            for box in find_card_cells(region, sprite_w)
        ]


def build_template(cell: np.ndarray, kind: str) -> np.ndarray:
    """Window exemplar for the template library. kind is 'rank' or 'suit'."""
    window = RANK_WINDOW if kind == "rank" else SUIT_WINDOW
    return glyph_window(cell, window)
