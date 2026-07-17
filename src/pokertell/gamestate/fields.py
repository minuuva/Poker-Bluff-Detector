"""Assign detected text boxes to HUD fields.

Pure logic: takes TextBox detections plus the HudLayout and produces one
PanelRead per player in the hand and the shared pot/blinds values.

Player panels stack in the left column, one per player in the hand (2 to 8),
in action order top to bottom. The stack's anchor and pitch drift slightly
across sessions and the acting player's panel indents right, so panels are
found by CLUSTERING detections into rows by y proximity and classifying each
row by content, not by fixed bands:

- row 1 of a panel: player name (leftmost alpha text) and win-equity '21%'
- row 2 of a panel: action text ('CALL $100', 'ALL IN $10,500',
  '$10,400 TO CALL'), stack money, and a position code (SB, BB, 3B, CO, ...)

'$X TO CALL' doubles as the to-act signal: the HUD renders it only on the
player currently facing a decision. Pot and stakes stay fixed-ROI (verified
stable across sessions).
"""

import re
from dataclasses import dataclass, field

from pokertell.gamestate.ocr import TextBox, parse_money, parse_stakes
from pokertell.gamestate.rois import HudLayout

PERCENT_RE = re.compile(r"^\d{1,3}%$")
MONEY_RE = re.compile(r"^\$[\d,]+(?:\.\d+)?[KM]?$")
POSITION_RE = re.compile(r"^(SB|BB|3B|4B|BTN|CO|HJ|LJ|MP\d?|UTG(\+\d)?|ST|STRADDLE)$")
ACTION_RE = re.compile(
    r"^(BET|RAISE(\s*TO)?|CALL|CALLS|ALL[\s-]?IN)\s*\$?[\d,]*[KM]?$|^(CHECK|FOLD)$"
)
TO_CALL_RE = re.compile(r"^\$[\d,]+[KM]?\s*TO\s*CALL$")

# Geometry of the left panel column (1080p HCL, 2026 overlay), intentionally
# loose. Panel text starts at x 65 (normal) or ~145 (acting-player indent);
# table-felt sponsor logos sit at x 400+ and are rejected as names.
PANEL_X_MAX = 590
PANEL_Y_MIN = 420
NAME_X_MAX = 330
ROW_GAP = 28
PANEL_PAIR_GAP = 85


@dataclass
class PanelRead:
    """Everything read from one player's panel at one snapshot."""

    slot: int
    name: str | None = None
    equity_pct: float | None = None
    stack: float | None = None
    position: str | None = None
    action_text: str | None = None
    action_amount: float | None = None
    to_call: float | None = None
    y_top: float = 0.0
    name_box: TextBox | None = None
    equity_box: TextBox | None = None
    raw: list[str] = field(default_factory=list)


@dataclass
class HudRead:
    """One snapshot's worth of assigned fields."""

    pot: float | None = None
    blinds: list[float] = field(default_factory=list)
    panels: dict[int, PanelRead] = field(default_factory=dict)
    unassigned: list[TextBox] = field(default_factory=list)


def _contains(roi, box: TextBox) -> bool:
    return roi.x <= box.cx <= roi.x + roi.w and roi.y <= box.cy <= roi.y + roi.h


def _cluster_rows(boxes: list[TextBox]) -> list[list[TextBox]]:
    """Group boxes into visual rows by y proximity."""
    rows: list[list[TextBox]] = []
    for box in sorted(boxes, key=lambda b: b.cy):
        if rows and box.cy - rows[-1][-1].cy < ROW_GAP:
            rows[-1].append(box)
        else:
            rows.append([box])
    return rows


def _classify_row(row: list[TextBox]) -> str:
    """Label a row cluster as 'row1' (name/equity), 'row2' (action/stack/pos),
    or 'junk'."""
    has_row2 = any(
        TO_CALL_RE.match(t) or ACTION_RE.match(t) or POSITION_RE.match(t) or MONEY_RE.match(t)
        for t in (b.text.upper().strip() for b in row)
    )
    if has_row2:
        return "row2"
    has_name = any(
        len(b.text) >= 2 and any(c.isalpha() for c in b.text) and b.x < NAME_X_MAX
        for b in row
    )
    has_equity = any(PERCENT_RE.match(b.text.strip()) for b in row)
    if has_name or has_equity:
        return "row1"
    return "junk"


def _fill_row1(panel: PanelRead, row: list[TextBox]) -> None:
    for box in row:
        text = box.text.strip()
        panel.raw.append(text)
        if PERCENT_RE.match(text):
            panel.equity_pct = float(text.rstrip("%"))
            panel.equity_box = box
        elif len(text) >= 2 and any(c.isalpha() for c in text) and box.x < NAME_X_MAX:
            # Card-rank bleed-through ('10', 'K 9') is digits/single letters;
            # require a real word. Leftmost qualifying text wins.
            if panel.name_box is None or box.x < panel.name_box.x:
                panel.name = text
                panel.name_box = box


def _fill_row2(panel: PanelRead, row: list[TextBox]) -> None:
    for box in row:
        text = box.text.upper().strip()
        panel.raw.append(box.text.strip())
        if TO_CALL_RE.match(text):
            panel.to_call = parse_money(text.split("TO")[0])
            panel.action_text = text
        elif ACTION_RE.match(text):
            panel.action_text = text
            panel.action_amount = parse_money(text)
        elif POSITION_RE.match(text):
            panel.position = text
        elif MONEY_RE.match(text):
            panel.stack = parse_money(text)


def assign_fields(boxes: list[TextBox], layout: HudLayout) -> HudRead:
    """Assign every detection to a field; keep the leftovers for debugging."""
    read = HudRead()
    pot_roi = layout.rois.get("pot")
    stakes_roi = layout.rois.get("stakes")

    panel_candidates: list[TextBox] = []
    for box in boxes:
        text = box.text.upper()
        if pot_roi is not None and _contains(pot_roi, box) and MONEY_RE.match(text):
            read.pot = parse_money(box.text)
            continue
        if stakes_roi is not None and _contains(stakes_roi, box):
            levels = parse_stakes(box.text)
            if levels:
                read.blinds = levels
                continue
        if box.cx < PANEL_X_MAX and box.cy > PANEL_Y_MIN:
            panel_candidates.append(box)
            continue
        read.unassigned.append(box)

    rows = _cluster_rows(panel_candidates)
    labeled = [(r, _classify_row(r)) for r in rows]

    slot = 0
    i = 0
    while i < len(labeled):
        row, kind = labeled[i]
        if kind == "row1":
            panel = PanelRead(slot=slot, y_top=min(b.y for b in row))
            _fill_row1(panel, row)
            # Pair with an immediately following row2.
            if i + 1 < len(labeled) and labeled[i + 1][1] == "row2":
                next_row = labeled[i + 1][0]
                if min(b.cy for b in next_row) - max(b.cy for b in row) < PANEL_PAIR_GAP:
                    _fill_row2(panel, next_row)
                    i += 1
            if panel.name is not None:
                read.panels[slot] = panel
                slot += 1
            else:
                read.unassigned.extend(row)
        elif kind == "row2":
            # Orphan row2 (row1 missed): keep it visible for debugging.
            read.unassigned.extend(row)
        else:
            read.unassigned.extend(row)
        i += 1
    return read
