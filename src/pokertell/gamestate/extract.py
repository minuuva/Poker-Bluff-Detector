"""Video to HudSnapshot stream: the extract-state pipeline stage.

Samples frames at a fixed interval, skips frames whose HUD regions have not
changed (cheap thumbnail diff, so the 0.26 s OCR only runs on new state),
then runs OCR field assignment plus card template matching and emits one
HudSnapshot per processed frame.

Output is JSONL (one snapshot per line) under data/hands/, consumed by the
hand-history assembler.
"""

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from pokertell.gamestate.cards import CardMatcher
from pokertell.gamestate.fields import HudRead, PanelRead, assign_fields
from pokertell.gamestate.ocr import HudReader
from pokertell.gamestate.rois import HudLayout
from pokertell.gamestate.statemachine import HudSnapshot

GATE_THRESH = 1.5
THUMB_W = 96
CARD_BAND_ABOVE = 6
CARD_BAND_BELOW = 58
CARD_X_PAD = 3
BOARD_PAD = 10


@dataclass
class ExtractStats:
    sampled: int = 0
    gated: int = 0
    processed: int = 0
    snapshots: int = 0
    unmatched_cards: int = 0


class SnapshotExtractor:
    def __init__(
        self,
        layout: HudLayout,
        template_dir: Path,
        unmatched_dir: Path | None = None,
    ) -> None:
        """unmatched_dir: if set, unrecognized card cells are saved there for
        labeling (the template library grows via this loop; ranks 3-5 were
        absent from the initial harvest)."""
        self.layout = layout
        self.reader = HudReader(layout)
        self.matcher = CardMatcher(template_dir)
        self.unmatched_dir = unmatched_dir
        self._last_thumb: np.ndarray | None = None
        self._t: float = 0.0

    def _hud_thumb(self, frame: np.ndarray) -> np.ndarray:
        """Small grayscale digest of the HUD regions for the change gate."""
        thumbs = []
        for cx, cy, cw, ch in self.reader._regions:
            crop = frame[cy : cy + ch, cx : cx + cw]
            h = max(1, int(THUMB_W * crop.shape[0] / max(1, crop.shape[1])))
            thumbs.append(cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (THUMB_W, h)))
        return np.vstack(thumbs).astype(np.int16)

    def _changed(self, frame: np.ndarray) -> bool:
        thumb = self._hud_thumb(frame)
        if self._last_thumb is None or thumb.shape != self._last_thumb.shape:
            self._last_thumb = thumb
            return True
        diff = float(np.abs(thumb - self._last_thumb).mean())
        if diff < GATE_THRESH:
            return False
        self._last_thumb = thumb
        return True

    def _read_cards(self, region: np.ndarray) -> tuple[list[str], int]:
        """Identify all card cells in a region; save misses for labeling."""
        from pokertell.gamestate.cards import find_card_cells, normalize_cell

        boxes = find_card_cells(region)
        cards = []
        for box in boxes:
            cell = normalize_cell(region, box)
            m = self.matcher.match_cell(cell)
            if m.card:
                cards.append(m.card)
                continue
            self._misses += 1
            if self.unmatched_dir is not None:
                self.unmatched_dir.mkdir(parents=True, exist_ok=True)
                name = f"t{self._t:07.1f}_x{box.x}_r{m.rank_score:.2f}_s{m.suit_score:.2f}.png"
                cv2.imwrite(str(self.unmatched_dir / name), cell)
        return cards, len(boxes)

    def _panel_cards(self, frame: np.ndarray, panel: PanelRead) -> list[str]:
        """Hole cards for one panel: template match between name and equity."""
        if panel.name_box is None:
            return []
        y0 = max(0, int(panel.y_top) - CARD_BAND_ABOVE)
        y1 = min(frame.shape[0], int(panel.y_top) + CARD_BAND_BELOW)
        x0 = panel.name_box.x + panel.name_box.w + CARD_X_PAD
        x1 = panel.equity_box.x - CARD_X_PAD if panel.equity_box else x0 + 135
        x1 = min(x1, frame.shape[1])
        if x1 - x0 < 40 or y1 - y0 < 20:
            return []
        cards, found = self._read_cards(frame[y0:y1, x0:x1])
        return cards if len(cards) == found else []

    def _board_cards(self, frame: np.ndarray) -> list[str]:
        roi = self.layout.rois["board"]
        y0, y1 = roi.y - BOARD_PAD, roi.y + roi.h + BOARD_PAD
        x0, x1 = roi.x - BOARD_PAD, roi.x + roi.w + BOARD_PAD
        cards, _ = self._read_cards(frame[y0:y1, x0:x1])
        return cards

    def _to_snapshot(self, t: float, frame: np.ndarray, read: HudRead) -> HudSnapshot:
        snap = HudSnapshot(t=round(t, 2), pot=read.pot, blinds=read.blinds)
        snap.board = self._board_cards(frame)
        for panel in read.panels.values():
            if panel.name is None:
                continue
            name = panel.name
            if panel.stack is not None:
                snap.stacks[name] = panel.stack
            if panel.action_amount is not None:
                snap.bets[name] = panel.action_amount
            if panel.action_text is not None:
                snap.actions[name] = panel.action_text
            if panel.position is not None:
                snap.positions[name] = panel.position
            if panel.equity_pct is not None:
                snap.equities[name] = panel.equity_pct
            if panel.to_call is not None:
                snap.to_act = name
                snap.to_call = panel.to_call
            cards = self._panel_cards(frame, panel)
            if len(cards) == 2:
                snap.hole_cards[name] = cards
        return snap

    def run(
        self,
        video: Path,
        interval: float = 1.0,
        t_start: float = 0.0,
        t_end: float | None = None,
    ) -> Iterator[tuple[HudSnapshot, "ExtractStats"]]:
        """Yield (snapshot, running stats) for each processed frame."""
        stats = ExtractStats()
        self._misses = 0
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = n_frames / fps
        end = min(t_end, duration) if t_end is not None else duration

        t = t_start
        while t < end:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if not ok:
                break
            stats.sampled += 1
            self._t = t
            if not self._changed(frame):
                stats.gated += 1
                t += interval
                continue
            stats.processed += 1
            read = assign_fields(self.reader.read_frame(frame), self.layout)
            snap = self._to_snapshot(t, frame, read)
            stats.snapshots += 1
            stats.unmatched_cards = self._misses
            yield snap, stats
            t += interval
        cap.release()


def write_snapshots(snapshots: list[HudSnapshot], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for snap in snapshots:
            f.write(json.dumps(dataclasses.asdict(snap)) + "\n")
