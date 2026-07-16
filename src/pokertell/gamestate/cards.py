"""Card recognition from HUD sprites via template matching.

The broadcast hole-card and board overlays are rendered from a fixed sprite
set, so OpenCV template matching (cv2.matchTemplate, TM_CCOEFF_NORMED) against
52 reference crops is more reliable than OCR for these fields. Templates are
harvested once per show layout from frames where the cards are known.

Card strings use treys format: rank then suit, e.g. "As", "Td", "9c".
"""

from pathlib import Path

import numpy as np

MATCH_THRESHOLD = 0.85


class CardMatcher:
    """Match a card ROI crop against the 52-card template library.

    TODO(day 2): implement.
        build(template_dir) -> load / harvest reference sprites
        match(crop) -> card string or None if below MATCH_THRESHOLD
    Keep a confusion log: 6/9 and suit-color confusions are the expected
    failure modes and belong in the OCR validation report.
    """

    def __init__(self, template_dir: Path) -> None:
        self.template_dir = Path(template_dir)
        raise NotImplementedError("CardMatcher lands in the day 2 milestone")

    def match(self, crop: np.ndarray) -> str | None:
        raise NotImplementedError
