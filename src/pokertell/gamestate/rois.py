"""Regions of interest on the broadcast frame.

The stream HUD is a deterministic graphics render: for a given show layout,
every field (pot, stacks, names, bets, hole cards, board) sits at a fixed
pixel rectangle. Reading those rectangles is far more reliable than full-frame
OCR. Layouts are defined in configs/*.yaml, one file per show format.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class Roi:
    """A named rectangle in pixel coordinates on the full-resolution frame."""

    name: str
    x: int
    y: int
    w: int
    h: int

    def crop(self, frame: np.ndarray) -> np.ndarray:
        return frame[self.y : self.y + self.h, self.x : self.x + self.w]


@dataclass(frozen=True)
class HudLayout:
    """All ROIs for one show format plus layout metadata."""

    name: str
    frame_width: int
    frame_height: int
    rois: dict[str, Roi]

    def crop_all(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        self._check_frame(frame)
        return {name: roi.crop(frame) for name, roi in self.rois.items()}

    def _check_frame(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        if (w, h) != (self.frame_width, self.frame_height):
            raise ValueError(
                f"frame is {w}x{h} but layout '{self.name}' expects "
                f"{self.frame_width}x{self.frame_height}; resize first or recalibrate"
            )


def load_layout(path: Path) -> HudLayout:
    """Load a HUD layout from YAML. See configs/hcl_rois.yaml for the schema."""
    raw = yaml.safe_load(Path(path).read_text())
    rois = {
        name: Roi(name=name, x=box["x"], y=box["y"], w=box["w"], h=box["h"])
        for name, box in raw["rois"].items()
    }
    return HudLayout(
        name=raw["name"],
        frame_width=raw["frame"]["width"],
        frame_height=raw["frame"]["height"],
        rois=rois,
    )
