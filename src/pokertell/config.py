"""Project paths and configuration loading.

All pipeline stages read and write under a single data root so that any stage
can be re-run in isolation. The root defaults to ./data relative to the
working directory and can be overridden with the POKERTELL_DATA env var.
"""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def raw(self) -> Path:
        """Downloaded session videos (never committed, never redistributed)."""
        return self.root / "raw"

    @property
    def frames(self) -> Path:
        """Sampled frames and ROI crops used for OCR calibration."""
        return self.root / "frames"

    @property
    def hands(self) -> Path:
        """Reconstructed hand histories, one parquet/JSON file per session."""
        return self.root / "hands"

    @property
    def features(self) -> Path:
        """Per-decision feature tables (betting and behavioral)."""
        return self.root / "features"

    @property
    def models(self) -> Path:
        """Trained model artifacts."""
        return self.root / "models"

    @property
    def reports(self) -> Path:
        """Evaluation outputs: metrics tables, calibration plots."""
        return self.root / "reports"

    @property
    def demo(self) -> Path:
        """Rendered demo overlay clips (local only, never committed)."""
        return self.root / "demo"

    def ensure(self) -> "Paths":
        """Create all data subdirectories if missing and return self."""
        for p in (
            self.raw, self.frames, self.hands, self.features,
            self.models, self.reports, self.demo,
        ):
            p.mkdir(parents=True, exist_ok=True)
        return self


def default_paths() -> Paths:
    return Paths(root=Path(os.environ.get("POKERTELL_DATA", "data")))
