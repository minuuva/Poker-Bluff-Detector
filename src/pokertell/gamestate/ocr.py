"""OCR of HUD text via PaddleOCR full-frame detection.

Strategy (validated in the day 1 spike): run det+rec once per sampled frame,
then assign each detected text box to a HUD field by position (which ROI band
contains it) plus a pattern check (see fields.py). Detection is indent-proof:
the HCL overlay shifts the acting player's panel right by ~75 px, which broke
tight per-field crops but leaves detection untouched.

Spike results on real HCL footage (2026 overlay): every HUD field detected
with rec scores 0.93 to 1.0. Card suits do NOT come through OCR; ranks
sometimes do. All card reading is template matching in cards.py.

Accuracy requirement: the whole study inherits label quality from this stage.
Validate against hand-transcribed ground truth (30 to 50 hands) before
trusting any downstream number.
"""

import re
from dataclasses import dataclass

import numpy as np

_MONEY_RE = re.compile(r"[^0-9.]")


def parse_money(text: str) -> float | None:
    """Parse a HUD money string like '$12,400' or '1.2M' into a float.

    Not for stakes strings ('$25/50 NL'): use parse_stakes, this would
    concatenate the digits.
    """
    text = text.strip().upper()
    if not text or "/" in text:
        return None
    multiplier = 1.0
    if text.endswith("M"):
        multiplier = 1_000_000.0
    elif text.endswith("K"):
        multiplier = 1_000.0
    cleaned = _MONEY_RE.sub("", text)
    if not cleaned:
        return None
    try:
        return float(cleaned) * multiplier
    except ValueError:
        return None


def parse_stakes(text: str) -> list[float]:
    """Parse '$25/50 NL' or '$25/50/100' into blind levels."""
    levels = []
    for part in re.findall(r"[\d,]+(?:\.\d+)?", text.split("NL")[0]):
        value = parse_money(part)
        if value is not None:
            levels.append(value)
    return levels


@dataclass(frozen=True)
class TextBox:
    """One detected text line with its bounding box on the full frame."""

    text: str
    score: float
    x: int
    y: int
    w: int
    h: int

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2


class HudReader:
    """Region-cropped det+rec via PaddleOCR (PP-OCRv6).

    OCR runs only on the two HUD regions (left panel column, right pot/board/
    stakes block), not the full frame, and detections are mapped back to
    full-frame coordinates. Model tier is configurable; day 1 benchmarks on
    an M-series MacBook CPU (full 1080p frame, medium tier: 7.4 s/frame;
    combined HUD crops: medium 4.0 s, small 1.25 s, tiny 0.41 s).
    """

    def __init__(self, layout=None, tier: str = "tiny", min_score: float = 0.8) -> None:
        try:
            from paddleocr import PaddleOCR
        except ImportError as e:
            raise ImportError("PaddleOCR is not installed. Run: uv sync --extra ocr") from e
        self.min_score = min_score
        self._regions = self._hud_regions(layout)
        kwargs = dict(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            # oneDNN's PIR path crashes CPU inference on x86 Linux
            # (ConvertPirAttribute2RuntimeAttribute NotImplementedError);
            # harmless elsewhere since oneDNN is x86-only.
            enable_mkldnn=False,
        )
        if tier != "medium":
            kwargs["text_detection_model_name"] = f"PP-OCRv6_{tier}_det"
            kwargs["text_recognition_model_name"] = f"PP-OCRv6_{tier}_rec"
        self._ocr = PaddleOCR(**kwargs)

    @staticmethod
    def _hud_regions(layout) -> list[tuple[int, int, int, int]]:
        """(x, y, w, h) crop regions; falls back to the whole frame."""
        if layout is None:
            return [(0, 0, 10**6, 10**6)]
        left = layout.rois["panels_left"]
        right_rois = [layout.rois[k] for k in ("pot", "board", "stakes") if k in layout.rois]
        pad = 20
        rx = min(r.x for r in right_rois) - pad
        ry = min(r.y for r in right_rois) - pad
        rx2 = max(r.x + r.w for r in right_rois) + pad
        ry2 = max(r.y + r.h for r in right_rois) + pad
        return [
            (left.x, left.y, left.w, left.h),
            (rx, ry, rx2 - rx, ry2 - ry),
        ]

    def read_frame(self, frame: np.ndarray) -> list[TextBox]:
        """Detect and recognize HUD text; boxes are in full-frame coordinates."""
        boxes: list[TextBox] = []
        fh, fw = frame.shape[:2]
        for cx, cy, cw, ch in self._regions:
            x0, y0 = max(0, cx), max(0, cy)
            crop = frame[y0 : min(fh, cy + ch), x0 : min(fw, cx + cw)]
            if crop.size == 0:
                continue
            for res in self._ocr.predict(crop):
                data = res.json["res"] if hasattr(res, "json") else res
                texts = data.get("rec_texts", [])
                scores = data.get("rec_scores", [])
                polys = data.get("rec_boxes", data.get("dt_polys", []))
                for text, score, poly in zip(texts, scores, polys):
                    if score < self.min_score or not text.strip():
                        continue
                    pts = np.asarray(poly).reshape(-1, 2)
                    x, y = int(pts[:, 0].min()), int(pts[:, 1].min())
                    w = int(pts[:, 0].max()) - x
                    h = int(pts[:, 1].max()) - y
                    boxes.append(
                        TextBox(
                            text=text.strip(),
                            score=float(score),
                            x=x + x0,
                            y=y + y0,
                            w=w,
                            h=h,
                        )
                    )
        return boxes
