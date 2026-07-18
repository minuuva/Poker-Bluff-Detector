"""Face behavior features from MediaPipe Face Landmarker blendshapes.

Feature definitions (computed over a decision window, then z-scored per
player before modeling):

- blink_rate: blinks per second. A blink is a threshold crossing of the
  eyeBlink blendshape pair (both above BLINK_ON, then both below BLINK_OFF),
  debounced to at most one blink per BLINK_DEBOUNCE_S.
- smile_asymmetry: mean absolute difference between mouthSmileLeft and
  mouthSmileRight. Motivated by Geel's observation that a symmetric smile
  is usually more genuine; the literature says face signal is weak to
  deceptive (Slepian 2013: face-only accuracy r = -.07), so this may earn
  a null result and that is fine.
- smile_mean: mean of the larger smile blendshape.
- gaze_dispersion: mean temporal standard deviation of the eight eyeLook
  blendshapes, a proxy for how much the eyes wander during the decision.
- brow_activity: mean of browDown and browInnerUp magnitudes.
- face_coverage: fraction of window frames with a detected face. A quality
  gate, not a model feature: cutaway shots leave the seat crop faceless.

A caution from Slepian 2013: fusing face and arm signals naively cancelled
the arm signal (upper-body judgments were at chance while arms-only beat
chance). Face features stay in their own ablation column group.
"""

from dataclasses import dataclass

import numpy as np

BLINK_ON = 0.5
BLINK_OFF = 0.3
BLINK_DEBOUNCE_S = 0.1

FACE_FEATURES = [
    "blink_rate",
    "smile_asymmetry",
    "smile_mean",
    "gaze_dispersion",
    "brow_activity",
]

_GAZE_KEYS = [
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeLookDownLeft", "eyeLookDownRight",
]
_BROW_KEYS = ["browDownLeft", "browDownRight", "browInnerUp"]


def count_blinks(
    blink_left: np.ndarray, blink_right: np.ndarray, fps: float
) -> int:
    """Count blinks in paired eyeBlink blendshape series (pure, testable)."""
    both = np.minimum(np.asarray(blink_left), np.asarray(blink_right))
    blinks = 0
    armed = True
    last_blink_t = -1e9
    for i, v in enumerate(both):
        t = i / fps
        if armed and v > BLINK_ON and t - last_blink_t > BLINK_DEBOUNCE_S:
            blinks += 1
            last_blink_t = t
            armed = False
        elif not armed and v < BLINK_OFF:
            armed = True
    return blinks


def summarize_blendshapes(frames: list[dict[str, float] | None], fps: float) -> dict:
    """Aggregate per-frame blendshape dicts (None = no face) into features."""
    present = [f for f in frames if f is not None]
    coverage = len(present) / len(frames) if frames else 0.0
    out = {f: float("nan") for f in FACE_FEATURES}
    out["face_coverage"] = coverage
    if len(present) < max(4, int(0.2 * len(frames))):
        return out

    def series(key: str) -> np.ndarray:
        return np.array([f.get(key, 0.0) for f in present])

    duration = len(frames) / fps
    out["blink_rate"] = (
        count_blinks(series("eyeBlinkLeft"), series("eyeBlinkRight"), fps) / duration
    )
    smile_l, smile_r = series("mouthSmileLeft"), series("mouthSmileRight")
    out["smile_asymmetry"] = float(np.mean(np.abs(smile_l - smile_r)))
    out["smile_mean"] = float(np.mean(np.maximum(smile_l, smile_r)))
    out["gaze_dispersion"] = float(np.mean([series(k).std() for k in _GAZE_KEYS]))
    out["brow_activity"] = float(np.mean([series(k).mean() for k in _BROW_KEYS]))
    return out


@dataclass
class FaceTracker:
    """Run MediaPipe Face Landmarker over a frame sequence (one seat crop).

    VIDEO running mode requires monotonically increasing timestamps; create
    one tracker per decision window.
    """

    def __post_init__(self) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        from pokertell.behavior.models import ensure_model

        self._mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model("face_landmarker.task"))),
            running_mode=vision.RunningMode.VIDEO,
            output_face_blendshapes=True,
            num_faces=1,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def process(
        self, frame_bgr: np.ndarray, t_ms: int
    ) -> tuple[dict[str, float], float] | None:
        """(blendshapes, face_height_fraction) for one face, or None.

        The height fraction lets callers reject implausible detections:
        camera cutaways put fragments of a giant closeup face inside the
        seat crop, and those must not be attributed to the seated player.
        """
        rgb = frame_bgr[..., ::-1].copy()
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, t_ms)
        if not result.face_blendshapes or not result.face_landmarks:
            return None
        ys = [p.y for p in result.face_landmarks[0]]
        height_frac = max(ys) - min(ys)
        shapes = {c.category_name: c.score for c in result.face_blendshapes[0]}
        return shapes, float(height_frac)

    def close(self) -> None:
        self._landmarker.close()
