"""Face behavior features from MediaPipe Face Landmarker blendshapes.

Feature definitions (all computed over a decision window, then z-scored per
player before modeling):

- blink_rate: blinks per second. A blink is a threshold crossing of the
  eyeBlinkLeft/eyeBlinkRight blendshape pair (both above BLINK_ON, then both
  below BLINK_OFF), debounced to at most one blink per 100 ms.
- smile_asymmetry: mean absolute difference between mouthSmileLeft and
  mouthSmileRight. Motivated by Geel's observation that a symmetric smile is
  usually more genuine; the literature says face signal is weak to deceptive
  (Slepian 2013: face-only accuracy r = -.07), so this may earn a null result.
- gaze_dispersion: variance of the eyeLookIn/Out/Up/Down blendshape vector,
  a proxy for how much the eyes wander during the decision.
- brow_activity: mean of browDownLeft/Right and browInnerUp magnitudes.

A caution from Slepian 2013: fusing face and arm signals naively cancelled the
arm signal (upper-body judgments were at chance while arms-only beat chance).
Keep face features in a separate column group so ablations can drop them.
"""

BLINK_ON = 0.5
BLINK_OFF = 0.3
BLINK_DEBOUNCE_S = 0.1

FACE_FEATURES = [
    "blink_rate",
    "smile_asymmetry",
    "gaze_dispersion",
    "brow_activity",
]


class FaceTracker:
    """Run MediaPipe Face Landmarker over a decision window's frames.

    TODO(day 4): implement against real footage. Interface:
        track(frames, fps) -> per-frame blendshape dicts
        summarize(blendshapes, fps) -> dict of FACE_FEATURES
    Uses the face_landmarker task with output_face_blendshapes=True; the
    52-category blendshape vector includes everything needed above.
    """

    def __init__(self) -> None:
        raise NotImplementedError("FaceTracker lands in the day 4 milestone")
