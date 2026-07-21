"""Behavioral event detectors composed from the face and pose streams.

Iteration 3 features, inspired by the event counters in the reference
overlay (ChipGlance, NearFace, Freeze, Shuffle, posture). Each is a pure
function over per-frame series the extractor already collects, so they
add no new model inference cost, and each passed a bet-size leakage
review: everything here is a rate, fraction, ratio, or geometry in
shoulder-width units, never an amplitude or amount.

- gaze_down_rate: downward-gaze excursions per second from the eyeLook
  blendshapes, with hysteresis and debounce. This is the honest proxy
  for "glancing at chips" until chip-stack localization exists: at a
  poker table the chips and hole cards live below the eyeline.
- near_face_frac: fraction of tracked frames with a visible wrist close
  to the face box (hand-to-face contact is a classic self-touch cue).
- freeze_frac and freeze_longest_s: how still the player goes. Motion
  energy is the mean landmark speed in shoulder-widths per second; a
  frame is frozen below FREEZE_THRESH and the longest frozen streak is
  reported in seconds.
- shuffle_score: periodicity of wrist speed in the chip-shuffling band
  (1.5 to 4.5 Hz), as peak band power over total non-DC power.
- lean_fwd: signed forward-lean geometry, mean of (nose drop below the
  shoulder line) in shoulder widths.

All features are z-scored per player downstream like the rest of the
behavior table, so absolute thresholds only need to be sane, not exact.
"""

import numpy as np

EVENT_FEATURES = [
    "gaze_down_rate",
    "near_face_frac",
    "freeze_frac",
    "freeze_longest_s",
    "shuffle_score",
    "lean_fwd",
]

GAZE_DOWN_ON = 0.35
GAZE_DOWN_OFF = 0.20
NEAR_FACE_DIST_W = 1.4
# Calibrated on real footage with One-Euro smoothed trajectories: an active
# fiddling window has p5 ~ 0.10 sw/s, so 0.10 marks its calmest tail while a
# genuinely motionless player sits well below. Raw unsmoothed jitter alone is
# ~0.5 sw/s, which is why smoothing is not optional here.
FREEZE_THRESH = 0.10
SHUFFLE_BAND_HZ = (1.5, 4.5)
MIN_FRAMES = 6

NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
MOTION_LANDMARKS = (
    NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST
)
MIN_VISIBILITY = 0.5


def gaze_down_series(blend_frames: list[dict | None]) -> np.ndarray:
    """Signed downward-gaze intensity per present face frame."""
    present = [f for f in blend_frames if f is not None]
    return np.array(
        [
            (f.get("eyeLookDownLeft", 0.0) + f.get("eyeLookDownRight", 0.0)) / 2
            - (f.get("eyeLookUpLeft", 0.0) + f.get("eyeLookUpRight", 0.0)) / 2
            for f in present
        ]
    )


def gaze_down_rate(blend_frames: list[dict | None], eff_fps: float) -> float:
    """Downward-gaze excursions per second, hysteresis-debounced."""
    gy = gaze_down_series(blend_frames)
    if len(gy) < MIN_FRAMES or eff_fps <= 0:
        return float("nan")
    events = 0
    armed = True
    for v in gy:
        if armed and v > GAZE_DOWN_ON:
            events += 1
            armed = False
        elif not armed and v < GAZE_DOWN_OFF:
            armed = True
    duration = len(blend_frames) / eff_fps
    return events / duration if duration > 0 else float("nan")


def _shoulder_width(lm: np.ndarray) -> float | None:
    ls, rs = lm[LEFT_SHOULDER], lm[RIGHT_SHOULDER]
    if ls[2] < MIN_VISIBILITY or rs[2] < MIN_VISIBILITY:
        return None
    return float(np.linalg.norm(ls[:2] - rs[:2])) + 1e-6


def near_face_frac(
    bbox_frames: list[tuple[int, int, int, int] | None],
    pose_frames: list[np.ndarray | None],
) -> float:
    """Fraction of jointly-tracked frames with a wrist near the face box."""
    near = 0
    total = 0
    for bbox, lm in zip(bbox_frames, pose_frames):
        if bbox is None or lm is None:
            continue
        x, y, w, h = bbox
        cx, cy = x + w / 2, y + h / 2
        total += 1
        for idx in (LEFT_WRIST, RIGHT_WRIST):
            wx, wy, vis = lm[idx]
            if vis >= MIN_VISIBILITY and np.hypot(wx - cx, wy - cy) < NEAR_FACE_DIST_W * max(w, 1):
                near += 1
                break
    return near / total if total >= MIN_FRAMES else float("nan")


def motion_energy(pose_frames: list[np.ndarray | None], fps: float) -> np.ndarray:
    """Mean upper-body landmark speed per frame gap, in shoulder-widths/s.

    Landmark trajectories are One-Euro filtered first (the project rule:
    raw per-frame detector jitter is ~0.5 shoulder-widths/s, which would
    swamp any stillness signal). NaN where either endpoint is untracked.
    """
    from pokertell.behavior.smoothing import smooth_trajectory

    n = len(pose_frames)
    out = np.full(max(0, n - 1), np.nan)
    if n < 2:
        return out
    widths = [w for lm in pose_frames if lm is not None and (w := _shoulder_width(lm))]
    if not widths:
        return out
    scale = float(np.median(widths))

    speed_sets: list[list[float]] = [[] for _ in range(n - 1)]
    for j in MOTION_LANDMARKS:
        pts = [
            lm[j, :2] if lm is not None and lm[j, 2] >= MIN_VISIBILITY else None
            for lm in pose_frames
        ]
        valid_idx = [i for i, p in enumerate(pts) if p is not None]
        if len(valid_idx) < MIN_FRAMES:
            continue
        filled, last = [], pts[valid_idx[0]]
        for p in pts:
            last = p if p is not None else last
            filled.append(last)
        arr = np.array(filled, dtype=float)
        sm = np.column_stack(
            [smooth_trajectory(list(arr[:, 0]), fps), smooth_trajectory(list(arr[:, 1]), fps)]
        )
        d = np.linalg.norm(np.diff(sm, axis=0), axis=1) * fps / scale
        for i in range(n - 1):
            if pts[i] is not None and pts[i + 1] is not None:
                speed_sets[i].append(float(d[i]))
    for i, speeds in enumerate(speed_sets):
        if speeds:
            out[i] = float(np.mean(speeds))
    return out


def freeze_features(energy: np.ndarray, fps: float) -> tuple[float, float]:
    """(freeze_frac, freeze_longest_s) from a motion-energy series."""
    valid = energy[~np.isnan(energy)]
    if len(valid) < MIN_FRAMES or fps <= 0:
        return float("nan"), float("nan")
    frozen = valid < FREEZE_THRESH
    frac = float(frozen.mean())
    longest = 0
    run = 0
    for f in frozen:
        run = run + 1 if f else 0
        longest = max(longest, run)
    return frac, longest / fps


def shuffle_score(speed: np.ndarray, fps: float) -> float:
    """Peak power in the chip-shuffle band over total non-DC power."""
    speed = np.asarray(speed, dtype=float)
    speed = speed[~np.isnan(speed)]
    if len(speed) < 4 * MIN_FRAMES or fps <= 0 or np.allclose(speed, 0):
        return float("nan")
    speed = speed - speed.mean()
    mag = np.abs(np.fft.rfft(speed)) ** 2
    freqs = np.fft.rfftfreq(len(speed), d=1.0 / fps)
    total = mag[1:].sum()
    if total <= 0:
        return float("nan")
    lo, hi = SHUFFLE_BAND_HZ
    band = mag[(freqs >= lo) & (freqs <= hi)]
    return float(band.max() / total) if len(band) else 0.0


def lean_fwd(pose_frames: list[np.ndarray | None]) -> float:
    """Mean signed nose drop below the shoulder line, in shoulder widths."""
    vals = []
    for lm in pose_frames:
        if lm is None or lm[NOSE, 2] < MIN_VISIBILITY:
            continue
        width = _shoulder_width(lm)
        if width is None:
            continue
        mid_y = (lm[LEFT_SHOULDER, 1] + lm[RIGHT_SHOULDER, 1]) / 2
        vals.append(float((lm[NOSE, 1] - mid_y) / width))
    return float(np.mean(vals)) if len(vals) >= MIN_FRAMES else float("nan")


def acting_wrist_speed(pose_frames: list[np.ndarray | None], fps: float) -> np.ndarray:
    """Speed profile of the wrist with the larger total path, NaN-gapped."""
    best = None
    best_path = -1.0
    for idx in (LEFT_WRIST, RIGHT_WRIST):
        speeds = np.full(max(0, len(pose_frames) - 1), np.nan)
        for i in range(1, len(pose_frames)):
            a, b = pose_frames[i - 1], pose_frames[i]
            if a is None or b is None:
                continue
            if a[idx, 2] < MIN_VISIBILITY or b[idx, 2] < MIN_VISIBILITY:
                continue
            speeds[i - 1] = float(np.linalg.norm(b[idx, :2] - a[idx, :2])) * fps
        path = np.nansum(speeds) / fps if len(speeds) else 0.0
        if path > best_path:
            best_path = path
            best = speeds
    return best if best is not None else np.array([])


def compute_event_features(
    blend_frames: list[dict | None],
    bbox_frames: list[tuple[int, int, int, int] | None],
    pose_frames: list[np.ndarray | None],
    fps: float,
    face_stride: int,
) -> dict:
    """All EVENT_FEATURES for one decision window."""
    energy = motion_energy(pose_frames, fps)
    frac, longest = freeze_features(energy, fps)
    return {
        "gaze_down_rate": gaze_down_rate(blend_frames, fps / max(1, face_stride)),
        "near_face_frac": near_face_frac(bbox_frames, pose_frames),
        "freeze_frac": frac,
        "freeze_longest_s": longest,
        "shuffle_score": shuffle_score(acting_wrist_speed(pose_frames, fps), fps),
        "lean_fwd": lean_fwd(pose_frames),
    }
