"""Body pose tracking and motion smoothness metrics.

The single strongest behavioral cue in the literature is the smoothness of the
betting motion (Slepian et al. 2013: rated smoothness predicted hand quality at
r = .29, versus .07 to .15 for arm motions generally and -.07 for the face).
That paper never defined smoothness kinematically, so we operationalize it two
standard ways from motor control research and let the model decide:

1. RMS jerk of the wrist trajectory during the chip push (lower = smoother).
2. Spectral arc length (SPARC) of the velocity profile (closer to zero =
   smoother). Reference: Balasubramanian et al., "On the analysis of movement
   smoothness", J NeuroEngineering Rehabil 2015.

Trajectories must be One-Euro filtered (see smoothing.py) before these metrics
are computed, or they will measure detector jitter instead of motion.
"""

import numpy as np


def velocity_profile(traj_xy: np.ndarray, fps: float) -> np.ndarray:
    """Speed over time from an (N, 2) pixel trajectory sampled at fps."""
    if traj_xy.ndim != 2 or traj_xy.shape[1] != 2:
        raise ValueError("traj_xy must have shape (N, 2)")
    d = np.diff(traj_xy, axis=0)
    return np.linalg.norm(d, axis=1) * fps


def jerk_rms(traj_xy: np.ndarray, fps: float) -> float:
    """Root mean square jerk (third derivative of position) of a trajectory.

    Units are pixels per second cubed. Compare only within a player and a
    fixed camera layout; z-score per player before modeling.
    """
    if len(traj_xy) < 4:
        return float("nan")
    v = np.diff(traj_xy, axis=0) * fps
    a = np.diff(v, axis=0) * fps
    j = np.diff(a, axis=0) * fps
    return float(np.sqrt(np.mean(np.sum(j**2, axis=1))))


def spectral_arc_length(
    speed: np.ndarray,
    fps: float,
    pad_level: int = 4,
    max_freq: float = 10.0,
    amp_threshold: float = 0.05,
) -> float:
    """SPARC smoothness of a speed profile. Always negative; closer to 0 is smoother.

    Args:
        speed: 1D speed profile from velocity_profile().
        fps: sampling frequency of the profile in Hz.
        pad_level: zero padding factor exponent for the FFT.
        max_freq: analysis band upper limit in Hz.
        amp_threshold: adaptive cutoff as a fraction of peak magnitude.
    """
    speed = np.asarray(speed, dtype=float)
    if len(speed) < 4 or np.allclose(speed, 0):
        return float("nan")

    n_fft = int(2 ** np.ceil(np.log2(len(speed)) + pad_level))
    mag = np.abs(np.fft.rfft(speed, n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fps)

    band = freqs <= max_freq
    mag, freqs = mag[band], freqs[band]
    if mag.max() == 0:
        return float("nan")
    mag = mag / mag.max()

    above = np.nonzero(mag >= amp_threshold)[0]
    if len(above) == 0:
        return float("nan")
    cut = slice(0, above[-1] + 1)
    mag, freqs = mag[cut], freqs[cut]

    df = np.diff(freqs / freqs[-1])
    dm = np.diff(mag)
    return float(-np.sum(np.sqrt(df**2 + dm**2)))


def log_dimensionless_jerk(traj_xy: np.ndarray, fps: float) -> float:
    """Amplitude-invariant smoothness: negative log dimensionless jerk.

    Bet-size leakage guard: raw jerk scales with movement amplitude, and a
    bigger chip push could correlate with bet size. This variant normalizes
    by duration and path length (Hogan and Sternad), so it measures HOW the
    motion was executed, not how big it was. Higher (closer to zero) is
    smoother.
    """
    if len(traj_xy) < 6:
        return float("nan")
    v = np.diff(traj_xy, axis=0) * fps
    speed = np.linalg.norm(v, axis=1)
    path = float(speed.sum() / fps)
    if path < 1e-6:
        return float("nan")
    a = np.diff(v, axis=0) * fps
    j = np.diff(a, axis=0) * fps
    duration = len(traj_xy) / fps
    integral = float(np.sum(np.sum(j**2, axis=1)) / fps)
    return -float(np.log(integral * duration**5 / path**2 + 1e-12))


POSE_FEATURES = [
    "wrist_peak_speed_norm",
    "wrist_jerk_ldj",
    "wrist_sparc",
    "lean_std",
    "head_motion",
]

# MediaPipe Pose landmark indices.
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_WRIST, RIGHT_WRIST = 15, 16
MIN_VISIBILITY = 0.5


def summarize_pose(
    landmarks: list[np.ndarray | None], fps: float, commit_frames: int
) -> dict:
    """Aggregate per-frame pose landmark arrays into window features.

    landmarks: per frame, an (33, 3) array of (x_px, y_px, visibility) or
    None when no pose was detected. Wrist smoothness is computed on the
    trailing commit_frames (the chip-push happens at the window's end); the
    wrist with the larger path in that segment is the acting hand. Posture
    features use the whole window.
    """
    from pokertell.behavior.smoothing import smooth_trajectory

    out = {f: float("nan") for f in POSE_FEATURES}
    present = [lm for lm in landmarks if lm is not None]
    out["pose_coverage"] = len(present) / len(landmarks) if landmarks else 0.0
    if len(present) < max(4, int(0.2 * len(landmarks))):
        return out

    def visible_series(idx: int) -> np.ndarray | None:
        pts = [
            lm[idx, :2] if lm is not None and lm[idx, 2] >= MIN_VISIBILITY else None
            for lm in landmarks
        ]
        valid = [p for p in pts if p is not None]
        if len(valid) < max(6, int(0.3 * len(landmarks))):
            return None
        # Forward-fill gaps so filtering and derivatives stay defined.
        filled, last = [], valid[0]
        for p in pts:
            last = p if p is not None else last
            filled.append(last)
        arr = np.array(filled, dtype=float)
        return np.column_stack(
            [smooth_trajectory(list(arr[:, 0]), fps), smooth_trajectory(list(arr[:, 1]), fps)]
        )

    # Posture over the whole window.
    shoulders_l = visible_series(LEFT_SHOULDER)
    shoulders_r = visible_series(RIGHT_SHOULDER)
    nose = visible_series(NOSE)
    if shoulders_l is not None and shoulders_r is not None:
        mid_y = (shoulders_l[:, 1] + shoulders_r[:, 1]) / 2
        width = np.abs(shoulders_r[:, 0] - shoulders_l[:, 0]).mean() + 1e-6
        out["lean_std"] = float(mid_y.std() / width)
    if nose is not None:
        speeds = velocity_profile(nose, fps)
        scale = 1.0
        if shoulders_l is not None and shoulders_r is not None:
            scale = np.abs(shoulders_r[:, 0] - shoulders_l[:, 0]).mean() + 1e-6
        out["head_motion"] = float(np.mean(speeds) / scale)

    # Chip-push smoothness on the commit segment, acting wrist only.
    best_path = 0.0
    for idx in (LEFT_WRIST, RIGHT_WRIST):
        traj = visible_series(idx)
        if traj is None or len(traj) < commit_frames:
            continue
        seg = traj[-commit_frames:]
        speeds = velocity_profile(seg, fps)
        path = float(speeds.sum() / fps)
        if path <= best_path:
            continue
        best_path = path
        scale = 1.0
        if shoulders_l is not None and shoulders_r is not None:
            scale = np.abs(shoulders_r[:, 0] - shoulders_l[:, 0]).mean() + 1e-6
        out["wrist_peak_speed_norm"] = float(speeds.max() / scale)
        out["wrist_jerk_ldj"] = log_dimensionless_jerk(seg, fps)
        out["wrist_sparc"] = spectral_arc_length(speeds, fps)
    return out


class PoseTracker:
    """Run MediaPipe Pose Landmarker on per-frame crops.

    IMAGE mode (stateless): the crop follows the identified face across
    zooming, panning cameras, so there is no temporally consistent image
    stream for VIDEO-mode tracking to exploit. Landmarks are returned in
    the crop's pixel coordinates; callers map them to a common frame.
    """

    def __init__(self) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        from pokertell.behavior.models import ensure_model

        self._mp = mp
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(ensure_model("pose_landmarker_lite.task"))
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def process(self, frame_bgr: np.ndarray, t_ms: int = 0) -> np.ndarray | None:
        """(33, 3) array of (x_px, y_px, visibility) for one pose, or None."""
        h, w = frame_bgr.shape[:2]
        rgb = frame_bgr[..., ::-1].copy()
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(image)
        if not result.pose_landmarks:
            return None
        lm = result.pose_landmarks[0]
        return np.array([[p.x * w, p.y * h, p.visibility] for p in lm])

    def close(self) -> None:
        self._landmarker.close()
