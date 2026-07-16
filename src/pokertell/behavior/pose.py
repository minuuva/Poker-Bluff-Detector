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


class WristTracker:
    """Extract wrist trajectories from video frames with MediaPipe Pose.

    TODO(day 4): implement against real footage. Interface:
        track(frames, fps) -> dict with left/right wrist (N, 2) arrays and
        per-frame visibility, One-Euro filtered.
    MediaPipe Pose landmark indices: 15 = left wrist, 16 = right wrist.
    """

    def __init__(self) -> None:
        raise NotImplementedError("WristTracker lands in the day 4 milestone")
