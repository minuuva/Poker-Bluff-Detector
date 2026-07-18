"""Behavioral feature computation (pure logic; MediaPipe runs are validated
via the CLI on real footage, not in unit tests)."""

import numpy as np

from pokertell.behavior.face import count_blinks, summarize_blendshapes
from pokertell.behavior.pose import (
    jerk_rms,
    log_dimensionless_jerk,
    summarize_pose,
)

FPS = 30.0


def _blink_series(n=90, blink_at=(20, 50)):
    left = np.zeros(n)
    right = np.zeros(n)
    for i in blink_at:
        left[i : i + 4] = 0.9
        right[i : i + 4] = 0.9
    return left, right


def test_count_blinks_counts_distinct_blinks():
    left, right = _blink_series()
    assert count_blinks(left, right, FPS) == 2


def test_count_blinks_requires_both_eyes():
    left, _ = _blink_series()
    assert count_blinks(left, np.zeros_like(left), FPS) == 0


def test_count_blinks_debounces_sustained_closure():
    left = np.zeros(60)
    right = np.zeros(60)
    left[10:40] = 0.9  # one long closure, not many blinks
    right[10:40] = 0.9
    assert count_blinks(left, right, FPS) == 1


def _shape_frame(**kw):
    base = {
        "eyeBlinkLeft": 0.05, "eyeBlinkRight": 0.05,
        "mouthSmileLeft": 0.2, "mouthSmileRight": 0.2,
        "browDownLeft": 0.1, "browDownRight": 0.1, "browInnerUp": 0.1,
    }
    for k in ("In", "Out", "Up", "Down"):
        base[f"eyeLook{k}Left"] = 0.2
        base[f"eyeLook{k}Right"] = 0.2
    base.update(kw)
    return base


def test_summarize_blendshapes_features():
    frames = [_shape_frame(mouthSmileLeft=0.6, mouthSmileRight=0.2)] * 30
    out = summarize_blendshapes(frames, FPS)
    assert out["face_coverage"] == 1.0
    assert abs(out["smile_asymmetry"] - 0.4) < 1e-6
    assert abs(out["smile_mean"] - 0.6) < 1e-6
    assert out["gaze_dispersion"] < 1e-9


def test_summarize_blendshapes_low_coverage_gives_nan():
    frames = [None] * 28 + [_shape_frame()] * 2
    out = summarize_blendshapes(frames, FPS)
    assert out["face_coverage"] < 0.1
    assert np.isnan(out["blink_rate"])


def _push(n=45, scale=1.0, seed=None):
    t = np.linspace(-3, 3, n)
    x = scale * 200 / (1 + np.exp(-t))
    y = np.zeros(n)
    traj = np.column_stack([x, y])
    if seed is not None:
        rng = np.random.default_rng(seed)
        traj += rng.normal(0, 4 * scale, traj.shape)
    return traj


def test_ldj_is_amplitude_invariant_but_jerk_rms_is_not():
    small, big = _push(scale=1.0), _push(scale=3.0)
    assert abs(log_dimensionless_jerk(small, FPS) - log_dimensionless_jerk(big, FPS)) < 0.1
    assert jerk_rms(big, FPS) > 2 * jerk_rms(small, FPS)


def test_ldj_orders_smooth_above_jerky():
    assert log_dimensionless_jerk(_push(), FPS) > log_dimensionless_jerk(_push(seed=1), FPS)


def _pose_frame(wrist_x, wrist_y=200.0, visible=0.9):
    lm = np.zeros((33, 3))
    lm[:, 2] = 0.9
    lm[0] = [100, 60, 0.9]     # nose
    lm[11] = [60, 120, 0.9]    # left shoulder
    lm[12] = [140, 120, 0.9]   # right shoulder
    lm[15] = [wrist_x, wrist_y, visible]
    lm[16] = [80, 220, 0.2]    # other wrist barely visible
    return lm


def test_summarize_pose_uses_moving_wrist():
    frames = [_pose_frame(60.0)] * 30 + [
        _pose_frame(60.0 + 8 * i) for i in range(15)
    ]
    out = summarize_pose(frames, FPS, commit_frames=15)
    assert out["pose_coverage"] == 1.0
    assert not np.isnan(out["wrist_peak_speed_norm"])
    assert out["wrist_peak_speed_norm"] > 0


def test_summarize_pose_no_pose_gives_nan():
    out = summarize_pose([None] * 30, FPS, commit_frames=10)
    assert out["pose_coverage"] == 0.0
    assert np.isnan(out["lean_std"])


def test_shot_signature_is_normalized_and_matches_self():
    from pokertell.behavior.extract import shot_signature

    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (1080, 1920, 3), dtype=np.uint8)
    sig = shot_signature(frame)
    assert abs(sig.mean()) < 1e-9
    assert abs(np.dot(sig, sig) / len(sig) - 1.0) < 1e-6


def test_load_seats_schema(tmp_path):
    from pokertell.behavior.extract import load_seats

    p = tmp_path / "seats.yaml"
    p.write_text(
        "seats:\n"
        "  PLAYERX:\n"
        "    - {shot_t: 12.0, x: 10, y: 20, w: 300, h: 400}\n"
        "    - {shot_t: 99.0, x: 5, y: 5, w: 100, h: 100}\n"
    )
    seats = load_seats(p)
    assert len(seats["PLAYERX"]) == 2
    assert seats["PLAYERX"][0].shot_t == 12.0
    assert seats["PLAYERX"][1].w == 100
