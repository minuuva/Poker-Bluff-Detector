"""Event detector features (iteration 3)."""

import math

import numpy as np

from pokertell.behavior.events import (
    EVENT_FEATURES,
    LEFT_SHOULDER,
    LEFT_WRIST,
    NOSE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    compute_event_features,
    freeze_features,
    gaze_down_rate,
    lean_fwd,
    motion_energy,
    near_face_frac,
    shuffle_score,
)


def _pose(nose=(500, 300), wrists=((400, 600), (600, 600)), shoulders=((400, 400), (600, 400))):
    lm = np.zeros((33, 3))
    lm[NOSE] = (*nose, 0.9)
    lm[LEFT_SHOULDER] = (*shoulders[0], 0.9)
    lm[RIGHT_SHOULDER] = (*shoulders[1], 0.9)
    lm[LEFT_WRIST] = (*wrists[0], 0.9)
    lm[RIGHT_WRIST] = (*wrists[1], 0.9)
    return lm


def test_gaze_down_rate_counts_excursions():
    down = {"eyeLookDownLeft": 0.6, "eyeLookDownRight": 0.6}
    up = {"eyeLookUpLeft": 0.1, "eyeLookUpRight": 0.1}
    frames = [up] * 5 + [down] * 3 + [up] * 5 + [down] * 3 + [up] * 4
    rate = gaze_down_rate(frames, eff_fps=10.0)
    assert math.isclose(rate, 2 / 2.0)  # two excursions over 20 frames at 10 fps


def test_gaze_down_rate_needs_data():
    assert math.isnan(gaze_down_rate([None, None], eff_fps=10.0))


def test_near_face_frac():
    face = (450, 250, 100, 100)  # center (500, 300)
    near = _pose(wrists=((510, 320), (600, 600)))
    far = _pose(wrists=((400, 600), (600, 600)))
    frames = [near] * 6 + [far] * 6
    boxes = [face] * 12
    assert math.isclose(near_face_frac(boxes, frames), 0.5)
    assert math.isnan(near_face_frac([None] * 12, frames))


def test_freeze_features_from_still_series():
    fps = 30.0
    still = [_pose() for _ in range(40)]
    energy = motion_energy(still, fps)
    frac, longest = freeze_features(energy, fps)
    assert frac == 1.0
    assert longest > 1.0

    moving = [
        _pose(wrists=((400 + 20 * i, 600), (600, 600)), nose=(500 + 10 * i, 300))
        for i in range(40)
    ]
    frac_m, _ = freeze_features(motion_energy(moving, fps), fps)
    assert frac_m < 0.5


def test_shuffle_score_periodic_vs_flat():
    fps = 30.0
    t = np.arange(300) / fps
    periodic = 50 + 40 * np.sin(2 * np.pi * 3.0 * t)  # 3 Hz, in-band
    rng = np.random.default_rng(0)
    noise = 50 + 40 * rng.standard_normal(300)
    assert shuffle_score(periodic, fps) > 0.5
    assert shuffle_score(noise, fps) < shuffle_score(periodic, fps)


def test_lean_fwd_sign():
    upright = [_pose(nose=(500, 300)) for _ in range(10)]
    hunched = [_pose(nose=(500, 420)) for _ in range(10)]  # nose beyond shoulder line
    assert lean_fwd(hunched) > lean_fwd(upright)


def test_compute_event_features_keys_match():
    frames = [_pose() for _ in range(20)]
    shapes = [{"eyeLookDownLeft": 0.1, "eyeLookDownRight": 0.1}] * 10
    out = compute_event_features(shapes, [None] * 20, frames, 30.0, 2)
    assert list(out.keys()) == EVENT_FEATURES
