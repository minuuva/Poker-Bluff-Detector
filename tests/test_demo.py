"""Pure helpers of the demo overlay renderer."""

import numpy as np

from pokertell.behavior.pose import LEFT_WRIST, RIGHT_WRIST
from pokertell.demo.overlay import (
    FACE_EDGE_L,
    FACE_EDGE_R,
    NOSE_TIP,
    ActingWrist,
    LiveBlink,
    face_tags,
    fmt_cards,
    gaze_vector,
    head_dir,
    phase_label,
    state_lines,
    window_progress,
)

DECISION = {
    "t_start": 100.0,
    "t_end": 108.0,
    "player": "AIRBALL",
    "position": "CO",
    "street": "preflop",
    "action": "raise",
    "amount": 1500.0,
    "pot_before": 1200.0,
}
HAND = {"hand_id": "s#0001"}


def test_fmt_cards():
    assert fmt_cards(["8h", "4s"]) == "8h 4s"
    assert fmt_cards(None) == "??"


def test_phase_label_transitions():
    assert phase_label(99.0, DECISION) == "waiting"
    assert phase_label(104.0, DECISION) == "to act"
    assert phase_label(109.0, DECISION) == "RAISE $1,500"


def test_state_lines_include_pot_and_actor():
    lines = state_lines(DECISION, HAND, 103.0)
    assert "pot $1,200" in lines[1]
    assert "AIRBALL" in lines[2]


def test_window_progress_clips():
    assert window_progress(99.0, 100.0, 108.0) == 0.0
    assert window_progress(104.0, 100.0, 108.0) == 0.5
    assert window_progress(120.0, 100.0, 108.0) == 1.0
    assert window_progress(5.0, 10.0, 10.0) == 0.0


def test_gaze_vector_signs():
    # Subject looks toward their own left = image right in broadcast view.
    right_img = {"eyeLookOutLeft": 0.8, "eyeLookInRight": 0.8}
    gx, gy = gaze_vector(right_img)
    assert gx > 0.5 and abs(gy) < 1e-9
    down = {"eyeLookDownLeft": 0.6, "eyeLookDownRight": 0.6}
    gx, gy = gaze_vector(down)
    assert gy > 0.5 and abs(gx) < 1e-9


def test_face_tags_thresholds():
    assert face_tags({"browInnerUp": 0.5}) == ["brow raised"]
    assert face_tags({"jawOpen": 0.5}) == ["mouth open"]
    assert face_tags({"jawOpen": 0.05, "browInnerUp": 0.1}) == []


def test_head_dir_normalized_offset():
    lm = np.zeros((478, 2))
    lm[FACE_EDGE_L] = (0.0, 0.0)
    lm[FACE_EDGE_R] = (100.0, 0.0)
    lm[NOSE_TIP] = (70.0, 10.0)
    dx, dy = head_dir(lm)
    assert abs(dx - 0.2) < 1e-6
    assert abs(dy - 0.1) < 1e-6


def test_live_blink_counts_with_debounce():
    lb = LiveBlink()
    fps = 30.0
    series = [0.1] * 5 + [0.8] * 3 + [0.1] * 5 + [0.8] * 3 + [0.1] * 5
    for i, v in enumerate(series):
        lb.update(v, v, i / fps)
    assert lb.blinks == 2
    assert lb.rate > 0


def test_acting_wrist_picks_moving_hand():
    aw = ActingWrist(fps=30.0)
    for i in range(30):
        lm = np.zeros((33, 3))
        lm[LEFT_WRIST] = (100.0, 100.0, 0.9)          # still
        lm[RIGHT_WRIST] = (200.0 + 8 * i, 300.0, 0.9)  # moving
        aw.update(lm, i / 30.0)
    trail = aw.trail()
    assert len(trail) > 10
    xs = [p[0] for p in trail]
    assert max(xs) - min(xs) > 50  # the moving wrist's spread, not the still one

    aw.reset()
    assert aw.trail() == []
