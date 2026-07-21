"""Demo overlay rendering: burn the pipeline's view onto a short clip.

Iteration 2 of the demo artifact. For one decision window it draws the
assembled game state with a decision-time bar, the face re-identification
box with its chip score, facial detail from the landmarks the pipeline
already computes (eye, brow and lip contours, a gaze arrow from the
eyeLook blendshapes, a head-facing arrow from the nose direction, and
expression tags), a One-Euro smoothed trail of the acting wrist, live
blink counting, the window's behavioral z-scores labeled as such, the
held-out six-class hand-strength distribution, and P(bluff) for the
betting baseline next to baseline + behavior. The clip ends on a freeze
frame that reveals the hole cards and the equity label.

Data policy note: demo clips are for the portfolio writeup and are not
committed to the repo; rendered media stays local under data/demo (see
README, Ethics and data).
"""

import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from pokertell.behavior.events import (
    FREEZE_THRESH,
    GAZE_DOWN_OFF,
    GAZE_DOWN_ON,
    NEAR_FACE_DIST_W,
    gaze_down_series,
    motion_energy,
)
from pokertell.behavior.extract import (
    CHIP_MATCH_THRESH,
    BehaviorExtractor,
    chip_similarity,
    face_chip,
    load_seats,
)
from pokertell.behavior.face import (
    BLINK_DEBOUNCE_S,
    BLINK_OFF,
    BLINK_ON,
    FaceTracker,
)
from pokertell.behavior.pose import LEFT_WRIST, RIGHT_WRIST, PoseTracker
from pokertell.behavior.smoothing import OneEuroFilter

PRE_ROLL_S = 2.0
POST_ROLL_S = 3.0
FREEZE_S = 3.5
TRAIL_LEN = 36
TRAIL_EXPIRE_MS = 500
WRIST_SWITCH_RATIO = 1.5

GREEN = (80, 220, 80)
GRAY = (170, 170, 170)
GOLD = (60, 190, 235)
WHITE = (235, 235, 235)
RED = (70, 70, 230)
CYAN = (220, 200, 80)
MAGENTA = (200, 80, 220)
FONT = cv2.FONT_HERSHEY_SIMPLEX

STRENGTH_ORDER = ["bluff", "weak_draw", "medium", "strong_draw", "strong", "monster"]
STRENGTH_COLORS = {
    "bluff": (60, 60, 230),
    "weak_draw": (60, 140, 245),
    "medium": (60, 200, 245),
    "strong_draw": (90, 220, 170),
    "strong": (80, 220, 80),
    "monster": (60, 180, 40),
}

NOSE_TIP = 1
FACE_EDGE_L, FACE_EDGE_R = 234, 454


def fmt_cards(cards: list[str] | None) -> str:
    """['8h', '4s'] -> '8h 4s'."""
    return " ".join(c[0].upper() + c[1] for c in cards) if cards else "??"


def phase_label(t: float, decision: dict) -> str:
    """What the acting player is doing at clip time t."""
    if t < decision["t_start"]:
        return "waiting"
    if t <= decision["t_end"]:
        return "to act"
    amount = decision.get("amount")
    action = (decision.get("action") or "").upper()
    return f"{action} ${amount:,.0f}" if amount else action


def state_lines(decision: dict, hand: dict, t: float) -> list[str]:
    """Text block for the game-state panel at clip time t."""
    pot = decision.get("pot_before")
    return [
        f"{hand['hand_id']}  {decision['street'].upper()}  t={t:7.1f}s",
        f"pot ${pot:,.0f}" if pot else "pot unknown",
        f"{decision['player']} ({decision.get('position') or '?'}): {phase_label(t, decision)}",
    ]


def window_progress(t: float, t_start: float, t_end: float) -> float:
    """Fraction of the decision window elapsed at time t, clipped to [0, 1]."""
    if t_end <= t_start:
        return 0.0
    return max(0.0, min(1.0, (t - t_start) / (t_end - t_start)))


def gaze_vector(shapes: dict) -> tuple[float, float]:
    """Image-space gaze direction from the eyeLook blendshapes.

    Blendshape left/right follow the SUBJECT, so in an unmirrored broadcast
    view the subject's left is image right. Positive x is image right,
    positive y is image down; each component lands in [-1, 1].
    """
    def s(k):
        return shapes.get(k, 0.0)

    gx = ((s("eyeLookOutLeft") + s("eyeLookInRight"))
          - (s("eyeLookInLeft") + s("eyeLookOutRight"))) / 2
    gy = ((s("eyeLookDownLeft") + s("eyeLookDownRight"))
          - (s("eyeLookUpLeft") + s("eyeLookUpRight"))) / 2
    return gx, gy


def face_tags(shapes: dict) -> list[str]:
    """Short expression states worth printing next to the face box."""
    def s(k):
        return shapes.get(k, 0.0)

    tags = []
    if s("browInnerUp") > 0.35:
        tags.append("brow raised")
    if (s("browDownLeft") + s("browDownRight")) / 2 > 0.35:
        tags.append("brow furrowed")
    if s("jawOpen") > 0.2:
        tags.append("mouth open")
    if (s("mouthPressLeft") + s("mouthPressRight")) / 2 > 0.4:
        tags.append("lip press")
    if max(s("mouthSmileLeft"), s("mouthSmileRight")) > 0.4:
        tags.append("smile")
    return tags


def head_dir(landmarks: np.ndarray) -> tuple[float, float]:
    """2D facing direction: nose tip offset from the face-edge midpoint,
    normalized by face width. Roughly (0, 0) when facing the camera."""
    mid = (landmarks[FACE_EDGE_L] + landmarks[FACE_EDGE_R]) / 2
    width = float(np.linalg.norm(landmarks[FACE_EDGE_R] - landmarks[FACE_EDGE_L])) + 1e-6
    d = (landmarks[NOSE_TIP] - mid) / width
    return float(d[0]), float(d[1])


class LiveBlink:
    """Streaming blink counter matching count_blinks() semantics."""

    def __init__(self) -> None:
        self.blinks = 0
        self._armed = True
        self._last_t = -1e9
        self._t0: float | None = None
        self._t: float = 0.0

    def update(self, blink_left: float, blink_right: float, t: float) -> None:
        self._t = t
        if self._t0 is None:
            self._t0 = t
        v = min(blink_left, blink_right)
        if self._armed and v > BLINK_ON and t - self._last_t > BLINK_DEBOUNCE_S:
            self.blinks += 1
            self._last_t = t
            self._armed = False
        elif not self._armed and v < BLINK_OFF:
            self._armed = True

    @property
    def rate(self) -> float:
        if self._t0 is None or self._t <= self._t0:
            return 0.0
        return self.blinks / (self._t - self._t0)


class LiveEvents:
    """Streaming event counters mirroring the events.py feature logic."""

    def __init__(self, fps: float) -> None:
        self._fps = fps
        self.gaze_downs = 0
        self._gaze_armed = True
        self.near_face = False
        self.frozen_s = 0.0
        self._freeze_run = 0
        self._lm_buf: deque = deque(maxlen=12)

    def update_face(self, shapes: dict) -> None:
        gy = float(gaze_down_series([shapes])[0])
        if self._gaze_armed and gy > GAZE_DOWN_ON:
            self.gaze_downs += 1
            self._gaze_armed = False
        elif not self._gaze_armed and gy < GAZE_DOWN_OFF:
            self._gaze_armed = True

    def update_pose(self, lm: np.ndarray, face_bbox: tuple[int, int, int, int]) -> None:
        from pokertell.behavior.events import LEFT_WRIST as LW
        from pokertell.behavior.events import MIN_VISIBILITY, RIGHT_WRIST as RW

        x, y, w, h = face_bbox
        cx, cy = x + w / 2, y + h / 2
        self.near_face = any(
            lm[i, 2] >= MIN_VISIBILITY
            and np.hypot(lm[i, 0] - cx, lm[i, 1] - cy) < NEAR_FACE_DIST_W * max(w, 1)
            for i in (LW, RW)
        )
        self._lm_buf.append(lm)
        if len(self._lm_buf) >= 8:
            e = motion_energy(list(self._lm_buf), self._fps)
            last = e[~np.isnan(e)]
            if len(last) and last[-1] < FREEZE_THRESH:
                self._freeze_run += 1
            else:
                self._freeze_run = 0
        self.frozen_s = self._freeze_run / self._fps

    @property
    def line(self) -> str:
        hands = "hand at face" if self.near_face else "hands down"
        return (
            f"gaze-downs {self.gaze_downs}   {hands}   "
            f"still {self.frozen_s:.1f}s"
        )


class ActingWrist:
    """Smoothed trails for both wrists; exposes the acting one.

    Points pass through One-Euro filters at append time so the drawn trail
    matches how the feature pipeline treats trajectories. The acting wrist
    is the one with the larger smoothed path over the trail window, with
    hysteresis so the choice does not flip frame to frame.
    """

    def __init__(self, fps: float) -> None:
        self._fps = fps
        self._trails = {i: deque(maxlen=TRAIL_LEN) for i in (LEFT_WRIST, RIGHT_WRIST)}
        self._filters = {}
        self._current = LEFT_WRIST
        self.reset()

    def reset(self) -> None:
        for i in (LEFT_WRIST, RIGHT_WRIST):
            self._trails[i].clear()
            self._filters[i] = (OneEuroFilter(self._fps), OneEuroFilter(self._fps))

    def update(self, landmarks: np.ndarray, t: float) -> None:
        """landmarks: (33, 3) full-frame pose points with visibility."""
        for i in (LEFT_WRIST, RIGHT_WRIST):
            x, y, vis = landmarks[i]
            if vis < 0.5:
                continue
            fx, fy = self._filters[i]
            self._trails[i].append((fx(float(x), t), fy(float(y), t)))

    @staticmethod
    def _path(trail) -> float:
        pts = np.array(trail)
        return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()) if len(pts) > 1 else 0.0

    def trail(self) -> list[tuple[float, float]]:
        paths = {i: self._path(self._trails[i]) for i in self._trails}
        other = LEFT_WRIST if self._current == RIGHT_WRIST else RIGHT_WRIST
        if paths[other] > WRIST_SWITCH_RATIO * paths[self._current]:
            self._current = other
        return list(self._trails[self._current])


def feature_lines(behavior_row: dict) -> list[str]:
    """Window-level behavioral z-scores, labeled as baseline-relative."""
    def val(key):
        v = behavior_row.get(key)
        try:
            return f"{v:+.1f}" if v is not None and not np.isnan(v) else "n/a"
        except TypeError:
            return "n/a"

    return [
        f"coverage: face {behavior_row.get('face_coverage', 0):.2f}"
        f"  pose {behavior_row.get('pose_coverage', 0):.2f}",
        "window vs his own baseline (sd):",
        f"  blink {val('blink_rate')}   smile asym {val('smile_asymmetry')}",
        f"  wrist smoothness {val('wrist_jerk_ldj')}   lean {val('lean_std')}",
        f"  head motion {val('head_motion')}   gaze disp {val('gaze_dispersion')}",
        f"  gaze-down {val('gaze_down_rate')}   near-face {val('near_face_frac')}",
        f"  freeze {val('freeze_frac')}   shuffle {val('shuffle_score')}"
        f"   lean fwd {val('lean_fwd')}",
    ]


def _panel(frame: np.ndarray, x: int, y: int, w: int, h: int) -> None:
    overlay = frame[y : y + h, x : x + w].copy()
    cv2.rectangle(frame, (x, y), (x + w, y + h), (25, 25, 25), -1)
    frame[y : y + h, x : x + w] = cv2.addWeighted(
        overlay, 0.35, frame[y : y + h, x : x + w], 0.65, 0
    )
    cv2.rectangle(frame, (x, y), (x + w, y + h), (90, 90, 90), 1)


def _text(frame, s, org, scale=0.55, color=WHITE, thick=1):
    cv2.putText(frame, s, org, FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(frame, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def _prob_bar(frame, label, p, x, y, w=300, color=None):
    _text(frame, f"{label} {p:5.1%}", (x, y - 6), 0.5)
    cv2.rectangle(frame, (x, y), (x + w, y + 14), (60, 60, 60), -1)
    fill = int(w * max(0.0, min(1.0, p)))
    bar = color if color is not None else (RED if p > 0.5 else GOLD)
    cv2.rectangle(frame, (x, y), (x + fill, y + 14), bar, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + 14), (110, 110, 110), 1)


def _draw_strength_panel(frame, strength_probs: dict, x: int, y: int) -> None:
    _panel(frame, x, y, 396, 40 + 26 * len(STRENGTH_ORDER))
    _text(frame, "hand-strength model (held-out)", (x + 16, y + 26), 0.5, GOLD)
    for i, cls in enumerate(STRENGTH_ORDER):
        p = strength_probs.get(cls, 0.0)
        yy = y + 44 + 26 * i
        _text(frame, f"{cls:<11s}", (x + 16, yy + 10), 0.45)
        cv2.rectangle(frame, (x + 140, yy), (x + 340, yy + 12), (60, 60, 60), -1)
        cv2.rectangle(
            frame, (x + 140, yy),
            (x + 140 + int(200 * min(1.0, p)), yy + 12),
            STRENGTH_COLORS[cls], -1,
        )
        _text(frame, f"{p:5.1%}", (x + 344, yy + 11), 0.42, GRAY)


_CONTOUR_SETS = None


def _contour_sets():
    global _CONTOUR_SETS
    if _CONTOUR_SETS is None:
        from mediapipe.tasks.python.vision.face_landmarker import (
            FaceLandmarksConnections as FC,
        )

        _CONTOUR_SETS = [
            (FC.FACE_LANDMARKS_LEFT_EYE, CYAN),
            (FC.FACE_LANDMARKS_RIGHT_EYE, CYAN),
            (FC.FACE_LANDMARKS_LEFT_EYEBROW, GOLD),
            (FC.FACE_LANDMARKS_RIGHT_EYEBROW, GOLD),
            (FC.FACE_LANDMARKS_LIPS, MAGENTA),
        ]
    return _CONTOUR_SETS


def _draw_face_detail(frame, landmarks_full: np.ndarray, shapes: dict, bbox_full) -> None:
    """Contours, gaze arrow, head-facing arrow, and expression tags."""
    pts = landmarks_full.astype(np.int32)
    for connections, color in _contour_sets():
        for conn in connections:
            cv2.line(
                frame, tuple(pts[conn.start]), tuple(pts[conn.end]), color, 1, cv2.LINE_AA
            )

    x, y, w, h = bbox_full
    eye_mid = ((pts[FACE_EDGE_L] + pts[FACE_EDGE_R]) / 2).astype(int)
    gx, gy = gaze_vector(shapes)
    if abs(gx) + abs(gy) > 0.08:
        tip = (int(eye_mid[0] + gx * 1.2 * w), int(eye_mid[1] + gy * 1.2 * w))
        cv2.arrowedLine(frame, tuple(eye_mid), tip, MAGENTA, 2, cv2.LINE_AA, tipLength=0.25)
        _text(frame, "gaze", (tip[0] + 6, tip[1]), 0.45, MAGENTA)

    dx, dy = head_dir(landmarks_full)
    nose = tuple(pts[NOSE_TIP])
    tip = (int(nose[0] + dx * 2.2 * w), int(nose[1] + dy * 2.2 * w))
    cv2.arrowedLine(frame, nose, tip, GOLD, 2, cv2.LINE_AA, tipLength=0.3)

    for i, tag in enumerate(face_tags(shapes)):
        _text(frame, tag, (x + w + 10, y + 18 + 22 * i), 0.5, GOLD)


def _draw_trail(frame, trail: list[tuple[float, float]]) -> None:
    """Acting-wrist trail, fading with age."""
    n = len(trail)
    if n < 2:
        return
    for i in range(1, n):
        age = i / n
        color = tuple(int(c * (0.3 + 0.7 * age)) for c in GOLD)
        p0 = (int(trail[i - 1][0]), int(trail[i - 1][1]))
        p1 = (int(trail[i][0]), int(trail[i][1]))
        cv2.line(frame, p0, p1, color, 1 + int(2 * age), cv2.LINE_AA)


def render_demo(
    video: Path,
    hands_path: Path,
    seats_path: Path,
    out_path: Path,
    hand_id: str,
    player: str,
    t_end: float,
    probs: dict,
    behavior_row: dict,
    target: str = "is_bluff",
) -> Path:
    """Render one decision window to out_path. Returns out_path."""
    hand = next(
        h
        for h in (json.loads(line) for line in Path(hands_path).open())
        if h["hand_id"] == hand_id
    )
    decision = next(
        d
        for d in hand["decisions"]
        if d["player"] == player and abs(d["t_end"] - t_end) < 0.51
    )

    seats = load_seats(seats_path)
    ref = seats[player]
    extractor = BehaviorExtractor(video, seats)  # builds reference chips
    cap = extractor.cap
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    face = FaceTracker()
    pose = PoseTracker()
    wrist = ActingWrist(fps)
    blink = LiveBlink()
    events = LiveEvents(fps)
    last_accept_ms = -10_000
    sx, sy = ref.search[0], ref.search[1]

    t0 = max(0.0, decision["t_start"] - PRE_ROLL_S)
    t1 = decision["t_end"] + POST_ROLL_S
    cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000)
    frame = None
    ts_ms = 0
    try:
        while True:
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            ok, got = cap.read()
            if not ok or t > t1:
                break
            frame = got
            crop = extractor._search_crop(frame, ref)
            ts_ms += int(1000 / fps)
            accepted = None
            score = 0.0
            for obs in face.detect(crop, ts_ms):
                chip = face_chip(crop, obs.bbox)
                if chip is None:
                    continue
                s = max(chip_similarity(chip, c) for c in ref.chips)
                if s >= CHIP_MATCH_THRESH:
                    accepted, score = obs, s
                    break
            if accepted is not None:
                last_accept_ms = ts_ms
                x, y, bw, bh = accepted.bbox
                bbox_full = (sx + x, sy + y, bw, bh)
                cv2.rectangle(
                    frame, bbox_full[:2],
                    (sx + x + bw, sy + y + bh), GREEN, 2,
                )
                _text(frame, f"{player} re-id {score:.2f}", (sx + x, sy + y - 8), 0.5, GREEN)
                lm_full = accepted.landmarks + np.array([sx, sy])
                _draw_face_detail(frame, lm_full, accepted.shapes, bbox_full)
                blink.update(
                    accepted.shapes.get("eyeBlinkLeft", 0.0),
                    accepted.shapes.get("eyeBlinkRight", 0.0),
                    t,
                )
                events.update_face(accepted.shapes)
                px0, py0 = max(0, x - int(1.6 * bw)), max(0, y - int(0.6 * bh))
                px1 = min(crop.shape[1], x + bw + int(1.6 * bw))
                py1 = min(crop.shape[0], y + bh + int(3.2 * bh))
                lm = pose.process(crop[py0:py1, px0:px1])
                if lm is not None:
                    lm = lm.copy()
                    lm[:, 0] += sx + px0
                    lm[:, 1] += sy + py0
                    wrist.update(lm, t)
                    events.update_pose(lm, bbox_full)
            if ts_ms - last_accept_ms > TRAIL_EXPIRE_MS:
                wrist.reset()
            _draw_trail(frame, wrist.trail())

            _panel(frame, w // 2 - 300, h - 168, 600, 144)
            for i, line in enumerate(state_lines(decision, hand, t)):
                _text(frame, line, (w // 2 - 284, h - 138 + 26 * i), 0.6)
            frac = window_progress(t, decision["t_start"], decision["t_end"])
            if 0.0 < frac < 1.0:
                bx, by = w // 2 - 284, h - 52
                _text(frame, f"decision time {t - decision['t_start']:4.1f}s", (bx, by - 6), 0.5, RED)
                cv2.rectangle(frame, (bx, by), (bx + 560, by + 12), (60, 60, 60), -1)
                cv2.rectangle(frame, (bx, by), (bx + int(560 * frac), by + 12), RED, -1)

            _draw_strength_panel(frame, probs.get("strength", {}), w - 420, 24)
            py = 24 + 40 + 26 * len(STRENGTH_ORDER) + 12
            _panel(frame, w - 420, py, 396, 292)
            _text(frame, f"P({target}) leave-one-session-out", (w - 404, py + 26), 0.5, GOLD)
            _prob_bar(frame, "betting only ", probs["base"], w - 404, py + 56)
            _prob_bar(frame, "with behavior", probs["full"], w - 404, py + 102)
            _text(
                frame,
                f"blinks {blink.blinks}  ({blink.rate:.2f}/s live)",
                (w - 404, py + 140), 0.45, WHITE,
            )
            _text(frame, events.line, (w - 404, py + 160), 0.45, WHITE)
            for i, line in enumerate(feature_lines(behavior_row)):
                _text(frame, line, (w - 404, py + 184 + 17 * i), 0.42, GRAY)
            writer.write(frame)

        if frame is not None:
            _panel(frame, w // 2 - 330, h // 2 - 110, 660, 220)
            eq = decision.get("equity_mc") or behavior_row.get("equity_mc")
            label = behavior_row.get(target)
            reveal = [
                f"{player} shows {fmt_cards(decision.get('hole_cards'))}",
                f"equity vs random: {eq:.1%}" if eq is not None else "equity unknown",
                f"label: {'BLUFF' if label else 'not a bluff'}",
            ]
            for i, line in enumerate(reveal):
                _text(frame, line, (w // 2 - 300, h // 2 - 60 + 56 * i), 0.9, WHITE, 2)
            for _ in range(int(FREEZE_S * fps)):
                writer.write(frame)
    finally:
        face.close()
        pose.close()
        extractor.close()
        writer.release()
    return out_path
