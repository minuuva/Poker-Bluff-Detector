"""Demo overlay rendering: burn the pipeline's view onto a short clip.

The demo artifact is a 20 to 40 second clip of one decision window with
everything the pipeline knows drawn on top: the assembled game state, the
face re-identification box with its chip score, the wrist trail feeding the
smoothness features, the behavioral readouts, and the LOSO-consistent
P(bluff) from the betting baseline next to baseline + behavior. The clip
ends on a freeze frame that reveals the hole cards and the equity label.

Data policy note: demo clips are for the portfolio writeup and are not
committed to the repo; rendered media stays local under data/demo (see
README, Ethics and data).
"""

import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from pokertell.behavior.extract import (
    CHIP_MATCH_THRESH,
    BehaviorExtractor,
    chip_similarity,
    face_chip,
    load_seats,
)
from pokertell.behavior.face import FaceTracker
from pokertell.behavior.pose import LEFT_WRIST, RIGHT_WRIST, PoseTracker

PRE_ROLL_S = 2.0
POST_ROLL_S = 3.0
FREEZE_S = 3.5
TRAIL_LEN = 36

GREEN = (80, 220, 80)
GRAY = (150, 150, 150)
GOLD = (60, 190, 235)
WHITE = (235, 235, 235)
RED = (70, 70, 230)
FONT = cv2.FONT_HERSHEY_SIMPLEX


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
    lines = [
        f"{hand['hand_id']}  {decision['street'].upper()}",
        f"pot ${pot:,.0f}" if pot else "pot unknown",
        f"{decision['player']} ({decision.get('position') or '?'}): {phase_label(t, decision)}",
    ]
    if decision["t_start"] <= t <= decision["t_end"]:
        lines.append(f"decision window {t - decision['t_start']:4.1f}s")
    return lines


def feature_lines(behavior_row: dict) -> list[str]:
    def val(key, fmt="{:.2f}"):
        v = behavior_row.get(key)
        try:
            return fmt.format(v) if v is not None and not np.isnan(v) else "n/a"
        except TypeError:
            return "n/a"

    return [
        f"face cov {val('face_coverage')}  pose cov {val('pose_coverage')}",
        f"blink rate {val('blink_rate')}/s  smile asym {val('smile_asymmetry')}",
        f"wrist smoothness (LDJ) {val('wrist_jerk_ldj')}",
        f"lean std {val('lean_std')}  head motion {val('head_motion')}",
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


def _prob_bar(frame, label, p, x, y, w=300):
    _text(frame, f"{label} {p:5.1%}", (x, y - 6), 0.5)
    cv2.rectangle(frame, (x, y), (x + w, y + 14), (60, 60, 60), -1)
    fill = int(w * max(0.0, min(1.0, p)))
    cv2.rectangle(frame, (x, y), (x + fill, y + 14), RED if p > 0.5 else GOLD, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + 14), (110, 110, 110), 1)


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
    trail: deque = deque(maxlen=TRAIL_LEN)
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
            for _, bbox in face.process(crop, ts_ms):
                chip = face_chip(crop, bbox)
                if chip is None:
                    continue
                s = max(chip_similarity(chip, c) for c in ref.chips)
                if s >= CHIP_MATCH_THRESH:
                    accepted, score = bbox, s
                    break
            if accepted is not None:
                last_accept_ms = ts_ms
                x, y, bw, bh = accepted
                cv2.rectangle(
                    frame, (sx + x, sy + y), (sx + x + bw, sy + y + bh), GREEN, 2
                )
                _text(
                    frame, f"{player} re-id {score:.2f}", (sx + x, sy + y - 8), 0.5, GREEN
                )
                px0, py0 = max(0, x - int(1.6 * bw)), max(0, y - int(0.6 * bh))
                px1 = min(crop.shape[1], x + bw + int(1.6 * bw))
                py1 = min(crop.shape[0], y + bh + int(3.2 * bh))
                lm = pose.process(crop[py0:py1, px0:px1])
                if lm is not None:
                    wrists = [lm[i] for i in (LEFT_WRIST, RIGHT_WRIST) if lm[i][2] > 0.5]
                    if wrists:
                        wx, wy = max(wrists, key=lambda p: p[2])[:2]
                        trail.append((int(sx + px0 + wx), int(sy + py0 + wy)))
            # The trail is only meaningful while the player is being tracked;
            # a camera cut would otherwise leave it smeared across the new shot.
            if ts_ms - last_accept_ms > 500:
                trail.clear()
            if len(trail) > 1:
                pts = np.array(trail, dtype=np.int32)
                cv2.polylines(frame, [pts], False, GOLD, 2, cv2.LINE_AA)

            _panel(frame, w // 2 - 280, h - 150, 560, 126)
            for i, line in enumerate(state_lines(decision, hand, t)):
                _text(frame, line, (w // 2 - 264, h - 118 + 26 * i), 0.6)

            _panel(frame, w - 420, 24, 396, 200)
            _text(frame, f"P({target}) leave-one-session-out", (w - 404, 52), 0.5, GOLD)
            _prob_bar(frame, "betting only ", probs["base"], w - 404, 84)
            _prob_bar(frame, "with behavior", probs["full"], w - 404, 132)
            for i, line in enumerate(feature_lines(behavior_row)):
                _text(frame, line, (w - 404, 162 + 18 * i), 0.42, GRAY)
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
