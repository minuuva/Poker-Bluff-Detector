"""Behavioral feature extraction over decision windows.

For each Decision of a seat-mapped player, reads the window's frames, crops
the player's seat region, runs face and pose tracking, and emits one row of
behavioral features. Output joins with betting features on (hand_id,
player, t_end).

Identity is the hard part of this stage. HCL has no master wide shot: the
broadcast cycles through fixed camera angles, each framing 2-3 seats, plus
graphics and replays. A seat crop is therefore only valid in its angle, so
the per-session seat map (configs/seats/<session>.yaml) gives each target
player a LIST of views: a reference timestamp identifying the camera angle
plus the player's bbox in that angle. At extraction time every frame is
matched against the angle signatures (correlation of the frame's top-half
thumbnail, where the static camera background dominates and the HUD does
not reach) and only matching frames contribute, with the matched view's
box. Two further gates keep bad frames out: face detections must be a
plausible size for the crop, and every row records face/pose coverage so
low-coverage windows can be filtered downstream.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from pokertell.behavior.face import FaceTracker, summarize_blendshapes
from pokertell.behavior.pose import PoseTracker, summarize_pose

FACE_MIN_H_FRAC = 0.10
FACE_MAX_H_FRAC = 0.65
WINDOW_PRE_PAD_S = 0.5
COMMIT_PAD_S = 2.0
COMMIT_SEGMENT_S = 6.0
MAX_WINDOW_S = 90.0
FACE_FRAME_STRIDE = 2
SHOT_MATCH_THRESH = 0.62
SIG_SIZE = (48, 15)
SIG_TOP_FRAC = 0.55


@dataclass(frozen=True)
class SeatView:
    """One camera angle in which a player is visible."""

    shot_t: float
    x: int
    y: int
    w: int
    h: int


def load_seats(path: Path) -> dict[str, list[SeatView]]:
    raw = yaml.safe_load(Path(path).read_text())
    return {
        name: [SeatView(**view) for view in views]
        for name, views in raw.get("seats", {}).items()
    }


def shot_signature(frame: np.ndarray) -> np.ndarray:
    """Normalized thumbnail of the frame's top half (camera-angle identity)."""
    top = frame[: int(frame.shape[0] * SIG_TOP_FRAC)]
    g = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)
    s = cv2.resize(g, SIG_SIZE).astype(float).ravel()
    return (s - s.mean()) / (s.std() + 1e-9)


class BehaviorExtractor:
    def __init__(self, video: Path, seats: dict[str, list[SeatView]]) -> None:
        self.video = Path(video)
        self.seats = seats
        self.cap = cv2.VideoCapture(str(video))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._signatures: dict[float, np.ndarray] = {}
        for views in seats.values():
            for view in views:
                if view.shot_t not in self._signatures:
                    self.cap.set(cv2.CAP_PROP_POS_MSEC, view.shot_t * 1000)
                    ok, frame = self.cap.read()
                    if not ok:
                        raise ValueError(f"cannot read reference frame at t={view.shot_t}")
                    self._signatures[view.shot_t] = shot_signature(frame)

    def close(self) -> None:
        self.cap.release()

    def _match_view(self, frame: np.ndarray, views: list[SeatView]) -> SeatView | None:
        sig = shot_signature(frame)
        best, best_corr = None, SHOT_MATCH_THRESH
        for view in views:
            ref = self._signatures[view.shot_t]
            corr = float(np.dot(ref, sig) / len(sig))
            if corr > best_corr:
                best, best_corr = view, corr
        return best

    def extract_decision(self, decision: dict) -> dict | None:
        player = decision["player"]
        views = self.seats.get(player)
        if not views:
            return None
        t1 = decision["t_end"] + COMMIT_PAD_S
        t0 = max(0.0, decision["t_start"] - WINDOW_PRE_PAD_S)
        if t1 - t0 > MAX_WINDOW_S:
            t0 = t1 - MAX_WINDOW_S

        face = FaceTracker()
        pose = PoseTracker()
        blend_frames: list[dict | None] = []
        pose_frames: list[np.ndarray | None] = []
        n = 0
        n_shot_matched = 0
        try:
            self.cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000)
            while True:
                t = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                ok, frame = self.cap.read()
                if not ok or t > t1:
                    break
                t_ms = int(t * 1000)
                view = self._match_view(frame, views)
                crop = None
                if view is not None:
                    n_shot_matched += 1
                    crop = frame[view.y : view.y + view.h, view.x : view.x + view.w]
                if n % FACE_FRAME_STRIDE == 0:
                    shapes = None
                    if crop is not None:
                        hit = face.process(crop, t_ms)
                        if hit is not None:
                            s, h_frac = hit
                            if FACE_MIN_H_FRAC <= h_frac <= FACE_MAX_H_FRAC:
                                shapes = s
                    blend_frames.append(shapes)
                pose_frames.append(pose.process(crop, t_ms) if crop is not None else None)
                n += 1
        finally:
            face.close()
            pose.close()
        if n == 0:
            return None

        face_feats = summarize_blendshapes(blend_frames, self.fps / FACE_FRAME_STRIDE)
        commit_frames = min(n, int(COMMIT_SEGMENT_S * self.fps))
        pose_feats = summarize_pose(pose_frames, self.fps, commit_frames)
        return {
            "hand_id": decision["hand_id"],
            "player": player,
            "t_start": decision["t_start"],
            "t_end": decision["t_end"],
            "window_s": round(t1 - t0, 2),
            "n_frames": n,
            "shot_coverage": round(n_shot_matched / n, 3),
            **face_feats,
            **pose_feats,
        }


def extract_session_behavior(
    video: Path,
    hands_path: Path,
    seats_path: Path,
    progress=None,
) -> pd.DataFrame:
    """One row of behavioral features per seat-mapped decision."""
    seats = load_seats(seats_path)
    hands = [json.loads(line) for line in Path(hands_path).open()]
    extractor = BehaviorExtractor(video, seats)
    rows = []
    try:
        for hand in hands:
            for decision in hand["decisions"]:
                row = extractor.extract_decision(decision)
                if row is not None:
                    rows.append(row)
                    if progress is not None:
                        progress(row)
    finally:
        extractor.close()
    return pd.DataFrame(rows)
