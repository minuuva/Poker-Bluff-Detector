"""Behavioral feature extraction over decision windows.

For each Decision of a mapped player, reads the window's frames, finds the
player by FACE RE-IDENTIFICATION, and runs face and pose tracking into one
feature row per decision. Output joins with betting features on (hand_id,
player, t_end).

Identity design (v2): HCL's cameras are operated: they zoom and pan, so
camera-angle signatures do not survive contact with real footage. What is
stable within a session is the player's own appearance. The seat map gives
each player a search region and a few reference timestamps; at init the
extractor builds equalized grayscale face chips from those references, and
at extraction time every detected face's chip must correlate with the
player's references above CHIP_MATCH_THRESH to be attributed (measured
separation on real footage: same player ~0.74, different players ~0.25).
The pose crop then follows the accepted face, which makes the pipeline
zoom-proof. Rows record chip/face/pose coverage for downstream filters.

Caveat recorded honestly: mid-window zoom changes distort pixel
trajectories; wrist speeds are shoulder-width normalized, and the
amplitude-invariant smoothness metrics limit but do not eliminate this.
"""

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from pokertell.behavior.face import FACE_FEATURES, FaceTracker, summarize_blendshapes
from pokertell.behavior.pose import POSE_FEATURES, PoseTracker, summarize_pose
from pokertell.checkpoint import trim_partial_line

CHIP_SIZE = 48
CHIP_MATCH_THRESH = 0.55
MIN_FACE_W = 60
WINDOW_PRE_PAD_S = 0.5
COMMIT_PAD_S = 2.0
COMMIT_SEGMENT_S = 6.0
MAX_WINDOW_S = 90.0
FACE_STRIDE = 2


@dataclass
class PlayerRef:
    """Search region plus face-chip references for one player."""

    search: tuple[int, int, int, int]
    face_refs: list[dict]
    chips: list[np.ndarray] = field(default_factory=list)


def load_seats(path: Path) -> dict[str, PlayerRef]:
    raw = yaml.safe_load(Path(path).read_text())
    out = {}
    for name, spec in raw.get("seats", {}).items():
        s = spec["search"]
        out[name] = PlayerRef(
            search=(s["x"], s["y"], s["w"], s["h"]),
            face_refs=list(spec.get("face_refs", [])),
        )
    return out


def face_chip(crop_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
    """Equalized grayscale chip for a detected face bbox (crop coords)."""
    x, y, w, h = bbox
    if w < MIN_FACE_W:
        return None
    pad = int(0.15 * w)
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1 = min(crop_bgr.shape[1], x + w + pad)
    y1 = min(crop_bgr.shape[0], y + h + pad)
    if x1 - x0 < 24 or y1 - y0 < 24:
        return None
    gray = cv2.cvtColor(crop_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(cv2.resize(gray, (CHIP_SIZE, CHIP_SIZE)))


def chip_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED).max())


class BehaviorExtractor:
    def __init__(self, video: Path, seats: dict[str, PlayerRef]) -> None:
        self.video = Path(video)
        self.seats = seats
        self.cap = cv2.VideoCapture(str(video))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._build_reference_chips()

    def _build_reference_chips(self) -> None:
        face = FaceTracker()
        try:
            for name, ref in self.seats.items():
                for spec in ref.face_refs:
                    self.cap.set(cv2.CAP_PROP_POS_MSEC, float(spec["t"]) * 1000)
                    ok, frame = self.cap.read()
                    if not ok:
                        continue
                    crop = self._search_crop(frame, ref)
                    faces = face.process(crop, int(spec["t"] * 1000))
                    if not faces:
                        continue
                    sx = ref.search[0]
                    target_cx = float(spec["cx"]) - sx
                    _, bbox = min(
                        faces, key=lambda fb: abs(fb[1][0] + fb[1][2] / 2 - target_cx)
                    )
                    chip = face_chip(crop, bbox)
                    if chip is not None:
                        ref.chips.append(chip)
                if not ref.chips:
                    raise ValueError(f"no reference chips built for {name}")
        finally:
            face.close()

    @staticmethod
    def _search_crop(frame: np.ndarray, ref: PlayerRef) -> np.ndarray:
        x, y, w, h = ref.search
        return frame[y : y + h, x : x + w]

    def close(self) -> None:
        self.cap.release()

    def extract_decision(self, decision: dict) -> dict | None:
        player = decision["player"]
        ref = self.seats.get(player)
        if ref is None:
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
        n_identified = 0
        last_bbox: tuple[int, int, int, int] | None = None
        try:
            self.cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000)
            while True:
                t = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                ok, frame = self.cap.read()
                if not ok or t > t1:
                    break
                crop = self._search_crop(frame, ref)
                run_face = n % FACE_STRIDE == 0
                accepted_bbox = None
                if run_face:
                    shapes = None
                    for cand_shapes, bbox in face.process(crop, int(t * 1000)):
                        chip = face_chip(crop, bbox)
                        if chip is None:
                            continue
                        score = max(chip_similarity(chip, c) for c in ref.chips)
                        if score >= CHIP_MATCH_THRESH:
                            shapes, accepted_bbox = cand_shapes, bbox
                            break
                    blend_frames.append(shapes)
                    if accepted_bbox is not None:
                        n_identified += 1
                        last_bbox = accepted_bbox
                # Pose follows the most recent identified face.
                if last_bbox is not None:
                    x, y, w, h = last_bbox
                    px0 = max(0, x - int(1.6 * w))
                    py0 = max(0, y - int(0.6 * h))
                    px1 = min(crop.shape[1], x + w + int(1.6 * w))
                    py1 = min(crop.shape[0], y + h + int(3.2 * h))
                    sub = crop[py0:py1, px0:px1]
                    lm = pose.process(sub)
                    if lm is not None:
                        lm = lm.copy()
                        lm[:, 0] += px0
                        lm[:, 1] += py0
                    pose_frames.append(lm)
                else:
                    pose_frames.append(None)
                n += 1
        finally:
            face.close()
            pose.close()
        if n == 0:
            return None

        face_feats = summarize_blendshapes(blend_frames, self.fps / FACE_STRIDE)
        commit_frames = min(n, int(COMMIT_SEGMENT_S * self.fps))
        pose_feats = summarize_pose(pose_frames, self.fps, commit_frames)
        n_face_frames = max(1, len(blend_frames))
        return {
            "hand_id": decision["hand_id"],
            "player": player,
            "t_start": decision["t_start"],
            "t_end": decision["t_end"],
            "window_s": round(t1 - t0, 2),
            "n_frames": n,
            "shot_coverage": round(n_identified / n_face_frames, 3),
            **face_feats,
            **pose_feats,
        }


# Fixed CSV schema: every extract_decision row has exactly these keys, and a
# stable order lets interrupted runs append to the same file safely.
BEHAVIOR_COLUMNS = [
    "hand_id",
    "player",
    "t_start",
    "t_end",
    "window_s",
    "n_frames",
    "shot_coverage",
    *FACE_FEATURES,
    "face_coverage",
    *POSE_FEATURES,
    "pose_coverage",
]


def decision_key(hand_id: str, player: str, t_end: float) -> tuple[str, str, float]:
    return (str(hand_id), str(player), round(float(t_end), 2))


def load_done_keys(out_path: Path) -> set[tuple[str, str, float]]:
    """Keys of decisions already present in a partial output CSV.

    Raises if the file's schema does not match BEHAVIOR_COLUMNS: mixing
    rows from an older extractor version into a resume would silently
    corrupt the dataset (the v1 shot-signature extractor left such a file
    behind once).
    """
    out_path = Path(out_path)
    if not out_path.exists() or out_path.stat().st_size == 0:
        return set()
    df = pd.read_csv(out_path)
    if list(df.columns) != BEHAVIOR_COLUMNS:
        raise ValueError(
            f"{out_path} has a different column schema (older extractor version?); "
            "delete it or rerun with a fresh output path"
        )
    return {
        decision_key(r.hand_id, r.player, r.t_end)
        for r in df.itertuples(index=False)
    }


def extract_session_behavior(
    video: Path,
    hands_path: Path,
    seats_path: Path,
    out_path: Path | None = None,
    progress=None,
) -> pd.DataFrame:
    """One row of behavioral features per mapped decision.

    With out_path set, rows append to the CSV as they are computed
    (flushed per row, so a kill loses at most the decision in flight)
    and decisions already present in the file are skipped, which makes
    rerunning the command a resume. Returns the full table either way.
    """
    seats = load_seats(seats_path)
    hands = [json.loads(line) for line in Path(hands_path).open()]

    done: set[tuple[str, str, float]] = set()
    writer = None
    out_file = None
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        trim_partial_line(out_path)
        done = load_done_keys(out_path)
        new_file = not out_path.exists() or out_path.stat().st_size == 0
        out_file = out_path.open("a", newline="")
        writer = csv.DictWriter(out_file, fieldnames=BEHAVIOR_COLUMNS)
        if new_file:
            writer.writeheader()
            out_file.flush()

    extractor = BehaviorExtractor(video, seats)
    rows = []
    try:
        for hand in hands:
            for decision in hand["decisions"]:
                if decision["player"] not in seats:
                    continue
                key = decision_key(
                    decision["hand_id"], decision["player"], decision["t_end"]
                )
                if key in done:
                    continue
                row = extractor.extract_decision(decision)
                if row is None:
                    continue
                rows.append(row)
                if writer is not None:
                    writer.writerow(row)
                    out_file.flush()
                if progress is not None:
                    progress(row)
    finally:
        extractor.close()
        if out_file is not None:
            out_file.close()
    if out_path is not None:
        return pd.read_csv(out_path)
    return pd.DataFrame(rows)
