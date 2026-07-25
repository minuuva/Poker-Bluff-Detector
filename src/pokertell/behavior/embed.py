"""Pretrained-embedding extraction for decision windows (prereg round 2).

Experiment A of docs/prereg_round2.md: replace hand-crafted behavioral
summaries with a frozen DINOv2 ViT-S/14 representation and re-run the
same evaluation. This module produces one row per covered decision:
identity-gated face crops and their pose crops are embedded per frame
(stride EMBED_STRIDE), and each stream is pooled to the per-window mean
and standard deviation of the CLS embedding, giving 4 x 384 = 1536
columns. PCA reduction happens at analysis time, fit on training folds
only, per the preregistration.

Torch is an optional dependency (the embed extra); everything importable
without it stays importable, and the embedder is injectable so the
pipeline logic is testable without model weights.
"""

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from pokertell.behavior.extract import (
    BehaviorExtractor,
    decision_key,
    face_chip,
    load_seats,
    match_face,
)
from pokertell.behavior.face import FaceTracker
from pokertell.checkpoint import trim_partial_line

EMBED_STRIDE = 4
EMBED_DIM = 384
BATCH = 32
INPUT_PX = 224
WINDOW_PRE_PAD_S = 0.5
COMMIT_PAD_S = 2.0
MAX_WINDOW_S = 90.0
MIN_EMBED_FRAMES = 4

ID_COLUMNS = ["hand_id", "player", "t_start", "t_end", "n_embedded"]


def embed_columns() -> list[str]:
    cols = list(ID_COLUMNS)
    for stream in ("face", "pose"):
        for stat in ("mean", "std"):
            cols += [f"e_{stream}_{stat}_{i}" for i in range(EMBED_DIM)]
    return cols


class DinoEmbedder:
    """Frozen DINOv2 ViT-S/14 CLS embeddings for BGR crops."""

    def __init__(self, device: str | None = None) -> None:
        import torch

        self._torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self.model.eval().to(self.device)
        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __call__(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        torch = self._torch
        out = []
        with torch.no_grad():
            for i in range(0, len(crops_bgr), BATCH):
                batch = []
                for c in crops_bgr[i : i + BATCH]:
                    rgb = cv2.cvtColor(
                        cv2.resize(c, (INPUT_PX, INPUT_PX)), cv2.COLOR_BGR2RGB
                    )
                    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
                    batch.append((t - self._mean) / self._std)
                x = torch.stack(batch).to(self.device)
                out.append(self.model(x).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, EMBED_DIM))


def pool_stream(embs: np.ndarray) -> np.ndarray:
    """Mean and std over frames; NaNs when too few frames."""
    if len(embs) < MIN_EMBED_FRAMES:
        return np.full(2 * EMBED_DIM, np.nan)
    return np.concatenate([embs.mean(axis=0), embs.std(axis=0)])


def _clamped(crop: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray | None:
    """Bbox slice clamped to the crop; None when degenerate. Landmark-derived
    boxes can poke outside the frame and produce empty arrays otherwise."""
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(crop.shape[1], x + w), min(crop.shape[0], y + h)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return crop[y0:y1, x0:x1]


def _pose_crop(crop: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = bbox
    px0, py0 = max(0, x - int(1.6 * w)), max(0, y - int(0.6 * h))
    px1 = min(crop.shape[1], x + w + int(1.6 * w))
    py1 = min(crop.shape[0], y + h + int(3.2 * h))
    return crop[py0:py1, px0:px1]


def extract_session_embeddings(
    video: Path,
    hands_path: Path,
    seats_path: Path,
    out_path: Path,
    embedder=None,
    progress=None,
) -> pd.DataFrame:
    """One pooled-embedding row per mapped decision, appended with resume."""
    if embedder is None:
        embedder = DinoEmbedder()
    seats = load_seats(seats_path)
    hands = [json.loads(line) for line in Path(hands_path).open()]
    extractor = BehaviorExtractor(video, seats)  # builds gated reference chips
    cap = extractor.cap

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trim_partial_line(out_path)
    done = set()
    if out_path.exists() and out_path.stat().st_size > 0:
        prev = pd.read_csv(out_path, usecols=["hand_id", "player", "t_end"])
        done = {
            decision_key(r.hand_id, r.player, r.t_end)
            for r in prev.itertuples(index=False)
        }
    new_file = not out_path.exists() or out_path.stat().st_size == 0
    cols = embed_columns()
    f = out_path.open("a", newline="")
    writer = csv.writer(f)
    if new_file:
        writer.writerow(cols)
        f.flush()

    rows = []
    try:
        for hand in hands:
            for decision in hand["decisions"]:
                player = decision["player"]
                ref = seats.get(player)
                if ref is None:
                    continue
                key = decision_key(decision["hand_id"], player, decision["t_end"])
                if key in done:
                    continue
                t1 = decision["t_end"] + COMMIT_PAD_S
                t0 = max(0.0, decision["t_start"] - WINDOW_PRE_PAD_S)
                if t1 - t0 > MAX_WINDOW_S:
                    t0 = t1 - MAX_WINDOW_S
                face = FaceTracker()
                face_crops, pose_crops = [], []
                n = 0
                try:
                    cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000)
                    while True:
                        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                        ok, frame = cap.read()
                        if not ok or t > t1:
                            break
                        if n % EMBED_STRIDE == 0:
                            crop = extractor._search_crop(frame, ref)
                            faces = face.process(crop, int(t * 1000))
                            chips = [face_chip(crop, b) for _, b in faces]
                            m = match_face(chips, ref)
                            if m is not None:
                                bbox = faces[m[0]][1]
                                fc = _clamped(crop, *bbox)
                                pc = _pose_crop(crop, bbox)
                                if fc is not None and pc.size:
                                    face_crops.append(fc)
                                    pose_crops.append(pc)
                        n += 1
                finally:
                    face.close()
                row_vec = np.concatenate(
                    [pool_stream(embedder(face_crops)), pool_stream(embedder(pose_crops))]
                )
                row = [
                    decision["hand_id"], player, decision["t_start"],
                    decision["t_end"], len(face_crops), *np.round(row_vec, 5),
                ]
                writer.writerow(row)
                f.flush()
                rows.append(row)
                if progress is not None:
                    progress(decision["hand_id"], player, len(face_crops))
    finally:
        f.close()
        extractor.close()
    return pd.read_csv(out_path)
