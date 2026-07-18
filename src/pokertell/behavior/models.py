"""MediaPipe task model files: local cache with on-demand download.

Model blobs stay out of git (data/ is ignored); they download once from
Google's model storage into data/models/.
"""

import ssl
import urllib.request
from pathlib import Path

import certifi

from pokertell.config import default_paths

_BASE = "https://storage.googleapis.com/mediapipe-models"
MODEL_URLS = {
    "face_landmarker.task": (
        f"{_BASE}/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
    ),
    "pose_landmarker_lite.task": (
        f"{_BASE}/pose_landmarker/pose_landmarker_lite/float16/latest/"
        "pose_landmarker_lite.task"
    ),
}


def ensure_model(name: str) -> Path:
    """Return the local path of a model file, downloading it if missing."""
    if name not in MODEL_URLS:
        raise KeyError(f"unknown model {name}; known: {sorted(MODEL_URLS)}")
    target = default_paths().root / "models" / name
    if target.exists() and target.stat().st_size > 1_000_000:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".download")
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(MODEL_URLS[name], context=context) as resp:
        tmp.write_bytes(resp.read())
    tmp.rename(target)
    return target
