"""Crash-safe incremental output for the long-running extraction stages.

The heavy stages (extract-state, extract-behavior) used to buffer all
results in memory and write once at the end, so a kill or a reclaimed
spot instance lost hours of work. These helpers give both stages the
same contract: append one record at a time, flush as you go, record how
far sampling got in a sidecar file, and on restart trim any half-written
trailing line and continue from the recorded position.

The sidecar lives next to the output as <output>.progress and holds one
JSON object. It is written atomically (temp file + os.replace) so a kill
can never leave a corrupt sidecar, only a slightly stale one, which at
worst re-does a few seconds of work.
"""

import json
import os
from pathlib import Path


def progress_path(out_path: Path) -> Path:
    return Path(str(out_path) + ".progress")


def save_progress(out_path: Path, next_t: float) -> None:
    """Record that sampling is complete up to (but not including) next_t."""
    target = progress_path(out_path)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps({"next_t": round(next_t, 3)}))
    os.replace(tmp, target)


def load_progress(out_path: Path) -> float | None:
    """The recorded resume timestamp for out_path, or None if absent."""
    p = progress_path(out_path)
    if not p.exists():
        return None
    try:
        return float(json.loads(p.read_text())["next_t"])
    except (ValueError, KeyError, json.JSONDecodeError):
        return None


def clear_progress(out_path: Path) -> None:
    progress_path(out_path).unlink(missing_ok=True)


def trim_partial_line(path: Path) -> None:
    """Drop a truncated trailing line left by a kill mid-write.

    Records are newline-terminated and never contain embedded newlines,
    so a file whose last byte is not a newline ends in a partial record.
    Everything up to the last newline is kept.
    """
    path = Path(path)
    if not path.exists():
        return
    size = path.stat().st_size
    if size == 0:
        return
    with path.open("rb+") as f:
        f.seek(-1, os.SEEK_END)
        if f.read(1) == b"\n":
            return
        # Walk back in blocks to find the last newline.
        block = 4096
        pos = size
        keep = 0
        while pos > 0:
            step = min(block, pos)
            pos -= step
            f.seek(pos)
            chunk = f.read(step)
            idx = chunk.rfind(b"\n")
            if idx != -1:
                keep = pos + idx + 1
                break
        f.truncate(keep)
