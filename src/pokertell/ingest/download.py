"""Download session videos with yt-dlp.

Data policy (see README, Ethics and data): footage stays local, is never
committed, and is never redistributed in any form. Only streams where players
consented to hole-card broadcast (e.g. Hustler Casino Live) are used, and all
analysis is post hoc.
"""

import subprocess
from pathlib import Path


def download_session(url: str, out_dir: Path, height: int = 1080) -> Path:
    """Download one session video at a fixed resolution.

    A fixed height matters: ROI layouts are calibrated in pixel coordinates,
    so every session must decode at the same frame size.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    template = str(out_dir / "%(id)s.%(ext)s")
    cmd = [
        "yt-dlp",
        "--format",
        f"bestvideo[height={height}]+bestaudio/best[height={height}]",
        "--merge-output-format",
        "mp4",
        "--output",
        template,
        "--print",
        "after_move:filepath",
        "--no-simulate",
        url,
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    filepath = result.stdout.strip().splitlines()[-1]
    return Path(filepath)
