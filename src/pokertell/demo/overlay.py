"""Burn a prediction overlay onto a short clip with ffmpeg.

The demo artifact: a 60 to 90 second clip with a hand-strength probability
panel (the six classes with bars), the live behavioral feature readouts, and
the decision-window timer, in the style of the broadcast graphic.

Data policy note: demo clips are for the portfolio writeup and are not
committed to the repo; the pipeline outputs stay local (see README, Ethics
and data).
"""

from pathlib import Path


def render_overlay(
    video_in: Path,
    predictions_json: Path,
    video_out: Path,
    t_start: float,
    t_end: float,
) -> None:
    """Render the demo clip.

    TODO(day 7): implement. Plan: draw the panel per frame with OpenCV or
    Pillow onto transparent PNGs, then composite with:
        ffmpeg -i clip.mp4 -framerate FPS -i overlay/%05d.png
               -filter_complex overlay -c:a copy out.mp4
    Pure drawtext gets unreadable fast; prerendered PNGs keep layout control.
    """
    raise NotImplementedError("render_overlay lands in the day 7 milestone")
