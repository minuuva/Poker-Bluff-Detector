"""Command line interface for the pokertell pipeline.

One subcommand per pipeline stage so any stage can be re-run in isolation:

    pokertell download URL          fetch a session video (stays local)
    pokertell calibrate VIDEO       sample frames and check the ROI layout
    pokertell extract-state VIDEO   OCR the HUD into hand histories
    pokertell extract-behavior      face/pose features per decision window
    pokertell label                 hand-strength classes from hole cards
    pokertell train                 fit the ablation pair
    pokertell report                metrics, bootstrap CIs, calibration
    pokertell demo                  burn the overlay demo clip
"""

from pathlib import Path

import typer

from pokertell.config import default_paths

app = typer.Typer(no_args_is_help=True, add_completion=False)


def _todo(stage: str, day: str) -> None:
    typer.echo(f"'{stage}' is scheduled for the {day} milestone and is not implemented yet.")
    raise typer.Exit(code=1)


@app.command()
def download(url: str, height: int = 1080) -> None:
    """Download one session video into data/raw (local only, never committed)."""
    from pokertell.ingest.download import download_session

    paths = default_paths().ensure()
    path = download_session(url, paths.raw, height=height)
    typer.echo(f"saved: {path}")


@app.command()
def calibrate(
    video: Path,
    layout: Path = Path("configs/hcl_rois.yaml"),
    t: float = typer.Option(600.0, help="timestamp (s) of the frame to sample"),
) -> None:
    """Save one frame plus every ROI crop to data/frames for visual inspection."""
    import cv2

    from pokertell.gamestate.rois import load_layout

    paths = default_paths().ensure()
    hud = load_layout(layout)
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        typer.echo(f"could not read a frame at t={t}s from {video}")
        raise typer.Exit(code=1)

    out = paths.frames / video.stem
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("roi_*.png"):
        stale.unlink()
    cv2.imwrite(str(out / "full_frame.png"), frame)
    for name, crop in hud.crop_all(frame).items():
        cv2.imwrite(str(out / f"roi_{name}.png"), crop)

    annotated = frame.copy()
    for name, roi in hud.rois.items():
        cv2.rectangle(
            annotated, (roi.x, roi.y), (roi.x + roi.w, roi.y + roi.h), (0, 255, 0), 2
        )
        cv2.putText(
            annotated,
            name,
            (roi.x, max(15, roi.y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
    cv2.imwrite(str(out / "annotated.png"), annotated)
    typer.echo(f"wrote full frame, annotated overlay, and {len(hud.rois)} ROI crops to {out}")


@app.command("ocr-frame")
def ocr_frame(
    video: Path,
    layout: Path = Path("configs/hcl_rois.yaml"),
    t: float = typer.Option(300.0, help="timestamp (s) of the frame to read"),
) -> None:
    """Debug: OCR one frame and print the assigned HUD fields."""
    import cv2

    from pokertell.gamestate.fields import assign_fields
    from pokertell.gamestate.ocr import HudReader
    from pokertell.gamestate.rois import load_layout

    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        typer.echo(f"could not read a frame at t={t}s from {video}")
        raise typer.Exit(code=1)

    hud = load_layout(layout)
    read = assign_fields(HudReader(hud).read_frame(frame), hud)
    typer.echo(f"pot: {read.pot}  blinds: {read.blinds}")
    for slot in sorted(read.panels):
        p = read.panels[slot]
        typer.echo(
            f"slot{slot}: name={p.name!r} equity={p.equity_pct} stack={p.stack} "
            f"pos={p.position} action={p.action_text!r} to_call={p.to_call}"
        )
    typer.echo(f"unassigned: {[b.text for b in read.unassigned]}")


@app.command("extract-state")
def extract_state(video: Path) -> None:
    """OCR the HUD stream into hand histories (day 2-3)."""
    _todo("extract-state", "day 2-3")


@app.command("extract-behavior")
def extract_behavior() -> None:
    """Extract face/pose features over decision windows (day 4)."""
    _todo("extract-behavior", "day 4")


@app.command()
def label() -> None:
    """Compute equity and strength-class labels from exposed hole cards (day 5)."""
    _todo("label", "day 5")


@app.command()
def train() -> None:
    """Fit the betting-only baseline and the behavior-augmented model (day 5)."""
    _todo("train", "day 5")


@app.command()
def report() -> None:
    """Produce the ablation report with bootstrap CIs and calibration (day 6)."""
    _todo("report", "day 6")


@app.command()
def demo() -> None:
    """Render the overlay demo clip (day 7)."""
    _todo("demo", "day 7")


if __name__ == "__main__":
    app()
