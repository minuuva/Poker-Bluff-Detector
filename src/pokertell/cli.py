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
def extract_state(
    video: Path,
    layout: Path = Path("configs/hcl_rois.yaml"),
    templates: Path = Path("configs/templates/hcl"),
    interval: float = typer.Option(1.0, help="seconds between sampled frames"),
    t_start: float = 0.0,
    t_end: float = typer.Option(None, help="stop timestamp (s); default = full video"),
) -> None:
    """OCR the HUD stream into a HudSnapshot JSONL under data/hands."""
    from pokertell.gamestate.extract import SnapshotExtractor, write_snapshots
    from pokertell.gamestate.rois import load_layout

    paths = default_paths().ensure()
    extractor = SnapshotExtractor(
        load_layout(layout), templates, unmatched_dir=paths.frames / "unmatched" / video.stem
    )
    snapshots = []
    stats = None
    for snap, stats in extractor.run(video, interval=interval, t_start=t_start, t_end=t_end):
        snapshots.append(snap)
        if stats.processed % 50 == 0:
            typer.echo(
                f"t={snap.t:8.1f}s sampled={stats.sampled} gated={stats.gated} "
                f"processed={stats.processed}"
            )
    out = paths.hands / f"{video.stem}.snapshots.jsonl"
    write_snapshots(snapshots, out)
    if stats is None:
        typer.echo("no frames processed; check the video path and time range")
        raise typer.Exit(code=1)
    typer.echo(
        f"done: {stats.sampled} sampled, {stats.gated} gated out, "
        f"{stats.processed} OCRed, {stats.unmatched_cards} unmatched card cells"
    )
    typer.echo(f"wrote {len(snapshots)} snapshots to {out}")


@app.command()
def assemble(snapshots_file: Path) -> None:
    """Fold a snapshot JSONL into Hand records with decisions."""
    import dataclasses
    import json

    from pokertell.gamestate.extract import read_snapshots
    from pokertell.gamestate.statemachine import assemble_session

    paths = default_paths().ensure()
    session_id = snapshots_file.stem.replace(".snapshots", "")
    snaps = read_snapshots(snapshots_file)
    hands, report = assemble_session(snaps, session_id)

    out = paths.hands / f"{session_id}.hands.jsonl"
    with out.open("w") as f:
        for hand in hands:
            f.write(json.dumps(dataclasses.asdict(hand)) + "\n")

    for key, value in report.items():
        typer.echo(f"{key}: {value}")
    typer.echo(f"wrote {len(hands)} hands to {out}")


@app.command("extract-behavior")
def extract_behavior(
    video: Path,
    hands_file: Path,
    seats: Path,
) -> None:
    """Extract face/pose features over decision windows into data/features."""
    from pokertell.behavior.extract import extract_session_behavior

    paths = default_paths().ensure()
    session_id = hands_file.stem.replace(".hands", "")

    def progress(row: dict) -> None:
        typer.echo(
            f"{row['hand_id']} {row['player'][:14]:14s} window={row['window_s']:6.1f}s "
            f"face_cov={row.get('face_coverage', 0):.2f} pose_cov={row.get('pose_coverage', 0):.2f}"
        )

    df = extract_session_behavior(video, hands_file, seats, progress=progress)
    out = paths.features / f"{session_id}.behavior.csv"
    df.to_csv(out, index=False)
    typer.echo(f"wrote {len(df)} rows to {out}")


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
