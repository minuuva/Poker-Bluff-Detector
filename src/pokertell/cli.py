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
    """Build labeled decision tables from every hands file in data/hands."""
    import json

    from pokertell.labels.build import build_decision_table

    paths = default_paths().ensure()
    hands_files = sorted(paths.hands.glob("*.hands.jsonl"))
    if not hands_files:
        typer.echo("no hands files found; run extract-state and assemble first")
        raise typer.Exit(code=1)
    for hands_file in hands_files:
        session_id = hands_file.stem.replace(".hands", "")
        hands = [json.loads(line) for line in hands_file.open()]
        df = build_decision_table(hands)
        out = paths.features / f"{session_id}.decisions.csv"
        df.to_csv(out, index=False)
        labeled = int(df["equity_mc"].notna().sum())
        typer.echo(f"{session_id}: {len(df)} decisions, {labeled} with equity labels -> {out}")


@app.command()
def train(
    target: str = typer.Option("is_bluff", help="is_bluff (aggressive only) or is_weak"),
    model: str = typer.Option("logreg", help="logreg or xgb"),
    min_coverage: float = typer.Option(0.5, help="min face/pose coverage for ablation rows"),
    min_ablation_rows: int = typer.Option(30, help="min behavior rows to attempt ablation"),
) -> None:
    """Fit the betting-only baseline; run the ablation where behavior exists."""
    import pandas as pd

    from pokertell.behavior.face import FACE_FEATURES
    from pokertell.behavior.features import zscore_per_player
    from pokertell.behavior.pose import POSE_FEATURES
    from pokertell.models.baseline import BETTING_FEATURES
    from pokertell.models.train import loso_ablation, loso_baseline

    paths = default_paths().ensure()
    dec_files = sorted(paths.features.glob("*.decisions.csv"))
    if not dec_files:
        typer.echo("no decision tables found; run label first")
        raise typer.Exit(code=1)
    df = pd.concat([pd.read_csv(f) for f in dec_files], ignore_index=True)

    beh_files = sorted(paths.features.glob("*.behavior.csv"))
    behavior_cols = FACE_FEATURES + POSE_FEATURES
    if beh_files:
        beh = pd.concat([pd.read_csv(f) for f in beh_files], ignore_index=True)
        df = df.merge(
            beh.drop(columns=["t_start", "window_s", "n_frames"], errors="ignore"),
            on=["hand_id", "player", "t_end"],
            how="left",
        )
        df = zscore_per_player(df, [c for c in behavior_cols if c in df.columns])

    labeled = df[df[target].notna()].copy()
    typer.echo(
        f"decisions: {len(df)} total, {len(labeled)} labeled for {target} "
        f"(base rate {labeled[target].mean():.2f}) across "
        f"{labeled['session_id'].nunique()} sessions"
    )

    typer.echo("\n== betting-only baseline (LOSO) ==")
    base = loso_baseline(labeled, target, BETTING_FEATURES, model_kind=model)
    for k, v in base.items():
        typer.echo(f"  {k}: {v}")

    coverage = df.get("face_coverage")
    covered = (
        labeled[
            (labeled.get("face_coverage", 0).fillna(0) >= min_coverage)
            | (labeled.get("pose_coverage", 0).fillna(0) >= min_coverage)
        ]
        if coverage is not None
        else labeled.iloc[0:0]
    )
    typer.echo(
        f"\nbehavior-covered labeled decisions: {len(covered)} "
        f"(need {min_ablation_rows} for the ablation)"
    )
    if len(covered) >= min_ablation_rows and covered[target].nunique() == 2:
        typer.echo("\n== ablation: baseline vs baseline+behavior (LOSO) ==")
        cols = [c for c in behavior_cols if c in covered.columns]
        result = loso_ablation(covered, target, BETTING_FEATURES, cols, model_kind=model)
        for k, v in result.items():
            typer.echo(f"  {k}: {v}")
    else:
        typer.echo(
            "ablation skipped: not enough behavior-covered rows yet. The honest "
            "path is more footage and more seat maps, not a smaller bar."
        )


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
