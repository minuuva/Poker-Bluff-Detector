"""Checkpoint and resume behavior for the long-running extraction stages."""

import json

import pytest

from pokertell.behavior.extract import (
    BEHAVIOR_COLUMNS,
    decision_key,
    load_done_keys,
)
from pokertell.behavior.face import summarize_blendshapes
from pokertell.behavior.pose import summarize_pose
from pokertell.checkpoint import (
    clear_progress,
    load_progress,
    save_progress,
    trim_partial_line,
)
from pokertell.gamestate.extract import (
    last_snapshot_t,
    read_snapshots,
    write_snapshots,
)
from pokertell.gamestate.statemachine import HudSnapshot


def test_progress_round_trip(tmp_path):
    out = tmp_path / "x.snapshots.jsonl"
    assert load_progress(out) is None
    save_progress(out, 123.4)
    assert load_progress(out) == 123.4
    save_progress(out, 200.0)
    assert load_progress(out) == 200.0
    clear_progress(out)
    assert load_progress(out) is None


def test_load_progress_ignores_corrupt_sidecar(tmp_path):
    out = tmp_path / "x.jsonl"
    save_progress(out, 10.0)
    (tmp_path / "x.jsonl.progress").write_text("{half")
    assert load_progress(out) is None


def test_trim_partial_line(tmp_path):
    p = tmp_path / "f.jsonl"
    p.write_text('{"t": 1.0}\n{"t": 2.0}\n{"t": 3.')
    trim_partial_line(p)
    assert p.read_text() == '{"t": 1.0}\n{"t": 2.0}\n'
    # Idempotent on a clean file.
    trim_partial_line(p)
    assert p.read_text() == '{"t": 1.0}\n{"t": 2.0}\n'


def test_trim_partial_line_edge_cases(tmp_path):
    trim_partial_line(tmp_path / "missing.jsonl")
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    trim_partial_line(empty)
    assert empty.read_text() == ""
    # A file that is one giant partial line truncates to nothing.
    only_partial = tmp_path / "partial.jsonl"
    only_partial.write_text('{"t": 1')
    trim_partial_line(only_partial)
    assert only_partial.read_text() == ""


def test_read_snapshots_tolerates_truncated_tail(tmp_path):
    p = tmp_path / "s.jsonl"
    write_snapshots([HudSnapshot(t=1.0), HudSnapshot(t=2.0)], p)
    with p.open("a") as f:
        f.write('{"t": 3.0, "pot"')
    snaps = read_snapshots(p)
    assert [s.t for s in snaps] == [1.0, 2.0]


def test_read_snapshots_raises_on_mid_file_corruption(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text('{"t": 1.0, broken\n' + json.dumps({"t": 2.0}) + "\n")
    with pytest.raises(json.JSONDecodeError):
        read_snapshots(p)


def test_last_snapshot_t(tmp_path):
    p = tmp_path / "s.jsonl"
    write_snapshots([HudSnapshot(t=5.0), HudSnapshot(t=17.0)], p)
    assert last_snapshot_t(p) == 17.0


def test_behavior_columns_match_feature_outputs():
    """The fixed CSV schema must equal the keys extract_decision emits."""
    from pokertell.behavior.events import compute_event_features

    face = summarize_blendshapes([None] * 5, 15.0)
    pose = summarize_pose([None] * 5, 30.0, 10)
    events = compute_event_features([None] * 5, [None] * 5, [None] * 5, 30.0, 2)
    row_keys = [
        "hand_id", "player", "t_start", "t_end", "window_s", "n_frames",
        "shot_coverage", *face.keys(), *pose.keys(), *events.keys(),
    ]
    assert row_keys == BEHAVIOR_COLUMNS


def test_load_done_keys_resume(tmp_path):
    out = tmp_path / "b.behavior.csv"
    assert load_done_keys(out) == set()

    header = ",".join(BEHAVIOR_COLUMNS)
    row = {c: "" for c in BEHAVIOR_COLUMNS}
    row.update(hand_id="s#0001", player="AIRBALL", t_start="10.0", t_end="25.5")
    line = ",".join(row[c] for c in BEHAVIOR_COLUMNS)
    out.write_text(header + "\n" + line + "\n" + line[: len(line) // 2])

    trim_partial_line(out)
    assert load_done_keys(out) == {decision_key("s#0001", "AIRBALL", 25.5)}


def test_load_done_keys_rejects_foreign_schema(tmp_path):
    out = tmp_path / "old.behavior.csv"
    out.write_text("hand_id,player,t_end,some_v1_feature\ns#0001,AIRBALL,25.5,0.1\n")
    with pytest.raises(ValueError, match="schema"):
        load_done_keys(out)
