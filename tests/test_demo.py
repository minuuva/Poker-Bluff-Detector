"""Pure helpers of the demo overlay renderer."""

from pokertell.demo.overlay import fmt_cards, phase_label, state_lines

DECISION = {
    "t_start": 100.0,
    "t_end": 108.0,
    "player": "AIRBALL",
    "position": "CO",
    "street": "preflop",
    "action": "raise",
    "amount": 1500.0,
    "pot_before": 1200.0,
}
HAND = {"hand_id": "s#0001"}


def test_fmt_cards():
    assert fmt_cards(["8h", "4s"]) == "8h 4s"
    assert fmt_cards(None) == "??"


def test_phase_label_transitions():
    assert phase_label(99.0, DECISION) == "waiting"
    assert phase_label(104.0, DECISION) == "to act"
    assert phase_label(109.0, DECISION) == "RAISE $1,500"


def test_state_lines_window_timer_only_inside_window():
    inside = state_lines(DECISION, HAND, 103.0)
    after = state_lines(DECISION, HAND, 110.0)
    assert any("decision window" in line for line in inside)
    assert not any("decision window" in line for line in after)
    assert "pot $1,200" in inside[1]
