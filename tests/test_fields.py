"""Field assignment on real detections captured in the day 1 OCR spike.

Both fixtures are actual PaddleOCR output on real HCL frames (box sizes
approximated; assignment keys off box centers and text):
- HEADSUP: session osFhAW7BFMs t=300, two players, one facing a bet.
- MULTIWAY: session loAuriiBRCk t=300, five stacked panels, straddle game,
  one player all-in (stack cell shows a glyph, not money), acting player's
  panel indented right by ~75 px.
"""

from pathlib import Path

from pokertell.gamestate.fields import assign_fields
from pokertell.gamestate.ocr import TextBox, parse_money, parse_stakes
from pokertell.gamestate.rois import load_layout

LAYOUT = load_layout(Path(__file__).parent.parent / "configs" / "hcl_rois.yaml")


def _box(text, x, y, score=0.99):
    return TextBox(text=text, score=score, x=x, y=y, w=10, h=10)


HEADSUP = [
    _box("TRADE ON", 21, 27),
    _box("HUSTLER", 1786, 35),
    _box("Polymarket", 445, 733),
    _box("94%", 400, 832),
    _box("JEFF THE CASH", 73, 835),
    _box("$14,950", 1676, 835),
    _box("9", 345, 836),
    _box("POT", 1585, 842),
    _box("BET $8,100", 87, 880),
    _box("$26,600", 272, 881),
    _box("SB", 410, 884),
    _box("2", 1625, 899),
    _box("JACKC", 176, 947),
    _box("$25/50 NL", 1630, 947),
    _box("6%", 478, 948),
    _box("$8,100", 354, 999),
    _box("CO", 482, 1002),
    _box("$8,100 TO CALL", 147, 1004),
]

MULTIWAY = [
    _box("WPT", 60, 45),
    _box("AIRBALL", 95, 480),
    _box("8", 285, 481),
    _box("21%", 400, 483),
    _box("CALL $100", 100, 535),
    _box("$168,125", 275, 537),
    _box("SB", 430, 538),
    _box("BIG MIKE", 95, 605),
    _box("9", 285, 606),
    _box("7%", 405, 608),
    _box("CALL $100", 100, 658),
    _box("$30,400", 278, 660),
    _box("BB", 430, 660),
    _box("RICH ASIAN BRO", 85, 728),
    _box("104", 300, 730),
    _box("27%", 405, 730),
    _box("ALL IN $10,500", 85, 780),
    _box("3B", 430, 782),
    _box("FRANCISCO", 150, 845),
    _box("$11,000", 1685, 848),
    _box("36%", 480, 848),
    _box("POT", 1590, 852),
    _box("$10,400 TO CALL", 155, 898),
    _box("$41,875", 355, 900),
    _box("LJ", 500, 900),
    _box("$25/50/100 NL", 1610, 955),
    _box("SUITED SUPERMAN", 78, 968),
    _box("9%", 445, 970),
    _box("CALL $100", 100, 1015),
    _box("$10,075", 300, 1017),
    _box("HJ", 450, 1018),
]


def test_headsup_pot_and_blinds():
    read = assign_fields(HEADSUP, LAYOUT)
    assert read.pot == 14950
    assert read.blinds == [25.0, 50.0]


def test_headsup_slot0_full_panel():
    panel = assign_fields(HEADSUP, LAYOUT).panels[0]
    assert panel.name == "JEFF THE CASH"
    assert panel.equity_pct == 94
    assert panel.action_text == "BET $8,100"
    assert panel.action_amount == 8100
    assert panel.stack == 26600
    assert panel.position == "SB"
    assert panel.to_call is None


def test_headsup_slot1_facing_bet():
    panel = assign_fields(HEADSUP, LAYOUT).panels[1]
    assert panel.name == "JACKC"
    assert panel.equity_pct == 6
    assert panel.to_call == 8100
    assert panel.stack == 8100
    assert panel.position == "CO"


def test_headsup_card_rank_noise_not_named():
    read = assign_fields(HEADSUP, LAYOUT)
    assert read.panels[0].name == "JEFF THE CASH"  # not overwritten by "9"


def test_headsup_branding_unassigned():
    read = assign_fields(HEADSUP, LAYOUT)
    leftovers = {b.text for b in read.unassigned}
    assert "Polymarket" in leftovers
    assert "TRADE ON" in leftovers
    assert "POT" in leftovers  # label text, not a value
    assert len(read.panels) == 2


def test_multiway_finds_all_five_panels():
    read = assign_fields(MULTIWAY, LAYOUT)
    names = [read.panels[i].name for i in sorted(read.panels)]
    assert names == ["AIRBALL", "BIG MIKE", "RICH ASIAN BRO", "FRANCISCO", "SUITED SUPERMAN"]


def test_multiway_straddle_blinds():
    assert assign_fields(MULTIWAY, LAYOUT).blinds == [25.0, 50.0, 100.0]


def test_multiway_positions_and_stacks():
    read = assign_fields(MULTIWAY, LAYOUT)
    assert read.panels[0].position == "SB"
    assert read.panels[0].stack == 168125
    assert read.panels[2].position == "3B"
    assert read.panels[2].stack is None  # all-in: stack cell shows a glyph


def test_multiway_all_in_action():
    panel = assign_fields(MULTIWAY, LAYOUT).panels[2]
    assert panel.action_text == "ALL IN $10,500"
    assert panel.action_amount == 10500


def test_multiway_to_act_is_indented_panel():
    read = assign_fields(MULTIWAY, LAYOUT)
    to_act = [p for p in read.panels.values() if p.to_call is not None]
    assert len(to_act) == 1
    assert to_act[0].name == "FRANCISCO"
    assert to_act[0].to_call == 10400


def test_parse_money():
    assert parse_money("$12,400") == 12400
    assert parse_money("1.2M") == 1_200_000
    assert parse_money("$83K") == 83_000
    assert parse_money("BET $8,100") == 8100
    assert parse_money("$25/50") is None
    assert parse_money("") is None


def test_parse_stakes():
    assert parse_stakes("$25/50 NL") == [25.0, 50.0]
    assert parse_stakes("$100/200/400") == [100.0, 200.0, 400.0]


def test_parse_money_repairs_separator_misreads():
    from pokertell.gamestate.ocr import parse_money

    assert parse_money("$12,400") == 12400.0
    assert parse_money("$51.575") == 51575.0
    assert parse_money("$1.750") == 1750.0
    assert parse_money("1.2M") == 1_200_000.0
    assert parse_money("$575") == 575.0


def test_repair_money_on_stored_values():
    from pokertell.gamestate.statemachine import repair_money

    assert repair_money(51.575) == 51575.0
    assert repair_money(1.75) == 1750.0
    assert repair_money(20750.0) == 20750.0
    assert repair_money(None) is None
