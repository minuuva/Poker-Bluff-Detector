"""Fold the HudSnapshot stream into Hand records with validated decisions.

This is the least glamorous and most load-bearing module in the project.
Input is the per-second snapshot JSONL from extract-state; output is one
Hand record per dealt hand, each carrying its action events and Decision
windows (the unit of analysis for the whole study).

How the HUD's rendering rules become structure:

- Panels exist only for players still in the hand, so a player's panel
  disappearing mid-hand is a fold, and stretches with no panels at all
  separate hands.
- Each panel's action cell shows the player's LAST action and persists, so
  action events are transitions of that text over time.
- '$X TO CALL' renders only on the player facing a decision, so a run of
  snapshots with to_act == P is P's decision window: action-on-player at
  the run's start, action-committed at the snapshot where P's action text
  changes. This is the TimeToAct window behavioral features extract from.
- Card sprites persist for the whole hand, so hole cards and board are
  majority votes across the hand's snapshots, which absorbs the partial
  reads that dealing animations cause.

Every reconstructed session must be validated against a hand-transcribed
sample before its data feeds the model.
"""

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum

from pokertell.gamestate.names import NameResolver


class Street(str, Enum):
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"


class ActionType(str, Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"
    ALL_IN = "all_in"


AGGRESSIVE_ACTIONS = {ActionType.BET, ActionType.RAISE, ActionType.ALL_IN}

STREET_BY_BOARD_LEN = {0: Street.PREFLOP, 3: Street.FLOP, 4: Street.TURN, 5: Street.RIVER}

# Segmentation thresholds
DEAD_GAP = 3            # consecutive panel-less snapshots that separate hands
POT_DROP_RATIO = 0.7    # pot falling below this fraction of the span max = new hand
MIN_SPAN_SNAPSHOTS = 3
MIN_PLAYER_SNAPSHOTS = 2
WINDOW_COMMIT_SLACK_S = 4.0


def street_from_board(board: list[str]) -> Street:
    """Map board card count to street. Raises on impossible counts."""
    n = len(board)
    if n not in STREET_BY_BOARD_LEN:
        raise ValueError(f"impossible board size {n}: {board}")
    return STREET_BY_BOARD_LEN[n]


@dataclass
class HudSnapshot:
    """Everything the HUD said at one sampled frame time t (seconds).

    bets holds the amount shown in each player's action cell (their last
    action's size as displayed, e.g. from 'BET $8,100' or 'CALL $100');
    actions holds the raw action text; equities the broadcast's own win
    percentages (a later cross-check for our card reads and labels).
    """

    t: float
    pot: float | None = None
    board: list[str] = field(default_factory=list)
    stacks: dict[str, float] = field(default_factory=dict)
    bets: dict[str, float] = field(default_factory=dict)
    hole_cards: dict[str, list[str]] = field(default_factory=dict)
    to_act: str | None = None
    to_call: float | None = None
    positions: dict[str, str] = field(default_factory=dict)
    actions: dict[str, str] = field(default_factory=dict)
    equities: dict[str, float] = field(default_factory=dict)
    blinds: list[float] = field(default_factory=list)

    @property
    def live(self) -> bool:
        return bool(self.positions or self.stacks or self.actions)

    def players(self) -> set[str]:
        return (
            set(self.positions) | set(self.stacks) | set(self.actions) | set(self.hole_cards)
        )


@dataclass
class Action:
    player: str
    action: ActionType
    amount: float | None
    street: Street
    t: float


@dataclass
class Decision:
    """One player decision: one row in the final feature table.

    The window [t_start, t_end] is where behavioral features get extracted.
    window_source is 'to_act' when the HUD's facing marker delimited the
    window, or 'action_only' when only the committing action was observed
    (window start then falls back to the previous event and is less exact).
    """

    hand_id: str
    player: str
    street: Street
    action: ActionType
    amount: float | None
    t_start: float
    t_end: float
    window_source: str
    pot_before: float | None
    to_call: float | None
    stack_before: float | None
    position: str | None
    equity_pct: float | None
    board: list[str]
    hole_cards: list[str] | None


@dataclass
class Hand:
    hand_id: str
    session_id: str
    t_start: float
    t_end: float
    players: list[str]
    display_names: dict[str, str]
    positions: dict[str, str]
    hole_cards: dict[str, list[str]]
    board: list[str]
    blinds: list[float]
    actions: list[Action]
    decisions: list[Decision]
    n_snapshots: int
    flags: list[str] = field(default_factory=list)


def parse_action_text(text: str) -> tuple[ActionType, float | None] | None:
    """Parse a HUD action cell into (type, amount). None for non-actions
    (the '$X TO CALL' facing marker, unrecognized text)."""
    from pokertell.gamestate.ocr import parse_money

    t = text.upper().strip()
    if "TO CALL" in t:
        return None
    for prefix, kind in (
        ("ALL", ActionType.ALL_IN),
        ("BET", ActionType.BET),
        ("RAISE", ActionType.RAISE),
        ("CALL", ActionType.CALL),
        ("CHECK", ActionType.CHECK),
        ("FOLD", ActionType.FOLD),
    ):
        if t.startswith(prefix):
            return kind, parse_money(t)
    return None


def normalize_snapshots(
    snaps: list[HudSnapshot], resolver: NameResolver
) -> list[HudSnapshot]:
    """Rewrite all player-keyed fields onto canonical identities."""
    out = []
    for s in snaps:
        out.append(
            HudSnapshot(
                t=s.t,
                pot=s.pot,
                board=list(s.board),
                stacks={resolver.resolve(k): v for k, v in s.stacks.items()},
                bets={resolver.resolve(k): v for k, v in s.bets.items()},
                hole_cards={resolver.resolve(k): v for k, v in s.hole_cards.items()},
                to_act=resolver.resolve(s.to_act) if s.to_act else None,
                to_call=s.to_call,
                positions={resolver.resolve(k): v for k, v in s.positions.items()},
                actions={resolver.resolve(k): v for k, v in s.actions.items()},
                equities={resolver.resolve(k): v for k, v in s.equities.items()},
                blinds=list(s.blinds),
            )
        )
    return out


def segment_hands(snaps: list[HudSnapshot]) -> list[list[HudSnapshot]]:
    """Split the normalized snapshot stream into per-hand spans."""
    spans: list[list[HudSnapshot]] = []
    span: list[HudSnapshot] = []
    span_positions: dict[str, str] = {}
    span_max_pot = 0.0
    dead = 0

    def close() -> None:
        nonlocal span, span_positions, span_max_pot
        if len(span) >= MIN_SPAN_SNAPSHOTS:
            spans.append(span)
        span = []
        span_positions = {}
        span_max_pot = 0.0

    for s in snaps:
        if not s.live:
            dead += 1
            if dead >= DEAD_GAP:
                close()
            continue
        dead = 0

        boundary = False
        if span:
            prev = span[-1]
            if len(prev.board) >= 3 and len(s.board) == 0:
                boundary = True
            for p, pos in s.positions.items():
                if span_positions.get(p, pos) != pos:
                    boundary = True
            if (
                s.pot is not None
                and span_max_pot > 0
                and s.pot < span_max_pot * POT_DROP_RATIO
                and s.pot < span_max_pot - 100
            ):
                boundary = True
        if boundary:
            close()
        span.append(s)
        span_positions.update(s.positions)
        if s.pot:
            span_max_pot = max(span_max_pot, s.pot)
    close()
    return spans


MERGE_MAX_GAP_S = 120.0


def _span_signature(span: list[HudSnapshot]) -> tuple[dict[str, str], dict[str, tuple]]:
    positions: dict[str, Counter] = {}
    cards: dict[str, Counter] = {}
    for s in span:
        for p, pos in s.positions.items():
            positions.setdefault(p, Counter())[pos] += 1
        for p, c in s.hole_cards.items():
            cards.setdefault(p, Counter())[tuple(c)] += 1
    return (
        {p: c.most_common(1)[0][0] for p, c in positions.items()},
        {p: c.most_common(1)[0][0] for p, c in cards.items()},
    )


def merge_spans(spans: list[list[HudSnapshot]]) -> list[list[HudSnapshot]]:
    """Re-join spans that belong to the same hand.

    Camera cutaways hide the HUD for many seconds, which the dead-gap
    separator reads as a hand boundary. But positions rotate and hole cards
    change every hand, so two adjacent spans whose common players hold the
    same positions AND the same hole cards are the same hand.
    """
    if not spans:
        return spans
    merged = [spans[0]]
    for span in spans[1:]:
        prev = merged[-1]
        gap = span[0].t - prev[-1].t
        prev_pos, prev_cards = _span_signature(prev)
        cur_pos, cur_cards = _span_signature(span)
        common = set(prev_pos) & set(cur_pos)
        common_cards = set(prev_cards) & set(cur_cards)
        same_positions = bool(common) and all(prev_pos[p] == cur_pos[p] for p in common)
        same_cards = bool(common_cards) and all(
            prev_cards[p] == cur_cards[p] for p in common_cards
        )
        if gap <= MERGE_MAX_GAP_S and same_positions and (same_cards or not common_cards):
            merged[-1] = prev + span
        else:
            merged.append(span)
    return merged


def _vote(values: list) -> object | None:
    if not values:
        return None
    return Counter(values).most_common(1)[0][0]


def _vote_board(span: list[HudSnapshot]) -> tuple[list[str], list[str]]:
    """Majority-vote the board. Returns (board, flags).

    Dealing animations produce brief misreads at each street's start, so
    the vote anchors on the reading with the most total support (a stable
    turn seen 76 times beats a garbled flop seen 3 times) and extends to
    longer streets only when their prefix agrees.
    """
    flags = []
    by_len: dict[int, Counter] = {}
    for s in span:
        if len(s.board) in (3, 4, 5) and len(set(s.board)) == len(s.board):
            by_len.setdefault(len(s.board), Counter())[tuple(s.board)] += 1
    # A reading seen once is animation junk unless it is all the evidence
    # that street has.
    for n, counter in by_len.items():
        confirmed = Counter({k: v for k, v in counter.items() if v >= 2})
        if confirmed:
            by_len[n] = confirmed
    by_len = {n: c for n, c in by_len.items() if c}
    if not by_len:
        return [], flags

    winners = {n: c.most_common(1)[0] for n, c in by_len.items()}
    anchor_len = max(winners, key=lambda n: winners[n][1])
    board = list(winners[anchor_len][0])
    for n in sorted(winners):
        if n <= anchor_len:
            continue
        cand, count = winners[n]
        if list(cand[: len(board)]) == board and count >= 2:
            board = list(cand)
        else:
            flags.append(f"board_prefix_conflict_len{n}")
    # Shorter-street conflicts with the anchor are worth flagging too.
    for n in sorted(winners):
        if n < anchor_len and list(winners[n][0]) != board[:n]:
            flags.append(f"board_prefix_conflict_len{n}")
    return board, flags


def _street_at(span: list[HudSnapshot], t: float) -> tuple[Street, list[str]]:
    """Street and stable board prefix in effect at time t."""
    n = 0
    board: list[str] = []
    for s in span:
        if s.t > t:
            break
        if len(s.board) in (3, 4, 5) and len(s.board) > n:
            n = len(s.board)
            board = list(s.board)
    return STREET_BY_BOARD_LEN[n], board


def _last_before(span: list[HudSnapshot], t: float, get) -> object | None:
    value = None
    for s in span:
        if s.t > t:
            break
        v = get(s)
        if v is not None:
            value = v
    return value


def assemble_hand(span: list[HudSnapshot], session_id: str, index: int) -> Hand:
    hand_id = f"{session_id}#{index:04d}"
    flags: list[str] = []

    counts = Counter(p for s in span for p in s.players())
    min_seen = MIN_PLAYER_SNAPSHOTS if len(span) >= 4 else 1
    players = sorted(p for p, n in counts.items() if n >= min_seen)
    if not players:
        flags.append("no_players")

    positions = {
        p: _vote([s.positions[p] for s in span if p in s.positions]) for p in players
    }
    positions = {p: v for p, v in positions.items() if v is not None}
    hole_cards = {}
    for p in players:
        votes = [tuple(s.hole_cards[p]) for s in span if p in s.hole_cards]
        if votes:
            hole_cards[p] = list(_vote(votes))
    board, board_flags = _vote_board(span)
    flags.extend(board_flags)
    # Single-level blind readings are misreads (aftermath graphics bleed
    # into the stakes ROI); real stakes always have at least two levels.
    blinds = _vote([tuple(s.blinds) for s in span if len(s.blinds) >= 2]) or ()

    # Action events: transitions of each player's action text.
    actions: list[Action] = []
    last_text: dict[str, str] = {}
    for s in span:
        for p, text in s.actions.items():
            if p not in players or last_text.get(p) == text:
                continue
            last_text[p] = text
            parsed = parse_action_text(text)
            if parsed is None:
                continue
            kind, amount = parsed
            street, _ = _street_at(span, s.t)
            actions.append(Action(player=p, action=kind, amount=amount, street=street, t=s.t))

    # Folds: a player's panel disappears while the hand continues.
    seen_players = set()
    absent_since: dict[str, float] = {}
    folded: set[str] = set()
    explicit = {a.player for a in actions if a.action == ActionType.FOLD}
    for i, s in enumerate(span):
        present = s.players()
        seen_players |= present
        for p in list(absent_since):
            if p in present:
                del absent_since[p]
        for p in seen_players - present - folded - set(absent_since):
            absent_since[p] = s.t
        remaining = i + 1 < len(span) - 1
        for p, t_gone in list(absent_since.items()):
            if p in explicit or p not in players:
                continue
            gone_for = sum(1 for x in span if x.t >= t_gone and p not in x.players())
            if gone_for >= 2 and remaining and len(present) >= 1:
                street, _ = _street_at(span, t_gone)
                actions.append(
                    Action(player=p, action=ActionType.FOLD, amount=None, street=street, t=t_gone)
                )
                folded.add(p)
                del absent_since[p]
    actions.sort(key=lambda a: a.t)

    # Decision windows from to_act runs.
    decisions: list[Decision] = []
    used_actions: set[int] = set()
    runs: list[tuple[str, float, float, HudSnapshot]] = []
    for s in span:
        if s.to_act is None or s.to_act not in players:
            continue
        if runs and runs[-1][0] == s.to_act and s.t - runs[-1][2] <= WINDOW_COMMIT_SLACK_S:
            runs[-1] = (runs[-1][0], runs[-1][1], s.t, runs[-1][3])
        else:
            runs.append((s.to_act, s.t, s.t, s))

    def make_decision(p, t_start, t_end, source, kind, amount, first_snap):
        street, street_board = _street_at(span, t_start)
        return Decision(
            hand_id=hand_id,
            player=p,
            street=street,
            action=kind,
            amount=amount,
            t_start=t_start,
            t_end=t_end,
            window_source=source,
            pot_before=_last_before(span, t_start, lambda s: s.pot),
            to_call=first_snap.to_call if first_snap is not None else None,
            stack_before=_last_before(span, t_start, lambda s, p=p: s.stacks.get(p)),
            position=positions.get(p),
            equity_pct=_last_before(span, t_start, lambda s, p=p: s.equities.get(p)),
            board=street_board,
            hole_cards=hole_cards.get(p),
        )

    for p, t_first, t_last, first_snap in runs:
        commit = next(
            (
                (i, a)
                for i, a in enumerate(actions)
                if i not in used_actions
                and a.player == p
                and t_last - 0.5 <= a.t <= t_last + WINDOW_COMMIT_SLACK_S
            ),
            None,
        )
        if commit is None:
            flags.append(f"window_without_commit:{p}@{t_first}")
            continue
        i, a = commit
        used_actions.add(i)
        decisions.append(
            make_decision(p, t_first, a.t, "to_act", a.action, a.amount, first_snap)
        )

    # Actions that never had a facing marker (checks, quick actions between
    # samples): keep them as less exact action_only windows.
    prev_event_t = span[0].t
    for i, a in enumerate(sorted(actions, key=lambda a: a.t)):
        if i in used_actions or a.action == ActionType.FOLD and a.amount is None:
            prev_event_t = a.t
            continue
        if not any(d.player == a.player and abs(d.t_end - a.t) < 0.5 for d in decisions):
            decisions.append(
                make_decision(
                    a.player, max(prev_event_t, a.t - WINDOW_COMMIT_SLACK_S), a.t,
                    "action_only", a.action, a.amount, None,
                )
            )
        prev_event_t = a.t
    decisions.sort(key=lambda d: d.t_end)

    for p in players:
        if any(a.player == p for a in actions) and p not in hole_cards:
            flags.append(f"actor_without_cards:{p}")
    overlap = set(board) & {c for cards in hole_cards.values() for c in cards}
    if overlap:
        flags.append(f"board_hole_overlap:{sorted(overlap)}")

    resolver_names = {p: p for p in players}
    return Hand(
        hand_id=hand_id,
        session_id=session_id,
        t_start=span[0].t,
        t_end=span[-1].t,
        players=players,
        display_names=resolver_names,
        positions=positions,
        hole_cards=hole_cards,
        board=board,
        blinds=list(blinds),
        actions=actions,
        decisions=decisions,
        n_snapshots=len(span),
        flags=flags,
    )


def assemble_session(
    snaps: list[HudSnapshot], session_id: str
) -> tuple[list[Hand], dict]:
    """Normalize names, segment, and assemble all hands of one session."""
    raw_names = [k for s in snaps for k in s.players()]
    resolver = NameResolver(raw_names)
    normalized = normalize_snapshots(snaps, resolver)
    spans = merge_spans(segment_hands(normalized))
    hands = [assemble_hand(span, session_id, i + 1) for i, span in enumerate(spans)]
    for hand in hands:
        hand.display_names = {p: resolver.display(p) for p in hand.players}

    report = {
        "snapshots": len(snaps),
        "hands": len(hands),
        "decisions": sum(len(h.decisions) for h in hands),
        "decisions_to_act": sum(
            1 for h in hands for d in h.decisions if d.window_source == "to_act"
        ),
        "hands_with_board": sum(1 for h in hands if h.board),
        "hands_fully_carded": sum(
            1
            for h in hands
            if h.players and all(p in h.hole_cards for p in h.players)
        ),
        "flag_counts": dict(Counter(f.split(":")[0] for h in hands for f in h.flags)),
    }
    return hands, report
