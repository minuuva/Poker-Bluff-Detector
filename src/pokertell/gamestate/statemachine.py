"""Reconstruct structured hand histories from per-frame HUD observations.

This is the least glamorous and most load-bearing module in the project. The
OCR stage emits a HudSnapshot every sampled frame; this module folds that
stream into Hand records with per-street actions, and emits Decision windows
(the unit of analysis for the whole study).

Inference rules (first pass, tuned against real footage in the day 2/3
milestone):
- New hand: the board clears and the pot drops back to blind level.
- Street change: board card count steps through 0 / 3 / 4 / 5.
- Action: diff of the per-player bet fields between consecutive snapshots.
- Decision window: from the first snapshot where a player is marked to_act
  until the snapshot where their action is committed.

Every reconstructed session must be validated against a hand-transcribed
sample before its data feeds the model.
"""

from dataclasses import dataclass, field
from enum import Enum


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


def street_from_board(board: list[str]) -> Street:
    """Map board card count to street. Raises on impossible counts."""
    n = len(board)
    mapping = {0: Street.PREFLOP, 3: Street.FLOP, 4: Street.TURN, 5: Street.RIVER}
    if n not in mapping:
        raise ValueError(f"impossible board size {n}: {board}")
    return mapping[n]


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


@dataclass
class Action:
    player: str
    action: ActionType
    amount: float
    street: Street
    t: float


@dataclass
class Decision:
    """One player decision: one row in the final feature table.

    The window [t_start, t_end] is where behavioral features get extracted;
    the action and the game state at t_start feed the betting baseline.
    """

    hand_id: str
    player: str
    street: Street
    action: Action
    t_start: float
    t_end: float
    pot_before: float | None
    to_call: float
    hole_cards: list[str] | None
    board: list[str]


@dataclass
class Hand:
    hand_id: str
    session_id: str
    t_start: float
    t_end: float
    players: list[str]
    hole_cards: dict[str, list[str]]
    actions: list[Action]
    final_board: list[str]

    def decisions(self) -> list[Decision]:
        """Derive Decision windows from the action sequence.

        TODO(day 3): implement once action inference is validated. Needs the
        to_act timing from snapshots to set t_start (window opens when action
        reaches the player, not when the previous action ends).
        """
        raise NotImplementedError("Decision derivation lands in the day 3 milestone")


class HandAssembler:
    """Fold a stream of HudSnapshots into Hand records.

    Usage:
        assembler = HandAssembler(session_id)
        for snap in snapshots:
            done = assembler.push(snap)
            if done is not None:
                hands.append(done)
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._snapshots: list[HudSnapshot] = []
        self._hand_count = 0

    def push(self, snap: HudSnapshot) -> Hand | None:
        """Consume one snapshot; return a completed Hand at hand boundaries."""
        if self._snapshots and self._is_new_hand(self._snapshots[-1], snap):
            hand = self._finalize()
            self._snapshots = [snap]
            return hand
        self._snapshots.append(snap)
        return None

    def flush(self) -> Hand | None:
        """Finalize the trailing hand at end of stream."""
        if not self._snapshots:
            return None
        hand = self._finalize()
        self._snapshots = []
        return hand

    @staticmethod
    def _is_new_hand(prev: HudSnapshot, cur: HudSnapshot) -> bool:
        """A new hand starts when a non-empty board clears."""
        return len(prev.board) > 0 and len(cur.board) == 0

    def _finalize(self) -> Hand:
        snaps = self._snapshots
        self._hand_count += 1
        players = sorted({p for s in snaps for p in s.stacks})
        hole_cards = {}
        for s in snaps:
            for player, cards in s.hole_cards.items():
                if cards:
                    hole_cards.setdefault(player, cards)
        return Hand(
            hand_id=f"{self.session_id}#{self._hand_count:04d}",
            session_id=self.session_id,
            t_start=snaps[0].t,
            t_end=snaps[-1].t,
            players=players,
            hole_cards=hole_cards,
            actions=self._infer_actions(snaps),
            final_board=max((s.board for s in snaps), key=len),
        )

    @staticmethod
    def _infer_actions(snaps: list[HudSnapshot]) -> list[Action]:
        """Infer actions from bet-field deltas between consecutive snapshots.

        First pass: a player's bet increasing is a bet/raise/call depending on
        the table's current high bet; a stack hitting zero marks all-in; folds
        need the HUD's fold indicator (player panel grays out on HCL), which
        is a day 2 OCR item. Expect this to be rewritten against real output.
        """
        actions: list[Action] = []
        for prev, cur in zip(snaps, snaps[1:]):
            street = street_from_board(cur.board)
            high_before = max(prev.bets.values(), default=0.0)
            for player, bet in cur.bets.items():
                bet_before = prev.bets.get(player, 0.0)
                if bet <= bet_before:
                    continue
                if cur.stacks.get(player, 1.0) == 0.0:
                    kind = ActionType.ALL_IN
                elif bet <= high_before:
                    kind = ActionType.CALL
                elif high_before == 0.0:
                    kind = ActionType.BET
                else:
                    kind = ActionType.RAISE
                actions.append(
                    Action(player=player, action=kind, amount=bet, street=street, t=cur.t)
                )
        return actions
