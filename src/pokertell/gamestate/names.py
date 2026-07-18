"""Player name canonicalization.

OCR jitters names across frames ('JACK C' vs 'JACKC'), and a stray
character can slip in. Identity within a session is the canonical key
(alphanumeric-only uppercase); rare variants within edit distance 1 of a
frequent name are merged into it. Display names keep the most common raw
spelling.
"""

import re
from collections import Counter

_CANON_RE = re.compile(r"[^A-Z0-9]")


def canonical(name: str) -> str:
    return _CANON_RE.sub("", name.upper())


def _edit_distance(a: str, b: str, cap: int = 2) -> int:
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[-1]


class NameResolver:
    """Maps raw OCR names to stable canonical identities for one session.

    Build with all raw names seen (with repetition), then resolve() each.
    A canonical form seen fewer than merge_below times is merged into the
    most frequent canonical within edit distance 1, if any.
    """

    def __init__(self, raw_names: list[str], merge_below: int = 3) -> None:
        self._display: dict[str, str] = {}
        self._merge: dict[str, str] = {}
        canon_counts: Counter[str] = Counter()
        raw_by_canon: dict[str, Counter[str]] = {}
        for raw in raw_names:
            c = canonical(raw)
            if not c:
                continue
            canon_counts[c] += 1
            raw_by_canon.setdefault(c, Counter())[raw] += 1

        frequent = [c for c, n in canon_counts.most_common() if n >= merge_below]
        for c, n in canon_counts.items():
            if n < merge_below:
                for f in frequent:
                    if _edit_distance(c, f, cap=1) <= 1:
                        self._merge[c] = f
                        break
        for c, raws in raw_by_canon.items():
            target = self._merge.get(c, c)
            merged = self._display.get(target)
            best = raws.most_common(1)[0][0]
            if merged is None or canon_counts[c] > canon_counts.get(canonical(merged), 0):
                self._display[target] = best

    def resolve(self, raw: str) -> str:
        """Canonical identity for a raw OCR name."""
        c = canonical(raw)
        return self._merge.get(c, c)

    def display(self, canon: str) -> str:
        """Most common raw spelling for a canonical identity."""
        return self._display.get(canon, canon)
