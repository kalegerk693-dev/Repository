"""Is the AI writing its own stories, or copying yours?

The checker looks for passages in a new story that appear word for word (ignoring
capital letters and punctuation) in one of your stories. A run of MIN_RUN or more
identical words in a row counts as copied. Shorter overlaps are just ordinary
phrases ("and then she said to the") that any writer reuses.
"""

import re

import numpy as np

MIN_RUN = 8
WORD = re.compile(r"[\w']+")
_SEPARATOR, _UNKNOWN = -1, -2  # never equal to each other or to a real word id


def _window_hashes(ids, n):
    """A hash of every run of n word ids, and whether that run is clean (no
    separator or unknown word in it)."""
    count = len(ids) - n + 1
    if count <= 0:
        return np.zeros(0, np.uint64), np.zeros(0, bool)
    h = np.zeros(count, np.uint64)
    for k in range(n):  # wraps around modulo 2**64, which is fine for hashing
        h = h * np.uint64(1_000_003) + (ids[k:k + count] + 3).astype(np.uint64)
    bad = np.concatenate([[0], np.cumsum(ids < 0)])
    return h, (bad[n:n + count] - bad[:count]) == 0


class CopyChecker:
    def __init__(self, sources):
        """sources: list of (name, story text) to compare against."""
        self.names = [name for name, _ in sources]
        self.vocab = {}
        ids, self.starts = [], []
        for _, text in sources:
            self.starts.append(len(ids))
            ids.extend(self.vocab.setdefault(w, len(self.vocab)) for w in WORD.findall(text.lower()))
            ids.append(_SEPARATOR)  # copies can't run from one story into the next
        self.ids = np.array(ids, dtype=np.int64)
        self.starts = np.array(self.starts)
        hashes, clean = _window_hashes(self.ids, MIN_RUN)
        positions = np.flatnonzero(clean)
        order = np.argsort(hashes[positions], kind="stable")
        self.hashes, self.positions = hashes[positions][order], positions[order]

    def check(self, text):
        """How much of `text` is copied. Returns a dict with:
        words, copied_fraction (0-1), longest (words in the longest copied passage),
        passage (that passage as written), source (which story it came from), verdict."""
        matches = list(WORD.finditer(text.lower()))
        g = np.array([self.vocab.get(m.group(), _UNKNOWN) for m in matches], dtype=np.int64)
        covered = np.zeros(len(g), bool)
        best_len, best_at, best_pos = 0, 0, 0
        hashes, clean = _window_hashes(g, MIN_RUN)
        found = np.searchsorted(self.hashes, hashes)
        for i in np.flatnonzero(clean):
            j = found[i]
            # Every place in your stories with the same 8 words (hashes can collide,
            # so the words themselves are compared below).
            while j < len(self.hashes) and self.hashes[j] == hashes[i] and j < found[i] + 50:
                pos = self.positions[j]
                span = min(len(g) - i, len(self.ids) - pos)
                differ = g[i:i + span] != self.ids[pos:pos + span]
                run = int(differ.argmax()) if differ.any() else span
                if run >= MIN_RUN:
                    covered[i:i + run] = True
                    if run > best_len:
                        best_len, best_at, best_pos = run, i, pos
                j += 1

        result = {"words": len(g), "copied_fraction": float(covered.mean()) if len(g) else 0.0,
                  "longest": best_len, "passage": "", "source": None}
        if best_len:
            result["passage"] = text[matches[best_at].start():matches[best_at + best_len - 1].end()]
            result["source"] = self.names[int(np.searchsorted(self.starts, best_pos, "right")) - 1]
        result["verdict"] = verdict(result)
        return result


def verdict(r):
    if r["copied_fraction"] >= 0.5:
        return "copied"   # most of it is your stories, word for word
    if r["copied_fraction"] >= 0.15 or r["longest"] >= 30:
        return "partly copied"
    return "original"


def describe(r):
    """A one-line, plain-English summary of a check() result."""
    pct = round(100 * r["copied_fraction"])
    if r["verdict"] == "original":
        if r["longest"]:
            return (f"Original: {100 - pct}% its own words. The longest bit shared with your "
                    f"stories is {r['longest']} words: “{r['passage']}”")
        return f"Original: no passage of {MIN_RUN}+ words is copied from your stories."
    where = f" (from {r['source']})" if r["source"] else ""
    return (f"{r['verdict'].capitalize()}: {pct}% of this story is copied word for word from your "
            f"stories. Longest copied passage: {r['longest']} words{where}. "
            "More stories (or less training) makes it more original.")
