"""Is the AI writing its own stories, or copying yours?

Two checks compare a new story with every story the AI learned from:

1. Copied passages: MIN_RUN or more words in a row that match one of your stories
   (ignoring capitals and punctuation). Stock phrases that STOCK_STORIES or more of
   your stories share ("Once upon a time there was a little girl named Lily") don't
   count: repeating those is how stories sound, not copying a particular one.
2. Near-copies: a story that follows one of yours closely with a few words changed
   here and there, which slips past the first check.

Short overlaps ("he went to the king, and said") are ordinary phrases every writer uses.
"""

import difflib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np

MIN_RUN = 8          # this many identical words in a row is a copied passage
NEAR_RUN = 4         # matching stretches of this many words build up a near-copy
STOCK_STORIES = 3     # a phrase this many different stories share is a stock phrase...
STOCK_PHRASE_MAX = 25  # ...unless it runs this long, which is copying however common it is
WORD = re.compile(r"\w+(?:'\w+)*")
QUOTES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "`": "'", "´": "'", "“": '"', "”": '"'})
_SEPARATOR, _UNKNOWN = -1, -2  # never equal to each other or to a real word id


def _words(text):
    """[(word, start, end)]: words lower-cased for comparing, positions in `text` for quoting."""
    return [(m.group().lower(), m.start(), m.end()) for m in WORD.finditer(text)]


def _normalize(text):
    # NFC keeps the text looking the same but makes "é" one character however it was
    # typed; curly and straight apostrophes become the same.
    return unicodedata.normalize("NFC", text).translate(QUOTES)


def _window_hashes(ids, n):
    """A hash of every run of n word ids, and whether the run is usable: no story break
    or unknown word in it, and not just one or two words repeated ("ha ha ha ha")."""
    count = len(ids) - n + 1
    if count <= 0:
        return np.zeros(0, np.uint64), np.zeros(0, bool)
    h = np.zeros(count, np.uint64)
    for k in range(n):  # wraps around modulo 2**64, which is fine for hashing
        h = h * np.uint64(1_000_003) + (ids[k:k + count] + 3).astype(np.uint64)
    bad = np.concatenate([[0], np.cumsum(ids < 0)])
    usable = (bad[n:n + count] - bad[:count]) == 0
    windows = np.sort(np.lib.stride_tricks.sliding_window_view(ids, n), axis=1)
    distinct = 1 + (windows[:, 1:] != windows[:, :-1]).sum(axis=1)
    return h, usable & (distinct >= max(2, n // 2))


class _Index:
    """Every run of n words in the training stories, sorted by hash for fast lookup."""

    def __init__(self, ids, story_of, n):
        h, usable = _window_hashes(ids, n)
        pos = np.flatnonzero(usable)
        order = np.lexsort((story_of[pos], h[pos]))
        self.hashes, self.positions = h[pos][order], pos[order]
        # How many different stories each run appears in (1 = only one story has it).
        stories = story_of[self.positions]
        new_hash = np.r_[True, self.hashes[1:] != self.hashes[:-1]]
        new_story = new_hash | np.r_[True, stories[1:] != stories[:-1]]
        group = np.cumsum(new_hash) - 1
        self.spread = np.bincount(group, weights=new_story).astype(int)[group]

    def lookup(self, hashes):
        lo = np.searchsorted(self.hashes, hashes, "left")
        hi = np.searchsorted(self.hashes, hashes, "right")
        spread = self.spread[np.minimum(lo, len(self.spread) - 1)] if len(self.spread) else lo * 0
        return lo, hi, np.where(hi > lo, spread, 0)


class CopyChecker:
    def __init__(self, sources):
        """sources: list of (name, story text) to compare against."""
        self.names, self.vocab = [], {}
        ids, starts, self.short, seen = [], [], [], set()
        for name, text in sources:
            words = [w for w, _, _ in _words(_normalize(text))]
            if not words or tuple(words) in seen:
                continue  # an exact duplicate (a backup copy) is still just one story
            seen.add(tuple(words))
            self.names.append(name)
            starts.append(len(ids))
            ids.extend(self.vocab.setdefault(w, len(self.vocab)) for w in words)
            ids.append(_SEPARATOR)  # copies can't run from one story into the next
            if len(words) < MIN_RUN:  # too short to ever contain a copied passage
                self.short.append((len(self.names) - 1, ids[starts[-1]:-1]))
        self.ids = np.array(ids, dtype=np.int64)
        self.starts = np.array(starts, dtype=np.int64)
        story_of = np.searchsorted(self.starts, np.arange(len(self.ids)), "right") - 1
        self.story_of = story_of
        self.exact = _Index(self.ids, story_of, MIN_RUN)
        self.near = _Index(self.ids, story_of, NEAR_RUN)

    def _story_words(self, story):
        end = self.starts[story + 1] - 1 if story + 1 < len(self.starts) else len(self.ids) - 1
        return self.ids[self.starts[story]:end]

    def check(self, text):
        """How much of `text` is copied. Returns a dict with words, copied_fraction (0-1),
        longest (words in the longest copied passage), passage (as written), source
        (which story it came from), near_copy (0-1, see below) and verdict."""
        text = _normalize(text)
        words = _words(text)
        g = np.array([self.vocab.get(w, _UNKNOWN) for w, _, _ in words], dtype=np.int64)
        n = len(g)
        covered = np.zeros(n, bool)
        best_len, best_at, best_pos = 0, 0, 0

        # 1. Copied passages.
        hashes, usable = _window_hashes(g, MIN_RUN)
        lo, hi, spread = self.exact.lookup(hashes)
        stock = spread >= STOCK_STORIES
        for i in np.flatnonzero(usable & (hi > lo)):
            if covered[i:i + MIN_RUN].all():
                continue  # already part of a copied passage that was found
            for j in range(lo[i], min(hi[i], lo[i] + 50)):
                pos = int(self.exact.positions[j])
                span = min(n - i, len(self.ids) - pos)
                differ = g[i:i + span] != self.ids[pos:pos + span]
                forward = int(differ.argmax()) if differ.any() else span
                back = 0
                while i - back > 0 and pos - back > 0 and g[i - back - 1] == self.ids[pos - back - 1]:
                    back += 1
                start, run, src = i - back, back + forward, pos - back
                if run < MIN_RUN:
                    continue  # two different runs that happen to share a hash
                if run < STOCK_PHRASE_MAX and stock[start:start + run - MIN_RUN + 1].all():
                    continue  # a stock phrase many of your stories share
                covered[start:start + run] = True
                if run > best_len:
                    best_len, best_at, best_pos = run, start, src
        for story, short in self.short:  # whole tiny stories (a 6-word poem) recited word for word
            k = len(short)
            if k >= 4 and n >= k:
                windows = np.lib.stride_tricks.sliding_window_view(g, k)
                for i in np.flatnonzero((windows == short).all(axis=1)):
                    covered[i:i + k] = True
                    if k > best_len:
                        best_len, best_at, best_pos = k, int(i), int(self.starts[story])

        # 2. Near-copies: line the story up with the one of yours it shares the most
        # 4-word runs with, and count how much of it follows that story.
        near_copy, near_source = 0.0, None
        h4, usable4 = _window_hashes(g, NEAR_RUN)
        lo4, hi4, spread4 = self.near.lookup(h4)
        unique4 = usable4 & (spread4 > 0) & (spread4 < STOCK_STORIES)  # not stock phrases
        hits = np.flatnonzero(unique4)
        if len(hits) >= 3:
            votes, places = {}, {}
            for i in hits:
                found = self.near.positions[lo4[i]:min(hi4[i], lo4[i] + 20)]
                for s in set(self.story_of[found].tolist()):
                    votes[s] = votes.get(s, 0) + 1
                for s, place in zip(self.story_of[found].tolist(), found.tolist()):
                    places.setdefault(s, []).append(place)
            story = max(votes, key=votes.get)
            # Only the part of their story around the shared runs needs lining up.
            middle = int(np.median(places[story])) - int(self.starts[story])
            a = max(0, middle - 3 * n // 2)
            theirs = self._story_words(story)[a:a + 3 * n + NEAR_RUN]
            matcher = difflib.SequenceMatcher(None, g.tolist(), theirs.tolist(), autojunk=False)
            # Stretches that are only stock phrases (shared by several stories) don't count.
            matched = sum(b.size for b in matcher.get_matching_blocks()
                          if b.size >= NEAR_RUN and unique4[b.a:b.a + b.size - NEAR_RUN + 1].any())
            near_copy, near_source = matched / n, self.names[story]

        result = {"words": n, "copied_fraction": float(covered.mean()) if n else 0.0,
                  "longest": best_len, "passage": "", "source": None,
                  "near_copy": float(near_copy), "near_source": near_source}
        if best_len:
            result["passage"] = text[words[best_at][1]:words[best_at + best_len - 1][2]]
            story = int(np.searchsorted(self.starts, best_pos, "right")) - 1
            result["source"] = self.names[story]
        result["verdict"] = verdict(result)
        return result


def verdict(r):
    if r["copied_fraction"] >= 0.5 or r["near_copy"] >= 0.6:
        return "copied"   # most of it is one of your stories
    if r["copied_fraction"] >= 0.15 or r["longest"] >= 15 or r["near_copy"] >= 0.3:
        return "partly copied"
    return "original"


def describe(r):
    """A one-line, plain-English summary of a check() result."""
    pct = round(100 * r["copied_fraction"])
    near = round(100 * r["near_copy"])
    if r["verdict"] == "original":
        if r["longest"]:
            return (f"Original: {100 - pct}% its own words. The longest bit shared with your "
                    f"stories is {r['longest']} words: “{r['passage']}”")
        return f"Original: no passage of {MIN_RUN}+ words is copied from your stories."
    if r["copied_fraction"] < 0.15 and r["longest"] < 15:  # flagged as a near-copy
        return (f"{r['verdict'].capitalize()}: {near}% of this story follows {r['near_source']} "
                "closely, with only a few words changed. More stories (or less training) "
                "makes it more original.")
    where = f" (from {r['source']})" if r["source"] else ""
    return (f"{r['verdict'].capitalize()}: {pct}% of this story is copied word for word from your "
            f"stories. Longest copied passage: {r['longest']} words{where}. "
            "More stories (or less training) makes it more original.")


# ------------------------------------------------------- what the model learned from

SEEN_FILE = "trained_stories.jsonl"


def record_trained_stories(model_dir, sources):
    """Remember every story the model trains on, so a story you delete later is still
    checked against: the model may have memorised it."""
    from .data import story_id

    path = Path(model_dir) / SEEN_FILE
    known = {record["id"] for record in _read_records(path)}
    with path.open("a", encoding="utf-8") as f:
        for name, text in sources:
            if story_id(text) not in known:
                known.add(story_id(text))
                f.write(json.dumps({"id": story_id(text), "name": name, "text": text}) + "\n")


def checker_sources(stories_dir, model_dir):
    """Your current stories, plus any the model trained on that have since been removed."""
    from .data import story_id, story_sources

    sources = story_sources(stories_dir)
    have = {story_id(text) for _, text in sources}
    for record in _read_records(Path(model_dir) / SEEN_FILE):
        if record["id"] not in have:
            have.add(record["id"])
            sources.append((f"{record['name']} (since removed)", record["text"]))
    return sources


def _read_records(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
