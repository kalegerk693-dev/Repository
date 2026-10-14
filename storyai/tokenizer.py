"""A byte-level BPE (byte pair encoding) tokenizer, built from scratch.

How it works:
  1. Text becomes raw UTF-8 bytes, so every possible character (any language,
     emoji, ...) can be written with just 256 starting tokens.
  2. Training looks through your stories, finds the most common pair of
     neighbouring tokens and merges that pair into a brand new token. It
     repeats this until the vocabulary is full, so common pieces like " the",
     "ing" or " Once" end up as single tokens.
  3. Encoding replays those merges on new text. Decoding glues the bytes
     back together.
"""

import heapq
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

END_OF_STORY = "<|endofstory|>"

# Before BPE runs, text is cut into word-like chunks. Merges never cross a
# chunk boundary, so tokens stay meaningful (" dragon" rather than "n dra").
# Every character of the input lands in exactly one chunk.
SPLIT_PATTERN = re.compile(
    r"'(?:s|t|re|ve|m|ll|d)"  # common English contractions
    r"| ?[^\W\d_]+"           # a word, together with the space before it
    r"| ?\d+"                 # a number
    r"| ?[^\s\w]+"            # punctuation and symbols
    r"|_+"                    # underscores (\w, but neither letter nor digit)
    r"|\s+(?!\S)"             # whitespace, leaving the last space for the next word
    r"|\s+"                   # any remaining whitespace
)


# Very long runs with no spaces (a pasted web link, text without spaces) are cut into
# pieces of this many characters, so training and encoding never slow to a crawl.
MAX_CHUNK = 64


def split_chunks(text):
    for chunk in SPLIT_PATTERN.findall(text):
        for i in range(0, len(chunk), MAX_CHUNK):
            yield chunk[i:i + MAX_CHUNK]


def _merge(ids, pair, new_id):
    """Replace every occurrence of `pair` in `ids` with `new_id`."""
    a, b = pair
    out = []
    i, n = 0, len(ids)
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class Tokenizer:
    def __init__(self, merges=(), special_tokens=(END_OF_STORY,), trained_chars=None):
        # merges[i] is the pair of token ids that became token 256 + i.
        self.merges = [tuple(p) for p in merges]
        self.special_tokens = list(special_tokens)
        self.trained_chars = trained_chars  # how much text it learned from

        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]
        self.special_ids = {}
        for j, tok in enumerate(self.special_tokens):
            tid = 256 + len(self.merges) + j
            self.special_ids[tok] = tid
            self.vocab[tid] = tok.encode("utf-8")
        self._special_re = None
        if self.special_tokens:
            self._special_re = re.compile(
                "(" + "|".join(re.escape(t) for t in self.special_tokens) + ")"
            )
        self._cache = {}

    @property
    def vocab_size(self):
        return 256 + len(self.merges) + len(self.special_tokens)

    # ------------------------------------------------------------------ training

    @classmethod
    def train(cls, texts, vocab_size, special_tokens=(END_OF_STORY,), verbose=False, log=print):
        """Learn merges from `texts` until the vocabulary has `vocab_size` tokens."""
        num_merges = vocab_size - 256 - len(special_tokens)
        if num_merges < 0:
            raise ValueError(f"vocab_size must be at least {256 + len(special_tokens)}")

        chunk_counts = Counter()
        trained_chars = 0
        for text in texts:
            chunk_counts.update(split_chunks(text))
            trained_chars += len(text)
        words = [list(chunk.encode("utf-8")) for chunk in chunk_counts]
        freqs = list(chunk_counts.values())

        # How often each neighbouring pair appears, and which words contain it.
        # Both are updated incrementally after each merge, which is what makes
        # training fast enough to run on big story collections.
        pair_counts = defaultdict(int)
        where = defaultdict(set)
        for i, (w, f) in enumerate(zip(words, freqs)):
            for pair in zip(w, w[1:]):
                pair_counts[pair] += f
                where[pair].add(i)
        # Max-heap of (-count, pair). Entries go stale as counts change; a
        # popped entry is only trusted if it still matches pair_counts.
        heap = [(-c, p) for p, c in pair_counts.items()]
        heapq.heapify(heap)

        merges = []
        report_every = max(1, num_merges // 10)
        while len(merges) < num_merges and heap:
            neg_count, pair = heapq.heappop(heap)
            if pair_counts.get(pair, 0) != -neg_count:
                continue
            if -neg_count < 2:
                break  # nothing left that appears more than once
            new_id = 256 + len(merges)
            merges.append(pair)

            changed = set()
            for i in where.pop(pair):
                w, f = words[i], freqs[i]
                for p in zip(w, w[1:]):
                    pair_counts[p] -= f
                    changed.add(p)
                w = _merge(w, pair, new_id)
                words[i] = w
                for p in zip(w, w[1:]):
                    pair_counts[p] += f
                    where[p].add(i)
                    changed.add(p)
            for p in changed:
                if pair_counts[p] > 0:
                    heapq.heappush(heap, (-pair_counts[p], p))
                else:
                    del pair_counts[p]

            if verbose and len(merges) % report_every == 0:
                log(f"  learned {len(merges)}/{num_merges} merges")

        if verbose and len(merges) < num_merges:
            log(f"  stopped at {len(merges)} merges: no pair of tokens repeats any more")
        return cls(merges, special_tokens, trained_chars)

    # ------------------------------------------------------------ encode/decode

    def _encode_chunk(self, data):
        ids = list(data)
        while len(ids) >= 2:
            # Apply the earliest-learned merge that is available, just like training did.
            pair = min(zip(ids, ids[1:]), key=lambda p: self.ranks.get(p, float("inf")))
            rank = self.ranks.get(pair)
            if rank is None:
                break
            ids = _merge(ids, pair, 256 + rank)
        return ids

    def encode_ordinary(self, text):
        """Encode text, treating special token strings as plain text."""
        ids = []
        for chunk in split_chunks(text):
            cached = self._cache.get(chunk)
            if cached is None:
                cached = self._encode_chunk(chunk.encode("utf-8"))
                if len(self._cache) < 200_000:
                    self._cache[chunk] = cached
            ids.extend(cached)
        return ids

    def encode(self, text):
        """Encode text into token ids. Special tokens like <|endofstory|> are recognised."""
        if self._special_re is None:
            return self.encode_ordinary(text)
        ids = []
        for part in self._special_re.split(text):
            if part in self.special_ids:
                ids.append(self.special_ids[part])
            elif part:
                ids.extend(self.encode_ordinary(part))
        return ids

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    # ---------------------------------------------------------------- save/load

    def save(self, path):
        data = {
            "type": "byte-level-bpe",
            "version": 1,
            "special_tokens": self.special_tokens,
            "trained_chars": self.trained_chars,
            "merges": [list(p) for p in self.merges],
        }
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["merges"], data["special_tokens"], data.get("trained_chars"))
