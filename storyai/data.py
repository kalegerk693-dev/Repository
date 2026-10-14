"""Reading your stories from disk and turning them into training batches."""

import hashlib
import math
from pathlib import Path

import numpy as np

from .tokenizer import END_OF_STORY


def read_stories(folder):
    """Every .txt file in `folder` (including subfolders) is a story.

    One file can also hold many stories if they are separated by a line
    containing just <|endofstory|>.
    """
    stories = []
    for path in sorted(Path(folder).rglob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
        for part in text.split(END_OF_STORY):
            part = part.strip()
            if part:
                stories.append(part)
    return stories


def is_holdout(story):
    """About 1 in 10 stories is kept aside to check the model is really learning
    to write, not just memorising. The choice is fixed per story so it never
    sneaks into training later."""
    return hashlib.md5(story.encode("utf-8")).digest()[0] % 10 == 0


def auto_vocab_size(num_chars):
    """More text supports a bigger vocabulary."""
    return min(4096, max(512, int(2 * math.sqrt(num_chars))))


def encode_stories(stories, tokenizer):
    """One long token stream: <|endofstory|> story <|endofstory|> story ...

    The model learns that a new story starts after <|endofstory|>, and that a
    story can end by producing it.
    """
    eos = tokenizer.special_ids[END_OF_STORY]
    ids = [eos]
    for story in stories:
        ids.extend(tokenizer.encode(story))
        ids.append(eos)
    return np.array(ids, dtype=np.int32)


def get_batch(data, batch_size, block_size, rng):
    """Random windows of text (x) and the same windows shifted by one token (y)."""
    T = min(block_size, len(data) - 1)
    starts = rng.integers(0, len(data) - T, size=batch_size)
    offsets = starts[:, None] + np.arange(T)
    return data[offsets], data[offsets + 1]
