"""Reading your stories from disk and turning them into training batches."""

import hashlib
import math
import re
from datetime import datetime
from pathlib import Path

import numpy as np

from .tokenizer import END_OF_STORY


# Story separators: ours, plus the one GPT-2 style datasets (like TinyStories) use.
SEPARATORS = (END_OF_STORY, "<|endoftext|>")


def read_text_file(path):
    """Read a text file whatever it was saved as: UTF-8 (with or without a BOM),
    UTF-16 ("Unicode" in Windows Notepad), or Windows "ANSI". Line endings become \\n."""
    raw = Path(path).read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        text = raw.decode("utf-16", errors="replace")
    else:
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = raw.decode("cp1252", errors="replace")
    return text.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")


def story_files(folder):
    """Every .txt file in `folder` and its subfolders (or `folder` itself if it's a file).
    Hidden files and the junk macOS leaves behind (__MACOSX, ._name.txt) are skipped."""
    root = Path(folder)
    if root.is_file():
        return [root]
    if not root.is_dir():
        return []
    return [path for path in sorted(root.rglob("*"))
            if path.suffix.lower() == ".txt" and path.is_file()
            and not any(part.startswith(".") or part == "__MACOSX"
                        for part in path.relative_to(root).parts)]


def split_stories(text):
    """One file can hold many stories, separated by a line containing just <|endofstory|>."""
    for sep in SEPARATORS[1:]:
        text = text.replace(sep, END_OF_STORY)
    return [part.strip() for part in text.split(END_OF_STORY) if part.strip()]


def read_stories(folder):
    """All the stories in `folder`, in a stable order."""
    return [story for path in story_files(folder) for story in split_stories(read_text_file(path))]


def count_new_stories(folder, text):
    """How many of the stories in `text` aren't in `folder` yet."""
    existing = set(read_stories(folder))
    return sum(story not in existing for story in split_stories(text))


def story_id(story):
    return hashlib.md5(story.encode("utf-8")).hexdigest()[:12]


def save_story(folder, text, name=None):
    """Save one story as a new .txt file in `folder` and return its path.
    Never overwrites an existing file."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^\w-]+", "_", name).strip("_")[:60] if name else ""
    stem = stem or "story_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    path, n = folder / f"{stem}.txt", 1
    while path.exists():
        n += 1
        path = folder / f"{stem}_{n}.txt"
    path.write_text(text.replace("\ufeff", "").replace("\r\n", "\n").strip() + "\n",
                    encoding="utf-8")
    return path


def list_story_files(folder):
    """[(path, number of words, first words)] for every story file."""
    files = []
    for path in story_files(folder):
        words = read_text_file(path).split()
        files.append((path, len(words), " ".join(words[:12])))
    return files


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
