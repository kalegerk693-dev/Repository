"""Loading a trained model and having it write a story, piece by piece."""

import codecs
import re
from pathlib import Path

import numpy as np

from .checkpoint import load_checkpoint
from .tokenizer import END_OF_STORY, Tokenizer


# Raw bytes the model might produce that would only garble a terminal or web page
# (everything below a space except tab and newline, plus DEL).
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


class NoModelError(Exception):
    pass


class StoryWriter:
    def __init__(self, model_dir):
        model_dir = Path(model_dir)
        if not (model_dir / "model.npz").exists() or not (model_dir / "tokenizer.json").exists():
            raise NoModelError("No trained model yet. Feed it stories, then train it.")
        self.model, _, self.meta = load_checkpoint(model_dir / "model.npz")
        self.tokenizer = Tokenizer.load(model_dir / "tokenizer.json")
        if self.tokenizer.vocab_size != self.model.config.vocab_size:
            raise NoModelError("The saved model and tokenizer don't match. "
                               "Train again, starting over from scratch (--fresh).")
        self.eos = self.tokenizer.special_ids[END_OF_STORY]

    def write(self, prompt="", max_tokens=600, temperature=0.8, top_k=None, top_p=0.95,
              rng=None, min_tokens=20):
        """A new story. Iterate over it to get its text piece by piece, starting with
        the prompt. Afterwards, story.finished is True if the AI ended the story itself
        and False if it was cut off at max_tokens."""
        return Story(self, prompt, max_tokens, temperature, top_k, top_p, rng, min_tokens)


class Story:
    def __init__(self, writer, prompt, max_tokens, temperature, top_k, top_p, rng, min_tokens):
        self.writer = writer
        self.prompt = prompt.rstrip(" ")  # tokens carry their own leading space
        self.max_tokens = max_tokens
        self.sampling = dict(temperature=temperature, top_k=top_k, top_p=top_p, rng=rng,
                             min_new_tokens=min(min_tokens, max_tokens))
        self.finished = False
        self.generated = ""  # just the AI's part, without the prompt

    def __iter__(self):
        w = self.writer
        yield CONTROL_CHARS.sub("", self.prompt)
        # Tokens can split a multi-byte character; this only lets out complete ones.
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        written = 0
        ids = [w.eos] + w.tokenizer.encode_ordinary(self.prompt)
        for token in w.model.generate(ids, self.max_tokens, stop_id=w.eos, **self.sampling):
            written += 1
            piece = CONTROL_CHARS.sub("", decoder.decode(w.tokenizer.vocab[token]))
            if piece:
                self.generated += piece
                yield piece
        piece = CONTROL_CHARS.sub("", decoder.decode(b"", final=True))
        self.generated += piece
        yield piece
        self.finished = written < self.max_tokens
