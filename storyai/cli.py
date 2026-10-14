"""Command line: feed stories, train the AI, and have it write new ones."""

import argparse
import codecs
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STORIES = ROOT / "stories"
DEFAULT_MODEL_DIR = ROOT / "checkpoints"


def new_story_path(stories_dir):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path, n = stories_dir / f"story_{stamp}.txt", 1
    while path.exists():
        n += 1
        path = stories_dir / f"story_{stamp}_{n}.txt"
    return path


def cmd_feed(args):
    from .data import read_stories

    stories_dir = Path(args.stories)
    stories_dir.mkdir(parents=True, exist_ok=True)
    if args.file:
        source = Path(args.file)
        dest = stories_dir / source.name
        if dest.exists():
            dest = new_story_path(stories_dir)
        shutil.copyfile(source, dest)
        print(f"Copied {source} to {dest}")
    else:
        print("Type or paste your story. When you're done, type END on its own line and press Enter.\n")
        lines = []
        for line in sys.stdin:
            if line.strip() == "END":
                break
            lines.append(line)
        text = "".join(lines).strip()
        if not text:
            print("Nothing to save.")
            return
        dest = new_story_path(stories_dir)
        dest.write_text(text + "\n", encoding="utf-8")
        print(f"\nSaved your story ({len(text.split()):,} words) to {dest}")
    n = len(read_stories(stories_dir))
    print(f"Your AI has {n} {'story' if n == 1 else 'stories'} to learn from. "
          "Run `python story.py train` to teach it.")


def cmd_train(args):
    from .trainer import train

    train(args.stories, args.model_dir, size=args.size, steps=args.steps,
          batch_size=args.batch_size, lr=args.lr, vocab_size=args.vocab_size,
          fresh=args.fresh, holdout=not args.no_holdout, eval_every=args.eval_every,
          seed=args.seed)


def load_model_and_tokenizer(model_dir):
    from .checkpoint import load_checkpoint
    from .tokenizer import Tokenizer

    model_dir = Path(model_dir)
    if not (model_dir / "model.npz").exists():
        raise SystemExit("No trained model yet. Feed it stories, then run:  python story.py train")
    model, _, meta = load_checkpoint(model_dir / "model.npz")
    return model, Tokenizer.load(model_dir / "tokenizer.json"), meta


def cmd_write(args):
    from .tokenizer import END_OF_STORY

    model, tokenizer, meta = load_model_and_tokenizer(args.model_dir)
    eos = tokenizer.special_ids[END_OF_STORY]
    rng = np.random.default_rng(args.seed)
    prompt = args.prompt.rstrip(" ")  # tokens carry their own leading space

    for n in range(args.count):
        if args.count > 1:
            print(f"\n=== Story {n + 1} ===\n")
        print(prompt, end="", flush=True)
        # Tokens can split a multi-byte character; this prints only complete ones.
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        written = 0
        for token in model.generate([eos] + tokenizer.encode(prompt), args.max_tokens,
                                    temperature=args.temperature, top_k=args.top_k,
                                    top_p=args.top_p, stop_id=eos, rng=rng):
            print(decoder.decode(tokenizer.vocab[token]), end="", flush=True)
            written += 1
        print(decoder.decode(b"", final=True))
        if written == args.max_tokens:
            print("\n[...cut off at --max-tokens; raise it for longer stories]")
        else:
            print("\n— The End —")


def cmd_tokens(args):
    from .tokenizer import Tokenizer

    path = Path(args.model_dir) / "tokenizer.json"
    if not path.exists():
        raise SystemExit("No tokenizer yet. Run `python story.py train` first.")
    tokenizer = Tokenizer.load(path)
    ids = tokenizer.encode(args.text)
    print(f"{len(args.text)} characters -> {len(ids)} tokens\n")
    print("".join(f"[{tokenizer.decode([i])}]" for i in ids))
    print("\nids:", ids)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="story.py", description="A story-writing AI built completely from scratch.")
    parser.add_argument("--stories", default=DEFAULT_STORIES, help="folder of .txt stories")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR,
                        help="where the tokenizer and trained model are saved")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("feed", help="give the AI a new story to learn from")
    p.add_argument("--file", help="copy this .txt file in instead of typing/pasting")
    p.set_defaults(func=cmd_feed)

    p = sub.add_parser("train", help="teach the AI using your stories")
    p.add_argument("--size", choices=["tiny", "small", "medium"],
                   help="brain size for a new model (default: tiny)")
    p.add_argument("--steps", type=int, default=2000, help="how many training steps to run")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    p.add_argument("--vocab-size", type=int, help="tokenizer vocabulary size (default: picked from data size)")
    p.add_argument("--fresh", action="store_true",
                   help="throw away the current tokenizer and model and start from zero")
    p.add_argument("--no-holdout", action="store_true",
                   help="train on every story instead of keeping ~10%% aside as a quiz")
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--seed", type=int)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("write", help="have the AI write a story")
    p.add_argument("prompt", nargs="?", default="", help="how the story should begin (optional)")
    p.add_argument("--count", type=int, default=1, help="how many stories to write")
    p.add_argument("--max-tokens", type=int, default=600)
    p.add_argument("--temperature", type=float, default=0.8,
                   help="creativity: lower is safer, higher is wilder")
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int)
    p.set_defaults(func=cmd_write)

    p = sub.add_parser("tokens", help="see how the tokenizer chops up some text")
    p.add_argument("text")
    p.set_defaults(func=cmd_tokens)

    args = parser.parse_args(argv)
    args.func(args)
