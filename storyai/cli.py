"""Command line: feed stories, train the AI, and have it write new ones."""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STORIES = ROOT / "stories"
DEFAULT_MODEL_DIR = ROOT / "checkpoints"


def say(message=""):
    print(message, flush=True)


def cmd_feed(args):
    from .data import read_stories, save_story

    if args.file:
        source = Path(args.file)
        if not source.is_file():
            raise SystemExit(f"Can't find the file {source}")
        text = source.read_text(encoding="utf-8-sig", errors="replace").replace("\r\n", "\n")
        if not text.strip():
            raise SystemExit(f"{source} is empty.")
        dest = save_story(args.stories, text, name=source.stem)
        say(f"Copied {source} to {dest}")
    else:
        say("Type or paste your story. When you're done, type END on its own line and press Enter.\n")
        lines = []
        for line in sys.stdin:
            if line.strip() == "END":
                break
            lines.append(line)
        text = "".join(lines).strip()
        if not text:
            say("Nothing to save.")
            return
        dest = save_story(args.stories, text)
        say(f"\nSaved your story ({len(text.split()):,} words) to {dest}")
    n = len(read_stories(args.stories))
    say(f"Your AI has {n} {'story' if n == 1 else 'stories'} to learn from. "
        "Run `python story.py train` to teach it.")


def cmd_train(args):
    from .trainer import TrainingError, train

    try:
        train(args.stories, args.model_dir, size=args.size, steps=args.steps,
              batch_size=args.batch_size, lr=args.lr, vocab_size=args.vocab_size,
              fresh=args.fresh, holdout=not args.no_holdout, eval_every=args.eval_every,
              seed=args.seed, log=say)
    except TrainingError as e:
        raise SystemExit(f"{e}\nPut .txt files in {args.stories}/ or run:  python story.py feed")
    say("Make it write with:  python story.py write")


def cmd_write(args):
    from .writer import NoModelError, StoryWriter

    try:
        writer = StoryWriter(args.model_dir)
    except NoModelError as e:
        raise SystemExit(f"{e}\nRun:  python story.py train")
    rng = np.random.default_rng(args.seed)
    for n in range(args.count):
        if args.count > 1:
            say(f"\n=== Story {n + 1} ===\n")
        story = writer.write(args.prompt, args.max_tokens, temperature=args.temperature,
                             top_k=args.top_k, top_p=args.top_p, rng=rng)
        for piece in story:
            print(piece, end="", flush=True)
        say()
        say("\n— The End —" if story.finished
            else "\n[...cut off at --max-tokens; raise it for longer stories]")


def cmd_tokens(args):
    from .tokenizer import Tokenizer

    path = Path(args.model_dir) / "tokenizer.json"
    if not path.exists():
        raise SystemExit("No tokenizer yet. Run `python story.py train` first.")
    tokenizer = Tokenizer.load(path)
    ids = tokenizer.encode(args.text)
    say(f"{len(args.text)} characters -> {len(ids)} tokens\n")
    say("".join(f"[{tokenizer.decode([i])}]" for i in ids))
    say(f"\nids: {ids}")


def cmd_app(args):
    from .webapp import serve

    serve(args.stories, args.model_dir, host=args.host, port=args.port)


def main(argv=None):
    # Never crash on a character the terminal can't show (e.g. emoji on older Windows consoles).
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(
        prog="story.py", description="A story-writing AI built completely from scratch.")
    parser.add_argument("--stories", default=DEFAULT_STORIES, help="folder of .txt stories")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR,
                        help="where the tokenizer and trained model are saved")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("app", help="open the point-and-click web page (easiest way to use it)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1",
                   help="use 0.0.0.0 to let other devices on your network open it")
    p.set_defaults(func=cmd_app)

    p = sub.add_parser("feed", help="give the AI a new story to learn from")
    p.add_argument("--file", help="copy this text file in instead of typing/pasting")
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
