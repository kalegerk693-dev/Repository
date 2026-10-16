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
    from .data import count_new_stories, read_stories, read_text_file, save_story

    if args.file:
        source = Path(args.file)
        if not source.is_file():
            raise SystemExit(f"Can't find the file {source}. Check the name and folder "
                             "(tip: drag the file into this window to paste its path).")
        if source.suffix.lower() in (".rtf", ".doc", ".docx", ".odt", ".pdf", ".pages"):
            raise SystemExit("That's a word-processor file. Save it as plain text (.txt) first.")
        text = read_text_file(source)
        name = source.stem
    else:
        say("Type or paste your story. When you're done, type END on its own line and press Enter.\n")
        lines = []
        try:
            for line in sys.stdin:
                if line.strip().upper() == "END":
                    break
                lines.append(line)
        except KeyboardInterrupt:
            say()  # Ctrl+C also finishes the story, instead of throwing it away
        text, name = "".join(lines), None
    if not text.strip():
        say("Nothing to save.")
        return
    if count_new_stories(args.stories, text) == 0:
        say("That story is already in your collection, so it wasn't added again.")
        return
    dest = save_story(args.stories, text, name=name)
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


def load_writer(model_dir):
    from .writer import NoModelError, StoryWriter

    try:
        return StoryWriter(model_dir)
    except NoModelError as e:
        raise SystemExit(f"{e}\nRun:  python story.py train")


def cmd_write(args):
    from .data import story_sources
    from .originality import CopyChecker, describe

    writer = load_writer(args.model_dir)
    sources = [] if args.no_check else story_sources(args.stories)
    checker = CopyChecker(sources) if sources else None
    rng = np.random.default_rng(args.seed)
    for n in range(args.count):
        if args.count > 1:
            say(f"\n=== Story {n + 1} ===\n")
        story = writer.write(" ".join(args.prompt), args.max_tokens, temperature=args.temperature,
                             top_k=args.top_k, top_p=args.top_p, rng=rng)
        for piece in story:
            print(piece, end="", flush=True)
        say()
        say("\n— The End —" if story.finished
            else "\n[...cut off at --max-tokens; raise it for longer stories]")
        if checker:
            say("\n" + describe(checker.check(story.generated)))


def cmd_check(args):
    from .data import story_sources
    from .originality import MIN_RUN, CopyChecker

    writer = load_writer(args.model_dir)
    sources = story_sources(args.stories)
    if not sources:
        raise SystemExit(f"No stories in {args.stories}/ to compare with.")
    checker = CopyChecker(sources)
    rng = np.random.default_rng(args.seed)
    say(f"Writing {args.count} stories and comparing each one, word for word, with your "
        f"{len(sources)} {'story' if len(sources) == 1 else 'stories'}.\n"
        f"A run of {MIN_RUN}+ identical words in a row counts as copied.\n")
    results = []
    for n in range(args.count):
        story = writer.write("", args.max_tokens, temperature=args.temperature, rng=rng)
        for _ in story:
            pass
        r = checker.check(story.generated)
        results.append(r)
        say(f"  story {n + 1:>2}: {r['verdict']:<13} {round(100 * r['copied_fraction']):>3}% copied, "
            f"longest copied passage {r['longest']} words  ({r['words']} words long)")
    copied = sum(r["copied_fraction"] * r["words"] for r in results)
    total = sum(r["words"] for r in results) or 1
    original = sum(r["verdict"] == "original" for r in results)
    worst = max(results, key=lambda r: r["longest"])
    say(f"\n{original} of {len(results)} stories are original. "
        f"Overall, {100 * copied / total:.1f}% of the words it wrote are copied from your stories.")
    if worst["longest"]:
        say(f"Longest copied passage ({worst['longest']} words, from {worst['source']}):\n"
            f"  “{worst['passage']}”")
    if original == len(results):
        say("\nVerdict: it's writing its own stories, not copying yours.")
    elif original >= len(results) / 2:
        say("\nVerdict: mostly its own writing, with some copying. More stories will help.")
    else:
        say("\nVerdict: it's copying your stories. Feed it more stories, or train for fewer "
            "steps, so it has to learn to write instead of memorising.")


def cmd_tokens(args):
    from .tokenizer import Tokenizer

    path = Path(args.model_dir) / "tokenizer.json"
    if not path.exists():
        raise SystemExit("No tokenizer yet. Run `python story.py train` first.")
    tokenizer = Tokenizer.load(path)
    text = " ".join(args.text)
    ids = tokenizer.encode(text)
    say(f"{len(text)} characters -> {len(ids)} tokens\n")
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

    # --stories and --model-dir work both before and after the command name.
    folders = argparse.ArgumentParser(add_help=False)
    folders.add_argument("--stories", default=argparse.SUPPRESS,
                         help=f"folder of .txt stories (default: {DEFAULT_STORIES})")
    folders.add_argument("--model-dir", default=argparse.SUPPRESS,
                         help=f"where the trained AI is saved (default: {DEFAULT_MODEL_DIR})")
    parser = argparse.ArgumentParser(
        prog="story.py", description="A story-writing AI built completely from scratch.",
        parents=[folders])
    sub = parser.add_subparsers(dest="command", required=True)
    add_command = lambda name, help: sub.add_parser(name, help=help, parents=[folders])

    p = add_command("app", help="open the point-and-click web page (easiest way to use it)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1",
                   help="use 0.0.0.0 to let other devices on your network open it")
    p.set_defaults(func=cmd_app)

    p = add_command("feed", help="give the AI a new story to learn from")
    p.add_argument("--file", help="copy this text file in instead of typing/pasting")
    p.set_defaults(func=cmd_feed)

    p = add_command("train", help="teach the AI using your stories")
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
    p.add_argument("--eval-every", type=int, default=100,
                   help="check progress and save every this many steps (0 = only at the end)")
    p.add_argument("--seed", type=int)
    p.set_defaults(func=cmd_train)

    p = add_command("write", help="have the AI write a story")
    p.add_argument("prompt", nargs="*", help="how the story should begin (optional)")
    p.add_argument("--count", type=int, default=1, help="how many stories to write")
    p.add_argument("--max-tokens", type=int, default=600)
    p.add_argument("--temperature", type=float, default=0.8,
                   help="creativity: lower is safer, higher is wilder")
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--top-p", type=float, default=0.95,
                   help="only pick from the likeliest words covering this much probability (0-1)")
    p.add_argument("--seed", type=int)
    p.add_argument("--no-check", action="store_true",
                   help="don't compare the story with yours for copying")
    p.set_defaults(func=cmd_write)

    p = add_command("check", help="test whether it writes its own stories or copies yours")
    p.add_argument("--count", type=int, default=10, help="how many test stories to write")
    p.add_argument("--max-tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--seed", type=int)
    p.set_defaults(func=cmd_check)

    p = add_command("tokens", help="see how the tokenizer chops up some text")
    p.add_argument("text", nargs="+")
    p.set_defaults(func=cmd_tokens)

    args = parser.parse_args(argv)
    # (Defaults are filled in here: argparse shares option objects between the main
    # parser and the commands, so a parser default would overwrite a typed value.)
    args.stories = getattr(args, "stories", DEFAULT_STORIES)
    args.model_dir = getattr(args, "model_dir", DEFAULT_MODEL_DIR)
    args.func(args)
