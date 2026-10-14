import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from storyai.cli import main
from storyai.data import count_new_stories, read_stories, read_text_file
from storyai.tokenizer import END_OF_STORY

STORY = "Once upon a time a café owner named Zoë baked “perfect” bread — every day."


def run(*argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        main([str(a) for a in argv])
    return out.getvalue()


class ReadingStories(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_any_encoding(self):
        for name, data in {
            "utf8.txt": STORY.encode("utf-8"),
            "utf8_bom.txt": b"\xef\xbb\xbf" + STORY.encode("utf-8"),
            "utf16.txt": STORY.encode("utf-16"),  # Notepad "Unicode"
            "ansi.txt": STORY.encode("cp1252"),   # Notepad "ANSI"
            "crlf.txt": (STORY + "\r\n\r\nThe end.").encode("utf-8"),
        }.items():
            (self.dir / name).write_bytes(data)
            text = read_text_file(self.dir / name)
            self.assertTrue(text.startswith(STORY), name)
            self.assertNotIn("\r", text)
            self.assertNotIn("﻿", text)

    def test_which_files_count(self):
        (self.dir / "a.txt").write_text("Story A")
        (self.dir / "B.TXT").write_text("Story B")
        (self.dir / "notes.md").write_text("not a story")
        (self.dir / "._a.txt").write_bytes(b"\x00\x05\x16\x07 Mac junk")
        (self.dir / "__MACOSX").mkdir()
        (self.dir / "__MACOSX" / "c.txt").write_text("junk")
        (self.dir / "folder.txt").mkdir()
        self.assertEqual(sorted(read_stories(self.dir)), ["Story A", "Story B"])
        self.assertEqual(read_stories(self.dir / "a.txt"), ["Story A"])  # a single file works too

    def test_separators(self):
        (self.dir / "many.txt").write_text(f"One\n{END_OF_STORY}\nTwo\n<|endoftext|>\nThree\n")
        self.assertEqual(read_stories(self.dir), ["One", "Two", "Three"])
        self.assertEqual(count_new_stories(self.dir, f"Two{END_OF_STORY}Four"), 1)


class CommandLine(unittest.TestCase):
    def test_feed_variants(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            source = d / "Notepad Story.txt"
            source.write_bytes(STORY.encode("utf-16"))
            # --stories may come after the command name too
            out = run("feed", "--file", source, "--stories", d / "s")
            self.assertIn("1 story to learn from", out)
            self.assertEqual(read_stories(d / "s"), [STORY])
            self.assertIn("already in your collection", run("feed", "--file", source, "--stories", d / "s"))
            (d / "story.docx").write_bytes(b"PK")
            with self.assertRaises(SystemExit):
                run("--stories", d / "s", "feed", "--file", d / "story.docx")
            with self.assertRaises(SystemExit):
                run("--stories", d / "s", "feed", "--file", d / "missing.txt")

    def test_train_guards(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "s").mkdir()
            (d / "s" / "fox.txt").write_text("The fox ran to the river. " * 40)
            common = ["--stories", d / "s", "--model-dir", d / "m"]
            run(*common, "train", "--steps", 2, "--batch-size", 2, "--eval-every", 0, "--seed", 0)
            # A bad vocab size is refused before --fresh deletes anything.
            with self.assertRaises(SystemExit):
                run(*common, "train", "--fresh", "--vocab-size", 100)
            self.assertTrue((d / "m" / "model.npz").exists())
            # Unquoted multi-word prompts work.
            self.assertTrue(run(*common, "write", "The", "fox", "--max-tokens", 3).startswith("The fox"))
            # A model without its tokenizer is reported, not crashed on.
            (d / "m" / "tokenizer.json").unlink()
            with self.assertRaises(SystemExit):
                run(*common, "train", "--steps", 1)
            with self.assertRaises(SystemExit):
                run(*common, "write")


    def test_quiz_never_contains_studied_stories(self):
        from storyai.data import is_holdout
        stories = [f"Story {i}: the fox found {i} shiny stones by the river." for i in range(60)]
        quiz_ones = [t for t in stories if is_holdout(t)]
        self.assertTrue(quiz_ones)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "s").mkdir()
            (d / "s" / "all.txt").write_text(f"\n{END_OF_STORY}\n".join(stories[:30]))
            common = ["--stories", d / "s", "--model-dir", d / "m"]
            out = run(*common, "train", "--steps", 1, "--batch-size", 2, "--no-holdout")
            self.assertIn("Quiz turned off", out)
            # Every story was studied, so none of them can be quiz material now.
            out = run(*common, "train", "--steps", 1, "--batch-size", 2)
            self.assertIn("none of the stories picked for it are new", out)
            # New stories can be quizzed on.
            (d / "s" / "more.txt").write_text(f"\n{END_OF_STORY}\n".join(stories[30:]))
            new_quiz = [t for t in stories[30:] if is_holdout(t)]
            out = run(*common, "train", "--steps", 1, "--batch-size", 2)
            if new_quiz:
                self.assertIn(f"quizzing on {len(new_quiz)} held-out stories", out)


if __name__ == "__main__":
    unittest.main()
