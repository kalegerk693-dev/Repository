import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from storyai.cli import main

STORY = """Once upon a time a small fox lived at the edge of a dark wood.
Every morning the fox ran to the river to watch the fish jump.
One day the fox found a golden key in the mud, and the adventure began.
"""


def run(*argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        main([str(a) for a in argv])
    return out.getvalue()


class EndToEnd(unittest.TestCase):
    def test_feed_train_write(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            stories, model_dir = d / "stories", d / "model"
            source = d / "fox.txt"
            source.write_text(STORY * 3)
            common = ["--stories", stories, "--model-dir", model_dir]

            self.assertIn("1 story to learn from", run(*common, "feed", "--file", source))

            out = run(*common, "train", "--steps", 3, "--eval-every", 2, "--batch-size", 2,
                      "--vocab-size", 300, "--seed", 0)
            self.assertIn("Created a new 'tiny' model", out)
            self.assertTrue((model_dir / "model.npz").exists())
            self.assertTrue((model_dir / "tokenizer.json").exists())

            out = run(*common, "train", "--steps", 2, "--batch-size", 2, "--seed", 0)
            self.assertIn("Continuing to train your model (3 steps so far)", out)

            out = run(*common, "write", "Once upon a time", "--max-tokens", 5, "--seed", 0)
            self.assertTrue(out.startswith("Once upon a time"))

            out = run(*common, "tokens", "the fox")
            self.assertIn("[ fox]", out)  # " fox" is common enough to be one token


if __name__ == "__main__":
    unittest.main()
