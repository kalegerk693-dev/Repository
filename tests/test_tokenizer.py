import tempfile
import unittest
from pathlib import Path

from storyai.tokenizer import END_OF_STORY, SPLIT_PATTERN, Tokenizer

SAMPLE = """Once upon a time, there was a little dragon named Ember.
Ember loved the sea. Every day the little dragon flew over the sea and sang.
"I'll find the treasure," said Ember. They're brave; we've seen it.
The end... or is it? 42 dragons, 3.14 pies, snake_case_name, naïve café, 你好, 🐉🌊!
"""


class TokenizerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = Tokenizer.train([SAMPLE] * 3, vocab_size=300)

    def test_split_pattern_keeps_every_character(self):
        weird = SAMPLE + "  \t\n\n   x__y   ́ ²½ \r\n end   "
        self.assertEqual("".join(SPLIT_PATTERN.findall(weird)), weird)

    def test_roundtrip(self):
        for text in [SAMPLE, "", " ", "Unseen wörds & symbols ∑ 🦄\n\n", "a" * 50, "x__y", END_OF_STORY]:
            self.assertEqual(self.tok.decode(self.tok.encode(text)), text)

    def test_vocab_size_and_compression(self):
        self.assertEqual(self.tok.vocab_size, 300)
        self.assertLess(len(self.tok.encode(SAMPLE)), len(SAMPLE.encode("utf-8")) * 0.7)
        self.assertEqual(len(self.tok.encode(" dragon")), 1)

    def test_special_token(self):
        ids = self.tok.encode(f"The end.{END_OF_STORY}Once")
        self.assertIn(self.tok.special_ids[END_OF_STORY], ids)
        self.assertEqual(self.tok.special_ids[END_OF_STORY], self.tok.vocab_size - 1)
        self.assertNotIn(self.tok.special_ids[END_OF_STORY], self.tok.encode_ordinary(END_OF_STORY))

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "tok.json"
            self.tok.save(path)
            loaded = Tokenizer.load(path)
        self.assertEqual(loaded.merges, self.tok.merges)
        self.assertEqual(loaded.encode(SAMPLE), self.tok.encode(SAMPLE))

    def test_incremental_training_matches_naive_bpe(self):
        """The fast incremental trainer must learn exactly what plain, slow BPE learns."""
        text = SAMPLE * 2
        chunks = [list(c.encode("utf-8")) for c in SPLIT_PATTERN.findall(text)]
        naive = []
        for _ in range(60):
            counts = {}
            for w in chunks:
                for p in zip(w, w[1:]):
                    counts[p] = counts.get(p, 0) + 1
            best = min(counts, key=lambda p: (-counts[p], p))
            if counts[best] < 2:
                break
            naive.append(best)
            new_id = 255 + len(naive)
            merged = []
            for w in chunks:
                out, i = [], 0
                while i < len(w):
                    if i + 1 < len(w) and (w[i], w[i + 1]) == best:
                        out.append(new_id)
                        i += 2
                    else:
                        out.append(w[i])
                        i += 1
                merged.append(out)
            chunks = merged
        fast = Tokenizer.train([text], vocab_size=256 + 60 + 1)
        self.assertEqual(fast.merges, naive)


if __name__ == "__main__":
    unittest.main()
