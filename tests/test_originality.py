import unittest

from storyai.originality import CopyChecker, describe

FOX = ("Once upon a time a small fox lived at the edge of a dark wood. Every morning the fox "
       "ran down to the river to watch the silver fish jump, and every evening she counted "
       "the stars until she fell asleep under the old oak tree.")
OWL = ("The owl had never seen the sea. One winter night she packed a bag of acorns, said "
       "goodbye to her brothers, and flew south over the mountains until the air smelled of salt.")


class CopyCheckerTest(unittest.TestCase):
    def setUp(self):
        self.checker = CopyChecker([("fox.txt", FOX), ("owl.txt", OWL)])

    def test_exact_copy(self):
        r = self.checker.check(FOX)
        self.assertEqual(r["verdict"], "copied")
        self.assertEqual(r["copied_fraction"], 1.0)
        self.assertEqual(r["source"], "fox.txt")
        self.assertEqual(r["passage"], FOX[:-1])  # as written, minus the final "."

    def test_copy_with_different_case_and_punctuation(self):
        sneaky = FOX.upper().replace(",", ";").replace(".", "!!")
        self.assertEqual(self.checker.check(sneaky)["verdict"], "copied")

    def test_copied_passage_inside_new_text(self):
        text = ("Far away, a tired bear woke up hungry and grumpy and wandered off. "
                "Every morning the fox ran down to the river to watch the silver fish jump. "
                "The bear followed, hoping for breakfast, but the clever fish hid under rocks.")
        r = self.checker.check(text)
        self.assertEqual(r["longest"], 15)
        self.assertEqual(r["passage"], "Every morning the fox ran down to the river to watch the silver fish jump")
        self.assertEqual(r["source"], "fox.txt")
        self.assertEqual(r["verdict"], "partly copied")

    def test_original_text(self):
        text = ("A dragon who was afraid of the dark lived in a lighthouse, and every night he "
                "polished the great lamp until it shone so brightly that ships sailed home safely.")
        r = self.checker.check(text)
        self.assertEqual((r["verdict"], r["longest"], r["copied_fraction"]), ("original", 0, 0.0))
        self.assertIn("Original", describe(r))

    def test_short_shared_phrases_are_not_copying(self):
        r = self.checker.check("Once upon a time a small fox found a golden key in the snow.")
        self.assertEqual(r["longest"], 0)  # "once upon a time a small fox" is only 7 words

    def test_runs_do_not_cross_stories_or_unknown_words(self):
        end_fox_start_owl = "fell asleep under the old oak tree The owl had never seen the sea"
        self.assertLess(self.checker.check(end_fox_start_owl)["longest"], 15)
        self.assertEqual(self.checker.check("the fox ran down to the zzz river to watch")["longest"], 0)

    def test_empty(self):
        self.assertEqual(self.checker.check("")["verdict"], "original")
        self.assertEqual(CopyChecker([]).check(FOX)["longest"], 0)


if __name__ == "__main__":
    unittest.main()
