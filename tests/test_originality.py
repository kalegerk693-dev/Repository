import tempfile
import unittest
from pathlib import Path

from storyai.originality import CopyChecker, checker_sources, describe, record_trained_stories

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


MIRA = ("Mira kept a jar of thunder under her bed. Every night it rumbled softly, and every "
        "morning she fed it a spoonful of rain from the gutter. One stormy afternoon the lid "
        "cracked, and the thunder rolled out across the meadow, shaking apples from the trees "
        "and waking the sleepy cows. Mira chased it all the way to the hills, where it curled "
        "up inside a cloud and purred like a cat. She went home with an empty jar, but every "
        "summer evening, when the sky grumbled, she knew it was saying hello.")
OPENING = "Once upon a time, there was a little girl named Lily. She loved to play outside."


class AttacksOnTheChecker(unittest.TestCase):
    """Ways of copying (or of looking like copying) that the checker must get right."""

    def test_a_long_copied_passage_is_never_called_original(self):
        own = " ".join(f"word{i}" for i in range(180))
        copied = " ".join(MIRA.split()[10:39])  # 29 words
        r = CopyChecker([("mira.txt", MIRA)]).check(f"{own} {copied}")
        self.assertEqual(r["longest"], 29)
        self.assertNotEqual(r["verdict"], "original")

    def test_stock_phrases_shared_by_several_stories_are_not_copying(self):
        sources = [(f"s{i}.txt", f"{OPENING} {plot}") for i, plot in enumerate([
            "One day she found a red ball in the park and gave it to her friend Sam.",
            "Her mom made cookies, and Lily shared them with the birds in the yard.",
            "When it rained she jumped in puddles with her brave dog called Max."])]
        mine = (f"{OPENING} She met a frog who wanted to learn to sing, so Lily taught him a "
                "song about the moon, and they sang it together until the stars came out.")
        r = CopyChecker(sources).check(mine)
        self.assertEqual(r["verdict"], "original", describe(r))
        # ...but copying one particular story still counts.
        self.assertNotEqual(CopyChecker(sources).check(sources[1][1])["verdict"], "original")

    def test_copies_of_a_story_that_is_in_the_folder_twice_still_count(self):
        draft = MIRA.replace("purred like a cat", "slept like a stone")
        copy = " ".join(MIRA.split()[5:29])  # 24 words
        for sources in ([("mira.txt", MIRA), ("backup/mira.txt", MIRA)],
                        [("mira.txt", MIRA), ("mira_draft.txt", draft)]):
            r = CopyChecker(sources).check("A goose waddled by and honked twice. " + copy)
            self.assertNotEqual(r["verdict"], "original", (sources[1][0], describe(r)))
            tweaked = " ".join("zebra" if i % 7 == 6 else w for i, w in enumerate(MIRA.split()))
            self.assertNotEqual(CopyChecker(sources).check(tweaked)["verdict"], "original")

    def test_one_shared_sentence_in_a_short_story_is_not_copying(self):
        sources = [("park.txt", "One day, he went to the park with his mom. They fed the ducks "
                                "and ate ice cream on a bench by the pond.")]
        mine = ("Ben liked rainy days best. One day, he went to the park with his mom. Big drops "
                "fell, and Ben jumped in every puddle he could find until his boots were full of "
                "water and his mom was laughing so hard she had to sit down on the wet grass. "
                "On the way home they saw a rainbow over the bakery, and Ben said it was the best "
                "puddle day of the whole year.")
        self.assertEqual(CopyChecker(sources).check(mine)["verdict"], "original")

    def test_a_copy_with_a_word_changed_here_and_there_is_caught(self):
        words = MIRA.split()
        for every in (5, 7):
            tweaked = " ".join("zebra" if i % every == every - 1 else w for i, w in enumerate(words))
            r = CopyChecker([("mira.txt", MIRA)]).check(tweaked)
            self.assertEqual(r["verdict"], "copied", (every, describe(r)))
            self.assertIn("mira.txt", describe(r))

    def test_apostrophes_and_quotes_do_not_hide_a_copy(self):
        story = "“Don’t touch Dad’s boat,” she said. “It’s the only one we’ve got, and it’s old.”"
        plain = story.replace("’", "'").replace("“", '"').replace("”", '"')
        self.assertEqual(CopyChecker([("boat.txt", story)]).check(plain)["verdict"], "copied")

    def test_a_long_copy_after_a_very_common_header_is_measured_in_full(self):
        header = "This story was written for the village library so that every child could read it."
        sources = [(f"t{i}.txt", f"{header} Tale number {i} is about something else entirely.")
                   for i in range(60)]
        sources.append(("lighthouse.txt", f"{header} {MIRA}"))
        r = CopyChecker(sources).check(f"Hello there. {header} {MIRA}")
        self.assertGreaterEqual(r["longest"], len(MIRA.split()))
        self.assertEqual(r["source"], "lighthouse.txt")

    def test_whole_tiny_stories_recited_word_for_word(self):
        poems = [("poem1.txt", "Red sky at night, sailor's delight."), ("poem2.txt", "Snow falls without a sound")]
        r = CopyChecker(poems).check("Red sky at night, sailor's delight. Snow falls without a sound.")
        self.assertEqual(r["verdict"], "copied")

    def test_quoted_passage_keeps_its_letters(self):
        r = CopyChecker([("mira.txt", MIRA)]).check("İzmir İpek İlker came along. " + MIRA)
        self.assertTrue(r["passage"].startswith("Mira kept a jar of thunder"), r["passage"])

    def test_repeated_laughter_is_not_copying(self):
        checker = CopyChecker([("ha.txt", "Ha ha ha ha ha ha ha ha ha ha ha ha, laughed the giant.")])
        r = checker.check("The goblin told a joke and the troll went ha ha ha ha ha ha ha ha ha until he cried.")
        self.assertEqual(r["verdict"], "original", describe(r))

    def test_stories_removed_after_training_are_still_checked(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "stories").mkdir()
            (d / "stories" / "mira.txt").write_text(MIRA)
            (d / "model").mkdir()
            record_trained_stories(d / "model", [("mira.txt", MIRA)])
            (d / "stories" / "mira.txt").unlink()
            sources = checker_sources(d / "stories", d / "model")
            self.assertEqual(sources, [("mira.txt (since removed)", MIRA)])
            self.assertEqual(CopyChecker(sources).check(MIRA)["verdict"], "copied")


if __name__ == "__main__":
    unittest.main()
