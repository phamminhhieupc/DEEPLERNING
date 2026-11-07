from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from merge_datasets import checked_box, pixel_box, group_key, split_for, NAMES


class MergeTests(unittest.TestCase):
    def test_head_not_violation(self):
        self.assertNotEqual(NAMES.index("head"), NAMES.index("no_helmet"))

    def test_person_with_helmet_not_helmet(self):
        self.assertNotEqual(NAMES.index("person_with_helmet"), NAMES.index("helmet"))

    def test_pixel_conversion(self):
        self.assertEqual(pixel_box("helmet", [10, 20, 50, 60], 100, 100),
                         (NAMES.index("helmet"), 0.3, 0.4, 0.4, 0.4000000000000001))

    def test_invalid_box(self):
        for box in [(0.5, 0.5, 0, 1), (0.1, 0.1, 0.8, 0.8), (float("nan"), 0.5, 0.1, 0.1)]:
            with self.assertRaises(ValueError):
                checked_box("helmet", box)

    def test_video_group(self):
        self.assertEqual(group_key(Path("Inside-merge_mov-58_jpg.rf.abcd.jpg")),
                         group_key(Path("Inside-merge_mov-90_jpg.rf.efgh.jpg")))

    def test_reproducible_split(self):
        self.assertEqual(split_for("same"), split_for("same"))
        self.assertIn(split_for("same"), ["train", "val", "test"])


if __name__ == "__main__":
    unittest.main()