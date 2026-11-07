import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_ppe import PlateauMonitor, select_checkpoint_by_validation, validate_dataset, write_json


class MonitorTests(unittest.TestCase):
    def test_plateau_and_ties(self):
        monitor = PlateauMonitor(2)
        self.assertFalse(monitor.update(1, 0.1))
        self.assertFalse(monitor.update(2, 0.2))
        self.assertFalse(monitor.update(3, 0.2))
        self.assertTrue(monitor.update(4, 0.19))
        self.assertEqual(monitor.best_epoch, 2)
        self.assertEqual(monitor.best, 0.2)

    def test_improvement_resets_counter(self):
        monitor = PlateauMonitor(2)
        for epoch, value in enumerate([0, 0, 0.1, 0.1, 0.2], 1):
            self.assertFalse(monitor.update(epoch, value))
        self.assertEqual(monitor.bad_epochs, 0)

    def test_repeated_final_callback_is_ignored(self):
        monitor = PlateauMonitor(1)
        monitor.update(1, 0.3)
        monitor.update(1, 0.1)
        self.assertEqual(monitor.bad_epochs, 0)
        self.assertEqual(monitor.best, 0.3)

    def test_invalid_values(self):
        with self.assertRaises(ValueError):
            PlateauMonitor(0)
        for value in [float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                PlateauMonitor(1).update(1, value)

    def test_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.json"
            write_json(path, {"mAP": 0.75})
            self.assertEqual(json.loads(path.read_text()), {"mAP": 0.75})

    def test_select_checkpoint_by_validation_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "weights").mkdir()
            (run / "weights" / "best.pt").write_bytes(b"native")
            (run / "weights" / "epoch1.pt").write_bytes(b"epoch1")
            (run / "weights" / "epoch2.pt").write_bytes(b"epoch2")
            (run / "results.csv").write_text(
                "epoch,metrics/mAP50-95(B)\n0,0.40\n1,0.70\n"
            )
            selected, details = select_checkpoint_by_validation(run)
            self.assertEqual(selected.name, "best_val.pt")
            self.assertEqual(selected.read_bytes(), b"epoch2")
            self.assertEqual(details["epoch"], 2)

    def test_select_checkpoint_falls_back_without_epoch_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "weights").mkdir()
            native = run / "weights" / "best.pt"
            native.write_bytes(b"native")
            selected, details = select_checkpoint_by_validation(run)
            self.assertEqual(selected, native)
            self.assertEqual(details["source"], "ultralytics_best_fallback")


@unittest.skipUnless(importlib.util.find_spec("yaml"), "Install requirements.txt for dataset tests")
class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = self.root / "data.yaml"
        self.config.write_text("train: images/train\nval: images/val\nnames: [person, helmet]\n")
        for split in ["train", "val"]:
            (self.root / "images" / split).mkdir(parents=True)
            (self.root / "labels" / split).mkdir(parents=True)
            (self.root / "images" / split / "a.jpg").write_bytes(split.encode())
            (self.root / "labels" / split / "a.txt").write_text("0 0.5 0.5 0.4 0.4\n")

    def test_valid(self):
        data, report = validate_dataset(self.config)
        self.assertEqual(report["train"]["objects_per_class"]["0"], 1)
        self.assertTrue(Path(data["train"]).is_absolute())

    def test_leakage(self):
        (self.root / "images/val/a.jpg").write_bytes(b"train")
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_dataset(self.config)

    def test_invalid_class(self):
        (self.root / "labels/val/a.txt").write_text("9 0.5 0.5 0.4 0.4")
        with self.assertRaisesRegex(ValueError, "invalid class"):
            validate_dataset(self.config)

    def test_outside_box(self):
        (self.root / "labels/val/a.txt").write_text("0 0.1 0.5 0.5 0.4")
        with self.assertRaisesRegex(ValueError, "outside"):
            validate_dataset(self.config)

    def test_no_labels(self):
        (self.root / "labels/val/a.txt").unlink()
        with self.assertRaisesRegex(ValueError, "no labeled"):
            validate_dataset(self.config)


if __name__ == "__main__":
    unittest.main()