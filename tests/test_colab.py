import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from colab_utils import pack_dataset, prepare_dataset, run_training


class ColabTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "data.yaml").write_text(
            f"path: {self.source}\ntrain: images/train\nval: images/val\nnames: [person]\n")
        for split in ("train", "val"):
            (self.source / "images" / split).mkdir(parents=True)
            (self.source / "labels" / split).mkdir(parents=True)
            (self.source / "images" / split / "a.jpg").write_bytes(split.encode())
            (self.source / "labels" / split / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")

    def test_roundtrip_and_relocation(self):
        archive = self.root / "dataset.zip"
        info = pack_dataset(self.source, archive)
        prepared = prepare_dataset(archive, self.root / "colab", info["sha256"])
        self.assertEqual(prepared["report"]["train"]["images"], 1)
        self.assertIn(str(self.root / "colab"), Path(prepared["data"]).read_text())
        self.assertTrue(archive.with_suffix(".zip.sha256").is_file())
        with self.assertRaises(FileExistsError):
            prepare_dataset(archive, self.root / "colab")

    def test_checksum(self):
        archive = self.root / "dataset.zip"
        pack_dataset(self.source, archive)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            prepare_dataset(archive, self.root / "out", "bad")

    def test_zip_traversal(self):
        archive = self.root / "unsafe.zip"
        with zipfile.ZipFile(archive, "w") as zipped:
            zipped.writestr("../escaped.txt", "unsafe")
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            prepare_dataset(archive, self.root / "out")
        self.assertFalse((self.root / "escaped.txt").exists())

    def test_run_name_validation(self):
        with self.assertRaises(ValueError):
            run_training(self.root, "data.yaml", self.root, "../bad")

    def test_logged_subprocess(self):
        repo = self.root / "repo"
        repo.mkdir()
        (repo / "train_ppe.py").write_text(
            "import argparse\nfrom pathlib import Path\n"
            "p=argparse.ArgumentParser()\np.add_argument('--project')\np.add_argument('--name')\n"
            "a,_=p.parse_known_args()\n(Path(a.project)/a.name).mkdir()\nprint('training smoke log')\n")
        project = self.root / "runs"
        output = run_training(repo, self.source / "data.yaml", project, "smoke", device="cpu")
        self.assertTrue(output.is_dir())
        self.assertIn("training smoke log", (project / "console_smoke.log").read_text())

    def test_failed_subprocess_is_reported(self):
        repo = self.root / "repo"
        repo.mkdir()
        (repo / "train_ppe.py").write_text("print('failed smoke log')\nraise SystemExit(2)\n")
        with self.assertRaisesRegex(RuntimeError, "exited with 2"):
            run_training(repo, self.source / "data.yaml", self.root / "runs", "failed", device="cpu")

    def test_notebook_python_syntax(self):
        notebook = json.loads((Path(__file__).resolve().parents[1] / "train_ppe_colab.ipynb").read_text())
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"cell_{index}", "exec")


if __name__ == "__main__":
    unittest.main()