"""Verify final on-disk dataset labels, pixels, splits and provenance."""
import collections
import hashlib
import json
from pathlib import Path
from PIL import Image
from train_ppe import validate_dataset, write_json

ROOT = Path(__file__).resolve().parent / "datasets"


def main():
    pending = ROOT / "merged_verified"
    if pending.exists():
        if not (pending / "merge_report.json").exists():
            raise RuntimeError("Merge still running; wait for merge_report.json")
        # Retain initial output for audit rather than deleting user data.
        if (ROOT / "merged").exists():
            (ROOT / "merged").rename(ROOT / "merged_initial")
        pending.rename(ROOT / "merged")
        import yaml
        config = ROOT / "merged/data.yaml"
        data = yaml.safe_load(config.read_text())
        data["path"] = str(ROOT / "merged")
        config.write_text(yaml.safe_dump(data, sort_keys=False))
    output = ROOT / "merged"
    _, report = validate_dataset(output / "data.yaml")
    seen, groups = {}, {}
    provenance = json.loads((output / "provenance.json").read_text())
    for record in provenance:
        image = output / record["image"]
        assert all("result" not in Path(s["image"]).parts for s in record["sources"])
        with Image.open(image) as im:
            rgb = im.convert("RGB")
            digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
        assert digest == record["digest"]
        assert digest not in seen, "Duplicate pixels in merged dataset"
        seen[digest] = record["split"]
        for source in record["sources"]:
            assert Path(source["image"]).is_file()
        if record["group"] in groups:
            assert groups[record["group"]] == record["split"], "Group leaked across splits"
        groups[record["group"]] = record["split"]
    assert sum(s["images"] for s in report.values()) == len(provenance)
    result = dict(status="passed", images=len(provenance), dataset_checks=report,
                  checks=["normalized labels", "class IDs", "pixel deduplication", "provenance",
                          "no result images", "group split isolation"])
    write_json(output / "verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()