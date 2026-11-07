"""Conservative dataset merge: explicit class mapping, provenance, deduplication."""
import collections
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET
from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parent / "datasets"
NAMES = ["person", "helmet", "vest", "no_helmet", "no_vest", "gloves", "mask", "no_mask",
         "head", "person_with_helmet"]
JOY = {"Person": "person", "Hardhat": "helmet", "Safety Vest": "vest", "NO-Hardhat": "no_helmet",
       "NO-Safety Vest": "no_vest", "Gloves": "gloves", "Mask": "mask", "NO-Mask": "no_mask"}
VOC = {"helmet": "helmet", "head": "head", "person": "person", "hardhat": "helmet",
       "person hardhat": "person_with_helmet"}


def checked_box(name, xywh):
    x, y, w, h = xywh
    if not all(math.isfinite(v) for v in xywh) or w <= 0 or h <= 0:
        raise ValueError("nonfinite or zero-size box")
    if min(x - w / 2, y - h / 2) < -0.002 or max(x + w / 2, y + h / 2) > 1.002:
        raise ValueError("box outside image")
    # Clip small rounding overflow only, not major malformed boxes.
    left, top = max(0, x - w / 2), max(0, y - h / 2)
    right, bottom = min(1, x + w / 2), min(1, y + h / 2)
    return (NAMES.index(name), (left + right) / 2, (top + bottom) / 2, right - left, bottom - top)


def pixel_box(name, coordinates, width, height):
    x1, y1, x2, y2 = coordinates
    return checked_box(name, ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
                              (x2 - x1) / width, (y2 - y1) / height))


def group_key(path):
    stem = re.split(r"[._]rf[._]", path.stem)[0]
    # Keep exported augmentations and identifiable video frames together.
    stem = re.sub(r"[._](?:jpg|png|jpeg)$", "", stem, flags=re.I)
    match = re.match(r"(.*?(?:mov|mp4|avi|webm))(?:[-_].*)?$", stem, re.I)
    return match[1].lower() if match else stem.lower()


def split_for(group):
    bucket = int(hashlib.sha256(("42:" + group).encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 70 else "val" if bucket < 90 else "test"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "merged")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    raw = ROOT / "raw"
    records, issues = [], []

    def add(image, source, boxes, annotation):
        try:
            with Image.open(image) as im:
                rgb = im.convert("RGB")
                rgb.load()
                digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            records.append(dict(image=image, source=source, boxes=boxes, annotation=str(annotation),
                                digest=digest, group=group_key(image)))
        except Exception as exc:
            issues.append(dict(image=str(image), reason=str(exc)))

    for source, base, names in [
        ("ilaydadeger", raw / "ilaydadeger/dataset_small", ["head"]),
        ("JoyKarmoker", raw / "JoyKarmoker/Train/Construction_Site_Safety", None),
    ]:
        if names is None:
            names = yaml.safe_load((base / "data.yaml").read_text())["names"]
        for image in sorted(base.rglob("*.jpg")):
            label = image.parent.parent / "labels" / (image.stem + ".txt")
            if not label.is_file():
                issues.append(dict(image=str(image), reason="missing label; not assumed background"))
                continue
            try:
                boxes = []
                for line in label.read_text().splitlines():
                    if not line.strip():
                        continue
                    fields = line.split()
                    if len(fields) != 5:
                        raise ValueError("not a detection label")
                    name = names[int(fields[0])]
                    target = JOY.get(name) if source == "JoyKarmoker" else "head"
                    if target:
                        boxes.append(checked_box(target, tuple(map(float, fields[1:]))))
                add(image, source, boxes, label)
            except Exception as exc:
                issues.append(dict(image=str(image), reason=str(exc)))

    for source, base in [("usckiki82", raw / "usckiki82"), ("drive", raw / "drive")]:
        index = collections.defaultdict(list)
        for image in base.rglob("*"):
            if image.suffix.lower() in {".jpg", ".jpeg", ".png"} and "result" not in image.parts:
                index[image.name].append(image)
        annotated = set()
        for annotation in sorted(base.rglob("*.xml")):
            try:
                doc = ET.parse(annotation).getroot()
                filename = Path(doc.findtext("filename").replace("\\", "/")).name
                candidates = index[filename]
                if not candidates:
                    raise ValueError("XML image unavailable")
                # Never silently pair same-name images with different pixel data.
                image = candidates[0]
                if len({hashlib.sha256(p.read_bytes()).hexdigest() for p in candidates}) > 1:
                    raise ValueError("ambiguous image filename")
                with Image.open(image) as im:
                    width, height = im.size
                if (int(doc.findtext("size/width")), int(doc.findtext("size/height"))) != (width, height):
                    raise ValueError("XML dimensions mismatch")
                boxes = []
                for obj in doc.findall("object"):
                    name = obj.findtext("name").strip().lower()
                    if name not in VOC:
                        raise ValueError(f"unmapped XML class {name}")
                    coords = [float(obj.findtext("bndbox/" + k)) for k in ["xmin", "ymin", "xmax", "ymax"]]
                    boxes.append(pixel_box(VOC[name], coords, width, height))
                add(image, source, boxes, annotation)
                annotated.update(candidates)
            except Exception as exc:
                issues.append(dict(annotation=str(annotation), reason=str(exc)))
        # Original Drive training manifests have absolute image paths + pixel boxes.
        if source == "drive":
            for manifest in sorted((base / "model_data").glob("dataset_*.txt")):
                if manifest.name == "dataset_names.txt":
                    continue
                for line_number, line in enumerate(manifest.read_text().splitlines(), 1):
                    try:
                        match = re.match(r"^(.*?\.(?:png|jpg|jpeg))\s*(.*)$", line, re.I)
                        if not match:
                            raise ValueError("unrecognized pixel manifest line")
                        candidates = index[Path(match[1]).name]
                        if not candidates:
                            raise ValueError("manifest image unavailable")
                        image = candidates[0]
                        if image in annotated:
                            continue
                        with Image.open(image) as im:
                            width, height = im.size
                        boxes = []
                        source_names = (base / "model_data/dataset_names.txt").read_text().splitlines()
                        for field in match[2].split():
                            values = field.split(",")
                            if len(values) != 5:
                                raise ValueError("invalid manifest box")
                            name = VOC[source_names[int(values[4])].strip()]
                            boxes.append(pixel_box(name, list(map(float, values[:4])), width, height))
                        add(image, source, boxes, f"{manifest}:{line_number}")
                        annotated.update(candidates)
                    except Exception as exc:
                        issues.append(dict(annotation=f"{manifest}:{line_number}", reason=str(exc)))
        for candidates in index.values():
            for image in candidates:
                if image not in annotated:
                    issues.append(dict(image=str(image), reason="no usable annotation; excluded"))

    # Same decoded pixels are deduplicated; differing annotations are quarantined.
    grouped = collections.defaultdict(list)
    for record in records:
        grouped[record["digest"]].append(record)
    counts = collections.defaultdict(collections.Counter)
    provenance, duplicates, conflicts = [], [], []
    for digest, members in sorted(grouped.items()):
        signatures = {tuple(sorted(tuple(round(v, 4) for v in box) for box in m["boxes"])) for m in members}
        if len(signatures) > 1:
            conflicts.append(dict(digest=digest, images=[str(m["image"]) for m in members],
                                  reason="different annotations for identical pixels; manual review required"))
            continue
        record = members[0]
        # Identical image copies inherit one split, based on stable content hash.
        split = split_for(min(m["group"] for m in members))
        filename = digest + record["image"].suffix.lower()
        image_out = output / "images" / split / filename
        label_out = output / "labels" / split / (digest + ".txt")
        image_out.parent.mkdir(parents=True, exist_ok=True)
        label_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record["image"], image_out)
        label_out.write_text("".join(" ".join(str(v) if i == 0 else f"{v:.8f}" for i, v in enumerate(box)) + "\n"
                                     for box in record["boxes"]))
        for box in record["boxes"]:
            counts[split][NAMES[box[0]]] += 1
        provenance.append(dict(image=str(image_out.relative_to(output)), split=split, digest=digest,
                               group=record["group"], sources=[dict(source=m["source"], image=str(m["image"]),
                               annotation=m["annotation"]) for m in members]))
        if len(members) > 1:
            duplicates.append(dict(digest=digest, copies=len(members)))
    active = {i: name for i, name in enumerate(NAMES)}
    config = dict(path=str(output), train="images/train", val="images/val", test="images/test", names=active)
    (output / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    for name, value in [("provenance", provenance), ("excluded", issues), ("duplicates", duplicates),
                        ("annotation_conflicts", conflicts)]:
        (output / f"{name}.json").write_text(json.dumps(value, indent=2, ensure_ascii=False))
    report = dict(images=len(provenance), splits=dict(collections.Counter(p["split"] for p in provenance)),
                  boxes_per_split=dict(counts), duplicate_groups=len(duplicates), conflict_groups=len(conflicts),
                  excluded_records=len(issues), sources=dict(collections.Counter(
                      s["source"] for p in provenance for s in p["sources"])),
                  quarantined_sources={"dngsein": "No class-ID definition in repository; do not guess"},
                  warnings=["Annotations are partial across sources; relabel missing classes before production training",
                            "head is NOT no_helmet; person_with_helmet is a person-sized box, NOT helmet",
                            "Grouped identifiable video names/augmentations; other near-duplicate frames may remain",
                            "License unknown for dngsein, usckiki82 and Drive; review before redistribution"])
    (output / "merge_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()