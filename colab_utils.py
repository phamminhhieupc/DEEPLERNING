"""Portable dataset packaging/preparation and logged training for Colab."""
import argparse
import hashlib
import json
from pathlib import Path
import stat
import subprocess
import sys
import zipfile

import yaml
from train_ppe import validate_dataset


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pack_dataset(source, archive):
    source, archive = Path(source).resolve(), Path(archive).resolve()
    resolved, _ = validate_dataset(source / "data.yaml")
    if archive.exists() or source == archive or source in archive.parents:
        raise ValueError("Choose a new archive path outside the dataset")
    archive.parent.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((source / "data.yaml").read_text())
    config["path"] = "."
    for split in ("train", "val", "test"):
        if split in config:
            config[split] = str(Path(resolved[split]).relative_to(source))
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as zipped:
        zipped.writestr("data.yaml", yaml.safe_dump(config, sort_keys=False))
        for path in sorted(source.rglob("*")):
            if path.is_file() and path.name != "data.yaml" and path.suffix != ".cache":
                zipped.write(path, path.relative_to(source))
    checksum = sha256(archive)
    archive.with_suffix(archive.suffix + ".sha256").write_text(f"{checksum}  {archive.name}\n")
    return dict(archive=str(archive), sha256=checksum, bytes=archive.stat().st_size)


def prepare_dataset(archive, destination, expected_sha256=None):
    archive, destination = Path(archive).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Dataset destination already exists: {destination}; use a new directory")
    if expected_sha256 and sha256(archive) != expected_sha256.strip().lower():
        raise ValueError("Dataset ZIP SHA256 mismatch")
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            path = Path(member.filename)
            mode = member.external_attr >> 16
            if path.is_absolute() or ".." in path.parts or "\\" in member.filename or stat.S_ISLNK(mode):
                raise ValueError(f"Unsafe ZIP entry: {member.filename}")
        if zipped.testzip() is not None:
            raise ValueError("Corrupted ZIP")
        destination.mkdir(parents=True)
        zipped.extractall(destination)
    config_path = destination / "data.yaml"
    if not config_path.is_file():
        raise ValueError("ZIP must contain data.yaml, images/ and labels/ directly at its root")
    config = yaml.safe_load(config_path.read_text())
    config["path"] = str(destination)
    for split in ("train", "val", "test"):
        if split in config:
            relative = Path(config[split])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"{split} must be a relative directory inside ZIP")
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    _, report = validate_dataset(config_path)
    return dict(data=str(config_path), report=report)


def run_training(repo, data, project, name, model="yolo26s.pt", epochs=150,
                 patience=25, batch=8, imgsz=640, workers=2, device="0"):
    repo, project = Path(repo).resolve(), Path(project).resolve()
    if Path(name).name != name or name in ("", ".", ".."):
        raise ValueError("Run name must be one directory name")
    if (project / name).exists():
        raise FileExistsError("Run already exists; choose a new name")
    project.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", str(repo / "train_ppe.py"), "--data", str(Path(data).resolve()),
               "--project", str(project), "--name", name, "--model", model,
               "--epochs", str(epochs), "--patience", str(patience), "--batch", str(batch),
               "--imgsz", str(imgsz), "--workers", str(workers), "--device", device]
    with (project / f"console_{name}.log").open("x") as log:
        process = subprocess.Popen(command, cwd=repo, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        finally:
            process.stdout.close()
    if code:
        raise RuntimeError(f"Training exited with {code}; inspect {project / f'console_{name}.log'}")
    return project / name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pack = sub.add_parser("pack")
    pack.add_argument("--source", required=True)
    pack.add_argument("--archive", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--archive", required=True)
    prepare.add_argument("--destination", required=True)
    prepare.add_argument("--sha256")
    args = parser.parse_args()
    if args.command == "pack":
        result = pack_dataset(args.source, args.archive)
    else:
        result = prepare_dataset(args.archive, args.destination, args.sha256)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()