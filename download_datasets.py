"""Fetch pinned GitHub dataset files only. Never run downloaded code."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import tarfile
import requests
from inspect_sources import ROOT, REPOS


def download(repo):
    slug = repo.split("/")[0]
    tree = json.loads((ROOT / "metadata" / f"{slug}.tree.json").read_text())
    prefixes = {"ilaydadeger": ("dataset_small/",), "dngsein": ("train/", "test/", "valid/", "val/"),
                "usckiki82": ("hardhat/", "people/"),
                "JoyKarmoker": ("Train/Construction_Site_Safety/",)}[slug]
    files = [x for x in tree["tree"] if x["type"] == "blob" and x["path"].startswith(prefixes)
             and Path(x["path"]).suffix.lower() in {".jpg", ".jpeg", ".png", ".txt", ".xml", ".yaml"}]
    destination = ROOT / "raw" / slug

    def fetch(item):
        path = destination / item["path"]
        if path.is_file() and path.stat().st_size == item["size"]:
            return
        response = requests.get(f"https://raw.githubusercontent.com/{repo}/{tree['sha']}/{item['path']}", timeout=120)
        response.raise_for_status()
        content = response.content
        actual = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        if actual != item["sha"]:
            raise ValueError(f"Git hash mismatch {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    print(f"Downloading {slug}: {len(files)} dataset files", flush=True)
    # A pinned codeload tarball is substantially faster than thousands of requests.
    wanted = {x["path"]: x for x in files}
    if not all((destination / x["path"]).is_file() and
               (destination / x["path"]).stat().st_size == x["size"] for x in files):
        response = requests.get(f"https://codeload.github.com/{repo}/tar.gz/{tree['sha']}", stream=True, timeout=180)
        response.raise_for_status()
        with tarfile.open(fileobj=response.raw, mode="r|gz") as archive:
            for member in archive:
                relative = member.name.split("/", 1)[-1]
                if relative not in wanted or not member.isfile():
                    continue
                content = archive.extractfile(member).read()
                item = wanted[relative]
                if hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest() != item["sha"]:
                    raise ValueError(f"Hash mismatch {relative}")
                path = destination / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
        response.close()
    # Retry missing files individually if an archive is unexpectedly incomplete.
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(fetch, files))
    print(f"Completed {slug}: {len(files)} files", flush=True)


if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download, REPOS))