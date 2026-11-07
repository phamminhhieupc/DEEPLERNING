"""Download repository metadata only, without executing repository code."""
import collections
import concurrent.futures
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent / "datasets"
REPOS = ["ilaydadeger/proje24_yolo_hardhat", "dngsein/Hardhat",
         "usckiki82/hardhat-dataset", "JoyKarmoker/Construction-Site-Safety-PPE-Detection"]


def inspect(repo):
    response = requests.get(f"https://api.github.com/repos/{repo}/git/trees/HEAD?recursive=1", timeout=120)
    response.raise_for_status()
    tree = response.json()
    if tree.get("truncated"):
        raise RuntimeError(f"Truncated tree: {repo}")
    slug = repo.split("/")[0]
    (ROOT / "metadata").mkdir(parents=True, exist_ok=True)
    (ROOT / "metadata" / f"{slug}.tree.json").write_text(json.dumps(tree))
    files = [item for item in tree["tree"] if item["type"] == "blob"]
    print(repo, "sha", tree["sha"], "MiB", round(sum(x.get("size", 0) for x in files) / 2**20, 1),
          "extensions", collections.Counter(Path(x["path"]).suffix.lower() for x in files), flush=True)
    for item in files:
        path = item["path"]
        if (Path(path).suffix.lower() in [".yaml", ".yml", ".names"] or
                "readme" in path.lower() or "license" in path.lower()):
            if item.get("size", 0) < 30000:
                response = requests.get(f"https://raw.githubusercontent.com/{repo}/{tree['sha']}/{path}", timeout=60)
                response.raise_for_status()
                target = ROOT / "metadata" / slug / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(response.text)
                print(repo, path, "\n", response.text[:6000], flush=True)


if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(inspect, REPOS))