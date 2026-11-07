"""Download only images/annotations/class metadata from the supplied public Drive."""
import concurrent.futures
import json
from pathlib import Path
import re
import gdown

ROOT = Path(__file__).resolve().parent / "datasets"


def listing():
    records = json.loads((ROOT / "metadata/drive_listing.json").read_text())
    result = []
    for record in records:
        if isinstance(record, str):
            match = re.match(r"GoogleDriveFileToDownload\(id='([^']+)', path='([^']+)'", record)
            if not match:
                raise ValueError(record)
            result.append(dict(id=match[1], path=match[2]))
        else:
            result.append(record)
    return result


def fetch(record):
    path = ROOT / "raw/drive" / record["path"]
    if path.exists() and path.stat().st_size:
        return str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = str(path) + ".part"
    try:
        result = gdown.download(id=record["id"], output=temporary, quiet=True, use_cookies=False,
                                timeout=45, retries=2)
        if result is None:
            raise RuntimeError("Download returned no file")
        Path(temporary).replace(path)
        return str(path)
    except Exception as exc:
        return dict(path=record["path"], error=str(exc))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()
    records = listing()
    selected = [x for x in records if "result" not in Path(x["path"]).parts and (
                (x["path"].split("/")[0] in {"train", "validation", "test", "final_xml"}
                 and Path(x["path"]).suffix.lower() in {".jpg", ".jpeg", ".png", ".xml", ".txt"})
                or (x["path"].startswith("model_data/") and Path(x["path"]).suffix in {".txt", ".names"}))]
    if args.sample:
        selected = [x for x in selected if x["path"].startswith("model_data/")] + [
            next(x for x in selected if x["path"].endswith(".xml"))]
    write = ROOT / "metadata" / ("drive_sample_download.json" if args.sample else "drive_download.json")
    print(f"Selected {len(selected)} files", flush=True)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for i, result in enumerate(pool.map(fetch, selected), 1):
            results.append(result)
            write.write_text(json.dumps(results, indent=2))
            if len(results) >= 20 and all(isinstance(x, dict) for x in results[-20:]):
                print("Stopping after 20 consecutive failures; check Drive permissions/quota.", flush=True)
                pool.shutdown(wait=False, cancel_futures=True)
                break
            if i % 100 == 0 or isinstance(result, dict):
                print(f"{i}/{len(selected)}", str(result)[:300], flush=True)
    write.write_text(json.dumps(results, indent=2))
    print(f"Finished; failures: {sum(isinstance(x, dict) for x in results)}", flush=True)