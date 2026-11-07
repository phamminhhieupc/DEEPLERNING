"""Run a validation-only model sweep and publish the best PPE checkpoint."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from train_ppe import PRIMARY_METRIC, evaluate_checkpoint, validate_dataset, write_json


def _values(raw: str, kind: type) -> list[Any]:
    values = [kind(value.strip()) for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("Candidate list cannot be empty")
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--project", default="/content/drive/MyDrive/PPE/sweeps")
    parser.add_argument("--models", default="yolo26n.pt,yolo26s.pt")
    parser.add_argument("--imgszs", default="640,960")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--close-mosaic", type=int, default=15)
    parser.add_argument("--save-period", type=int, default=1)
    parser.add_argument("--cache", default="ram", choices=["ram", "disk", "False"])
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--require-all-classes-in-val", action="store_true")
    return parser


def _candidate_summary(path: Path) -> dict[str, Any]:
    summary_path = path / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"Candidate has no summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "completed":
        raise RuntimeError(f"Candidate did not complete: {path}")
    return summary


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    models = _values(args.models, str)
    imgszs = _values(args.imgszs, int)
    validate_dataset(args.data)
    project = Path(args.project).expanduser().resolve()
    project.mkdir(parents=True, exist_ok=True)
    sweep_name = "sweep_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    sweep_dir = project / sweep_name
    sweep_dir.mkdir()
    manifest: list[dict[str, Any]] = []
    train_script = Path(__file__).resolve().parent / "train_ppe.py"

    for index, (model, imgsz) in enumerate(
        ((model, size) for model in models for size in imgszs), 1
    ):
        candidate_name = f"candidate_{index:02d}_{Path(model).stem}_{imgsz}"
        command = [
            sys.executable, str(train_script), "--data", str(Path(args.data).resolve()),
            "--model", model, "--epochs", str(args.epochs), "--patience", str(args.patience),
            "--imgsz", str(imgsz), "--batch", str(args.batch), "--workers", str(args.workers),
            "--device", args.device, "--seed", str(args.seed), "--project", str(sweep_dir),
            "--name", candidate_name, "--optimizer", args.optimizer, "--lr0", str(args.lr0),
            "--lrf", str(args.lrf), "--weight-decay", str(args.weight_decay),
            "--warmup-epochs", str(args.warmup_epochs), "--close-mosaic", str(args.close_mosaic),
            "--save-period", str(args.save_period), "--cache", args.cache, "--skip-test",
        ]
        if args.no_amp:
            command.append("--no-amp")
        if args.require_all_classes_in_val:
            command.append("--require-all-classes-in-val")
        log_path = sweep_dir / f"{candidate_name}.log"
        print(f"\n=== Candidate {index}: {model}, imgsz={imgsz} ===", flush=True)
        with log_path.open("x", encoding="utf-8") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True)
        candidate_dir = sweep_dir / candidate_name
        record: dict[str, Any] = {
            "name": candidate_name, "model": model, "imgsz": imgsz,
            "returncode": result.returncode,
        }
        if result.returncode == 0:
            summary = _candidate_summary(candidate_dir)
            record["summary"] = summary
            record["score"] = summary["selection"]["score"]
        else:
            record["error_log"] = str(log_path)
            print(f"Candidate failed; see {log_path}", flush=True)
        manifest.append(record)
        write_json(sweep_dir / "sweep_manifest.json", manifest)

    completed = [item for item in manifest if "summary" in item]
    if not completed:
        raise RuntimeError(f"No candidate completed. See logs in {sweep_dir}")
    winner = max(
        completed,
        key=lambda item: (
            float(item["summary"]["selection"]["score"]),
            float(item["summary"]["evaluation"]["val"]["metrics"].get("metrics/mAP50(B)", 0)),
            float(item["summary"]["evaluation"]["val"]["metrics"].get("metrics/recall(B)", 0)),
        ),
    )
    source_dir = sweep_dir / winner["name"]
    champion = sweep_dir / "champion"
    (champion / "weights").mkdir(parents=True)
    selected_model = Path(winner["summary"]["best_model"])
    if not selected_model.is_file():
        raise RuntimeError(f"Winner checkpoint is missing: {selected_model}")
    shutil.copy2(selected_model, champion / "weights" / "best.pt")
    shutil.copy2(source_dir / "dataset.resolved.yaml", champion / "dataset.resolved.yaml")
    shutil.copy2(source_dir / "data.original.yaml", champion / "data.original.yaml")
    validation = evaluate_checkpoint(
        champion / "weights" / "best.pt", champion / "dataset.resolved.yaml", "val",
        champion / "best_val", winner["imgsz"], args.batch, args.device, args.workers,
    )
    testing = evaluate_checkpoint(
        champion / "weights" / "best.pt", champion / "dataset.resolved.yaml", "test",
        champion / "best_test", winner["imgsz"], args.batch, args.device, args.workers,
    )
    selection = {
        "status": "completed", "metric": PRIMARY_METRIC, "winner": winner,
        "champion_model": str(champion / "weights" / "best.pt"),
        "validation": validation, "test": testing,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(champion / "selection.json", selection)
    write_json(sweep_dir / "sweep_manifest.json", manifest)
    print("\n=== CHAMPION MODEL ===\n" + json.dumps(selection, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()