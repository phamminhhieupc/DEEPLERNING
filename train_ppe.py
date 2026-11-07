"""Reproducible Ultralytics PPE training with validation-based selection."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
PRIMARY_METRIC = "metrics/mAP50-95(B)"


def write_json(path: str | Path, value: Any) -> None:
    """Atomically write JSON and reject non-finite numbers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


class PlateauMonitor:
    """Strict validation-metric monitor; ties do not reset patience."""

    def __init__(self, patience: int):
        if patience < 1:
            raise ValueError("patience must be >= 1")
        self.patience = patience
        self.best: float | None = None
        self.best_epoch: int | None = None
        self.last_epoch = 0
        self.bad_epochs = 0

    def update(self, epoch: int, score: float) -> bool:
        if not math.isfinite(score):
            raise ValueError("Validation mAP is NaN/Inf; check data and training losses")
        if epoch <= self.last_epoch:
            return self.bad_epochs >= self.patience
        self.last_epoch = epoch
        if self.best is None or score > self.best:
            self.best, self.best_epoch, self.bad_epochs = score, epoch, 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def state(self) -> dict[str, Any]:
        return {
            "monitor": PRIMARY_METRIC,
            "mode": "max",
            "patience": self.patience,
            "best": self.best,
            "first_best_epoch": self.best_epoch,
            "completed_epochs": self.last_epoch,
            "epochs_without_improvement": self.bad_epochs,
            "early_stopped": self.bad_epochs >= self.patience,
        }


def _read_names(data: dict[str, Any]) -> dict[int, str]:
    names = data.get("names")
    if isinstance(names, list):
        names = dict(enumerate(names))
    if not isinstance(names, dict) or set(names) != set(range(len(names))) or not names:
        raise ValueError("names must be a nonempty list or dict with consecutive integer IDs from 0")
    normalized = {int(index): str(name) for index, name in names.items()}
    if len(set(normalized.values())) != len(normalized):
        raise ValueError("Class names must be unique")
    return normalized


def validate_dataset(yaml_path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate a directory-style YOLO detection dataset without changing labels."""
    import yaml

    yaml_path = Path(yaml_path).resolve()
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Dataset YAML must contain a mapping")
    names = _read_names(data)
    root = Path(data.get("path", yaml_path.parent))
    if not root.is_absolute():
        root = yaml_path.parent / root
    root = root.resolve()
    report: dict[str, Any] = {}
    image_sets: dict[str, set[str]] = {}

    for split in ("train", "val", "test"):
        if split not in data:
            if split != "test":
                raise ValueError(f"Missing {split} split")
            continue
        if not isinstance(data[split], str):
            raise ValueError("This script expects one images directory per split, not lists or txt manifests")
        directory = (root / data[split]).resolve()
        if not directory.is_dir():
            raise ValueError(f"Images directory does not exist: {directory}")
        images = sorted(p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            raise ValueError(f"No images in {directory}")

        counts = {str(index): 0 for index in names}
        missing = 0
        empty_labels = 0
        hashes: set[str] = set()
        for image in images:
            hashes.add(hashlib.sha256(image.read_bytes()).hexdigest())
            parts = list(image.parts)
            if "images" not in parts:
                raise ValueError(f"Use standard images/<split>, labels/<split> layout: {image}")
            index = len(parts) - 1 - parts[::-1].index("images")
            parts[index] = "labels"
            label = Path(*parts).with_suffix(".txt")
            if not label.exists():
                missing += 1
                continue
            label_lines = [line for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not label_lines:
                empty_labels += 1
            for line_number, line in enumerate(label_lines, 1):
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError(f"{label}:{line_number}: expected class x_center y_center width height")
                try:
                    cls = int(fields[0])
                    x, y, w, h = map(float, fields[1:])
                except ValueError as error:
                    raise ValueError(f"{label}:{line_number}: invalid numeric label") from error
                if cls not in names or not all(math.isfinite(v) for v in (x, y, w, h)):
                    raise ValueError(f"{label}:{line_number}: invalid class or nonfinite coordinates")
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    raise ValueError(f"{label}:{line_number}: coordinates must be normalized to [0,1]")
                if x - w / 2 < -1e-5 or y - h / 2 < -1e-5 or x + w / 2 > 1.00001 or y + h / 2 > 1.00001:
                    raise ValueError(f"{label}:{line_number}: box extends outside image")
                counts[str(cls)] += 1
        if not sum(counts.values()):
            raise ValueError(f"{split} contains no labeled objects")
        for previous, previous_hashes in image_sets.items():
            if hashes & previous_hashes:
                raise ValueError(f"Identical images found in {previous} and {split}: data leakage")
        image_sets[split] = hashes
        report[split] = {
            "images": len(images),
            "missing_label_files": missing,
            "empty_label_files": empty_labels,
            "objects_per_class": counts,
        }

    data["path"] = str(root)
    data["names"] = names
    for split in report:
        data[split] = str((root / data[split]).resolve())
    return data, report


def _metric_value(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Metric {key} is NaN/Inf")
    return value


def _result_metrics(result: Any) -> dict[str, float]:
    return {str(key): float(value) for key, value in result.results_dict.items()}


def select_checkpoint_by_validation(run_dir: str | Path) -> tuple[Path, dict[str, Any]]:
    """Select the saved epoch checkpoint with the highest validation mAP50-95.

    Ultralytics' native ``best.pt`` selection is version-dependent fitness.  Saving
    every epoch lets this pipeline enforce the project's explicit selection metric.
    If a version does not emit an epoch checkpoint, its native best checkpoint is a
    safe fallback and the summary records that fact.
    """
    run_dir = Path(run_dir)
    results_path = run_dir / "results.csv"
    native_best = run_dir / "weights" / "best.pt"
    if not results_path.is_file():
        if not native_best.is_file():
            raise RuntimeError(f"Missing training history and checkpoint in {run_dir}")
        return native_best, {"source": "ultralytics_best_fallback"}
    with results_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    weights_dir = run_dir / "weights"
    # Infer the filename convention once for the whole run.  This avoids treating
    # epoch1.pt as two different epochs when epoch0.pt is absent.
    zero_based = (weights_dir / "epoch0.pt").is_file()
    candidates = []
    for row in rows:
        score = _metric_value(row, PRIMARY_METRIC)
        if score is None:
            continue
        csv_epoch = int(row["epoch"])
        # Ultralytics has used both zero-based and one-based epoch filenames across
        # releases.  Accept either form and let the available file decide.
        epoch = csv_epoch if zero_based else csv_epoch + 1
        checkpoint = weights_dir / f"epoch{epoch}.pt"
        if checkpoint.is_file():
            candidates.append((score, epoch, checkpoint))
    if not candidates:
        if not native_best.is_file():
            raise RuntimeError(f"No usable checkpoint found in {run_dir}")
        return native_best, {"source": "ultralytics_best_fallback", "reason": "no epoch checkpoints"}
    score, epoch, checkpoint = max(candidates, key=lambda item: (item[0], item[1]))
    selected = run_dir / "weights" / "best_val.pt"
    if selected.resolve() != checkpoint.resolve():
        import shutil
        shutil.copy2(checkpoint, selected)
    return selected, {"source": "epoch_checkpoint", "epoch": epoch, "score": score}


def evaluate_checkpoint(checkpoint: str | Path, data_yaml: str | Path, split: str,
                        output_dir: str | Path, imgsz: int, batch: int,
                        device: str, workers: int) -> dict[str, Any]:
    """Evaluate one checkpoint and persist aggregate/per-class metrics."""
    from ultralytics import YOLO

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = YOLO(str(checkpoint)).val(
        data=str(data_yaml), split=split, imgsz=imgsz, batch=batch, device=device,
        workers=workers, project=str(output_dir), name="plots", plots=True, verbose=False,
    )
    metrics = _result_metrics(result)
    per_class = []
    for position, class_id in enumerate(getattr(result.box, "ap_class_index", [])):
        precision, recall, ap50, ap = result.box.class_result(position)
        per_class.append({
            "id": int(class_id), "name": result.names[int(class_id)],
            "precision": float(precision), "recall": float(recall),
            "mAP50": float(ap50), "mAP50_95": float(ap),
        })
    evaluation = {"split": split, "metrics": metrics, "per_class": per_class, "speed_ms": result.speed}
    write_json(output_dir.parent / f"metrics_{split}.json", evaluation)
    return evaluation


def _capture_environment() -> dict[str, Any]:
    import torch
    import ultralytics

    environment: dict[str, Any] = {
        "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "ultralytics": ultralytics.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        environment["git_commit"] = subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        environment["git_commit"] = None
    try:
        environment["pip_freeze"] = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True
        ).splitlines()
    except (OSError, subprocess.CalledProcessError):
        environment["pip_freeze"] = []
    return environment


def build_train_config(args: argparse.Namespace, resolved_yaml: Path, run_dir: Path) -> dict[str, Any]:
    """Build a conservative configuration suitable for small PPE objects."""
    return {
        "data": str(resolved_yaml), "epochs": args.epochs, "patience": args.patience,
        "imgsz": args.imgsz, "batch": args.batch, "workers": args.workers,
        "device": args.device, "seed": args.seed, "deterministic": True,
        "project": str(run_dir.parent), "name": run_dir.name, "exist_ok": True,
        "save": True, "save_period": args.save_period, "plots": True, "val": True,
        "pretrained": True, "cache": args.cache, "optimizer": args.optimizer,
        "lr0": args.lr0, "lrf": args.lrf, "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs, "cos_lr": True,
        "close_mosaic": args.close_mosaic,
        "amp": args.device != "cpu" and not args.no_amp,
        "degrees": 0.0, "translate": 0.1, "scale": 0.5,
        "fliplr": 0.5, "mosaic": 1.0, "mixup": 0.0,
    }


def train_one(args: argparse.Namespace) -> Path:
    """Train and evaluate one candidate; return its run directory."""
    import yaml
    import torch
    from ultralytics import YOLO

    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable. Select a Colab GPU or use --device cpu explicitly")
    if "," in args.device:
        raise ValueError("Use one GPU or cpu; this pipeline selects one checkpoint per run")
    data, report = validate_dataset(args.data)
    val_counts = report["val"]["objects_per_class"]
    missing_val = [name for index, name in data["names"].items() if val_counts[str(index)] == 0]
    if missing_val and args.require_all_classes_in_val:
        raise ValueError(f"Validation has no objects for classes: {missing_val}")

    run_dir = Path(args.project).expanduser().resolve() / args.name
    run_dir.mkdir(parents=True, exist_ok=False)
    resolved_yaml = run_dir / "dataset.resolved.yaml"
    resolved_yaml.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    write_json(run_dir / "dataset_report.json", report)
    (run_dir / "data.original.yaml").write_bytes(Path(args.data).read_bytes())
    environment = _capture_environment()
    write_json(run_dir / "environment.json", environment)
    (run_dir / "requirements.lock.txt").write_text(
        "\n".join(environment["pip_freeze"]) + "\n", encoding="utf-8"
    )
    config = build_train_config(args, resolved_yaml, run_dir)
    write_json(run_dir / "requested_config.json", {"model": args.model, "selection_metric": PRIMARY_METRIC, **config})
    if missing_val:
        write_json(run_dir / "quality_warnings.json", {"classes_missing_from_val": missing_val})

    monitor = PlateauMonitor(args.patience)

    def record_epoch(trainer: Any) -> None:
        score = _metric_value(getattr(trainer, "metrics", {}) or {}, PRIMARY_METRIC)
        if score is None:
            return
        epoch = int(getattr(trainer, "epoch", monitor.last_epoch - 1)) + 1
        if epoch <= monitor.last_epoch:
            return
        should_stop = monitor.update(epoch, score)
        write_json(run_dir / "early_stopping.json", monitor.state())
        print(f"[PPE] epoch={epoch} val_mAP50-95={score:.6f} best={monitor.best:.6f} "
              f"no_improvement={monitor.bad_epochs}/{monitor.patience}", flush=True)
        if should_stop:
            trainer.stop = True
            print("[PPE] Early stopping: validation mAP did not improve.", flush=True)

    try:
        model = YOLO(args.model)
        model.add_callback("on_fit_epoch_end", record_epoch)
        model.train(**config)
        best_path, checkpoint_selection = select_checkpoint_by_validation(run_dir)
        evaluations = {"val": evaluate_checkpoint(
            best_path, resolved_yaml, "val", run_dir / "best_val",
            args.imgsz, args.batch, args.device, args.workers,
        )}
        if not args.skip_test and "test" in data:
            evaluations["test"] = evaluate_checkpoint(
                best_path, resolved_yaml, "test", run_dir / "best_test",
                args.imgsz, args.batch, args.device, args.workers,
            )
        score = _metric_value(evaluations["val"]["metrics"], PRIMARY_METRIC)
        if score is None:
            raise RuntimeError(f"Validation did not return {PRIMARY_METRIC}")
        summary = {
            "status": "completed", "best_model": str(best_path),
            "last_model": str(run_dir / "weights/last.pt"),
            "selection": {"metric": PRIMARY_METRIC, "score": score, "split": "val",
                           "checkpoint": checkpoint_selection},
            "early_stopping": monitor.state(), "evaluation": evaluations,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        write_json(run_dir / "summary.json", summary)
        print("\n=== BEST MODEL / VALIDATION METRICS ===\n" + json.dumps(summary, indent=2, ensure_ascii=False))
    except (Exception, KeyboardInterrupt) as error:
        write_json(run_dir / "failure.json", {
            "status": "interrupted_or_failed", "error": repr(error),
            "early_stopping": monitor.state(),
        })
        raise
    return run_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="yolo26s.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", default="/content/drive/MyDrive/PPE/runs")
    parser.add_argument("--name", default="ppe_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--close-mosaic", type=int, default=15)
    parser.add_argument("--save-period", type=int, default=1)
    parser.add_argument("--cache", default=False, choices=["ram", "disk", "False"])
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--require-all-classes-in-val", action="store_true")
    parser.add_argument("--skip-test", action="store_true", help="Do not evaluate test after this run")
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.cache == "False":
        args.cache = False
    if args.epochs < 1 or args.patience < 1 or args.imgsz < 32 or args.batch < 1 or args.workers < 0:
        parser.error("epochs/patience/imgsz/batch must be positive and workers must be non-negative")
    if Path(args.name).name != args.name or args.name in ("", ".", ".."):
        parser.error("name must be one directory name")
    train_one(args)


if __name__ == "__main__":
    main()