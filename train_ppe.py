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
import time
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


def _float_value(value: Any) -> float | None:
    """Convert a scalar/tensor-like value to a finite float."""
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "item"):
            value = value.item()
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _numeric_values(values: Any) -> dict[str, float]:
    if not hasattr(values, "items"):
        return {}
    return {
        str(key): number
        for key, value in values.items()
        if (number := _float_value(value)) is not None
    }


def _loss_values(trainer: Any, value: Any, prefix: str) -> dict[str, float]:
    """Read loss values across Ultralytics versions.

    YOLOv8/YOLO11 commonly expose ``tloss`` as a tensor and provide
    ``label_loss_items`` to name it.  Other versions expose a mapping directly.
    Missing loss families, including objectness on YOLOv8/YOLO11, are omitted.
    """
    values = _numeric_values(value)
    if values:
        return values
    label_loss_items = getattr(trainer, "label_loss_items", None)
    if callable(label_loss_items):
        for kwargs in ({"prefix": prefix}, {}):
            try:
                values = _numeric_values(label_loss_items(value, **kwargs))
            except (TypeError, ValueError, RuntimeError):
                continue
            if values:
                return values
    return {}


def _first_value(*mappings: dict[str, float], keys: tuple[str, ...]) -> float | None:
    for mapping in mappings:
        for key in keys:
            if key in mapping:
                return mapping[key]
    return None


def build_epoch_record(trainer: Any, epoch: int) -> dict[str, Any]:
    """Build an auditable per-epoch train/validation/lr record.

    ``trainer.tloss`` is the epoch-average training loss.  Validation losses and
    detection metrics are taken from ``trainer.metrics`` when the installed
    Ultralytics version exposes them.  Raw mappings are retained so no precision
    is lost even when a particular version uses different metric names.
    """
    train_losses = _loss_values(trainer, getattr(trainer, "tloss", {}), "train")
    validation = _numeric_values(getattr(trainer, "metrics", {}))
    validator = getattr(trainer, "validator", None)
    val_losses = _loss_values(trainer, getattr(validator, "loss", {}), "val")
    val_loss_items = _loss_values(trainer, getattr(validator, "loss_items", {}), "val")
    val_losses = {**val_loss_items, **val_losses}
    validation_losses = {**val_losses, **{
        key: value for key, value in validation.items() if "loss" in key.lower()
    }}
    learning_rates = _numeric_values(getattr(trainer, "lr", {}))
    record: dict[str, Any] = {
        "epoch": epoch,
        "train_loss": train_losses,
        "validation": {**validation, **validation_losses},
        "learning_rate": learning_rates,
    }
    if train_losses:
        record["train_total_loss"] = sum(train_losses.values())
    aliases = {
        "train_box_loss": ("train/box_loss", "box_loss"),
        "train_cls_loss": ("train/cls_loss", "cls_loss"),
        "train_dfl_loss": ("train/dfl_loss", "dfl_loss"),
        "train_obj_loss": ("train/obj_loss", "obj_loss", "objectness_loss"),
        "val_box_loss": ("val/box_loss", "validation/box_loss", "box_loss"),
        "val_cls_loss": ("val/cls_loss", "validation/cls_loss", "cls_loss"),
        "val_dfl_loss": ("val/dfl_loss", "validation/dfl_loss", "dfl_loss"),
        "val_obj_loss": ("val/obj_loss", "validation/obj_loss", "obj_loss", "objectness_loss"),
        "precision": ("metrics/precision(B)", "metrics/precision"),
        "recall": ("metrics/recall(B)", "metrics/recall"),
        "mAP50": ("metrics/mAP50(B)", "metrics/mAP50"),
        "mAP50_95": ("metrics/mAP50-95(B)", "metrics/mAP50-95"),
    }
    for output, keys in aliases.items():
        value = _first_value(
            train_losses if output.startswith("train_") else validation_losses,
            validation if output in {"precision", "recall", "mAP50", "mAP50_95"} else {},
            keys=keys,
        )
        if value is not None:
            record[output] = value
    if "precision" in record and "recall" in record and record["precision"] + record["recall"] > 0:
        record["f1"] = 2 * record["precision"] * record["recall"] / (record["precision"] + record["recall"])
    val_loss_keys = ("val_box_loss", "val_cls_loss", "val_dfl_loss", "val_obj_loss")
    available_val_losses = [record[key] for key in val_loss_keys if key in record]
    if available_val_losses:
        record["val_total_loss"] = sum(available_val_losses)
    return record


def _shown(record: dict[str, Any], key: str) -> str:
    value = record.get(key)
    return f"{value:.6f}" if isinstance(value, (int, float)) else "N/A"


def _delta(record: dict[str, Any], previous: dict[str, Any] | None, key: str) -> str:
    current = record.get(key)
    old = previous.get(key) if previous else None
    if not isinstance(current, (int, float)) or not isinstance(old, (int, float)):
        return ""
    if current > old:
        return " ↑"
    if current < old:
        return " ↓"
    return " →"


def _seconds(value: Any) -> str:
    if not isinstance(value, (int, float)) or value < 0:
        return "N/A"
    value = int(round(value))
    hours, remainder = divmod(value, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    return f"{minutes}m {seconds:02d}s"


def _file_info(path: str | Path) -> tuple[str, str]:
    path = Path(path)
    if not path.is_file():
        return str(path), "N/A"
    return str(path), f"{path.stat().st_size / (1024 ** 2):.2f}"


def _gpu_memory(torch: Any) -> dict[str, Any]:
    """Read and reset CUDA peak memory without failing in CPU mode."""
    if not getattr(torch, "cuda", None) or not torch.cuda.is_available():
        return {"gpu_status": "CPU mode"}
    try:
        allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        values = {
            "gpu_status": torch.cuda.get_device_name(0),
            "gpu_memory_allocated_gb": allocated,
            "gpu_memory_reserved_gb": reserved,
            "gpu_memory_total_gb": total,
        }
    except (AttributeError, RuntimeError, TypeError):
        values = {"gpu_status": "GPU memory unavailable"}
    finally:
        try:
            torch.cuda.reset_peak_memory_stats()
        except (AttributeError, RuntimeError):
            pass
    return values


def format_epoch_summary(record: dict[str, Any], total_epochs: int, best: float | None,
                         bad_epochs: int, patience: int,
                         previous: dict[str, Any] | None = None) -> str:
    """Format a readable train/validation summary for one completed epoch."""
    epoch = int(record["epoch"])
    train_obj = f" obj={_shown(record, 'train_obj_loss')}{_delta(record, previous, 'train_obj_loss')}" \
        if "train_obj_loss" in record else ""
    val_obj = f" obj={_shown(record, 'val_obj_loss')}{_delta(record, previous, 'val_obj_loss')}" \
        if "val_obj_loss" in record else ""
    best_marker = record.get("best_marker", "")
    return (
        f"[EPOCH {epoch:03d}/{total_epochs}]\n"
        f"  TRAIN: box={_shown(record, 'train_box_loss')}{_delta(record, previous, 'train_box_loss')} "
        f"cls={_shown(record, 'train_cls_loss')}{_delta(record, previous, 'train_cls_loss')} "
        f"dfl={_shown(record, 'train_dfl_loss')}{_delta(record, previous, 'train_dfl_loss')}"
        f"{train_obj} "
        f"total={_shown(record, 'train_total_loss')}\n"
        f"  VAL  : box={_shown(record, 'val_box_loss')}{_delta(record, previous, 'val_box_loss')} "
        f"cls={_shown(record, 'val_cls_loss')}{_delta(record, previous, 'val_cls_loss')} "
        f"dfl={_shown(record, 'val_dfl_loss')}{_delta(record, previous, 'val_dfl_loss')}"
        f"{val_obj} "
        f"P={_shown(record, 'precision')} "
        f"R={_shown(record, 'recall')} "
        f"mAP50={_shown(record, 'mAP50')} "
        f"mAP50-95={_shown(record, 'mAP50_95')} F1={_shown(record, 'f1')}\n"
        f"  BEST : {best_marker} fitness={_shown(record, 'fitness')} "
        f"best={_shown({'value': record.get('best_fitness', best)}, 'value')} "
        f"epoch={record.get('best_fitness_epoch', 'N/A')} "
        f"mAP_no_improvement={bad_epochs}/{patience}\n"
        f"  SPEED: epoch={_seconds(record.get('epoch_seconds'))} "
        f"avg={_seconds(record.get('avg_epoch_seconds'))} "
        f"ETA={_seconds(record.get('eta_seconds'))} "
        f"iteration={_shown(record, 'iterations_per_second')} it/s\n"
        f"  GPU  : {record.get('gpu_status', 'CPU mode')} "
        f"allocated={_shown(record, 'gpu_memory_allocated_gb')}GB "
        f"reserved={_shown(record, 'gpu_memory_reserved_gb')}GB "
        f"total={_shown(record, 'gpu_memory_total_gb')}GB\n"
        f"  FILES: last.pt={record.get('last_checkpoint', 'N/A')} "
        f"({record.get('last_checkpoint_mb', 'N/A')}MB), "
        f"best.pt={record.get('best_checkpoint', 'N/A')} "
        f"({record.get('best_checkpoint_mb', 'N/A')}MB)\n"
        f"  RUNS : {record.get('save_dir', 'N/A')} "
        f"results={record.get('results_csv', 'N/A')} plots={record.get('plots_dir', 'N/A')}"
        + (f"\n  WARNING: {record['warning']}" if record.get("warning") else "")
    )


def append_epoch_record(path: str | Path, record: dict[str, Any],
                        csv_name: str | None = None,
                        write_jsonl: bool = True) -> None:
    """Append one full-precision JSONL and CSV epoch record for easy inspection."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if write_jsonl:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
    columns = [
        "epoch", "train_box_loss", "train_cls_loss", "train_dfl_loss",
        "train_obj_loss", "train_total_loss", "val_box_loss", "val_cls_loss",
        "val_dfl_loss", "val_obj_loss", "val_total_loss", "precision", "recall",
        "mAP50", "mAP50_95", "f1", "fitness", "best_fitness", "map_bad_epochs",
        "best_fitness_epoch", "best_marker", "epoch_seconds", "avg_epoch_seconds", "eta_seconds",
        "iterations_per_second", "gpu_memory_allocated_gb", "gpu_memory_reserved_gb",
        "gpu_memory_total_gb", "gpu_status", "last_checkpoint", "last_checkpoint_mb",
        "best_checkpoint", "best_checkpoint_mb", "save_dir", "results_csv", "plots_dir",
        "warning",
    ]
    csv_path = path.with_name(csv_name) if csv_name else path.with_suffix(".csv")
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        if handle.tell() == 0:
            writer.writeheader()
        writer.writerow({key: record.get(key, "") for key in columns})
        handle.flush()


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


def print_final_evaluation(evaluation: dict[str, Any]) -> None:
    """Print a compact aggregate and per-class report for a final split evaluation."""
    metrics = evaluation["metrics"]
    print(f"\n[FINAL {evaluation['split'].upper()}]", flush=True)
    print(
        "  precision={:.6f} recall={:.6f} mAP50={:.6f} mAP50-95={:.6f}".format(
            float(metrics.get("metrics/precision(B)", float("nan"))),
            float(metrics.get("metrics/recall(B)", float("nan"))),
            float(metrics.get("metrics/mAP50(B)", float("nan"))),
            float(metrics.get(PRIMARY_METRIC, float("nan"))),
        ),
        flush=True,
    )
    if evaluation["per_class"]:
        print("  per-class:", flush=True)
        for item in evaluation["per_class"]:
            print(
                "    [{:02d}] {:<24} P={:.4f} R={:.4f} mAP50={:.4f} mAP50-95={:.4f}".format(
                    item["id"], item["name"], item["precision"], item["recall"],
                    item["mAP50"], item["mAP50_95"],
                ),
                flush=True,
            )


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
    """Build the Ultralytics configuration used for one training run.

    ``--fast`` deliberately disables deterministic CUDA algorithms.  This is
    useful on Colab T4 where deterministic execution can reduce throughput;
    the seed is still recorded in the run metadata.
    """
    return {
        "data": str(resolved_yaml), "epochs": args.epochs, "patience": args.patience,
        "map_patience": args.map_patience,
        "imgsz": args.imgsz, "batch": args.batch, "workers": args.workers,
        "device": args.device, "seed": args.seed, "deterministic": not args.fast,
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
    if args.device != "cpu" and args.fast:
        # T4 benefits from Tensor Cores (AMP) and from allowing cuDNN to pick
        # the fastest convolution algorithm for a fixed image size.
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
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
    epoch_log = run_dir / "epoch_metrics.jsonl"
    epoch_csv = run_dir / "epoch_log.csv"
    previous_record: dict[str, Any] | None = None
    epoch_durations: list[float] = []
    last_callback = time.perf_counter()
    overfit_epochs = 0
    best_fitness: float | None = None
    best_fitness_epoch: int | None = None
    best_map: float | None = None
    map_bad_epochs = 0

    def record_epoch(trainer: Any) -> None:
        nonlocal previous_record, last_callback, overfit_epochs
        nonlocal best_fitness, best_fitness_epoch, best_map, map_bad_epochs
        epoch = int(getattr(trainer, "epoch", monitor.last_epoch - 1)) + 1
        if epoch <= monitor.last_epoch:
            return
        record = build_epoch_record(trainer, epoch)
        now = time.perf_counter()
        epoch_seconds = _float_value(getattr(trainer, "epoch_time", None)) or (now - last_callback)
        last_callback = now
        epoch_durations.append(max(epoch_seconds, 0.0))
        avg_seconds = sum(epoch_durations) / len(epoch_durations)
        record.update({
            "epoch_seconds": epoch_seconds,
            "avg_epoch_seconds": avg_seconds,
            "eta_seconds": max(args.epochs - epoch, 0) * avg_seconds,
        })
        try:
            iterations = len(getattr(trainer, "train_loader", None))
        except TypeError:
            iterations = None
        if iterations and epoch_seconds > 0:
            record["iterations_per_second"] = iterations / epoch_seconds
        record.update(_gpu_memory(torch))

        score = record.get("mAP50_95")
        if score is None:
            score = _metric_value(getattr(trainer, "metrics", {}) or {}, PRIMARY_METRIC)
        fitness = _float_value(getattr(trainer, "fitness", None))
        if fitness is None:
            fitness = score
        is_best = fitness is not None and (best_fitness is None or fitness > best_fitness)
        if is_best:
            best_fitness, best_fitness_epoch = fitness, epoch
        if score is not None and (best_map is None or score > best_map):
            best_map, map_bad_epochs = score, 0
        elif score is not None:
            map_bad_epochs += 1
        if fitness is None:
            print(f"[PPE] epoch={epoch} loss/metrics logged; {PRIMARY_METRIC} unavailable", flush=True)
            should_stop = False
        else:
            # Early stopping remains tied to validation mAP, not a version-specific
            # Ultralytics fitness formula.
            should_stop = monitor.update(epoch, score if score is not None else fitness)
            record["fitness"] = fitness
        record["best_fitness"] = best_fitness
        record["best_fitness_epoch"] = best_fitness_epoch
        record["map_bad_epochs"] = map_bad_epochs
        record["best_marker"] = "★ BEST" if is_best else ""

        if (previous_record and record.get("train_total_loss") is not None and
                record.get("val_total_loss") is not None and
                record["train_total_loss"] < previous_record.get("train_total_loss", float("inf")) and
                record["val_total_loss"] > previous_record.get("val_total_loss", float("-inf"))):
            overfit_epochs += 1
        else:
            overfit_epochs = 0
        warnings: list[str] = []
        if overfit_epochs >= 5:
            warnings.append(f"nghi overfitting: val loss tăng {overfit_epochs} epoch liên tục, train loss giảm")
        if map_bad_epochs >= args.map_patience:
            warnings.append(f"mAP không cải thiện {map_bad_epochs} epoch; sắp early stopping (ngưỡng={args.map_patience})")
        record["warning"] = " | ".join(warnings)

        weights_dir = run_dir / "weights"
        last_path, last_mb = _file_info(weights_dir / "last.pt")
        best_path, best_mb = _file_info(weights_dir / "best.pt")
        record.update({
            "last_checkpoint": last_path, "last_checkpoint_mb": last_mb,
            "best_checkpoint": best_path, "best_checkpoint_mb": best_mb,
            "save_dir": str(run_dir), "results_csv": str(run_dir / "results.csv"),
            "plots_dir": str(run_dir),
        })
        append_epoch_record(epoch_log, record)
        append_epoch_record(epoch_log, record, csv_name=epoch_csv.name, write_jsonl=False)
        write_json(run_dir / "early_stopping.json", monitor.state())
        print(format_epoch_summary(record, args.epochs, monitor.best,
                                   monitor.bad_epochs, monitor.patience, previous_record), flush=True)
        if is_best and fitness is not None:
            print(f"[PPE] best.pt vừa được cập nhật: fitness={fitness:.6f}, epoch={epoch}", flush=True)
        if record["warning"]:
            print(f"[PPE] WARNING: {record['warning']}", flush=True)
        if should_stop:
            trainer.stop = True
            print("[PPE] Early stopping: validation mAP did not improve.", flush=True)
        previous_record = record

    try:
        model = YOLO(args.model)
        model.add_callback("on_fit_epoch_end", record_epoch)
        model.train(**config)
        best_path, checkpoint_selection = select_checkpoint_by_validation(run_dir)
        evaluations = {"val": evaluate_checkpoint(
            best_path, resolved_yaml, "val", run_dir / "best_val",
            args.imgsz, args.batch, args.device, args.workers,
        )}
        print_final_evaluation(evaluations["val"])
        if not args.skip_test and "test" in data:
            evaluations["test"] = evaluate_checkpoint(
                best_path, resolved_yaml, "test", run_dir / "best_test",
                args.imgsz, args.batch, args.device, args.workers,
            )
            print_final_evaluation(evaluations["test"])
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
    parser.add_argument("--map-patience", type=int, default=10,
                        help="Warn after this many epochs without mAP improvement")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=4,
                        help="Batch size; use -1 for Ultralytics auto-batch")
    parser.add_argument("--workers", type=int, default=4,
                        help="DataLoader workers; 4 is a good Colab T4 starting point")
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
    parser.add_argument("--cache", default="ram", choices=["ram", "disk", "False"],
                        help="ram speeds repeated epochs but uses system RAM")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--fast", action="store_true",
                        help="Optimize CUDA throughput; disables deterministic algorithms")
    parser.add_argument("--require-all-classes-in-val", action="store_true")
    parser.add_argument("--skip-test", action="store_true", help="Do not evaluate test after this run")
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.cache == "False":
        args.cache = False
    if (args.epochs < 1 or args.patience < 1 or args.map_patience < 1 or args.imgsz < 32
            or args.batch < -1 or args.workers < 0):
        parser.error("epochs/patience/map-patience/imgsz must be positive, batch must be >= -1, workers must be non-negative")
    if Path(args.name).name != args.name or args.name in ("", ".", ".."):
        parser.error("name must be one directory name")
    train_one(args)


if __name__ == "__main__":
    main()