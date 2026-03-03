"""
dataset_setup.py
================
Validates the dataset folder structure and prints class distribution.
No files are copied — training reads directly from dataset/Training
and dataset/Testing at runtime.

Usage:
    python dataset_setup.py [--config configs/config.yaml]
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

import helper
helper.install_modules(["yaml"])

import yaml

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def count_images(class_dir: Path) -> int:
    return sum(1 for p in class_dir.rglob("*") if p.suffix.lower() in EXTENSIONS)

def validate_and_stats(dataset_root: Path, val_split: float) -> dict:
    training_dir = dataset_root / "Training"
    testing_dir  = dataset_root / "Testing"

    for d in (training_dir, testing_dir):
        if not d.exists():
            raise FileNotFoundError(f"Missing folder: {d}")

    classes = sorted(
        d.name for d in training_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )
    if not classes:
        raise RuntimeError(f"No class folders found in {training_dir}")

    missing = [c for c in classes if not (testing_dir / c).exists()]
    if missing:
        print(f"  [WARN] Testing is missing folders for: {missing}")
    else:
        print(f"  [OK]  Testing has all {len(classes)} class folders.")

    stats = {"classes": classes, "val_split": val_split, "distribution": {}}
    total_train = total_val = total_test = 0

    line = "─" * 56
    print(f"\n{line}")
    print(f"{'Class':<18} {'Train(~)'!s:>9} {'Val(~)'!s:>7} {'Test':>7}")
    print(line)

    for cls in classes:
        n_all  = count_images(training_dir / cls)
        n_val  = max(1, int(n_all * val_split))
        n_tr   = n_all - n_val
        n_test = count_images(testing_dir / cls) if (testing_dir / cls).exists() else 0

        print(f"{cls:<18} {n_tr:>9} {n_val:>7} {n_test:>7}")
        total_train += n_tr
        total_val   += n_val
        total_test  += n_test

        stats["distribution"][cls] = {
            "train_approx": n_tr,
            "val_approx":   n_val,
            "test":         n_test,
        }

    print(line)
    print(f"{'TOTAL':<18} {total_train:>9} {total_val:>7} {total_test:>7}")
    print(f"{line}\n")

    stats["totals"] = {
        "train_approx": total_train,
        "val_approx":   total_val,
        "test":         total_test,
    }
    return stats

def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset validator for HybridMemoryNet")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg          = load_config(args.config)
    dataset_root = Path(cfg["paths"]["dataset_root"])
    val_split    = cfg["data"].get("val_split", 0.20)

    print(f"\n=== Dataset Validation ===")
    print(f"  Root     : {dataset_root.resolve()}")
    print(f"  Val split: {val_split:.0%}\n")

    stats = validate_and_stats(dataset_root, val_split)

    out_path = dataset_root.parent / "data_stats.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  [SAVED] {out_path}")
    print("\nDataset validation complete. No files were copied.\n")

if __name__ == "__main__":
    main()
