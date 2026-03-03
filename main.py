"""
main.py
========
Entry point for HybridMemoryNet training and evaluation.

Usage:
    python main.py
    python main.py --config configs/config.yaml
    python main.py --skip-dataset-setup
    python main.py --resume checkpoints/last_epoch_10.pth
    python main.py --eval-only
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Tuple

import helper

REQUIRED_MODULES = [
    "numpy",
    "torch",
    "torchvision",
    "timm",
    "yaml",
    "PIL",
    "sklearn",
    "matplotlib",
    "tensorboard",
    "tqdm",
]
helper.install_modules(REQUIRED_MODULES)

PROJECT_STRUCTURE = [

    (".",             ["configs", "checkpoints", "logs",
                       "results", "data"],                           []),
    ("results",       ["gradcam"],                                   []),
    ("models",        [],                                            ["__init__.py"]),
    ("training",      [],                                            ["__init__.py"]),
]
helper.create_structure(PROJECT_STRUCTURE)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import yaml

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_transforms(cfg: dict, split: str) -> transforms.Compose:
    """
    Build augmentation / normalization pipeline.

    Train split gets additional augmentation for regularisation.
    Val / Test use only resize + normalize.
    """
    img_size = cfg["data"]["image_size"]
    aug      = cfg["data"].get("augment_train", True)

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if split == "train" and aug:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.2, hue=0.05),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

class _TransformSubset(torch.utils.data.Dataset):
    """Wraps a Subset and applies a different transform than the base dataset."""
    def __init__(self, subset: torch.utils.data.Subset, transform):
        self.subset    = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]

        if self.transform:
            img = self.transform(img)
        return img, label

def build_loaders(
    cfg: dict,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Build train / val / test DataLoaders directly from dataset/.

    - dataset/Training/  →  80 % train  +  20 % val  (in-memory split, no copying)
    - dataset/Testing/   →  test set

    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    from torch.utils.data import Subset
    dataset_root = Path(cfg["paths"]["dataset_root"])
    training_dir = dataset_root / "Training"
    testing_dir  = dataset_root / "Testing"

    for d in (training_dir, testing_dir):
        if not d.exists():
            sys.exit(f"[ERROR] {d} not found. Check 'dataset_root' in config.yaml.")

    batch_size  = cfg["data"]["batch_size"]
    num_workers = cfg["data"].get("num_workers", 4)
    pin_memory  = cfg["data"].get("pin_memory", True)
    val_split   = cfg["data"].get("val_split", 0.20)
    seed        = cfg.get("seed", 42)

    full_train_ds = ImageFolder(training_dir, transform=None)
    class_names   = full_train_ds.classes
    n_total       = len(full_train_ds)

    indices = list(range(n_total))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_val   = max(1, int(n_total * val_split))
    n_train = n_total - n_val

    train_subset = Subset(full_train_ds, indices[:n_train])
    val_subset   = Subset(full_train_ds, indices[n_train:])

    train_ds = _TransformSubset(train_subset, get_transforms(cfg, "train"))
    val_ds   = _TransformSubset(val_subset,   get_transforms(cfg, "val"))

    test_ds = ImageFolder(testing_dir, transform=get_transforms(cfg, "test"))

    print(f"  Classes : {class_names}")
    print(f"  Train   : {n_train} images")
    print(f"  Val     : {n_val} images")
    print(f"  Test    : {len(test_ds)} images\n")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader, test_loader, class_names

def build_model_and_optimizer(
    cfg: dict,
    device: torch.device,
):
    """Build model, loss, optimizer, scheduler."""
    from models.hybrid_model  import HybridMemoryNet
    from training.losses      import HybridLoss
    from training.scheduler   import build_scheduler

    model = HybridMemoryNet.from_config(cfg).to(device)

    t = cfg["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=t["lr"],
        weight_decay=t["weight_decay"],
    )
    criterion = HybridLoss.from_config(cfg)
    scheduler = build_scheduler(optimizer, cfg)

    return model, criterion, optimizer, scheduler

def print_gpu_info(device: torch.device) -> None:
    if device.type != "cuda":
        print("  Running on CPU\n")
        return
    props = torch.cuda.get_device_properties(device)
    total_gb = props.total_memory / 1024**3
    print(f"  GPU     : {props.name}")
    print(f"  VRAM    : {total_gb:.1f} GB")
    print(f"  CUDA    : {torch.version.cuda}\n")

def main() -> None:
    parser = argparse.ArgumentParser(description="HybridMemoryNet — Brain Tumor Classification")
    parser.add_argument("--config",              default="configs/config.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--skip-dataset-setup",  action="store_true",
                        help="Skip dataset_setup.py (use existing data/ directory)")
    parser.add_argument("--resume",              default=None,
                        help="Path to checkpoint .pth file to resume training")
    parser.add_argument("--eval-only",           action="store_true",
                        help="Only evaluate best model on test set (no training)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print("  HybridMemoryNet — Brain Tumor Classification")
    print(f"{'='*60}")
    print_gpu_info(device)

    if not args.skip_dataset_setup and not args.eval_only:
        import subprocess
        print("Running dataset_setup.py ...\n")
        result = subprocess.run(
            [sys.executable, "dataset_setup.py", "--config", args.config],
            check=True,
        )

    print("Building DataLoaders ...")
    train_loader, val_loader, test_loader, class_names = build_loaders(cfg)

    print("Building model ...")
    model, criterion, optimizer, scheduler = build_model_and_optimizer(cfg, device)
    print(f"  Trainable params: {model.get_num_params():,}\n")

    from training.trainer import Trainer

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        class_names=class_names,
        device=device,
    )

    if not args.eval_only:
        trainer.train(resume_from=args.resume)

    test_metrics = trainer.evaluate(test_loader, split="test")

    print("\n[Final Test Metrics]")
    for k, v in test_metrics.items():
        if k != "per_class_accuracy":
            print(f"  {k:<22}: {v}")

    results_dir = Path(cfg["paths"]["results_dir"])
    print(f"\nAll results saved to: {results_dir.resolve()}\n")

if __name__ == "__main__":
    main()
