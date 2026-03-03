"""
training/metrics.py
===================
Evaluation metrics and visualisation utilities.

Functions:
    compute_metrics     — accuracy, precision, recall, F1, ROC-AUC
    per_class_accuracy  — per-class accuracy breakdown
    save_metrics        — writes metrics.json
    plot_confusion_matrix
    plot_roc_curves
    ClassificationEvaluator — stateful accumulator across batches
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import label_binarize

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        y_true:      (N,)    ground-truth class indices
        y_pred:      (N,)    predicted class indices
        y_prob:      (N, C)  predicted probabilities (softmax)
        class_names: list of class name strings

    Returns:
        dict of scalar metric values
    """
    num_classes = len(class_names)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred,  average="macro", zero_division=0)
    f1   = f1_score(y_true,  y_pred,     average="macro", zero_division=0)

    try:
        y_bin = label_binarize(y_true, classes=list(range(num_classes)))
        roc_auc = roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")
    except Exception:
        roc_auc = float("nan")

    per_class_acc = per_class_accuracy(y_true, y_pred, num_classes, class_names)

    metrics = {
        "accuracy":          round(float(acc),  4),
        "precision_macro":   round(float(prec), 4),
        "recall_macro":      round(float(rec),  4),
        "f1_macro":          round(float(f1),   4),
        "roc_auc_macro":     round(float(roc_auc), 4),
        "per_class_accuracy": per_class_acc,
    }
    return metrics

def per_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    class_names: List[str],
) -> Dict[str, float]:
    """Per-class accuracy (TP / (TP + FN) for each class)."""
    result = {}
    for i, name in enumerate(class_names):
        mask = y_true == i
        if mask.sum() == 0:
            result[name] = float("nan")
        else:
            result[name] = round(float((y_pred[mask] == i).mean()), 4)
    return result

def save_metrics(
    metrics: Dict,
    results_dir: Path,
    split: str = "test",
) -> Path:
    """Save metrics dict to JSON."""
    out = results_dir / f"metrics_{split}.json"
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    return out

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    results_dir: Path,
    split: str = "test",
) -> Path:
    """Plot and save confusion matrix as PNG."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)

    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = "white" if cm_norm[i, j] > 0.6 else "black"
            ax.text(
                j, i,
                f"{cm[i, j]}\n({cm_norm[i, j]:.2f})",
                ha="center", va="center",
                fontsize=9, color=color,
            )

    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title(f"Confusion Matrix — {split}")
    plt.tight_layout()

    out = results_dir / f"confusion_matrix_{split}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    return out

def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    results_dir: Path,
    split: str = "test",
) -> Path:
    """Plot per-class ROC curves and save as PNG."""
    from sklearn.metrics import roc_curve, auc

    num_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, num_classes))

    for i, (name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves — {split}")
    ax.legend(loc="lower right")
    plt.tight_layout()

    out = results_dir / f"roc_curve_{split}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    return out

class ClassificationEvaluator:
    """
    Stateful accumulator for batched evaluation.

    Usage:
        ev = ClassificationEvaluator(class_names)
        for logits, targets in loader:
            ev.update(logits, targets)
        metrics = ev.compute()
    """

    def __init__(self, class_names: List[str]) -> None:
        self.class_names  = class_names
        self._all_preds:  List[np.ndarray] = []
        self._all_targets: List[np.ndarray] = []
        self._all_probs:  List[np.ndarray] = []

    def reset(self) -> None:
        self._all_preds   = []
        self._all_targets = []
        self._all_probs   = []

    def update(
        self,
        logits:  torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """
        Args:
            logits:  (B, C) raw logits (on any device)
            targets: (B,)   class indices
        """
        probs = F.softmax(logits.detach().cpu().float(), dim=1).numpy()
        preds = probs.argmax(axis=1)
        y     = targets.detach().cpu().numpy()

        self._all_probs.append(probs)
        self._all_preds.append(preds)
        self._all_targets.append(y)

    def compute(self) -> Dict[str, object]:
        y_true = np.concatenate(self._all_targets)
        y_pred = np.concatenate(self._all_preds)
        y_prob = np.concatenate(self._all_probs)
        return compute_metrics(y_true, y_pred, y_prob, self.class_names)

    def save_all(
        self,
        results_dir: Path,
        split: str = "test",
    ) -> None:
        """Compute metrics and save JSON + figures."""
        y_true = np.concatenate(self._all_targets)
        y_pred = np.concatenate(self._all_preds)
        y_prob = np.concatenate(self._all_probs)

        metrics = compute_metrics(y_true, y_pred, y_prob, self.class_names)
        save_metrics(metrics, results_dir, split)
        plot_confusion_matrix(y_true, y_pred, self.class_names, results_dir, split)
        plot_roc_curves(y_true, y_prob, self.class_names, results_dir, split)

        report = classification_report(y_true, y_pred, target_names=self.class_names)
        report_path = results_dir / f"classification_report_{split}.txt"
        report_path.write_text(report)

        print(f"\n[Metrics — {split}]")
        for k, v in metrics.items():
            if k != "per_class_accuracy":
                print(f"  {k:<22}: {v}")
        print("\n[Per-Class Accuracy]")
        for cls, acc in metrics["per_class_accuracy"].items():
            print(f"  {cls:<18}: {acc:.4f}")
        print()
