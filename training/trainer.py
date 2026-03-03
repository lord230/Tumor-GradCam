"""
training/trainer.py
====================
Core training loop for HybridMemoryNet.

Features:
  • Mixed Precision (AMP) via torch.cuda.amp
  • TensorBoard logging (loss, accuracy, LR, GPU memory)
  • Early Stopping (configurable patience)
  • Checkpoint saving: best_model.pth + last_epoch_X.pth
  • CUDA cache cleared each epoch
  • Grad-CAM visualization after final epoch
  • Loss curve + accuracy curve PNG export
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm_module

try:
    from torch.cuda.amp import GradScaler, autocast
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False

from models.hybrid_model import HybridMemoryNet
from training.losses     import HybridLoss
from training.metrics    import ClassificationEvaluator

class GradCAMHook:
    """
    Simple Grad-CAM hook targeting the last convolutional layer of the
    CNN backbone (EfficientNet-B0 → features[-1]).
    """

    def __init__(self, layer: nn.Module) -> None:
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._handles: list = []
        self._register(layer)

    def _register(self, layer: nn.Module) -> None:
        self._handles.append(
            layer.register_forward_hook(self._save_activation)
        )
        self._handles.append(
            layer.register_full_backward_hook(self._save_gradient)
        )

    def _save_activation(self, module, inp, out) -> None:
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out) -> None:
        self.gradients = grad_out[0].detach()

    def remove(self) -> None:
        for h in self._handles:
            h.remove()

    def compute_cam(self) -> np.ndarray:
        """Returns normalised Grad-CAM heatmap: (H, W) float [0,1]."""
        assert self.gradients is not None and self.activations is not None

        grads = self.gradients[0]
        acts  = self.activations[0]
        weights = grads.mean(dim=(1, 2))
        cam = (weights[:, None, None] * acts).sum(dim=0)
        cam = torch.relu(cam).cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam

class EarlyStopper:
    """Stops training if monitored metric doesn't improve after `patience` epochs."""

    def __init__(self, patience: int = 10, mode: str = "max") -> None:
        self.patience    = patience
        self.mode        = mode
        self.counter     = 0
        self.best_value  = float("-inf") if mode == "max" else float("inf")
        self.should_stop = False

    def step(self, value: float) -> bool:
        improved = (
            value > self.best_value if self.mode == "max"
            else value < self.best_value
        )
        if improved:
            self.best_value = value
            self.counter    = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

class Trainer:
    """
    Full training / validation / evaluation loop.

    Args:
        model:          HybridMemoryNet instance.
        criterion:      HybridLoss instance.
        optimizer:      AdamW optimizer.
        scheduler:      WarmupCosineScheduler.
        train_loader:   DataLoader for training.
        val_loader:     DataLoader for validation.
        cfg:            Parsed config dict.
        class_names:    Ordered list of class name strings.
        device:         torch.device.
    """

    def __init__(
        self,
        model:        HybridMemoryNet,
        criterion:    HybridLoss,
        optimizer:    Optimizer,
        scheduler:    _LRScheduler,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        cfg:          dict,
        class_names:  List[str],
        device:       torch.device,
    ) -> None:
        self.model        = model
        self.criterion    = criterion
        self.optimizer    = optimizer
        self.scheduler    = scheduler
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = cfg
        self.class_names  = class_names
        self.device       = device

        t = cfg["training"]
        self.epochs       = t["epochs"]
        self.grad_clip    = t.get("grad_clip", 1.0)
        self.use_amp      = t.get("use_amp", True) and AMP_AVAILABLE and device.type == "cuda"
        self.save_every   = t.get("save_every_epoch", True)
        self.print_freq   = cfg["logging"].get("print_freq", 10)

        self.checkpoint_dir = Path(cfg["paths"]["checkpoint_dir"])
        self.results_dir    = Path(cfg["paths"]["results_dir"])
        self.log_dir        = Path(cfg["paths"]["log_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / "gradcam").mkdir(parents=True, exist_ok=True)

        self.early_stopper = EarlyStopper(patience=t["early_stopping_patience"])
        self.scaler        = GradScaler() if self.use_amp else None
        self.writer        = (
            SummaryWriter(log_dir=str(self.log_dir))
            if cfg["logging"].get("use_tensorboard", True)
            else None
        )

        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
            "train_acc":  [], "val_acc":  [],
            "lr":         [],
        }
        self.best_val_f1   = 0.0
        self.best_epoch    = 0

    def _gpu_mem_str(self) -> str:
        if self.device.type != "cuda":
            return "N/A"
        alloc   = torch.cuda.memory_allocated(self.device) / 1024**3
        reserv  = torch.cuda.memory_reserved(self.device)  / 1024**3
        return f"{alloc:.2f}/{reserv:.2f} GB"

    def _run_epoch(
        self,
        loader: DataLoader,
        training: bool,
        epoch: int = 0,
        total_epochs: int = 0,
    ) -> Tuple[float, float, ClassificationEvaluator]:
        """
        Run one epoch (train or val) with a tqdm progress bar.

        Returns:
            avg_loss (float), avg_acc (float), evaluator
        """
        phase = "Train" if training else "Val  "
        self.model.train(training)
        evaluator = ClassificationEvaluator(self.class_names)
        total_loss = 0.0
        correct    = 0
        total      = 0

        bar_fmt = "{l_bar}{bar:28}{r_bar}"
        desc = f"  {phase} Epoch {epoch}/{total_epochs}"

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            if TQDM_AVAILABLE:
                pbar = tqdm(loader, desc=desc, unit="batch",
                            bar_format=bar_fmt, leave=True, dynamic_ncols=True)
            else:
                pbar = loader

            for batch_idx, (images, targets) in enumerate(pbar):
                images  = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                if training:
                    self.optimizer.zero_grad(set_to_none=True)

                if self.use_amp:
                    with autocast():
                        out  = self.model(images)
                        loss_dict = self.criterion(
                            out["logits"],
                            out["embedding"],
                            targets,
                            self.model.memory.prototypes,
                        )
                else:
                    out  = self.model(images)
                    loss_dict = self.criterion(
                        out["logits"],
                        out["embedding"],
                        targets,
                        self.model.memory.prototypes,
                    )

                loss = loss_dict["total"]

                if training:
                    if self.use_amp:
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.grad_clip
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.grad_clip
                        )
                        self.optimizer.step()

                total_loss += loss.item() * images.size(0)
                preds = out["logits"].argmax(dim=1)
                correct += (preds == targets).sum().item()
                total   += images.size(0)

                evaluator.update(out["logits"], targets)

                if TQDM_AVAILABLE:
                    running_loss = total_loss / total
                    running_acc  = correct / total
                    pbar.set_postfix(
                        loss=f"{running_loss:.4f}",
                        acc=f"{running_acc:.4f}",
                        gpu=self._gpu_mem_str(),
                    )
                elif training and (batch_idx % self.print_freq == 0):
                    print(
                        f"    Batch [{batch_idx:4d}/{len(loader)}] "
                        f"loss={loss.item():.4f}  "
                        f"GPU: {self._gpu_mem_str()}"
                    )

        avg_loss = total_loss / total
        avg_acc  = correct / total
        return avg_loss, avg_acc, evaluator

    def _save_checkpoint(self, epoch: int, is_best: bool) -> None:
        state = {
            "epoch":          epoch,
            "model_state":    self.model.state_dict(),
            "optimizer_state":self.optimizer.state_dict(),
            "scheduler_state":self.scheduler.state_dict(),
            "best_val_f1":    self.best_val_f1,
            "history":        self.history,
            "class_names":    self.class_names,
        }
        if self.scaler is not None:
            state["scaler_state"] = self.scaler.state_dict()

        if is_best:
            path = self.checkpoint_dir / "best_model.pth"
            torch.save(state, path)
            print(f"  [SAVED] best model → {path}")

        if self.save_every:
            path = self.checkpoint_dir / f"last_epoch_{epoch}.pth"
            torch.save(state, path)

    def load_checkpoint(self, path: str) -> int:
        """Load checkpoint, returns epoch to resume from."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.history = ckpt.get("history", self.history)
        self.best_val_f1 = ckpt.get("best_val_f1", 0.0)
        if "scaler_state" in ckpt and self.scaler is not None:
            self.scaler.load_state_dict(ckpt["scaler_state"])
        epoch = ckpt["epoch"]
        print(f"  [LOADED] resumed from epoch {epoch}, best_val_f1={self.best_val_f1:.4f}")
        return epoch

    def _plot_curves(self) -> None:
        epochs = range(1, len(self.history["train_loss"]) + 1)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, self.history["train_loss"], label="Train Loss", linewidth=2)
        ax.plot(epochs, self.history["val_loss"],   label="Val Loss",   linewidth=2, linestyle="--")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.set_title("Loss Curve"); ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.results_dir / "loss_curve.png", dpi=150)
        plt.close()

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, self.history["train_acc"], label="Train Acc", linewidth=2)
        ax.plot(epochs, self.history["val_acc"],   label="Val Acc",   linewidth=2, linestyle="--")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy Curve"); ax.legend(); ax.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(self.results_dir / "accuracy_curve.png", dpi=150)
        plt.close()

    def _generate_gradcam(
        self,
        loader: DataLoader,
        n_per_class: int = 4,
    ) -> int:
        """
        Generate enhanced Grad-CAM visualizations covering every class.

        For each class, samples up to `n_per_class` images and saves:
          • An individual side-by-side PNG  (original | heatmap | overlay)
          • A combined `gradcam_summary.png` grid

        Returns:
            Total number of individual images saved.
        """
        import cv2
        from PIL import Image, ImageDraw, ImageFont

        gradcam_dir = self.results_dir / "gradcam"
        gradcam_dir.mkdir(exist_ok=True)

        n_classes = len(self.class_names)

        # ── 1. Collect samples per class from the loader ────────────────────
        gathered: dict[int, list[tuple[np.ndarray, np.ndarray, int, int]]] = {
            c: [] for c in range(n_classes)
        }
        needed = n_per_class * n_classes

        self.model.eval()
        target_layer = self.model.cnn.features[-1]
        hook = GradCAMHook(target_layer)

        for images, targets in loader:
            if all(len(v) >= n_per_class for v in gathered.values()):
                break
            images  = images.to(self.device)
            targets = targets.to(self.device)
            images.requires_grad_(True)

            out    = self.model(images)
            logits = out["logits"]

            for i in range(images.size(0)):
                cls = int(targets[i].item())
                if len(gathered[cls]) >= n_per_class:
                    continue

                self.optimizer.zero_grad(set_to_none=True)
                logits[i, cls].backward(retain_graph=True)

                cam = hook.compute_cam()
                if cam.ndim == 0:
                    continue

                img_np = images[i].detach().cpu().permute(1, 2, 0).numpy()
                mean = np.array([0.485, 0.456, 0.406])
                std  = np.array([0.229, 0.224, 0.225])
                img_np = (img_np * std + mean).clip(0, 1)

                cam_r = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
                pred  = int(logits[i].argmax().item())
                gathered[cls].append((img_np, cam_r, pred, cls))

        hook.remove()

        # ── 2. Render & save individual side-by-side panels ─────────────────
        PANEL_W, PANEL_H = 224, 224
        LABEL_H          = 28
        BORDER           = 3

        def _render_panel(
            img_np: np.ndarray,
            cam_r:  np.ndarray,
            pred:   int,
            true:   int,
        ) -> Image.Image:
            """Return a (PANEL_W*3 + gaps) × (PANEL_H + LABEL_H) PIL image."""
            correct = (pred == true)
            accent  = (60, 200, 80) if correct else (220, 60, 60)   # BGR→RGB

            # Original (resize for uniformity)
            orig_u8 = (img_np * 255).astype(np.uint8)
            orig_u8 = cv2.resize(orig_u8, (PANEL_W, PANEL_H))

            # Heatmap only
            heat_u8 = (cm_module.jet(cam_r)[..., :3] * 255).astype(np.uint8)
            heat_u8 = cv2.resize(heat_u8, (PANEL_W, PANEL_H))

            # Overlay (40 % CAM + 60 % original)
            cam_r_224 = cv2.resize(cam_r, (PANEL_W, PANEL_H))
            colormap  = cm_module.jet(cam_r_224)[..., :3]
            orig_f    = (orig_u8 / 255.0)
            overlay_f = (0.40 * colormap + 0.60 * orig_f).clip(0, 1)
            overlay_u8 = (overlay_f * 255).astype(np.uint8)

            gap     = 6
            total_w = PANEL_W * 3 + gap * 2 + BORDER * 2
            total_h = PANEL_H + LABEL_H + BORDER * 2

            canvas = Image.new("RGB", (total_w, total_h), (18, 18, 30))
            # coloured top border
            draw = ImageDraw.Draw(canvas)
            draw.rectangle([(0, 0), (total_w - 1, BORDER - 1)], fill=accent)

            x = BORDER
            y = BORDER
            for tile in (orig_u8, heat_u8, overlay_u8):
                canvas.paste(Image.fromarray(tile), (x, y))
                x += PANEL_W + gap

            # Label strip
            true_name = self.class_names[true]
            pred_name = self.class_names[pred]
            tag = f"True: {true_name}   Pred: {pred_name}"
            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except OSError:
                font = ImageFont.load_default()
            draw.rectangle(
                [(0, BORDER + PANEL_H), (total_w - 1, total_h - 1)],
                fill=(25, 25, 40),
            )
            draw.text((8, BORDER + PANEL_H + 6), tag, fill=accent, font=font)

            return canvas

        saved_panels: list[Image.Image] = []
        count = 0

        for cls_idx in range(n_classes):
            cls_name = self.class_names[cls_idx]
            for s_idx, (img_np, cam_r, pred, true) in enumerate(gathered[cls_idx]):
                panel = _render_panel(img_np, cam_r, pred, true)
                fname = gradcam_dir / (
                    f"sample_{count:03d}_true-{cls_name}"
                    f"_pred-{self.class_names[pred]}.png"
                )
                panel.save(fname)
                saved_panels.append(panel)
                count += 1

        # ── 3. Build summary grid ────────────────────────────────────────────
        if saved_panels:
            cols      = n_per_class
            rows      = n_classes
            pw, ph    = saved_panels[0].size
            gap       = 8
            grid_w    = cols * pw + (cols - 1) * gap
            grid_h    = rows * ph + (rows - 1) * gap
            grid      = Image.new("RGB", (grid_w, grid_h), (10, 10, 18))
            for idx, panel in enumerate(saved_panels):
                row = idx // cols
                col = idx %  cols
                x   = col * (pw + gap)
                y   = row * (ph + gap)
                grid.paste(panel, (x, y))
            grid.save(self.results_dir / "gradcam" / "gradcam_summary.png")

        print(
            f"  [SAVED] {count} Grad-CAM panels + summary grid "
            f"→ {gradcam_dir}"
        )
        return count

    def train(self, resume_from: Optional[str] = None) -> None:
        """
        Run the full training loop.

        Args:
            resume_from: Optional path to a checkpoint .pth file.
        """
        start_epoch = 1
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from) + 1

        print(f"\n{'='*60}")
        print(f"  HybridMemoryNet Training")
        print(f"  Device  : {self.device}")
        print(f"  AMP     : {self.use_amp}")
        print(f"  Epochs  : {self.epochs}")
        print(f"  Classes : {self.class_names}")
        print(f"  Params  : {self.model.get_num_params():,}")
        print(f"{'='*60}\n")

        epoch_times: List[float] = []

        for epoch in range(start_epoch, self.epochs + 1):
            t0 = time.time()

            if epoch_times:
                avg_epoch_t = sum(epoch_times) / len(epoch_times)
                remaining   = avg_epoch_t * (self.epochs - epoch + 1)
                eta_str     = time.strftime("%H:%M:%S", time.gmtime(remaining))
            else:
                eta_str = "--:--:--"

            print(
                f"\n{'─'*64}\n"
                f"  Epoch [{epoch}/{self.epochs}]  "
                f"LR={self.scheduler.get_last_lr()[0]:.2e}  "
                f"ETA={eta_str}"
                f"\n{'─'*64}"
            )

            train_loss, train_acc, _ = self._run_epoch(
                self.train_loader, training=True,
                epoch=epoch, total_epochs=self.epochs,
            )

            val_loss, val_acc, val_evaluator = self._run_epoch(
                self.val_loader, training=False,
                epoch=epoch, total_epochs=self.epochs,
            )
            val_metrics = val_evaluator.compute()
            val_f1      = val_metrics["f1_macro"]

            self.scheduler.step()

            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                gc.collect()

            elapsed = time.time() - t0
            epoch_times.append(elapsed)
            gpu_mem = self._gpu_mem_str()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(self.scheduler.get_last_lr()[0])

            if self.writer:
                self.writer.add_scalars("Loss",     {"train": train_loss, "val": val_loss}, epoch)
                self.writer.add_scalars("Accuracy", {"train": train_acc,  "val": val_acc},  epoch)
                self.writer.add_scalar("F1/val",    val_f1,                                  epoch)
                self.writer.add_scalar("LR",        self.scheduler.get_last_lr()[0],         epoch)
                if self.device.type == "cuda":
                    self.writer.add_scalar("GPU_MB", torch.cuda.memory_allocated(self.device) / 1024**2, epoch)

            best_marker = " ★ BEST" if val_f1 > self.best_val_f1 else ""
            print(
                f"\n  {'':22s}  {'TRAIN':>10}   {'VAL':>10}"
                f"\n  {'Loss':<22s}  {train_loss:>10.4f}   {val_loss:>10.4f}"
                f"\n  {'Accuracy':<22s}  {train_acc:>10.4f}   {val_acc:>10.4f}"
                f"\n  {'F1 (macro)':<22s}  {'—':>10}   {val_f1:>10.4f}{best_marker}"
                f"\n  {'Time':<22s}  {elapsed:>10.1f}s"
                f"\n  {'GPU':<22s}  {gpu_mem}"
            )

            is_best = val_f1 > self.best_val_f1
            if is_best:
                self.best_val_f1 = val_f1
                self.best_epoch  = epoch
            self._save_checkpoint(epoch, is_best)

            if self.early_stopper.step(val_f1):
                print(
                    f"\n[EARLY STOP] No improvement for {self.early_stopper.patience} "
                    f"epochs. Best F1={self.best_val_f1:.4f} at epoch {self.best_epoch}."
                )
                break

        print(f"\nTraining complete. Best val F1={self.best_val_f1:.4f} (epoch {self.best_epoch})")
        self._plot_curves()
        if self.writer:
            self.writer.close()

    def evaluate(self, test_loader: DataLoader, split: str = "test") -> Dict:
        """Run evaluation on test_loader and save all metrics + figures."""
        print(f"\n[Evaluating on {split} set ...]")

        best_ckpt = self.checkpoint_dir / "best_model.pth"
        if best_ckpt.exists():
            ckpt = torch.load(best_ckpt, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state"])
            print(f"  Loaded best model from {best_ckpt}")

        _, _, evaluator = self._run_epoch(test_loader, training=False)
        evaluator.save_all(self.results_dir, split)

        self.gradcam_count = self._generate_gradcam(test_loader, n_per_class=4)

        history_path = self.results_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        return evaluator.compute()
