"""
gradcam_app/app.py
==================
Tkinter-based GUI for Brain Tumor Classification + Grad-CAM visualisation.

Fully responsive — canvases expand with the window.
Press F11 or the ⛶ button to toggle fullscreen.

Usage (from project root):
    python gradcam_app/app.py
    python gradcam_app/app.py --checkpoint checkpoints/best_model.pth
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
import torch
import yaml
from PIL import Image, ImageTk
from torchvision import transforms

try:
    from models.hybrid_model import HybridMemoryNet
    _gradcam_dir = Path(__file__).resolve().parent
    if str(_gradcam_dir) not in sys.path:
        sys.path.insert(0, str(_gradcam_dir))
    from gradcam import GradCAM
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    sys.exit(1)

CLASS_NAMES   = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_COLORS  = {
    "glioma":      "#E74C3C",
    "meningioma":  "#E67E22",
    "notumor":     "#2ECC71",
    "pituitary":   "#3498DB",
}

BG_DARK   = "#0F1117"
BG_PANEL  = "#1A1D27"
BG_CARD   = "#1E2133"
ACCENT    = "#6C63FF"
TEXT_PRI  = "#EAEAEA"
TEXT_SEC  = "#9A9DC2"
FONT_HEAD = ("Segoe UI", 18, "bold")
FONT_SUB  = ("Segoe UI", 10)
FONT_MONO = ("Consolas", 10)
FONT_TINY = ("Segoe UI", 9)

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

MIN_CANVAS = 260

def load_model(checkpoint_path: str, config_path: str, device: torch.device):
    print(checkpoint_path)
    print(config_path)
    cfg = yaml.safe_load(open(config_path))
    model = HybridMemoryNet.from_config(cfg)
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()

def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

class TumorApp(tk.Tk):
    def __init__(self, checkpoint: str, config: str) -> None:
        super().__init__()
        self.title("Brain Tumor Classifier  ·  Grad-CAM Visualiser")
        self.configure(bg=BG_DARK)
        self.minsize(860, 560)

        self._checkpoint  = checkpoint
        self._config      = config
        self._fullscreen  = False

        self._device   = None
        self._model    = None
        self._gradcam  = None

        self._orig_pil      : Image.Image | None = None
        self._orig_bgr      : np.ndarray  | None = None
        self._raw_heatmap   : np.ndarray  | None = None
        self._overlay_bgr   : np.ndarray  | None = None
        self._pred_class    : int | None = None
        self._probs         : list = []
        self._input_tensor  : torch.Tensor | None = None

        self._build_ui()
        self._load_model_async()

        self.bind("<F11>",    lambda e: self._toggle_fullscreen())
        self.bind("<Escape>", lambda e: self._exit_fullscreen())

    def _toggle_fullscreen(self):
        self._fullscreen = not self._fullscreen
        self.attributes("-fullscreen", self._fullscreen)
        self._fs_btn.configure(text="✕ Exit Fullscreen" if self._fullscreen else "⛶ Fullscreen")

    def _exit_fullscreen(self):
        if self._fullscreen:
            self._fullscreen = False
            self.attributes("-fullscreen", False)
            self._fs_btn.configure(text="⛶ Fullscreen")

    def _build_ui(self):

        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=0)
        self.columnconfigure(0, weight=1)

        topbar = tk.Frame(self, bg=BG_PANEL, pady=8, padx=16)
        topbar.grid(row=0, column=0, sticky="ew")
        topbar.columnconfigure(1, weight=1)

        tk.Label(topbar, text="Brain Tumor Grad-CAM", font=FONT_HEAD,
                 bg=BG_PANEL, fg=ACCENT).grid(row=0, column=0, sticky="w")

        self._fs_btn = tk.Button(
            topbar, text="⛶ Fullscreen", command=self._toggle_fullscreen,
            bg=BG_PANEL, fg=TEXT_SEC, activebackground=BG_PANEL,
            relief="flat", font=FONT_TINY, cursor="hand2", padx=8
        )
        self._fs_btn.grid(row=0, column=1, sticky="e")

        self._status_var = tk.StringVar(value="Loading model…")
        tk.Label(topbar, textvariable=self._status_var, font=FONT_TINY,
                 bg=BG_PANEL, fg=TEXT_SEC).grid(row=0, column=2, sticky="e", padx=(4, 0))

        body = tk.Frame(self, bg=BG_DARK)
        body.grid(row=1, column=0, sticky="nsew", padx=8, pady=6)
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=0)

        left = tk.Frame(body, bg=BG_DARK)
        left.grid(row=0, column=0, sticky="nsew")
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        left.columnconfigure(1, weight=1)

        self._canvas_orig, self._frame_orig = self._make_canvas(
            left, "Original MRI", 0, 0)
        self._canvas_overlay, self._frame_overlay = self._make_canvas(
            left, "Grad-CAM Overlay", 0, 1)

        right = tk.Frame(body, bg=BG_PANEL, width=270, padx=14, pady=14)
        right.grid(row=0, column=1, sticky="ns", padx=(8, 0))
        right.pack_propagate(False)
        right.grid_propagate(False)

        tk.Label(right, text="Results", font=("Segoe UI", 13, "bold"),
                 bg=BG_PANEL, fg=TEXT_PRI).pack(anchor="w", pady=(0, 10))

        self._pred_label_var = tk.StringVar(value="—")
        self._pred_label = tk.Label(right, textvariable=self._pred_label_var,
                                    font=("Segoe UI", 20, "bold"),
                                    bg=BG_PANEL, fg=TEXT_PRI)
        self._pred_label.pack(anchor="w")

        self._conf_var = tk.StringVar(value="Confidence: —")
        tk.Label(right, textvariable=self._conf_var, font=FONT_TINY,
                 bg=BG_PANEL, fg=TEXT_SEC).pack(anchor="w", pady=(2, 6))

        ttk.Style().configure("Conf.Horizontal.TProgressbar",
                              troughcolor=BG_CARD, background=ACCENT, thickness=12)
        self._conf_bar = ttk.Progressbar(right, orient="horizontal",
                                          style="Conf.Horizontal.TProgressbar",
                                          length=238, mode="determinate")
        self._conf_bar.pack(anchor="w", pady=(0, 14))

        tk.Label(right, text="Per-class probabilities",
                 font=("Segoe UI", 10, "bold"),
                 bg=BG_PANEL, fg=TEXT_SEC).pack(anchor="w", pady=(0, 4))

        self._class_bars: dict[str, ttk.Progressbar] = {}
        self._class_vals: dict[str, tk.StringVar]    = {}
        for name in CLASS_NAMES:
            row = tk.Frame(right, bg=BG_PANEL)
            row.pack(fill="x", pady=2)
            color = CLASS_COLORS[name]
            tk.Label(row, text=name, width=11, anchor="w",
                     font=FONT_TINY, bg=BG_PANEL, fg=color).pack(side="left")
            var = tk.StringVar(value="0.0 %")
            tk.Label(row, textvariable=var, width=7, anchor="e",
                     font=FONT_MONO, bg=BG_PANEL, fg=TEXT_SEC).pack(side="right")
            bar = ttk.Progressbar(row, orient="horizontal", length=90, mode="determinate")
            bar.pack(side="right", padx=4)
            self._class_bars[name] = bar
            self._class_vals[name] = var

        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=6)

        tk.Label(right, text="Visualize class:",
                 font=("Segoe UI", 9, "bold"),
                 bg=BG_PANEL, fg=TEXT_SEC).pack(anchor="w")

        CAM_OPTIONS = ["Auto (predicted)"] + [n.capitalize() for n in CLASS_NAMES]
        self._cam_class_var = tk.StringVar(value="Auto (predicted)")
        cam_menu = tk.OptionMenu(right, self._cam_class_var, *CAM_OPTIONS,
                                 command=self._on_cam_class_change)
        cam_menu.configure(bg=BG_CARD, fg=TEXT_PRI, activebackground=ACCENT,
                           activeforeground="white", relief="flat",
                           font=FONT_TINY, width=18, highlightthickness=0)
        cam_menu["menu"].configure(bg=BG_CARD, fg=TEXT_PRI,
                                   activebackground=ACCENT, activeforeground="white")
        cam_menu.pack(anchor="w", pady=(2, 8))

        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=4)

        self._info_var = tk.StringVar(value="Upload an MRI image to begin.")
        tk.Label(right, textvariable=self._info_var, font=FONT_TINY,
                 bg=BG_PANEL, fg=TEXT_SEC, wraplength=240, justify="left").pack(anchor="w")

        ckpt_name = Path(self._checkpoint).name
        tk.Label(right, text=f"ckpt: {ckpt_name}", font=("Segoe UI", 7),
                 bg=BG_PANEL, fg="#444466").pack(side="bottom", anchor="w")

        bottom = tk.Frame(self, bg=BG_DARK, padx=12, pady=8)
        bottom.grid(row=2, column=0, sticky="ew")

        tk.Label(bottom, text="Overlay alpha:", font=FONT_TINY,
                 bg=BG_DARK, fg=TEXT_SEC).pack(side="left", padx=(2, 4))
        self._alpha_var = tk.DoubleVar(value=0.50)
        self._alpha_label_var = tk.StringVar(value="0.50")
        tk.Label(bottom, textvariable=self._alpha_label_var, width=4,
                 font=FONT_MONO, bg=BG_DARK, fg=TEXT_PRI).pack(side="left")
        tk.Scale(bottom, from_=0.0, to=1.0, resolution=0.01,
                 orient="horizontal", variable=self._alpha_var,
                 bg=BG_DARK, fg=TEXT_PRI, troughcolor=BG_CARD,
                 highlightthickness=0, bd=0, length=200, showvalue=False,
                 command=self._on_alpha_change).pack(side="left", padx=(0, 16))

        self._btn_upload = self._make_btn(bottom, "⬆  Upload Image",  self._upload,  ACCENT)
        self._btn_run    = self._make_btn(bottom, "▶  Run Inference", self._run,     "#27AE60")
        self._btn_save   = self._make_btn(bottom, "💾  Save Overlay",  self._save,    "#2980B9")

        self._btn_run["state"]  = "disabled"
        self._btn_save["state"] = "disabled"

    def _make_canvas(self, parent, title: str, row: int, col: int):
        frame = tk.Frame(parent, bg=BG_CARD, padx=2, pady=2)
        frame.grid(row=row, column=col, sticky="nsew", padx=4, pady=4)
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1)

        tk.Label(frame, text=title, font=FONT_TINY,
                 bg=BG_CARD, fg=TEXT_SEC).grid(row=0, column=0, pady=(4, 2))

        canvas = tk.Canvas(frame, bg="#0A0C14", highlightthickness=0)
        canvas.grid(row=1, column=0, sticky="nsew")

        canvas.bind("<Configure>", lambda e, c=canvas: self._on_canvas_resize(c))
        self._draw_placeholder(canvas)
        return canvas, frame

    @staticmethod
    def _make_btn(parent, text: str, cmd, color: str) -> tk.Button:
        btn = tk.Button(
            parent, text=text, command=cmd,
            bg=color, fg="white", activebackground=color,
            font=("Segoe UI", 10, "bold"),
            relief="flat", padx=12, pady=6, cursor="hand2",
        )
        btn.pack(side="left", padx=5)
        return btn

    def _on_canvas_resize(self, canvas: tk.Canvas):
        """Redraw whichever image belongs to this canvas at new size."""

        if not hasattr(self, '_canvas_orig') or not hasattr(self, '_canvas_overlay'):
            return

        if canvas.winfo_width() < 10:
            return
        if canvas is self._canvas_orig and self._orig_pil is not None:
            self._show_pil(canvas, self._orig_pil)
        elif canvas is self._canvas_overlay and self._overlay_bgr is not None:
            self._show_pil(canvas, bgr_to_pil(self._overlay_bgr))

    def _load_model_async(self):
        self._btn_upload.configure(state="disabled")

        def _worker():
            try:
                dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = load_model(self._checkpoint, self._config, dev)
                self._device  = dev
                self._model   = model
                self._gradcam = GradCAM(model, dev)
                label = f"Model ready  [{str(dev).upper()}]"
                self.after(0, lambda: self._status_var.set(label))
                self.after(0, lambda: self._btn_upload.configure(state="normal"))
            except Exception as exc:
                self.after(0, lambda: self._status_var.set(f"Load error: {exc}"))
                self.after(0, lambda: messagebox.showerror("Model Load Error", str(exc)))

        threading.Thread(target=_worker, daemon=True).start()

    def _upload(self):
        path = filedialog.askopenfilename(
            title="Select Brain MRI Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                       ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self._orig_pil = Image.open(path).convert("RGB")
            self._orig_bgr = cv2.cvtColor(np.array(self._orig_pil), cv2.COLOR_RGB2BGR)
        except Exception as exc:
            messagebox.showerror("Image Load Error", str(exc))
            return

        self._raw_heatmap = self._overlay_bgr = None
        self._pred_class  = None
        self._probs       = []

        self._show_pil(self._canvas_orig, self._orig_pil)
        self._draw_placeholder(self._canvas_overlay)
        self._reset_results()
        self._btn_run["state"]  = "normal"
        self._btn_save["state"] = "disabled"
        self._info_var.set(f"Loaded: {Path(path).name}")

    def _run(self):
        if self._orig_pil is None or self._gradcam is None:
            return
        self._btn_run["state"] = "disabled"
        self._status_var.set("Running inference…")

        def _worker():
            try:
                tensor = TRANSFORM(self._orig_pil).unsqueeze(0)
                self._input_tensor = tensor
                orig_h, orig_w = self._orig_bgr.shape[:2]

                cam_class_idx = self._get_cam_class_idx(None)

                heatmap, pred_idx, _ = self._gradcam.generate(
                    tensor, class_idx=cam_class_idx, orig_hw=(orig_h, orig_w))
                self._raw_heatmap = heatmap

                alpha = self._alpha_var.get()
                overlay = cv2.addWeighted(
                    self._orig_bgr, 1.0 - alpha, heatmap, alpha, 0)
                self._overlay_bgr = overlay

                with torch.no_grad():
                    out   = self._model(tensor.to(self._device))
                    probs = torch.softmax(out["logits"], dim=1)[0].cpu().tolist()

                self._pred_class = pred_idx
                self._probs      = probs
                self.after(0, self._update_results)
            except Exception as exc:
                import traceback; traceback.print_exc()
                self.after(0, lambda: messagebox.showerror("Inference Error", str(exc)))
                self.after(0, lambda: self._status_var.set("Error."))
                self.after(0, lambda: self._btn_run.configure(state="normal"))

        threading.Thread(target=_worker, daemon=True).start()

    def _rerun_cam(self):
        if self._input_tensor is None or self._gradcam is None:
            return
        self._status_var.set("Recomputing CAM…")

        def _worker():
            try:
                orig_h, orig_w = self._orig_bgr.shape[:2]
                cam_idx = self._get_cam_class_idx(self._pred_class)
                heatmap, _, _ = self._gradcam.generate(
                    self._input_tensor, class_idx=cam_idx, orig_hw=(orig_h, orig_w))
                self._raw_heatmap = heatmap
                alpha = self._alpha_var.get()
                overlay = cv2.addWeighted(
                    self._orig_bgr, 1.0 - alpha, heatmap, alpha, 0)
                self._overlay_bgr = overlay
                self.after(0, lambda: self._show_pil(self._canvas_overlay, bgr_to_pil(overlay)))
                self.after(0, lambda: self._status_var.set("CAM updated."))
            except Exception as exc:
                import traceback; traceback.print_exc()
                self.after(0, lambda: self._status_var.set(f"CAM error: {exc}"))

        threading.Thread(target=_worker, daemon=True).start()

    def _get_cam_class_idx(self, pred_class: int | None) -> int | None:
        """Translate dropdown selection to a class index (or None for auto)."""
        sel = self._cam_class_var.get()
        if sel == "Auto (predicted)":
            return None

        lookup = {n.capitalize(): i for i, n in enumerate(CLASS_NAMES)}
        return lookup.get(sel, None)

    def _on_cam_class_change(self, _val=None):
        """Called when user changes the class dropdown — recompute CAM immediately."""
        if self._input_tensor is not None:
            self._rerun_cam()

    def _update_results(self):
        if self._overlay_bgr is None:
            return
        self._show_pil(self._canvas_overlay, bgr_to_pil(self._overlay_bgr))

        idx   = self._pred_class
        name  = CLASS_NAMES[idx]
        conf  = self._probs[idx]
        color = CLASS_COLORS[name]

        self._pred_label_var.set(name.capitalize())
        self._pred_label.configure(fg=color)
        self._conf_var.set(f"Confidence: {conf*100:.1f} %")
        self._conf_bar["value"] = conf * 100

        for i, cls in enumerate(CLASS_NAMES):
            p = self._probs[i]
            self._class_bars[cls]["value"] = p * 100
            self._class_vals[cls].set(f"{p*100:.1f} %")

        tumor = name != "notumor"
        self._info_var.set(
            f"⚠ Tumor detected: {name.capitalize()}\nGrad-CAM highlights the region the model focused on."
            if tumor else "✅ No tumor detected.\nActivations shown for 'No Tumor' class."
        )
        self._btn_run["state"]  = "normal"
        self._btn_save["state"] = "normal"
        self._status_var.set("Inference complete.")

    def _on_alpha_change(self, val):
        self._alpha_label_var.set(f"{float(val):.2f}")
        if self._raw_heatmap is not None and self._orig_bgr is not None:
            alpha = float(val)
            overlay = cv2.addWeighted(
                self._orig_bgr, 1.0 - alpha, self._raw_heatmap, alpha, 0)
            self._overlay_bgr = overlay
            self._show_pil(self._canvas_overlay, bgr_to_pil(overlay))

    def _save(self):
        if self._overlay_bgr is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("JPEG image", "*.jpg")],
        )
        if path:
            cv2.imwrite(path, self._overlay_bgr)
            self._info_var.set(f"Saved: {Path(path).name}")

    def _show_pil(self, canvas: tk.Canvas, img: Image.Image):
        canvas.update_idletasks()
        w = max(canvas.winfo_width(),  MIN_CANVAS)
        h = max(canvas.winfo_height(), MIN_CANVAS)

        img_copy = img.copy()
        img_copy.thumbnail((w, h), Image.LANCZOS)

        tk_img = ImageTk.PhotoImage(img_copy)
        canvas.delete("all")
        canvas._tk_img = tk_img
        canvas.create_image(w // 2, h // 2, anchor="center", image=tk_img)

    @staticmethod
    def _draw_placeholder(canvas: tk.Canvas):
        canvas.delete("all")
        canvas.update_idletasks()
        w = max(canvas.winfo_width(),  MIN_CANVAS)
        h = max(canvas.winfo_height(), MIN_CANVAS)
        canvas.create_text(w // 2, h // 2, text="No image",
                           fill="#333355", font=("Segoe UI", 13), anchor="center")

    def _reset_results(self):
        self._pred_label_var.set("—")
        self._pred_label.configure(fg=TEXT_PRI)
        self._conf_var.set("Confidence: —")
        self._conf_bar["value"] = 0
        for name in CLASS_NAMES:
            self._class_bars[name]["value"] = 0
            self._class_vals[name].set("0.0 %")
        self._info_var.set("Press ▶ Run Inference to analyse.")

def _resolve_checkpoint(ckpt_arg: str | None) -> str:
    """Resolve a checkpoint path from argument or auto-selection.
    
    Search order:
    1. As-is (absolute or relative to CWD)
    2. Inside PROJECT_ROOT/checkpoints/  (so bare names like 'best_model.pth' work)
    3. Auto-select: best_model.pth first, then highest last_epoch_N.pth
    4. File-picker dialog if nothing found
    """
    ckpt_dir = PROJECT_ROOT / "checkpoints"

    if ckpt_arg:

        p = Path(ckpt_arg)
        if p.exists():
            return str(p)

        p2 = ckpt_dir / p.name
        if p2.exists():
            print(f"[INFO] Resolved '{ckpt_arg}' → {p2}")
            return str(p2)

        print(f"[WARN] Checkpoint not found at '{ckpt_arg}' or '{p2}', opening picker…")

    best = ckpt_dir / "best_model.pth"
    if best.exists():
        print(f"[INFO] Using best_model.pth: {best}")
        return str(best)

    candidates = []
    for p in ckpt_dir.glob("last_epoch_*.pth"):
        try:
            candidates.append((int(p.stem.split("_")[-1]), p))
        except ValueError:
            pass
    if candidates:
        _, latest = max(candidates)
        print(f"[INFO] Using latest epoch checkpoint: {latest}")
        return str(latest)

    import tkinter as _tk
    from tkinter import filedialog as _fd
    _root = _tk.Tk(); _root.withdraw()
    picked = _fd.askopenfilename(
        title="Select model checkpoint (.pth)",
        initialdir=str(ckpt_dir),
        filetypes=[("PyTorch checkpoint", "*.pth *.pt"), ("All files", "*.*")],
    )
    _root.destroy()
    if not picked:
        sys.exit("[ERROR] No checkpoint selected.")
    return picked

def main():
    parser = argparse.ArgumentParser(description="Brain Tumor Grad-CAM App")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to .pth checkpoint, or bare filename inside checkpoints/")
    parser.add_argument("--config",     default="configs/config.yaml")
    args = parser.parse_args()

    config_path = str(PROJECT_ROOT / args.config)
    if not Path(config_path).exists():
        sys.exit(f"[ERROR] Config not found: {config_path}")

    checkpoint_path = _resolve_checkpoint(args.checkpoint)
    print(f"[INFO] Final checkpoint: {checkpoint_path}")

    if not Path(checkpoint_path).exists():
        sys.exit(f"[ERROR] Checkpoint not found: {checkpoint_path}")

    app = TumorApp(checkpoint=checkpoint_path, config=config_path)
    app.geometry("1100x680")
    app.mainloop()

if __name__ == "__main__":
    main()
