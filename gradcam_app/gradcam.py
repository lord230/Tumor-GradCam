"""
gradcam_app/gradcam.py
======================
LayerCAM implementation for HybridMemoryNet.

Method: LayerCAM (Jiang et al., 2021)
  Per-pixel gradient × activation gives much sharper localisation than
  standard Grad-CAM for small objects (e.g. pituitary tumours).

      L = ReLU( Σ_k  ReLU(∂y^c/∂A^k_ij) · A^k_ij )

Target layer: model.cnn.features[8]  (last Conv block, 7×7, 1280 channels)
  The deepest layer with spatial maps — highest semantic content.
  features[7] (320ch) also works but has less discriminative power.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import cv2

class GradCAM:
    """
    LayerCAM for HybridMemoryNet's EfficientNet-B0 CNN backbone.

    Args:
        model:  Loaded HybridMemoryNet (eval mode).
        device: torch.device.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device) -> None:
        self.model  = model
        self.device = device
        self.model.eval()

        self._activations: torch.Tensor | None = None
        self._gradients:   torch.Tensor | None = None

        target_layer = model.cnn.features[8]

        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def generate(
        self,
        input_tensor: torch.Tensor,
        class_idx: int | None = None,
        orig_hw: tuple[int, int] = (224, 224),
    ) -> tuple[np.ndarray, int, float]:
        """
        Generate a LayerCAM heatmap.

        Args:
            input_tensor : (1, 3, H, W) preprocessed image tensor.
            class_idx    : Target class index. None → argmax (predicted class).
            orig_hw      : (H, W) of the original image for upscaling.

        Returns:
            heatmap_bgr  : (H, W, 3) uint8 BGR heatmap.
            pred_class   : int — predicted class index.
            confidence   : float — softmax probability of predicted class.
        """
        input_tensor = input_tensor.to(self.device)

        self.model.zero_grad()
        output = self.model(input_tensor)
        logits = output["logits"]

        probs      = torch.softmax(logits, dim=1)
        pred_class = int(logits.argmax(dim=1).item())
        confidence = float(probs[0, pred_class].item())

        if class_idx is None:
            class_idx = pred_class

        self.model.zero_grad()
        logits[0, class_idx].backward()

        acts  = self._activations[0]
        grads = self._gradients[0]

        grad_relu = F.relu(grads)
        cam = (grad_relu * acts).sum(dim=0)
        cam = F.relu(cam)
        cam = cam.detach().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        h, w = orig_hw
        cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
        cam_resized = np.clip(cam_resized, 0, 1)

        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam_resized), cv2.COLORMAP_JET
        )

        return heatmap, pred_class, confidence
