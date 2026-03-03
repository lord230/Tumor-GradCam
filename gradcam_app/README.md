# Brain Tumor Grad-CAM App

Tkinter-based desktop application for brain tumour classification with Grad-CAM visualisation.

## Usage

Run from the **project root**:

```bash
# Auto-detects latest checkpoint
python gradcam_app/app.py

# Specify checkpoint + config explicitly
python gradcam_app/app.py --checkpoint checkpoints/best_model.pth --config configs/config.yaml
```

## Features

| Feature | Details |
|---|---|
| Image upload | JPG, PNG, BMP, TIFF |
| Model | HybridMemoryNet (EfficientNet-B0 + Swin-Tiny) |
| GradCAM | Hooks `model.cnn.features[-1]` (last MBConv block) |
| Overlay alpha | Adjustable slider (0–1) with live re-blend |
| Results panel | Predicted class · Confidence bar · Per-class probabilities |
| Save | Export overlay as PNG or JPEG |

## Classes

`glioma` · `meningioma` · `notumor` · `pituitary`
