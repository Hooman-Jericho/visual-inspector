"""Heatmap generation and visualization utilities for defect localization."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageDraw


def apply_jet_colormap(v: np.ndarray) -> np.ndarray:
    """Compute Jet colormap directly using pure numpy without external dependencies."""
    r = np.clip(1.5 - np.abs(4.0 * v - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * v - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * v - 1.0), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0).astype(np.uint8)


def _to_numpy_image(image: Image.Image | np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert input image into a (H, W, 3) uint8 numpy array."""
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    elif isinstance(image, torch.Tensor):
        img_t = image.detach().cpu().float()
        if img_t.ndim == 3 and img_t.shape[0] in (1, 3):
            # (C, H, W) -> (H, W, C)
            img_t = img_t.permute(1, 2, 0)
        img_np = img_t.numpy()
        # If normalized with ImageNet stats, denormalize approximately
        if img_np.min() < 0:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = np.clip((img_np * std + mean) * 255.0, 0, 255)
        elif img_np.max() <= 1.0:
            img_np = np.clip(img_np * 255.0, 0, 255)
        return img_np.astype(np.uint8)
    elif isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[0] in (1, 3):
            image = np.transpose(image, (1, 2, 0))
        if image.dtype != np.uint8 and image.max() <= 1.0:
            image = (image * 255.0).clip(0, 255)
        return image.astype(np.uint8)
    raise ValueError(f"Unsupported image type: {type(image)}")


def _to_numpy_map(anomaly_map: Image.Image | np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert anomaly map into a 2D float32 numpy array."""
    if isinstance(anomaly_map, Image.Image):
        return np.asarray(anomaly_map.convert("L"), dtype=np.float32) / 255.0
    if isinstance(anomaly_map, torch.Tensor):
        anomaly_map = anomaly_map.detach().cpu().squeeze().numpy()
    arr = np.asarray(anomaly_map, dtype=np.float32).squeeze()
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D map, got shape {arr.shape}")
    return arr


def create_heatmap_overlay(
    image: Image.Image | np.ndarray | torch.Tensor,
    anomaly_map: np.ndarray | torch.Tensor,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> Image.Image:
    """Create a blended heatmap overlay on top of the original image.

    Args:
        image: Original RGB image.
        anomaly_map: 2D anomaly score map.
        alpha: Blending weight for heatmap overlay (0.0 to 1.0).
        colormap: Matplotlib colormap name ('jet', 'plasma', 'turbo').

    Returns:
        PIL Image of blended result.
    """
    img_np = _to_numpy_image(image)
    h, w = img_np.shape[:2]

    map_np = _to_numpy_map(anomaly_map)

    # Resize map to image size if different
    if map_np.shape != (h, w):
        map_img = Image.fromarray(map_np)
        map_img = map_img.resize((w, h), Image.Resampling.BILINEAR)
        map_np = np.array(map_img)

    # Normalize to [0, 1]
    min_v, max_v = float(map_np.min()), float(map_np.max())
    norm_map = (map_np - min_v) / (max_v - min_v) if max_v > min_v else np.zeros_like(map_np)

    # Apply colormap
    heatmap_rgb = apply_jet_colormap(norm_map)

    # Alpha blend
    blended = (1.0 - alpha) * img_np + alpha * heatmap_rgb
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return Image.fromarray(blended)


def create_comparison_image(
    image: Image.Image | np.ndarray | torch.Tensor,
    mask: Image.Image | np.ndarray | torch.Tensor | None,
    anomaly_map: np.ndarray | torch.Tensor,
    score: float | None = None,
    threshold: float | None = None,
    label: str | None = None,
) -> Image.Image:
    """Create a side-by-side comparison: [Input] | [Ground Truth] | [Heatmap Overlay].

    Args:
        image: Input image.
        mask: Ground truth mask (optional).
        anomaly_map: Anomaly score map.
        score: Image anomaly score (optional).
        threshold: Decision threshold (optional).
        label: Custom label header (optional).

    Returns:
        PIL Image of 3-panel comparison.
    """
    img_np = _to_numpy_image(image)
    h, w = img_np.shape[:2]

    # Overlay
    overlay_img = create_heatmap_overlay(img_np, anomaly_map)

    # Mask
    if mask is not None:
        mask_np = _to_numpy_map(mask)
        if mask_np.shape != (h, w):
            m_img = Image.fromarray((mask_np * 255).astype(np.uint8))
            mask_np = np.array(m_img.resize((w, h), Image.Resampling.NEAREST)) / 255.0
        mask_rgb = np.repeat((mask_np > 0.5)[:, :, np.newaxis] * 255, 3, axis=2).astype(np.uint8)
    else:
        mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # Assemble 3 panels horizontally with 4px border
    border = 4
    canvas_w = w * 3 + border * 4
    canvas_h = h + 40 + border * 2  # header space for titles

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(30, 30, 35))
    draw = ImageDraw.Draw(canvas)

    # Paste panels
    x0 = border
    x1 = x0 + w + border
    x2 = x1 + w + border
    y_img = 35

    canvas.paste(Image.fromarray(img_np), (x0, y_img))
    canvas.paste(Image.fromarray(mask_rgb), (x1, y_img))
    canvas.paste(overlay_img, (x2, y_img))

    # Header labels
    draw.text((x0 + 10, 10), "Input Image", fill=(220, 220, 220))
    draw.text((x1 + 10, 10), "Ground Truth Mask", fill=(220, 220, 220))

    title_overlay = "Heatmap Overlay"
    if score is not None:
        title_overlay += f" (Score: {score:.3f})"
    draw.text((x2 + 10, 10), title_overlay, fill=(220, 220, 220))

    return canvas
