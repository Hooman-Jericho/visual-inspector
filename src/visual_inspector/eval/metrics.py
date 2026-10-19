"""Evaluation metrics for anomaly detection: Image AUROC, Pixel AUROC, and AU-PRO.

References:
- Bergmann et al., MVTec AD -- A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection, IJCV 2021.
- Roth et al., Towards Total Recall in Industrial Anomaly Detection, CVPR 2022.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label as label_components
from sklearn.metrics import roc_auc_score


def compute_image_auroc(labels: list[int] | np.ndarray, scores: list[float] | np.ndarray) -> float:
    """Compute image-level Area Under the ROC Curve.

    Args:
        labels: Binary labels (0 = normal, 1 = anomalous).
        scores: Anomaly scores for each image.

    Returns:
        AUROC score in range [0.0, 1.0].
    """
    labels_arr = np.asarray(labels, dtype=int)
    scores_arr = np.asarray(scores, dtype=float)

    if len(np.unique(labels_arr)) < 2:
        return 1.0

    return float(roc_auc_score(labels_arr, scores_arr))


def compute_pixel_auroc(
    masks: list[np.ndarray] | np.ndarray,
    anomaly_maps: list[np.ndarray] | np.ndarray,
) -> float:
    """Compute pixel-level Area Under the ROC Curve.

    Args:
        masks: Ground-truth binary masks of shape (N, H, W) or (N, 1, H, W).
        anomaly_maps: Anomaly score maps of shape (N, H, W) or (N, 1, H, W).

    Returns:
        Pixel AUROC score in range [0.0, 1.0].
    """
    flat_masks = np.asarray(masks).squeeze().reshape(-1)
    flat_scores = np.asarray(anomaly_maps).squeeze().reshape(-1)

    # Binarize masks
    flat_masks = (flat_masks > 0.5).astype(int)

    if len(np.unique(flat_masks)) < 2:
        return 1.0

    # Subsample if extremely large to prevent OOM / slow computation (max 2M pixels)
    if len(flat_masks) > 2_000_000:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(flat_masks), size=2_000_000, replace=False)
        flat_masks = flat_masks[idx]
        flat_scores = flat_scores[idx]

    return float(roc_auc_score(flat_masks, flat_scores))


def compute_au_pro(
    masks: list[np.ndarray] | np.ndarray,
    anomaly_maps: list[np.ndarray] | np.ndarray,
    max_fpr: float = 0.30,
    n_thresholds: int = 500,
) -> float:
    """Compute Area Under the Per-Region Overlap (AU-PRO) up to max_fpr.

    MVTec protocol:
    1. Identify connected components in each anomalous image mask.
    2. Across a sweep of thresholds, compute:
       - PRO: average overlap ratio for each ground truth component.
       - FPR: false positive rate on normal (defect-free) pixels.
    3. Integrate PRO(FPR) curve from FPR=0 to max_fpr, normalized by max_fpr.

    Args:
        masks: Ground-truth masks of shape (N, H, W) or (N, 1, H, W).
        anomaly_maps: Predicted anomaly maps of shape (N, H, W) or (N, 1, H, W).
        max_fpr: Integration upper limit (default: 0.30 per MVTec protocol).
        n_thresholds: Number of threshold evaluation points.

    Returns:
        AU-PRO score in range [0.0, 1.0].
    """
    masks_arr = np.asarray(masks, dtype=float).squeeze()
    maps_arr = np.asarray(anomaly_maps, dtype=float).squeeze()

    if masks_arr.ndim == 2:
        masks_arr = masks_arr[np.newaxis, ...]
        maps_arr = maps_arr[np.newaxis, ...]

    binary_masks = (masks_arr > 0.5).astype(np.uint8)

    # 1. Extract connected component regions from anomalous masks
    structure = np.ones((3, 3), dtype=int)
    regions: list[tuple[int, np.ndarray, int]] = []  # (image_idx, bool_mask, area)

    for i in range(len(binary_masks)):
        mask_i = binary_masks[i]
        if not np.any(mask_i):
            continue
        labeled, num_features = label_components(mask_i, structure=structure)
        for comp_id in range(1, num_features + 1):
            comp_mask = labeled == comp_id
            area = int(np.sum(comp_mask))
            if area > 0:
                regions.append((i, comp_mask, area))

    if not regions:
        return 1.0

    # Normal pixels mask across all images
    normal_pixels_mask = binary_masks == 0
    total_normal_pixels = int(np.sum(normal_pixels_mask))

    if total_normal_pixels == 0:
        return 1.0

    # Threshold selection (from min to max of anomaly scores)
    min_score = float(np.min(maps_arr))
    max_score = float(np.max(maps_arr))

    if min_score == max_score:
        return 0.5

    thresholds = np.linspace(max_score, min_score, n_thresholds)

    # Evaluate thresholds
    fpr_to_pro: dict[float, float] = {}

    for th in thresholds:
        pred_binary = maps_arr >= th

        # Compute FPR on defect-free pixels
        false_positives = int(np.sum(pred_binary & normal_pixels_mask))
        fpr = round(false_positives / total_normal_pixels, 6)

        # Compute mean PRO over all regions
        region_overlaps: list[float] = []
        for img_idx, comp_mask, area in regions:
            overlap = np.sum(pred_binary[img_idx] & comp_mask) / area
            region_overlaps.append(float(overlap))

        mean_pro = float(np.mean(region_overlaps))

        if fpr not in fpr_to_pro or mean_pro > fpr_to_pro[fpr]:
            fpr_to_pro[fpr] = mean_pro

        if fpr >= max_fpr and mean_pro >= 0.999:
            break

    # Ensure (0.0, 0.0) exists if no threshold had FPR=0
    if 0.0 not in fpr_to_pro:
        fpr_to_pro[0.0] = 0.0

    sorted_fprs = sorted(fpr_to_pro.keys())
    fpr_arr = np.array(sorted_fprs, dtype=float)
    pro_arr = np.array([fpr_to_pro[f] for f in sorted_fprs], dtype=float)

    # Monotonically non-decreasing
    pro_arr = np.maximum.accumulate(pro_arr)

    # Interpolate to max_fpr
    if fpr_arr[-1] < max_fpr:
        fpr_arr = np.append(fpr_arr, max_fpr)
        pro_arr = np.append(pro_arr, pro_arr[-1])
    else:
        pro_at_max = float(np.interp(max_fpr, fpr_arr, pro_arr))
        valid = fpr_arr <= max_fpr
        fpr_arr = np.append(fpr_arr[valid], max_fpr)
        pro_arr = np.append(pro_arr[valid], pro_at_max)

    # Trapezoidal integration divided by max_fpr
    if hasattr(np, "trapezoid"):
        au_pro = float(np.trapezoid(pro_arr, fpr_arr)) / max_fpr
    else:
        from scipy.integrate import trapezoid as scipy_trapz

        au_pro = float(scipy_trapz(pro_arr, fpr_arr)) / max_fpr
    return float(np.clip(au_pro, 0.0, 1.0))
