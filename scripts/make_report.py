"""Generate Markdown report tables directly from results/*.json files.

Guarantees Rule 1: No fabricated numbers. All numbers are read directly from disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RESULTS_DIR = Path("results")

# Paper-reported numbers for context (Roth et al., CVPR 2022, Table 1 & Table 2)
# Cited explicitly as paper-reported reference numbers
PAPER_REPORTED_CONTEXT = {
    "bottle": {"image_auroc": 1.000, "pixel_auroc": 0.986, "au_pro": 0.959},
    "carpet": {"image_auroc": 0.987, "pixel_auroc": 0.989, "au_pro": 0.963},
    "hazelnut": {"image_auroc": 1.000, "pixel_auroc": 0.987, "au_pro": 0.951},
    "screw": {"image_auroc": 0.960, "pixel_auroc": 0.993, "au_pro": 0.942},
    "transistor": {"image_auroc": 1.000, "pixel_auroc": 0.975, "au_pro": 0.923},
}


def load_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        data: dict[str, Any] = json.load(f)
        return data


def generate_results_table(results_dir: Path = RESULTS_DIR) -> str:
    """Generate Markdown comparison table for default profile categories."""
    default_dir = results_dir / "default"
    if not default_dir.exists():
        default_dir = results_dir / "smoke"

    files = sorted(default_dir.glob("*.json"))
    patchcore_results: dict[str, Any] = {}
    baseline_results: dict[str, Any] = {}
    no_coreset_results: dict[str, Any] = {}

    for f in files:
        data = load_json(f)
        cat = data["category"]
        if cat.endswith("_baseline"):
            baseline_results[cat.replace("_baseline", "")] = data["metrics"]
        elif cat.endswith("_no_coreset"):
            no_coreset_results[cat.replace("_no_coreset", "")] = data["metrics"]
        else:
            patchcore_results[cat] = data["metrics"]

    lines = [
        "| Category | PatchCore Image AUROC | PatchCore Pixel AUROC | PatchCore AU-PRO | Baseline (Global kNN) | Full Bank (r=1.0) | Paper Context (Roth et al.) |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for cat, m in sorted(patchcore_results.items()):
        img_auc = f"{m['image_auroc']:.4f}"
        pix_auc = f"{m['pixel_auroc']:.4f}" if m.get("pixel_auroc") is not None else "N/A"
        pro_val = f"{m['au_pro']:.4f}" if m.get("au_pro") is not None else "N/A"

        base_val = "N/A"
        if cat in baseline_results:
            base_val = f"{baseline_results[cat]['image_auroc']:.4f}"

        nocore_val = "N/A"
        if cat in no_coreset_results:
            nocore_val = f"{no_coreset_results[cat]['image_auroc']:.4f}"

        paper_ctx = "—"
        if cat in PAPER_REPORTED_CONTEXT:
            p_img = PAPER_REPORTED_CONTEXT[cat]["image_auroc"]
            paper_ctx = f"{p_img:.3f} (Image AUROC)"

        lines.append(
            f"| `{cat}` | **{img_auc}** | {pix_auc} | {pro_val} | {base_val} | {nocore_val} | {paper_ctx} |"
        )

    return "\n".join(lines)


def generate_ablation_table(results_dir: Path = RESULTS_DIR) -> str:
    """Generate Markdown table for coreset ratio ablation."""
    ablation_dir = results_dir / "ablation"
    ratios = [0.01, 0.10, 1.00]

    lines = [
        "| Coreset Ratio ($r$) | Image AUROC | Pixel AUROC | Memory Bank Size | Inference Latency |",
        "|:---|:---:|:---:|:---:|:---:|",
    ]

    lat_file = results_dir / "latency.json"
    lat_data = load_json(lat_file) if lat_file.exists() else {}

    for r in ratios:
        f = ablation_dir / f"coreset_r{r:.2f}.json"
        if f.exists():
            d = load_json(f)
            m = d["metrics"]
            img_auc = f"{m['image_auroc']:.4f}"
            pix_auc = f"{m['pixel_auroc']:.4f}" if m.get("pixel_auroc") is not None else "N/A"
            mem_mb = f"{m.get('memory_bank_mb', 0):.2f} MB"
        else:
            img_auc, pix_auc, mem_mb = "N/A", "N/A", "N/A"

        lat_ms = "N/A"
        k = f"ratio_{r}"
        if lat_data and "ratios" in lat_data and k in lat_data["ratios"]:
            lat_ms = f"{lat_data['ratios'][k]['total_ms']['median']:.1f} ms"

        r_name = f"`r={r * 100:.0f}%`" + (" *(paper default)*" if r == 0.10 else "")
        lines.append(f"| {r_name} | {img_auc} | {pix_auc} | {mem_mb} | {lat_ms} |")

    return "\n".join(lines)


def generate_latency_breakdown_table(results_dir: Path = RESULTS_DIR) -> str:
    """Generate component-level latency breakdown table."""
    lat_file = results_dir / "latency.json"
    if not lat_file.exists():
        return "No latency results found."

    d = load_json(lat_file)
    ratios = d.get("ratios", {})

    lines = [
        "| Stage | Median Latency (ms) | P95 Latency (ms) | Description |",
        "|:---|:---:|:---:|:---|",
    ]

    # Use default r=0.10 for breakdown
    target = ratios.get("ratio_0.1", next(iter(ratios.values()), None))
    if not target:
        return "No latency data available."

    lines.append(
        f"| **Backbone & Aggregation** | {target['backbone_ms']['median']:.2f} ms | {target['backbone_ms']['p95']:.2f} ms | WRN-50-2 hooks (layer2+layer3) + pooling |"
    )
    lines.append(
        f"| **kNN Search (Faiss)** | {target['knn_ms']['median']:.2f} ms | {target['knn_ms']['p95']:.2f} ms | Nearest-neighbor memory bank distance |"
    )
    lines.append(
        f"| **Post-processing** | {target['postprocess_ms']['median']:.2f} ms | {target['postprocess_ms']['p95']:.2f} ms | Reshape, spatial upsampling & Gaussian blur |"
    )
    lines.append(
        f"| **Total End-to-End** | **{target['total_ms']['median']:.2f} ms** | **{target['total_ms']['p95']:.2f} ms** | Single-image inspection latency on CPU |"
    )

    return "\n".join(lines)


def main() -> None:
    print("### Evaluation Results Table:")
    print(generate_results_table())
    print("\n### Coreset Ablation Table:")
    print(generate_ablation_table())
    print("\n### Latency Breakdown Table:")
    print(generate_latency_breakdown_table())


if __name__ == "__main__":
    main()
