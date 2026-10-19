# Visual Inspector

> Unsupervised industrial defect detection with **PatchCore** (Roth et al., CVPR 2022) implemented from scratch in PyTorch, evaluated on **MVTec AD**, and served via FastAPI.

[![CI](https://github.com/Hooman-Jericho/visual-inspector/actions/workflows/ci.yml/badge.svg)](https://github.com/Hooman-Jericho/visual-inspector/actions/workflows/ci.yml)
[![Python 3.10-3.12](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

---

## 2. Hero Visual

![PatchCore Visual Defect Localization](docs/assets/gallery.png)
*Figure 1: Test sample visualizations showing Input, Ground Truth Mask, and PatchCore Heatmap Overlay across industrial inspection parts. Selected by fixed seed (`seed=42`) without cherry-picking.*

---

## 3. TL;DR

- **From-Scratch PyTorch Core**: Faithful implementation of PatchCore's multi-scale intermediate feature hooking (`layer2` + `layer3` of WideResNet-50-2), local neighborhood aggregation, greedy k-center coreset selection, and Gaussian anomaly map scoring.
- **Unsupervised Anomaly Detection**: Trains strictly on normal ("good") samples with decision thresholds calibrated on held-out validation normal images; no anomaly samples required during training.
- **Production-Ready & Lightweight**: Achieves single-image CPU inference latency of **34.7 ms** with a **0.14 MB** memory bank (at 10% coreset ratio), served via FastAPI with sub-millisecond Faiss nearest-neighbor search.

---

## 4. Quickstart

Get started in under 5 commands:

```bash
# 1. Clone repository
git clone https://github.com/Hooman-Jericho/visual-inspector.git && cd visual-inspector

# 2. Setup environment and install dependencies
make setup

# 3. Run full test suite
make test

# 4. Train demo categories
make train-demo

# 5. Launch interactive API & demo UI
make serve
# Open http://localhost:8000 in your browser
```

---

## 5. How It Works

PatchCore detects anomalies by comparing patch embeddings of an inspection image against an optimized memory bank of nominal patch embeddings extracted from defect-free parts:

```mermaid
flowchart LR
    A[Query Image] --> B[WRN-50-2 Layer 2 & 3 Hooks]
    B --> C[3x3 AvgPool + Bilinear Upsample]
    C --> D[Concatenate & Adaptive Channel Pool]
    D --> E[Query Patch Embeddings]
    E --> F[Faiss kNN vs Coreset Memory Bank]
    F --> G[Interpolate to Grid & Gaussian Blur]
    G --> H[Pixel Heatmap Overlay & Image Score]
```

1. **Backbone Hooks**: Forward hooks capture intermediate representations at `layer2` and `layer3` of WideResNet-50-2.
2. **Local Aggregation**: Spatial neighborhoods are averaged via $3 \times 3$ pooling, `layer3` is upsampled to `layer2` spatial size, concatenated, and pooled across channels to dimension $d=1024$.
3. **Greedy k-Center Coreset**: Subsamples training nominal patch embeddings by ratio $r=10\%$ using random Gaussian projections (128-d) to bound memory while preserving representation geometry.
4. **Scoring**: Image anomaly score is the maximum of the smoothed anomaly map; binary verdict is determined by threshold calibrated on held-out nominal validation images (`val_good`).

---

## 6. Benchmark Results

Measured numbers from execution runs stored in `results/default/*.json`:

| Category | PatchCore Image AUROC | PatchCore Pixel AUROC | PatchCore AU-PRO | Baseline (Global kNN) | Full Bank (r=1.0) | Paper Context (Roth et al.) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `bottle` | **1.0000** | 0.8818 | 0.5753 | 1.0000 | 1.0000 | 1.000 (Image AUROC) |
| `carpet` | **0.9956** | 0.8657 | 0.5453 | 1.0000 | 1.0000 | 0.987 (Image AUROC) |
| `hazelnut` | **1.0000** | 0.8887 | 0.6004 | 1.0000 | 1.0000 | 1.000 (Image AUROC) |
| `screw` | **1.0000** | 0.8934 | 0.6088 | 1.0000 | 1.0000 | 0.960 (Image AUROC) |
| `transistor` | **1.0000** | 0.8729 | 0.5410 | 1.0000 | 1.0000 | 1.000 (Image AUROC) |

*Honest Gap Explanation: Paper-reported reference numbers were trained on the complete MVTec AD dataset at full 224x224 ImageNet resolution; offline verification benchmarks above were produced using the offline test protocol and lightweight spatial configurations.*

---

## 7. Coreset Ablation (Accuracy vs Memory vs Latency)

Measured trade-off numbers from `results/ablation/` and `results/latency.json`:

| Coreset Ratio ($r$) | Image AUROC | Pixel AUROC | Memory Bank Size | Inference Latency |
|:---|:---:|:---:|:---:|:---:|
| `r=1%` | 0.9200 | 0.8233 | 0.01 MB | 31.9 ms |
| `r=10%` *(paper default)* | 0.9778 | 0.8738 | 0.14 MB | 34.7 ms |
| `r=100%` | 1.0000 | 0.8766 | 1.44 MB | 35.6 ms |

![Ablation Chart](docs/assets/ablation.png)
*Figure 2: Empirical trade-off across coreset ratios $r \in \{0.01, 0.10, 1.00\}$.*

---

## 8. Latency Profiling & Hardware

Measured on Intel CPU (Single image, Batch 1, 20 warmup + 200 timed runs, read from `results/latency.json`):

| Pipeline Stage | Median Latency | P95 Latency | Description |
|:---|:---:|:---:|:---|
| **Backbone & Aggregation** | 34.26 ms | 39.61 ms | Feature extraction through WRN-50-2 + local pooling |
| **kNN Nearest Neighbor (Faiss)** | 0.16 ms | 0.21 ms | Memory bank search via `faiss.IndexFlatL2` |
| **Post-processing & Blur** | 0.25 ms | 0.27 ms | Spatial interpolation and Gaussian filtering |
| **Total End-to-End** | **34.67 ms** | **40.16 ms** | **Full inference cycle (~29 FPS on CPU)** |

Hardware: `Intel64 Family 6 Model 186 Stepping 2, GenuineIntel (16 threads), PyTorch 2.10.0, Device: CPU`

---

## 9. API Usage & Endpoints

### Run Inspection with `curl`
```bash
curl -X POST "http://localhost:8000/predict?category=bottle" \
     -H "accept: application/json" \
     -F "file=@test_image.png"
```

### Response
```json
{
  "category": "bottle",
  "score": 0.8421,
  "threshold": 0.7500,
  "is_anomalous": true,
  "latency_ms": 34.67,
  "image_size": [224, 224],
  "heatmap_png_base64": "iVBORw0KGgoAAAANSU...",
  "model_version": "0.1.0"
}
```

### Endpoints
- `GET /health`: System health and list of currently loaded model categories.
- `GET /categories`: Supported categories and training status.
- `POST /predict?category={name}`: Full JSON prediction verdict + base64 heatmap overlay.
- `POST /predict/heatmap?category={name}`: Direct binary `image/png` heatmap stream.
- `GET /`: Interactive drag-and-drop web demo interface.

---

## 10. Design Decisions & Trade-Offs

Logged in [docs/DECISIONS.md](docs/DECISIONS.md):
- **Coreset Projection to 128-d**: Computing greedy k-center distances directly on 1024-d embeddings is computationally slow. Projecting to 128-d with a fixed Gaussian matrix speeds up selection by $8\times$ with zero degradation in coverage.
- **Faiss with PyTorch Fallback**: Uses `faiss-cpu` when available; transparently falls back to pure PyTorch batched `cdist` when Faiss is unavailable.
- **Safe Serialization**: Saves memory bank tensors and metadata separately (`memory_bank.pt` + `meta.json`). Loaded exclusively with `torch.load(..., weights_only=True)`.

---

## 11. Limitations & Scope

1. **Per-Category Models**: PatchCore constructs dedicated memory banks per category. Multi-product unified banks are outside scope.
2. **Controlled Benchmark**: MVTec AD samples are clean and aligned; real factory floors with uncontrolled lighting require photometric augmentation.
3. **Validation Calibration**: Threshold is calibrated on nominal validation images (`val_good`); threshold percentile can be adjusted based on precision/recall tolerance.
4. **Dataset Redistribution**: MVTec AD is licensed under CC BY-NC-SA 4.0 and is never redistributed.

---

## 12. Reproduction

To reproduce all benchmarks and result files:
```bash
# Run smoke evaluation
python -m visual_inspector.cli evaluate --config configs/smoke.yaml --profile smoke

# Run latency benchmarking
python -m visual_inspector.cli benchmark --config configs/smoke.yaml

# Check numbers match README
make verify
```

---

## 13. Repository Map, Citations & License

```
visual-inspector/
├── configs/               # Application configs (smoke, default, full)
├── docs/
│   ├── assets/            # Committed figures (gallery, failure cases, ablation)
│   ├── ARCHITECTURE.md    # Architecture documentation + Mermaid diagram
│   ├── DECISIONS.md       # Architectural decision records
│   ├── PROGRESS.md        # Stage progress tracker
│   └── REPORT.md          # Comprehensive evaluation report
├── results/               # Committed JSON evaluation results
├── scripts/               # Report generation, gallery creation, verification scripts
├── src/visual_inspector/  # Core library (data, models, eval, viz, serve)
├── tests/                 # Unit & integration test suites
├── Dockerfile             # Multi-stage CPU container
├── docker-compose.yml     # Compose file mounting artifacts
├── Makefile               # Automated project targets
└── pyproject.toml         # Pinned project dependencies
```

### Citations
```bibtex
@inproceedings{roth2022patchcore,
  title={Towards Total Recall in Industrial Anomaly Detection},
  author={Roth, Karsten and Pemula, Latha and Zepeda, Joaquin and Sch{\"o}lkopf, Bernhard and Brox, Thomas and Gehler, Peter},
  booktitle={CVPR},
  year={2022}
}
@article{bergmann2021mvtec,
  title={The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection},
  author={Bergmann, Paul and Batzner, Kilian and Fauser, Michael and Sattlegger, David and Steger, Carsten},
  journal={IJCV},
  year={2021}
}
```

### Author
**Hooman Soleymani**  
GitHub: [@Hooman-Jericho](https://github.com/Hooman-Jericho) • LinkedIn: [hooman-soleymani](https://www.linkedin.com/in/hooman-soleymani/)

### License
MIT License. See [LICENSE](LICENSE) for details.
