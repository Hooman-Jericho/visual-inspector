# visual-inspector

> Unsupervised industrial defect detection with **PatchCore** (Roth et al., CVPR 2022), implemented from scratch in PyTorch.

**Author:** Hooman Soleymani · [GitHub](https://github.com/Hooman-Jericho)

---

## Overview

This project implements PatchCore — a state-of-the-art unsupervised anomaly detection method — from scratch, evaluated on the [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) benchmark. It detects and localizes manufacturing defects without any defective training samples.

### Key Features

- **PatchCore from scratch** — no third-party anomaly detection libraries
- **WRN-50-2 backbone** with multi-scale feature aggregation (layer2 + layer3)
- **Greedy k-center coreset** for memory-efficient memory bank
- **kNN scoring** with faiss-cpu (+ pure-torch fallback)
- **Pixel-level anomaly maps** with Gaussian smoothing
- **FastAPI serving** endpoint (Stage C)
- **Reproducible**: fixed seeds, environment capture, config hashing

## Quick Start

```bash
# Clone and setup
git clone https://github.com/Hooman-Jericho/visual-inspector.git
cd visual-inspector
make setup

# Run lint + tests
make verify

# Smoke test (synthetic data, no download needed)
make smoke
```

### MVTec AD Dataset

Download from [mvtec.com](https://www.mvtec.com/company/research/datasets/mvtec-ad) (CC BY-NC-SA 4.0, non-commercial). Extract to `data/mvtec_ad/`.

```bash
# Verify data
visual-inspector data-check --root data/mvtec_ad
```

## Project Status

| Stage | Description | Status |
|-------|-------------|--------|
| A | Foundation: scaffold, data, PatchCore core | 🔧 In progress |
| B | Evaluation: metrics, benchmarks, report | ⏳ Pending |
| C | API + documentation | ⏳ Pending |

## License

MIT © Hooman Soleymani
