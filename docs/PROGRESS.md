# Progress

## Stage A — Foundation
- **Status:** COMPLETED

### Done
- [x] Phase 0: Scaffold (repo structure, pyproject.toml, Makefile, configs, CI, pre-commit)
- [x] Phase 1: Data (MVTec AD dataset with val-good holdout, synthetic dataset, CLI data-check)
- [x] Phase 2: PatchCore Core (WideResNet-50-2 feature extractor, local aggregation, greedy k-center coreset, unified Faiss/Torch kNN, scoring map & blur, thresholding, safe save/load, global kNN baseline)

### Gates
- Gate 0: Setup, lint (ruff + mypy), and tests pass cleanly.
- Gate 1: Data tests pass on synthetic data; data-check works with and without MVTec AD present.
- Gate 2: 45 unit & integration tests pass in ~14s on CPU with 90% coverage; end-to-end synthetic fit-score verified.

### Pending
- Stage B: Evaluation (metrics: AUROC / AU-PRO, baselines, ablation, latency, visuals)
- Stage C: API + documentation (FastAPI serve, Docker, report, audit)

### Known Issues
- None.

### Next Stage
Stage B requires: MVTec AD dataset downloaded to `data/mvtec_ad/` (instructions printed by `visual-inspector data-check`).
