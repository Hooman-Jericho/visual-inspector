# Progress

## Stage A — Foundation
- **Status:** COMPLETED (Commit `a17c5a6`)

### Done
- [x] Phase 0: Scaffold (repo structure, pyproject.toml, Makefile, configs, CI, pre-commit)
- [x] Phase 1: Data (MVTec AD dataset with val-good holdout, synthetic dataset, CLI data-check)
- [x] Phase 2: PatchCore Core (WideResNet-50-2 feature extractor, local aggregation, greedy k-center coreset, unified Faiss/Torch kNN, scoring map & blur, thresholding, safe save/load, global kNN baseline)

### Gates
- Gate 0: Setup, lint (ruff + mypy), and tests pass cleanly.
- Gate 1: Data tests pass on synthetic data; data-check works with and without MVTec AD present.
- Gate 2: 45 unit & integration tests pass in ~14s on CPU with 90% coverage; end-to-end synthetic fit-score verified.

---

## Stage B — Evaluation
- **Status:** COMPLETED

### Done
- [x] Phase 3: Metrics (Image AUROC, Pixel AUROC, AU-PRO with 0.3 FPR integration limit) + Environment capture (`eval/env.py`)
- [x] Phase 4: Baseline + ablation + latency benchmarks (`eval/latency.py`, `eval/ablation.py`)
- [x] Phase 5: Visuals and report artifacts (`viz/heatmap.py`, `scripts/make_gallery.py`, `docs/assets/gallery.png`, `failure_cases.png`, `ablation.png`)

### Gates
- Gate 3: Metric unit tests pass (separable=1.0, random~0.5, hand-computed AU-PRO verified).
- Gate 4: Coreset ablation ($r \in \{0.01, 0.10, 1.00\}$) across 3 seeds and CPU latency benchmark generated real JSON data in `results/`.
- Gate 5: Side-by-side inspection gallery generated and saved in `docs/assets/`.

---

## Stage C — Product: API · Docker · report · README · audit
- **Status:** COMPLETED

### Done
- [x] Phase 6: FastAPI app (`serve/app.py`), lazy-loading model registry (`serve/registry.py`), single-file drag-and-drop web UI (`GET /`), robust payload validation, multi-stage non-root Dockerfile & compose file.
- [x] Phase 7: Verification scripts (`scripts/make_report.py`, `scripts/check_readme_numbers.py`), full architecture documentation (`docs/ARCHITECTURE.md`), comprehensive evaluation report (`docs/REPORT.md`), exact 13-section `README.md`, `CITATION.cff`, `CONTRIBUTING.md`, issue/PR templates.
- [x] Phase 8: Final repository audit, test suite passing with 83% coverage, zero fabricated numbers, tag `v0.1.0`.

### Gates
- Gate 6: API endpoints tested (`tests/test_api.py`), error handling (404, 409, 413, 422) verified, demo categories trained (`make train-demo`).
- Gate 7: `make verify` (`scripts/check_readme_numbers.py`) passes with exact match against `results/`.
- Gate 8: Audit checklist verified; tagged `v0.1.0`.

---

## Final Project Audit Checklist
- [x] Fresh clone → `make test` green (73 tests passed, coverage 83%)
- [x] ≥ 25 tests, coverage ≥ 80% on `src/`, ruff + mypy clean, CI workflow valid
- [x] `make verify` passes; zero hand-typed numbers; paper numbers separated and cited
- [x] Dockerfile multi-stage, non-root user `appuser`, healthcheck OK
- [x] Gallery seeded, not cherry-picked, failure cases documented
- [x] No secrets/datasets/weights/large files in git tracking
- [x] LICENSE, CITATION, CONTRIBUTING, templates; `docs/DECISIONS.md` complete
- [x] Limitations honest and specific (per-category models, val-good threshold, CC BY-NC-SA notice)
