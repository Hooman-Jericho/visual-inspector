# Design Decisions

| # | Decision | Alternatives Considered | Reason |
|---|----------|------------------------|--------|
| 1 | Use `hatchling` as build backend | setuptools, flit | Simplest modern PEP 517 backend with good uv support |
| 2 | faiss-cpu as primary kNN, torch cdist fallback | annoy, scikit-learn BallTree | faiss is the standard for exact/approximate NN; torch fallback ensures zero extra deps |
| 3 | Adaptive avg pool for channel reduction (not PCA) | PCA, random projection | Simpler, no fitting step, works identically across runs; paper uses random projection but adaptive pool is equivalent for fixed architectures |
| 4 | Gaussian blur via scipy.ndimage | torch conv2d with gaussian kernel | scipy is already a dep; simpler than constructing a torch kernel |
| 5 | val_good split with torch.Generator for determinism | sklearn train_test_split | Fewer deps, torch Generator is seedable and deterministic across platforms |
| 6 | Save as memory_bank.pt + meta.json (no pickle) | Single pickle file | Security: weights_only=True; JSON metadata is human-readable |
| 7 | Pure NumPy + Pillow colormap & chart generator | Matplotlib dependency | Zero heavy plotting dependencies; resilient in offline/headless container builds |
| 8 | Trapezoidal integration for AU-PRO with 0.3 FPR limit | Discrete threshold summation | Exactly follows MVTec AD official evaluation protocol and Roth et al. 2022 |
| 9 | Lazy-loading singleton model registry | Eager loading all categories on startup | Minimal initial RAM footprint; models loaded on-demand per incoming API request |
| 10 | Non-root multi-stage CPU Docker image | Single-stage default root container | Production security hardening; minimal runtime image size without build tools |
| 11 | Automated check_readme_numbers.py gating in CI | Manual documentation updates | Enforces zero fabricated numbers by failing tests if README drifts from results/*.json |

