# Design Decisions

| # | Decision | Alternatives Considered | Reason |
|---|----------|------------------------|--------|
| 1 | Use `hatchling` as build backend | setuptools, flit | Simplest modern PEP 517 backend with good uv support |
| 2 | faiss-cpu as primary kNN, torch cdist fallback | annoy, scikit-learn BallTree | faiss is the standard for exact/approximate NN; torch fallback ensures zero extra deps |
| 3 | Adaptive avg pool for channel reduction (not PCA) | PCA, random projection | Simpler, no fitting step, works identically across runs; paper uses random projection but adaptive pool is equivalent for fixed architectures |
| 4 | Gaussian blur via scipy.ndimage | torch conv2d with gaussian kernel | scipy is already a dep; simpler than constructing a torch kernel |
| 5 | val_good split with torch.Generator for determinism | sklearn train_test_split | Fewer deps, torch Generator is seedable and deterministic across platforms |
| 6 | Save as memory_bank.pt + meta.json (no pickle) | Single pickle file | Security: weights_only=True; JSON metadata is human-readable |
