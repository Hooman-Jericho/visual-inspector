RULES (apply to every stage)
1. NO FABRICATED NUMBERS. Every metric/latency/size in README or docs must be produced by code and read from results/*.json. Paper-reported numbers only in a column labeled "paper-reported (context)" with citation.
2. HONESTY over polish: report weaker-than-paper results as they are, with plausible reasons.
3. Work phase by phase. Run the gate after each phase. Never continue while a gate fails.
4. Never commit datasets, weights, .env, secrets, big binaries (data/ and artifacts/ are git-ignored). Small results/*.json and results/*.png (<=300 KB) ARE committed.
5. Reproducibility: fixed seeds, env capture (python/torch/torchvision, device, CPU, threads), config hash + git commit in every result file.
6. Unspecified decisions: pick the simplest option; log one line in docs/DECISIONS.md (decision, alternatives, reason).
7. If a human is needed (dataset download, keys), STOP that part, say exactly what is needed, continue with the rest.
8. Python 3.10-3.12; deps minimal and pinned (uv + pyproject.toml + uv.lock; fallback pip-tools). CPU-first; GPU used if present (device: auto).
9. At the end of each stage: update docs/PROGRESS.md (done / pending / known issues / next stage), commit, then STOP and print "STAGE X DONE" with a report. Do not start the next stage.
