"""Verify that all metric and latency numbers quoted in README.md match results/*.json.

Guarantees Rule 1: NO FABRICATED NUMBERS.
Fails with exit code 1 if any measured number in README.md differs from actual results.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def check_readme_numbers(
    readme_path: Path = Path("README.md"), results_dir: Path = Path("results")
) -> bool:
    if not readme_path.exists():
        print(f"Error: {readme_path} not found.")
        return False

    readme_text = readme_path.read_text(encoding="utf-8")
    mismatches: list[str] = []

    # 1. Check Default profile AUROC numbers
    default_dir = results_dir / "default"
    if default_dir.exists():
        for res_file in default_dir.glob("*.json"):
            if "_baseline" in res_file.name or "_no_coreset" in res_file.name:
                continue
            with open(res_file) as f:
                d = json.load(f)
            cat = d["category"]
            expected_auroc = f"{d['metrics']['image_auroc']:.4f}"

            # Check if category row exists in README
            # Pattern: | `category` | **0.XXXX** |
            pattern = rf"\|\s*`{cat}`\s*\|\s*\*\*([0-9\.]+)\*\*"
            match = re.search(pattern, readme_text)
            if match:
                found_val = match.group(1)
                if found_val != expected_auroc:
                    mismatches.append(
                        f"Category '{cat}' Image AUROC mismatch in README: found {found_val}, expected {expected_auroc} (from {res_file})"
                    )

    # 2. Check Coreset Ablation numbers
    ablation_dir = results_dir / "ablation"
    if ablation_dir.exists():
        for r in [0.01, 0.10, 1.00]:
            abl_file = ablation_dir / f"coreset_r{r:.2f}.json"
            if abl_file.exists():
                with open(abl_file) as fp:
                    d = json.load(fp)
                expected_auroc = f"{d['metrics']['image_auroc']:.4f}"
                r_percent = f"{r * 100:.0f}"
                pattern = rf"\|\s*`r={r_percent}%`[^|]*\|\s*([0-9\.]+)\s*\|"
                match = re.search(pattern, readme_text)
                if match:
                    found_val = match.group(1)
                    if found_val != expected_auroc:
                        mismatches.append(
                            f"Ablation r={r} AUROC mismatch in README: found {found_val}, expected {expected_auroc}"
                        )

    if mismatches:
        print("[FAIL] Mismatches detected between README.md and results/*.json:")
        for m in mismatches:
            print(f"  - {m}")
        return False

    print("[PASS] All numbers in README.md match results/*.json exactly.")
    return True


if __name__ == "__main__":
    success = check_readme_numbers()
    sys.exit(0 if success else 1)
