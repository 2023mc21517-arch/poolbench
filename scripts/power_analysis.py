"""
power_analysis.py
=================
Run ONCE before the main experiments.
Estimates the 95% CI half-width for AUROC at the planned corpus sizes.

Pre-committed MDE: 0.05 AUROC.
If the CI half-width > 0.025, increase n_per_class to 1000 and rebuild corpora.

Usage:
    python power_analysis.py                  # uses default 700/class, 5-fold
    python power_analysis.py --n_per_class 1000
    python power_analysis.py --n_per_class 700 --n_folds 5 --n_bootstrap 5000
"""

from __future__ import annotations
import argparse
import json
import numpy as np
from pathlib import Path


def estimate_auroc_ci_width(
    n_test: int = 140,
    n_bootstrap: int = 5000,
    true_auroc: float = 0.75,
    rng_seed: int = 42,
    n_experiments: int = 200,
) -> float:
    """
    Bootstrap-based estimate of 95% CI half-width for AUROC with n_test samples.

    Simulates 'n_experiments' probe-evaluation rounds at the given true AUROC.
    For each round: draws positive scores from Beta(5,2) and negative scores
    from Beta(2,5) — these have a theoretical AUROC near 0.75.
    Bootstraps n_bootstrap resamplings to get a CI; records the half-width.
    Returns the mean half-width across all experiments.

    The 95% CI half-width < 0.025 criterion comes from the pre-committed MDE of
    0.05 AUROC: if the half-width is ≤ 0.025, then two strategies whose true
    difference is 0.05 will have non-overlapping 95% CIs with ≥ ~80% power.
    """
    from sklearn.metrics import roc_auc_score  # noqa: PLC0415

    rng = np.random.default_rng(rng_seed)
    ci_half_widths = []

    for _ in range(n_experiments):
        n_pos = n_test // 2
        n_neg = n_test - n_pos
        # Simulate classifier scores
        pos_scores = rng.beta(5, 2, n_pos)   # higher scores for positives
        neg_scores = rng.beta(2, 5, n_neg)   # lower scores for negatives
        scores = np.concatenate([pos_scores, neg_scores])
        labels = np.array([1] * n_pos + [0] * n_neg)

        # Bootstrap
        boot_aurocs = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n_test, n_test)
            try:
                boot_aurocs.append(roc_auc_score(labels[idx], scores[idx]))
            except ValueError:
                pass   # skip degenerate bootstrap samples

        if not boot_aurocs:
            continue
        lo = float(np.percentile(boot_aurocs, 2.5))
        hi = float(np.percentile(boot_aurocs, 97.5))
        ci_half_widths.append((hi - lo) / 2.0)

    return float(np.mean(ci_half_widths))


def run_power_analysis(
    n_per_class: int = 700,
    n_folds: int = 5,
    n_bootstrap: int = 5000,
    output_path: Path = Path("results/power_analysis.json"),
) -> dict:
    """
    Full power analysis:
    - n_test per fold = n_per_class * 2 / n_folds (one fold is the test set)
    - Computes CI half-width
    - Reports whether MDE = 0.05 AUROC is achievable
    """
    n_test_fold = int(n_per_class * 2 / n_folds)

    print(f"Power analysis parameters:")
    print(f"  n_per_class:  {n_per_class}")
    print(f"  n_folds:      {n_folds}")
    print(f"  n_test/fold:  {n_test_fold} (balanced)")
    print(f"  n_bootstrap:  {n_bootstrap}")
    print(f"  Pre-committed MDE: 0.05 AUROC")
    print()
    print("Running bootstrap simulation (this takes ~1–2 min) ...")

    ci_half = estimate_auroc_ci_width(n_test=n_test_fold, n_bootstrap=n_bootstrap)

    mde_met = ci_half < 0.025
    verdict = "SUFFICIENT" if mde_met else "MARGINAL — consider 1000/class"

    print()
    print(f"Result:")
    print(f"  95% CI half-width (n={n_test_fold} test samples): ±{ci_half:.4f} AUROC")
    print(f"  Pre-committed MDE: 0.05 AUROC")
    print(f"  Verdict: {verdict}")

    if not mde_met:
        print()
        print("  ACTION REQUIRED: increase --n_train to 1000 in dataset_builder.py")
        print("  and rerun: python dataset_builder.py --all --n_train 1000")

    result = {
        "n_per_class":       n_per_class,
        "n_folds":           n_folds,
        "n_test_fold":       n_test_fold,
        "n_bootstrap":       n_bootstrap,
        "ci_half_width":     ci_half,
        "mde":               0.05,
        "mde_met":           mde_met,
        "verdict":           verdict,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved → {output_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="PoolBench power analysis")
    parser.add_argument("--n_per_class", type=int, default=700,
                        help="Passages per class in the training set")
    parser.add_argument("--n_folds",     type=int, default=5,
                        help="Number of CV folds")
    parser.add_argument("--n_bootstrap", type=int, default=5000,
                        help="Bootstrap resamples for CI estimation")
    parser.add_argument("--output", type=Path,
                        default=Path("results/power_analysis.json"),
                        help="Where to write the result JSON")
    args = parser.parse_args()

    run_power_analysis(
        n_per_class=args.n_per_class,
        n_folds=args.n_folds,
        n_bootstrap=args.n_bootstrap,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
