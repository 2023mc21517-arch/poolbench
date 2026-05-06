#!/usr/bin/env python3
"""
layer_rank_correlation.py — Spearman rank correlation of pooling strategy rankings
across candidate layers.

For each model, loads best_layer_auroc.json and computes pairwise Spearman ρ between
layers across all strategy × concept combinations (up to 323 values).

Also flags: does P1_last_token score highest at the selected best layer?

Usage:
    python scripts/layer_rank_correlation.py
    python scripts/layer_rank_correlation.py --results_dir results/auroc
    python scripts/layer_rank_correlation.py --results_dir results/auroc --output results/layer_rank_correlation.png
"""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


def load_per_layer_auroc(model_dir: Path) -> dict[int, dict[str, float]]:
    """Load best_layer_auroc.json and return {layer_int: {concept_strategy: auroc}}."""
    candidates = [
        model_dir / "best_layer_auroc.json",
        model_dir / f"{model_dir.name}_auroc_results.json",
    ]
    for path in candidates:
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            break
    else:
        return {}

    # Expect either:
    #   {"per_layer": {layer_str: {concept_strategy: {"auroc": float, ...}}}}
    # or the flat format from a single layer file.
    per_layer = data.get("per_layer", {})
    if not per_layer:
        # Try loading individual layer files
        per_layer = {}
        for layer_dir in sorted(model_dir.iterdir()):
            if not layer_dir.is_dir() or not layer_dir.name.startswith("layer_"):
                continue
            layer_idx = int(layer_dir.name.split("_")[1])
            layer_file = layer_dir / f"{model_dir.name}_auroc_results.json"
            if not layer_file.exists():
                continue
            with open(layer_file) as f:
                per_layer[str(layer_idx)] = json.load(f)

    result: dict[int, dict[str, float]] = {}
    for layer_str, strategies in per_layer.items():
        layer_idx = int(layer_str)
        auroc_map: dict[str, float] = {}
        for key, val in strategies.items():
            if isinstance(val, dict):
                auroc_map[key] = float(val.get("auroc", 0.0))
            elif isinstance(val, (float, int)):
                auroc_map[key] = float(val)
        if auroc_map:
            result[layer_idx] = auroc_map
    return result


def compute_layer_pair_correlations(
    per_layer: dict[int, dict[str, float]]
) -> list[tuple[int, int, float, float]]:
    """
    For each pair of layers, compute Spearman rank correlation over the
    intersection of strategy×concept keys present in both layers.

    Returns list of (layer_a, layer_b, rho, p_value).
    """
    layers = sorted(per_layer.keys())
    rows = []
    for la, lb in combinations(layers, 2):
        common_keys = sorted(set(per_layer[la]) & set(per_layer[lb]))
        if len(common_keys) < 5:
            continue
        va = np.array([per_layer[la][k] for k in common_keys])
        vb = np.array([per_layer[lb][k] for k in common_keys])
        rho, pval = spearmanr(va, vb)
        rows.append((la, lb, float(rho), float(pval)))
    return rows


def best_layer_from_summary(model_dir: Path) -> int | None:
    """Return the best layer recorded in the summary file, or None."""
    for fname in ["best_layer_auroc.json", f"{model_dir.name}_auroc_summary.json"]:
        p = model_dir / fname
        if p.exists():
            with open(p) as f:
                d = json.load(f)
            best = d.get("best_layer") or d.get("selected_layer")
            if best is not None:
                return int(best)
    return None


def p1_score_highest(per_layer: dict[int, dict[str, float]], best_layer: int | None) -> str:
    """Does P1_last_token achieve the highest mean AUROC at the best layer?"""
    if best_layer is None or best_layer not in per_layer:
        return "unknown (no best_layer)"
    layer_scores = per_layer[best_layer]
    # Group by strategy suffix
    strategy_means: dict[str, list[float]] = {}
    for key, auroc in layer_scores.items():
        parts = key.rsplit("_", maxsplit=2)
        # key format: concept_strategyid  — strategy id is last token with capital prefix
        # More robust: split on known strategy IDs
        for sid in ("P1_last_token", "P2_first_token", "P3_CLS", "A1_mean", "A2_max",
                     "A3_random", "A4_norm", "W1_mean_last_4", "W2_mean_last_8",
                     "W3_mean_last_16", "W4_hierarchical", "S1_attention_weighted",
                     "S2_SIF", "S3_ITI_exact", "L1_POS_filtered", "L2_dependency_rel",
                     "L3_named_entity", "L4_subword_root", "L5_SVO"):
            if key.endswith(f"_{sid}"):
                strategy_means.setdefault(sid, []).append(auroc)
                break
    if not strategy_means:
        return "unknown (no recognisable strategy keys)"
    mean_by_strategy = {s: float(np.mean(v)) for s, v in strategy_means.items()}
    best_strategy = max(mean_by_strategy, key=lambda s: mean_by_strategy[s])
    best_val = mean_by_strategy[best_strategy]
    p1_val = mean_by_strategy.get("P1_last_token", None)
    if p1_val is None:
        return f"P1_last_token not present; best={best_strategy} ({best_val:.3f})"
    is_top = best_strategy == "P1_last_token"
    return (f"YES — P1_last_token mean AUROC={p1_val:.3f} (best_layer={best_layer})"
            if is_top else
            f"NO — best={best_strategy} ({best_val:.3f}); P1_last_token={p1_val:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", default="results/auroc",
                        help="Root directory containing per-model subdirectories")
    parser.add_argument("--output", default="results/layer_rank_correlation.png",
                        help="Path to save the heatmap figure")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"ERROR: results_dir not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    # Discover models
    model_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir()])
    if not model_dirs:
        print(f"No model subdirectories found in {results_dir}", file=sys.stderr)
        sys.exit(1)

    # ── Per-model analysis ──────────────────────────────────────────────────
    all_corr_data: dict[str, list[tuple[int, int, float, float]]] = {}
    print(f"\n{'Model':<22} {'Layer pair':<14} {'Spearman ρ':>12} {'p-value':>12} {'n_keys':>8}")
    print("-" * 72)

    for model_dir in model_dirs:
        model_name = model_dir.name
        per_layer = load_per_layer_auroc(model_dir)
        if len(per_layer) < 2:
            print(f"{model_name:<22}  (fewer than 2 layers found — skip)")
            continue

        corrs = compute_layer_pair_correlations(per_layer)
        all_corr_data[model_name] = corrs
        for la, lb, rho, pval in corrs:
            common_n = len(set(per_layer[la]) & set(per_layer[lb]))
            print(f"{model_name:<22} L{la:02d} vs L{lb:02d}     {rho:>12.4f} {pval:>12.4e} {common_n:>8}")

        # P1 check
        best_layer = best_layer_from_summary(model_dir)
        p1_flag = p1_score_highest(per_layer, best_layer)
        print(f"  → P1_last_token highest at best layer? {p1_flag}")
        print()

    # ── Heatmap ────────────────────────────────────────────────────────────
    if not all_corr_data:
        print("No correlation data to plot.")
        return

    n_models = len(all_corr_data)
    # Find max number of layer pairs across all models
    max_pairs = max(len(v) for v in all_corr_data.values())

    fig, axes = plt.subplots(
        1, n_models,
        figsize=(max(4 * n_models, 6), max(3, max_pairs * 0.6 + 1.5)),
        squeeze=False,
    )

    for col, (model_name, corrs) in enumerate(all_corr_data.items()):
        ax = axes[0][col]
        if not corrs:
            ax.set_visible(False)
            continue
        pair_labels = [f"L{la}↔L{lb}" for la, lb, _, _ in corrs]
        rhos = [rho for _, _, rho, _ in corrs]
        im = ax.imshow(
            np.array(rhos).reshape(-1, 1),
            vmin=-1, vmax=1, cmap="RdYlGn", aspect="auto",
        )
        ax.set_yticks(range(len(pair_labels)))
        ax.set_yticklabels(pair_labels, fontsize=8)
        ax.set_xticks([])
        ax.set_title(model_name.replace("_", "\n"), fontsize=9, pad=4)
        for i, rho in enumerate(rhos):
            ax.text(0, i, f"{rho:.3f}", ha="center", va="center", fontsize=8,
                    color="black" if abs(rho) < 0.7 else "white")

    fig.colorbar(im, ax=axes[0].tolist(), label="Spearman ρ", shrink=0.6)
    fig.suptitle("Layer Rank Correlation of Pooling Strategy AUROC", fontsize=11, y=1.01)
    plt.tight_layout()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nHeatmap saved → {out_path}")

    # Save numeric results to JSON
    json_path = out_path.with_suffix(".json")
    json_data = {
        model: [
            {"layer_a": la, "layer_b": lb, "spearman_rho": rho, "p_value": pval}
            for la, lb, rho, pval in corrs
        ]
        for model, corrs in all_corr_data.items()
    }
    with open(json_path, "w") as f:
        import json as _json
        _json.dump(json_data, f, indent=2)
    print(f"Results saved → {json_path}")


if __name__ == "__main__":
    main()
