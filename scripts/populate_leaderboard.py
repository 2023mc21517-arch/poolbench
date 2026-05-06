"""
populate_leaderboard.py
Read computed D1/D2/D3 results and write them into leaderboard/official/poolbench_v1.json.

Sources read per model:
    D1  results/auroc/{model}/best_layer_auroc.json       → mean AUROC per strategy
    D2  results/scp/{model}_scp.json                      → mean SCP_c per strategy
    D3  results/disentanglement/{model}_d3.json           → mean D3_LD and D3_LC per strategy

Model → leaderboard key mapping (3 models evaluated in PoolBench v1):
    llama3_8b  → model_A  (8B decoder-only)
    gemma2_9b  → model_B  (9B decoder-only)
    mistral_7b → model_C  (7B decoder-only)

Usage
-----
# Populate from all 3 models (skips any whose results are missing)
python scripts/populate_leaderboard.py

# Preview without writing
python scripts/populate_leaderboard.py --dry_run
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import mean

BASE_DIR         = Path(__file__).parent.parent
RESULTS_DIR      = Path(os.environ.get("POOLBENCH_RESULTS_DIR", BASE_DIR / "results"))
AUROC_DIR        = RESULTS_DIR / "auroc"
SCP_DIR          = RESULTS_DIR / "scp"
D3_DIR           = RESULTS_DIR / "disentanglement"
LEADERBOARD_PATH = BASE_DIR / "leaderboard" / "official" / "poolbench_v1.json"

# Internal model name → anonymous leaderboard key
MODEL_TO_LEADERBOARD_KEY: dict[str, str] = {
    "llama3_8b":  "model_A",
    "gemma2_9b":  "model_B",
    "mistral_7b": "model_C",
}


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ── D1: mean AUROC per strategy ───────────────────────────────────────────────

def _mean_auroc_per_strategy(model_name: str) -> tuple[int | None, dict[str, float | None]]:
    """Returns (best_layer, {strategy: mean_auroc})."""
    from poolbench.pooling_strategies import RANKED_STRATEGIES  # type: ignore[import]

    raw = _load_json(AUROC_DIR / model_name / "best_layer_auroc.json")
    if raw is None:
        return None, {}

    best_layer = raw.get("best_layer")
    layer_results: dict = raw.get("per_layer", {}).get(str(best_layer), {})

    per_strategy: dict[str, list[float]] = {s: [] for s in RANKED_STRATEGIES}
    for key, val in layer_results.items():
        if isinstance(val, dict):
            val = val.get("auroc")
        if val is None:
            continue
        for s in RANKED_STRATEGIES:
            if key.endswith(f"_{s}"):
                per_strategy[s].append(float(val))
                break

    return best_layer, {
        s: round(mean(vals), 5) if vals else None
        for s, vals in per_strategy.items()
    }


# ── D2: mean SCP_c per strategy ───────────────────────────────────────────────

def _mean_scp_per_strategy(model_name: str) -> dict[str, float | None]:
    """Returns {strategy: mean_SCP_c across concepts}."""
    from poolbench.pooling_strategies import RANKED_STRATEGIES  # type: ignore[import]

    raw = _load_json(SCP_DIR / f"{model_name}_scp.json")
    if raw is None:
        return {}

    per_strategy: dict[str, list[float]] = {s: [] for s in RANKED_STRATEGIES}
    for concept_data in raw.values():
        if not isinstance(concept_data, dict):
            continue
        for s in RANKED_STRATEGIES:
            entry = concept_data.get(s, {})
            if not isinstance(entry, dict):
                continue
            scp_c = entry.get("SCP_c")
            if scp_c is not None:
                per_strategy[s].append(float(scp_c))

    return {
        s: round(mean(vals), 5) if vals else None
        for s, vals in per_strategy.items()
    }


# ── D3: mean D3_LD and D3_LC per strategy ────────────────────────────────────

def _mean_d3_per_strategy(model_name: str) -> tuple[dict[str, float | None], dict[str, float | None]]:
    """Returns ({strategy: mean_D3_LD}, {strategy: mean_D3_LC})."""
    from poolbench.pooling_strategies import RANKED_STRATEGIES  # type: ignore[import]

    raw = _load_json(D3_DIR / f"{model_name}_d3.json")
    if raw is None:
        return {}, {}

    ld_vals: dict[str, list[float]] = {s: [] for s in RANKED_STRATEGIES}
    lc_vals: dict[str, list[float]] = {s: [] for s in RANKED_STRATEGIES}

    for concept_data in raw.values():
        if not isinstance(concept_data, dict):
            continue
        for s in RANKED_STRATEGIES:
            entry = concept_data.get(s, {})
            if not isinstance(entry, dict):
                continue
            if entry.get("D3_LD") is not None:
                ld_vals[s].append(float(entry["D3_LD"]))
            if entry.get("D3_LC") is not None:
                lc_vals[s].append(float(entry["D3_LC"]))

    mean_ld = {s: round(mean(v), 5) if v else None for s, v in ld_vals.items()}
    mean_lc = {s: round(mean(v), 5) if v else None for s, v in lc_vals.items()}
    return mean_ld, mean_lc


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Populate leaderboard/official/poolbench_v1.json from computed D1/D2/D3 results"
    )
    parser.add_argument("--dry_run", action="store_true",
                        help="Print what would be written without modifying the file")
    args = parser.parse_args()

    with open(LEADERBOARD_PATH) as f:
        leaderboard = json.load(f)

    updated_models: list[str] = []

    for model_name, lb_key in MODEL_TO_LEADERBOARD_KEY.items():
        print(f"\n{'─'*50}")
        print(f"  {model_name} → {lb_key}")

        best_layer, d1 = _mean_auroc_per_strategy(model_name)
        d2             = _mean_scp_per_strategy(model_name)
        d3_ld, d3_lc   = _mean_d3_per_strategy(model_name)

        if not d1 and not d2 and not d3_ld:
            print(f"  [skip] no results found for {model_name}")
            continue

        print(f"  best_layer = {best_layer}")
        print(f"\n  {'Strategy':<28s}  {'D1_auroc':>10}  {'D2_scp_c':>10}  {'D3_LD':>8}  {'D3_LC':>8}")
        from poolbench.pooling_strategies import RANKED_STRATEGIES  # type: ignore[import]
        for s in RANKED_STRATEGIES:
            print(f"  {s:<28s}  {str(d1.get(s)):>10}  {str(d2.get(s)):>10}  "
                  f"{str(d3_ld.get(s)):>8}  {str(d3_lc.get(s)):>8}")

        if not args.dry_run:
            m = leaderboard["models"][lb_key]
            m["best_layer"]                   = best_layer
            m["D1_mean_auroc_per_strategy"]   = d1
            m["D2_mean_scp_c_per_strategy"]   = d2
            m["D3_mean_ld_per_strategy"]       = d3_ld
            m["D3_mean_lc_per_strategy"]       = d3_lc
            updated_models.append(lb_key)

    if not args.dry_run and updated_models:
        with open(LEADERBOARD_PATH, "w") as f:
            json.dump(leaderboard, f, indent=2)
        print(f"\nWrote leaderboard → {LEADERBOARD_PATH}")
        print(f"Updated: {updated_models}")
    elif args.dry_run:
        print("\n[dry_run] No files written.")
    else:
        print("\nNo models had results available — leaderboard unchanged.")


if __name__ == "__main__":
    main()
