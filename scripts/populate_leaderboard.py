"""
populate_leaderboard.py
Read computed AUROC results and write them into leaderboard/official/poolbench_v1.json.

For each model the script reads:
    results/auroc/{model}/best_layer_auroc.json

which was written by run_model.py (Step 2) and has the structure:
    {
        "best_layer": <int>,
        "per_layer": {
            "<layer>": {
                "<concept>_<strategy_id>": <auroc_float>,
                ...
            },
            ...
        }
    }

The script computes mean AUROC per strategy (across all concepts present at the
best layer) and writes the result into the leaderboard JSON under the appropriate
model key.

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

BASE_DIR        = Path(__file__).parent.parent
RESULTS_DIR     = Path(os.environ.get("POOLBENCH_RESULTS_DIR", BASE_DIR / "results"))
AUROC_DIR       = RESULTS_DIR / "auroc"
LEADERBOARD_PATH = BASE_DIR / "leaderboard" / "official" / "poolbench_v1.json"

# Internal model name → anonymous leaderboard key
MODEL_TO_LEADERBOARD_KEY: dict[str, str] = {
    "llama3_8b":  "model_A",
    "gemma2_9b":  "model_B",
    "mistral_7b": "model_C",
}


def _load_auroc(model_name: str) -> dict | None:
    path = AUROC_DIR / model_name / "best_layer_auroc.json"
    if not path.exists():
        print(f"  [skip] {model_name}: no results at {path}")
        return None
    with open(path) as f:
        return json.load(f)


def _mean_auroc_per_strategy(best_layer_data: dict) -> dict[str, float | None]:
    """Return {strategy_id: mean_auroc_across_concepts} for the best layer."""
    best_layer = str(best_layer_data["best_layer"])
    layer_results: dict[str, float] = best_layer_data["per_layer"].get(best_layer, {})

    # Group by strategy: keys are "<concept>_<strategy_id>"
    # Strategy ids can themselves contain underscores, so we match from the right.
    from poolbench.pooling_strategies import RANKED_STRATEGIES  # type: ignore[import]

    per_strategy: dict[str, list[float]] = {s: [] for s in RANKED_STRATEGIES}
    for key, auroc_val in layer_results.items():
        for strategy_id in RANKED_STRATEGIES:
            if key.endswith(f"_{strategy_id}"):
                per_strategy[strategy_id].append(float(auroc_val))
                break

    return {
        s: round(mean(vals), 5) if vals else None
        for s, vals in per_strategy.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Populate leaderboard/official/poolbench_v1.json from computed AUROC results"
    )
    parser.add_argument("--dry_run", action="store_true",
                        help="Print what would be written without modifying the file")
    args = parser.parse_args()

    # Load current leaderboard
    with open(LEADERBOARD_PATH) as f:
        leaderboard = json.load(f)

    updated_models: list[str] = []

    for model_name, lb_key in MODEL_TO_LEADERBOARD_KEY.items():
        print(f"\n{model_name} → {lb_key}")
        raw = _load_auroc(model_name)
        if raw is None:
            continue

        best_layer = raw.get("best_layer")
        print(f"  best_layer = {best_layer}")

        per_strategy = _mean_auroc_per_strategy(raw)
        for strategy, val in per_strategy.items():
            print(f"  {strategy:<28s} {val}")

        if not args.dry_run:
            leaderboard["models"][lb_key]["mean_auroc_per_strategy"] = per_strategy
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
