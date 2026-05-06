"""
populate_leaderboard.py
Read computed D1/D2/D3 results and write three leaderboard files:

    leaderboard/official/poolbench_v1.json          — per-model summary (D1/D2/D3 means)
    leaderboard/official/poolbench_v1_full.json     — per-concept × per-model × per-strategy
    leaderboard/official/poolbench_v1_rankings.json — cross-model ranked strategy summary

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
python scripts/populate_leaderboard.py            # write all three files
python scripts/populate_leaderboard.py --dry_run  # preview without writing
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import mean

BASE_DIR            = Path(__file__).parent.parent
RESULTS_DIR         = Path(os.environ.get("POOLBENCH_RESULTS_DIR", BASE_DIR / "results"))
AUROC_DIR           = RESULTS_DIR / "auroc"
SCP_DIR             = RESULTS_DIR / "scp"
D3_DIR              = RESULTS_DIR / "disentanglement"
LEADERBOARD_DIR     = BASE_DIR / "leaderboard" / "official"
SUMMARY_PATH        = LEADERBOARD_DIR / "poolbench_v1.json"
FULL_PATH           = LEADERBOARD_DIR / "poolbench_v1_full.json"
RANKINGS_PATH       = LEADERBOARD_DIR / "poolbench_v1_rankings.json"

MODEL_TO_LEADERBOARD_KEY: dict[str, str] = {
    "llama3_8b":  "model_A",
    "gemma2_9b":  "model_B",
    "mistral_7b": "model_C",
}

# Strategy metadata (family, supervision) — mirrors STRATEGY_REGISTRY in pooling_strategies.py
STRATEGY_META: dict[str, dict] = {
    "P1_last_token":         {"family": "position_anchored",            "supervision": "unsupervised"},
    "P2_first_token":        {"family": "position_anchored",            "supervision": "unsupervised"},
    "P3_CLS":                {"family": "position_anchored",            "supervision": "unsupervised"},
    "A1_mean":               {"family": "uniform_aggregation",          "supervision": "unsupervised"},
    "A2_max":                {"family": "uniform_aggregation",          "supervision": "unsupervised"},
    "A3_random":             {"family": "uniform_aggregation",          "supervision": "unsupervised"},
    "A4_norm":               {"family": "uniform_aggregation",          "supervision": "unsupervised"},
    "W1_mean_last_4":        {"family": "window",                       "supervision": "unsupervised"},
    "W2_mean_last_8":        {"family": "window",                       "supervision": "unsupervised"},
    "W3_mean_last_16":       {"family": "window",                       "supervision": "unsupervised"},
    "W4_hierarchical":       {"family": "window",                       "supervision": "unsupervised"},
    "S1_attention_weighted": {"family": "saliency_weighted",            "supervision": "unsupervised"},
    "S2_SIF":                {"family": "saliency_weighted",            "supervision": "unsupervised"},
    "S3_ITI_exact":          {"family": "saliency_weighted_supervised", "supervision": "supervised"},
    "L1_POS_filtered":       {"family": "structural_linguistic",        "supervision": "unsupervised"},
    "L2_dependency_rel":     {"family": "structural_linguistic",        "supervision": "unsupervised"},
    "L3_named_entity":       {"family": "structural_linguistic",        "supervision": "unsupervised"},
    "L4_subword_root":       {"family": "structural_linguistic",        "supervision": "unsupervised"},
    "L5_SVO":                {"family": "structural_linguistic",        "supervision": "unsupervised"},
}


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _safe_mean(vals: list[float]) -> float | None:
    return round(mean(vals), 5) if vals else None


# ── Per-model data extraction ─────────────────────────────────────────────────

def _extract_model_data(model_name: str, ranked_strategies: list[str]) -> dict | None:
    """
    Returns a dict with:
      best_layer: int
      per_concept: {concept: {strategy: {D1, D2, D3_LD, D3_LC}}}
      means: {strategy: {D1, D2, D3_LD, D3_LC}}
    Returns None if no D1 data found.
    """
    auroc_raw = _load_json(AUROC_DIR / model_name / "best_layer_auroc.json")
    if auroc_raw is None:
        return None

    best_layer: int = auroc_raw["best_layer"]
    layer_results: dict = auroc_raw.get("per_layer", {}).get(str(best_layer), {})

    scp_raw  = _load_json(SCP_DIR / f"{model_name}_scp.json") or {}
    d3_raw   = _load_json(D3_DIR  / f"{model_name}_d3.json")  or {}

    # Build per-concept dict
    # First pass: collect all concept names from D1 keys
    concepts: set[str] = set()
    for key in layer_results:
        for s in ranked_strategies:
            if key.endswith(f"_{s}"):
                concepts.add(key[: -(len(s) + 1)])
                break

    per_concept: dict[str, dict] = {}
    for concept in sorted(concepts):
        concept_data: dict[str, dict] = {}
        for s in ranked_strategies:
            d1_key = f"{concept}_{s}"
            d1_entry = layer_results.get(d1_key)
            d1_val = None
            if isinstance(d1_entry, dict):
                d1_val = d1_entry.get("auroc")
            elif isinstance(d1_entry, (int, float)):
                d1_val = float(d1_entry)

            scp_entry = scp_raw.get(concept, {}).get(s, {})
            d2_val    = scp_entry.get("SCP_c") if isinstance(scp_entry, dict) else None

            d3_entry  = d3_raw.get(concept, {}).get(s, {})
            d3_ld_val = d3_entry.get("D3_LD")  if isinstance(d3_entry, dict) else None
            d3_lc_val = d3_entry.get("D3_LC")  if isinstance(d3_entry, dict) else None

            concept_data[s] = {
                "D1_auroc": round(float(d1_val), 5) if d1_val is not None else None,
                "D2_scp_c": round(float(d2_val), 5) if d2_val is not None else None,
                "D3_LD":    round(float(d3_ld_val), 5) if d3_ld_val is not None else None,
                "D3_LC":    round(float(d3_lc_val), 5) if d3_lc_val is not None else None,
            }
        per_concept[concept] = concept_data

    # Compute per-strategy means across concepts
    means: dict[str, dict] = {}
    for s in ranked_strategies:
        d1_vals  = [per_concept[c][s]["D1_auroc"] for c in per_concept if per_concept[c][s]["D1_auroc"] is not None]
        d2_vals  = [per_concept[c][s]["D2_scp_c"] for c in per_concept if per_concept[c][s]["D2_scp_c"] is not None]
        ld_vals  = [per_concept[c][s]["D3_LD"]    for c in per_concept if per_concept[c][s]["D3_LD"]    is not None]
        lc_vals  = [per_concept[c][s]["D3_LC"]    for c in per_concept if per_concept[c][s]["D3_LC"]    is not None]
        means[s] = {
            "D1_mean_auroc": _safe_mean(d1_vals),
            "D2_mean_scp_c": _safe_mean(d2_vals),
            "D3_mean_LD":    _safe_mean(ld_vals),
            "D3_mean_LC":    _safe_mean(lc_vals),
        }

    return {"best_layer": best_layer, "per_concept": per_concept, "means": means}


# ── Writers ───────────────────────────────────────────────────────────────────

def _write_summary(leaderboard: dict, model_name: str, lb_key: str,
                   model_data: dict, dry_run: bool) -> None:
    m = leaderboard["models"][lb_key]
    m["best_layer"]                 = model_data["best_layer"]
    m["D1_mean_auroc_per_strategy"] = {s: v["D1_mean_auroc"] for s, v in model_data["means"].items()}
    m["D2_mean_scp_c_per_strategy"] = {s: v["D2_mean_scp_c"] for s, v in model_data["means"].items()}
    m["D3_mean_ld_per_strategy"]    = {s: v["D3_mean_LD"]    for s, v in model_data["means"].items()}
    m["D3_mean_lc_per_strategy"]    = {s: v["D3_mean_LC"]    for s, v in model_data["means"].items()}


def _write_full(leaderboard: dict, model_name: str, lb_key: str,
                model_data: dict, dry_run: bool) -> None:
    leaderboard["models"][lb_key] = {
        "best_layer":  model_data["best_layer"],
        "per_concept": model_data["per_concept"],
    }


def _build_rankings(all_model_data: dict[str, dict],
                    ranked_strategies: list[str]) -> list[dict]:
    """Cross-model mean of each metric per strategy, sorted by D1 descending."""
    rows: list[dict] = []
    for s in ranked_strategies:
        d1_vals, d2_vals, ld_vals, lc_vals = [], [], [], []
        per_model: dict[str, dict] = {}
        for model_name, lb_key in MODEL_TO_LEADERBOARD_KEY.items():
            if model_name not in all_model_data:
                continue
            m = all_model_data[model_name]["means"][s]
            per_model[lb_key] = {
                "D1_mean_auroc": m["D1_mean_auroc"],
                "D2_mean_scp_c": m["D2_mean_scp_c"],
                "D3_mean_LD":    m["D3_mean_LD"],
                "D3_mean_LC":    m["D3_mean_LC"],
            }
            if m["D1_mean_auroc"] is not None: d1_vals.append(m["D1_mean_auroc"])
            if m["D2_mean_scp_c"] is not None: d2_vals.append(m["D2_mean_scp_c"])
            if m["D3_mean_LD"]    is not None: ld_vals.append(m["D3_mean_LD"])
            if m["D3_mean_LC"]    is not None: lc_vals.append(m["D3_mean_LC"])

        rows.append({
            "strategy":               s,
            "family":                 STRATEGY_META[s]["family"],
            "supervision":            STRATEGY_META[s]["supervision"],
            "cross_model_D1_auroc":   _safe_mean(d1_vals),
            "cross_model_D2_scp_c":   _safe_mean(d2_vals),
            "cross_model_D3_LD":      _safe_mean(ld_vals),
            "cross_model_D3_LC":      _safe_mean(lc_vals),
            "per_model":              per_model,
        })

    # Rank by cross-model D1 descending (None goes to bottom)
    rows.sort(key=lambda r: r["cross_model_D1_auroc"] or -1, reverse=True)
    for i, row in enumerate(rows, start=1):
        row["rank"] = i
        # Move rank to front
        rows[i - 1] = {"rank": i, **{k: v for k, v in row.items() if k != "rank"}}

    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Populate all three PoolBench leaderboard files (summary, full, rankings)"
    )
    parser.add_argument("--dry_run", action="store_true",
                        help="Print what would be written without modifying any file")
    args = parser.parse_args()

    from poolbench.pooling_strategies import RANKED_STRATEGIES  # type: ignore[import]

    # Load existing files
    summary_lb   = _load_json(SUMMARY_PATH)   or {}
    full_lb      = _load_json(FULL_PATH)      or {"models": {}}
    rankings_lb  = _load_json(RANKINGS_PATH)  or {}

    all_model_data: dict[str, dict] = {}
    updated: list[str] = []

    for model_name, lb_key in MODEL_TO_LEADERBOARD_KEY.items():
        print(f"\n{'─'*55}")
        print(f"  {model_name} → {lb_key}")
        data = _extract_model_data(model_name, RANKED_STRATEGIES)
        if data is None:
            print(f"  [skip] no D1 results found")
            continue

        all_model_data[model_name] = data
        updated.append(lb_key)

        print(f"  best_layer = {data['best_layer']}  "
              f"concepts = {len(data['per_concept'])}")
        print(f"\n  {'Strategy':<28s}  {'D1':>8}  {'D2':>8}  {'D3_LD':>8}  {'D3_LC':>8}")
        for s in RANKED_STRATEGIES:
            m = data["means"][s]
            print(f"  {s:<28s}  "
                  f"{str(m['D1_mean_auroc']):>8}  "
                  f"{str(m['D2_mean_scp_c']):>8}  "
                  f"{str(m['D3_mean_LD']):>8}  "
                  f"{str(m['D3_mean_LC']):>8}")

        if not args.dry_run:
            _write_summary(summary_lb, model_name, lb_key, data, args.dry_run)
            _write_full(full_lb,    model_name, lb_key, data, args.dry_run)

    if not updated:
        print("\nNo models had results — nothing written.")
        return

    # Build rankings across all computed models
    ranked = _build_rankings(all_model_data, RANKED_STRATEGIES)
    print(f"\n{'═'*55}")
    print("  Rankings (cross-model mean D1 descending):\n")
    print(f"  {'Rank':>4}  {'Strategy':<28s}  {'D1':>8}  {'D2':>8}  {'D3_LD':>8}  {'D3_LC':>8}")
    for r in ranked:
        print(f"  {r['rank']:>4}  {r['strategy']:<28s}  "
              f"{str(r['cross_model_D1_auroc']):>8}  "
              f"{str(r['cross_model_D2_scp_c']):>8}  "
              f"{str(r['cross_model_D3_LD']):>8}  "
              f"{str(r['cross_model_D3_LC']):>8}")

    if not args.dry_run:
        rankings_lb["ranked_strategies"] = ranked
        rankings_lb["models_included"]   = [MODEL_TO_LEADERBOARD_KEY[m] for m in all_model_data]

        with open(SUMMARY_PATH, "w") as f:
            json.dump(summary_lb, f, indent=2)
        with open(FULL_PATH, "w") as f:
            json.dump(full_lb, f, indent=2)
        with open(RANKINGS_PATH, "w") as f:
            json.dump(rankings_lb, f, indent=2)

        print(f"\nWrote:")
        print(f"  {SUMMARY_PATH.name}")
        print(f"  {FULL_PATH.name}")
        print(f"  {RANKINGS_PATH.name}")
    else:
        print("\n[dry_run] No files written.")


if __name__ == "__main__":
    main()


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
