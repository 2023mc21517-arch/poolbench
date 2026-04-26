"""
run_model.py
Per-model orchestration: activation extraction → pooling → AUROC → SCP → (optional) SAE.

Usage
-----
# Single model, all concepts
python run_model.py --model llama3_8b --device cuda:0

# Single model, single concept
python run_model.py --model llama3_8b --concept hedging --device cuda:0

# All models sequentially (for multi-GPU nodes, run them in parallel manually)
python run_model.py --all --device cuda:0

# Only pool + eval (skip extraction if activations already exist)
python run_model.py --model llama3_8b --skip_extraction --device cuda:0

# Run linearity checks only
python run_model.py --model llama3_8b --linearity_only --device cpu
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

import numpy as np

from poolbench.concepts           import CONCEPTS, CONCEPT_NAMES
from poolbench.pooling_strategies import (STRATEGY_REGISTRY, RANKED_STRATEGIES,
                                    compute_all_pooling_strategies)
from poolbench.construction_methods import DEFAULT_CONSTRUCTION
from poolbench.probe_training       import (compute_all_auroc, check_linearity_assumption,
                                      compute_layer_icc, nemenyi_strategy_significance,
                                      build_nemenyi_auroc_matrix)
from poolbench.extract_activations  import extract_activations_for_model, load_activations


# ── Model configs ─────────────────────────────────────────────────────────────

MODEL_CONFIGS: dict[str, dict] = {
    "llama3_8b": {
        "hf_id":            "meta-llama/Meta-Llama-3.1-8B",
        "d_model":          4096,
        "n_layers":         32,
        "candidate_layers": [16, 24, 31],  # early-mid | mid | final
        "architecture":     "causal_lm",
        "batch_size":       8,
    },
    "gemma2_9b": {
        "hf_id":            "google/gemma-2-9b",
        "d_model":          3584,
        "n_layers":         42,
        "candidate_layers": [14, 28, 41],
        "architecture":     "causal_lm",
        "batch_size":       6,
    },
    "mistral_7b": {
        "hf_id":            "mistralai/Mistral-7B-v0.3",
        "d_model":          4096,
        "n_layers":         32,
        "candidate_layers": [16, 24, 31],
        "architecture":     "causal_lm",
        "batch_size":       8,
    },
    "qwen25_7b": {
        "hf_id":            "Qwen/Qwen2.5-7B",
        "d_model":          3584,
        "n_layers":         28,
        "candidate_layers": [10, 18, 27],
        "architecture":     "causal_lm",
        "batch_size":       8,
    },
    "flan_t5_xl": {
        "hf_id":            "google/flan-t5-xl",
        "d_model":          2048,
        "n_layers":         24,
        "candidate_layers": [8, 16, 23],
        "architecture":     "encoder_decoder",
        "batch_size":       16,
        # Attention pooling: FLAN-T5 encoder self-attn is available but cross-attn is not.
        # S1/S4 strategies fall back to pool_mean for this model.
    },
    "mamba2_2b7": {
        "hf_id":            "state-spaces/mamba2-2.8b",
        "d_model":          2560,
        "n_layers":         64,
        "candidate_layers": [21, 42, 63],
        "architecture":     "ssm",
        "batch_size":       16,
        # No self-attention → S1/S4 fallback to pool_mean.
    },
    "bert_base_uncased": {
        "hf_id":            "bert-base-uncased",
        "d_model":          768,
        "n_layers":         12,
        "candidate_layers": [6, 9, 11],
        "architecture":     "encoder_only",
        "batch_size":       32,
    },
}

BASE_DIR      = Path(__file__).parent.parent
ACT_DIR       = BASE_DIR / "results" / "activations"
AUROC_DIR     = BASE_DIR / "results" / "auroc"
LINEARITY_DIR = BASE_DIR / "results" / "linearity"
NEMENYI_DIR   = BASE_DIR / "results" / "nemenyi"
ICC_DIR       = BASE_DIR / "results" / "icc"
CORPUS_DIR    = BASE_DIR / "data" / "corpora"


# ── Step 1: Extract activations ───────────────────────────────────────────────

def step_extract(model_name: str, device: str, skip_existing: bool = True,
                 concept_filter: str | None = None) -> None:
    cfg = MODEL_CONFIGS[model_name]
    print(f"\n=== Step 1: Extracting activations — {model_name} ===")
    extract_activations_for_model(
        model_name      = model_name,
        hf_id           = cfg["hf_id"],
        concept_corpus_dir = CORPUS_DIR if concept_filter is None else CORPUS_DIR / concept_filter,
        out_dir         = ACT_DIR,
        candidate_layers= cfg["candidate_layers"],
        batch_size      = cfg["batch_size"],
        device          = device,
        skip_existing   = skip_existing,
    )


# ── Step 2: Pool + compute AUROC per layer, select best ───────────────────────

def step_pool_and_auroc(model_name: str, construction_method: str = DEFAULT_CONSTRUCTION,
                        concept_filter: str | None = None) -> dict:
    """
    For each candidate layer, apply all 19 ranked strategies + compute AUROC.
    Select best layer per strategy (modal layer selection across all concepts).

    Returns consolidated auroc_results dict:
        {"best_layer": int, "per_layer": {layer: auroc_results_dict}}
    """
    cfg = MODEL_CONFIGS[model_name]
    print(f"\n=== Step 2: Pool + AUROC — {model_name} ===")
    concepts_to_run = {k: v for k, v in CONCEPTS.items()
                       if concept_filter is None or k == concept_filter}

    per_layer_results: dict[int, dict] = {}

    for layer_idx in cfg["candidate_layers"]:
        print(f"\n  Layer {layer_idx} …")
        layer_act_dir = ACT_DIR / model_name / f"layer_{layer_idx}"

        # Build pooled_results: {concept_strategy: {pos_pooled, neg_pooled}}
        pooled_results: dict = {}
        for concept_name in concepts_to_run:
            pos_acts = load_activations(ACT_DIR, model_name, layer_idx, concept_name, "pos")
            neg_acts = load_activations(ACT_DIR, model_name, layer_idx, concept_name, "neg")
            if pos_acts is None or neg_acts is None:
                print(f"  [pool] {concept_name} L{layer_idx}: activations missing — skip")
                continue
            # Apply all pooling strategies
            concept_dict = {concept_name: concepts_to_run[concept_name]}
            layer_pooled = compute_all_pooling_strategies(
                act_dir       = layer_act_dir,
                concepts      = concept_dict,
                tokenizer_name= cfg["hf_id"],
            )
            pooled_results.update(layer_pooled)

        layer_auroc_dir = AUROC_DIR / model_name / f"layer_{layer_idx}"
        auroc_res = compute_all_auroc(
            pooled_results       = pooled_results,
            model_name           = model_name,
            out_dir              = layer_auroc_dir,
            construction_method  = construction_method,
        )
        per_layer_results[layer_idx] = auroc_res

    # Modal layer selection: for each strategy, find best layer across concepts
    best_layer = _select_modal_layer(per_layer_results, cfg["candidate_layers"])
    print(f"\n  Best layer (modal): {best_layer}")

    # Save the best-layer results as the primary leaderboard input
    primary_out = AUROC_DIR / model_name / "best_layer_auroc.json"
    primary_out.parent.mkdir(parents=True, exist_ok=True)
    with open(primary_out, "w") as f:
        json.dump({
            "best_layer":    best_layer,
            "per_layer":     {str(k): v for k, v in per_layer_results.items()},
        }, f, indent=2)

    return {"best_layer": best_layer, "per_layer": per_layer_results}


def _select_modal_layer(per_layer_results: dict[int, dict],
                        candidate_layers: list[int]) -> int:
    """
    For each strategy × concept, record which layer achieves highest AUROC.
    Return the modal best layer (most frequent winner).
    """
    from collections import Counter  # noqa: PLC0415
    winner_counts: Counter = Counter()

    all_keys = set()
    for _, res in per_layer_results.items():
        all_keys.update(res.keys())

    for key in all_keys:
        best_auroc = -1.0
        best_layer = candidate_layers[0]
        for layer, res in per_layer_results.items():
            auroc = res.get(key, {}).get("auroc", 0.0)
            if auroc > best_auroc:
                best_auroc = auroc
                best_layer = layer
        winner_counts[best_layer] += 1

    return winner_counts.most_common(1)[0][0] if winner_counts else candidate_layers[-1]


# ── Step 3: Linearity validation (D3) ────────────────────────────────────────

def step_linearity(model_name: str, best_layer: int,
                   construction_method: str = DEFAULT_CONSTRUCTION,
                   concept_filter: str | None = None) -> None:
    """
    Run the linearity assumption check for each concept at the best layer.
    Saves results to results/linearity/{model_name}_linearity.json.
    """
    print(f"\n=== Step 3: Linearity check — {model_name} L{best_layer} ===")
    LINEARITY_DIR.mkdir(parents=True, exist_ok=True)
    concepts_to_run = CONCEPT_NAMES if concept_filter is None else [concept_filter]

    linearity_results = {}
    for concept_name in concepts_to_run:
        pos_acts = load_activations(ACT_DIR, model_name, best_layer, concept_name, "pos")
        neg_acts = load_activations(ACT_DIR, model_name, best_layer, concept_name, "neg")
        if pos_acts is None or neg_acts is None:
            continue

        pos_pooled = np.stack([item["hidden"].mean(0) for item in pos_acts])
        neg_pooled = np.stack([item["hidden"].mean(0) for item in neg_acts])

        result = check_linearity_assumption(pos_pooled, neg_pooled, concept_name,
                                             construction_method)
        linearity_results[concept_name] = result
        status = "✓" if result["passes"] else "✗ FAIL"
        print(f"  {status}  {concept_name}: linear={result['linear_auroc']:.3f} "
              f"mlp={result['mlp_auroc']:.3f} gap={result['gap']:.4f}")

    out_path = LINEARITY_DIR / f"{model_name}_linearity.json"
    with open(out_path, "w") as f:
        json.dump(linearity_results, f, indent=2)
    print(f"  Saved → {out_path}")

    n_fail = sum(1 for r in linearity_results.values() if not r["passes"])
    if n_fail > 0:
        print(f"\n  WARNING: {n_fail} concept(s) fail linearity check (gap ≥ 0.03). "
              f"Inspect results and consider using C3_logreg or C4_repe for those concepts.")


# ── Step 4: ICC computation ───────────────────────────────────────────────────

def step_icc(model_name: str, construction_method: str = DEFAULT_CONSTRUCTION,
             concept_filter: str | None = None) -> None:
    """
    Compute intraclass correlation across candidate layers (for N_eff correction).
    Requires best-layer auroc results already saved by step_pool_and_auroc.
    """
    print(f"\n=== Step 4: ICC computation — {model_name} ===")
    ICC_DIR.mkdir(parents=True, exist_ok=True)
    best_layer_path = AUROC_DIR / model_name / "best_layer_auroc.json"
    if not best_layer_path.exists():
        print(f"  [icc] {best_layer_path} not found — run step 2 first.")
        return

    with open(best_layer_path) as f:
        data = json.load(f)

    concepts_to_run = CONCEPT_NAMES if concept_filter is None else [concept_filter]
    # Build per-concept list of AUROC from best strategy (A1_mean as reference strategy)
    per_layer = data.get("per_layer", {})
    layer_aurocs_per_concept: dict[str, list[float]] = {}
    for concept_name in concepts_to_run:
        aucs = []
        for layer_str in sorted(per_layer.keys(), key=int):
            key = f"{concept_name}_A1_mean"
            auroc = per_layer[layer_str].get(key, {}).get("auroc")
            if auroc is not None:
                aucs.append(auroc)
        if aucs:
            layer_aurocs_per_concept[concept_name] = aucs

    cfg     = MODEL_CONFIGS[model_name]
    n_base  = 300   # test set size per class
    icc_res = compute_layer_icc(layer_aurocs_per_concept, n_base=n_base)
    icc_res["model_name"] = model_name

    out_path = ICC_DIR / f"{model_name}_icc.json"
    with open(out_path, "w") as f:
        json.dump(icc_res, f, indent=2)
    print(f"  mean_icc={icc_res['mean_icc']:.3f}  N_eff={icc_res['N_eff']}")
    print(f"  Saved → {out_path}")


# ── Step 5: Nemenyi strategy significance test ────────────────────────────────

def step_nemenyi(all_model_auroc_results: dict[str, dict],
                 concept_filter: str | None = None) -> None:
    """
    Run the Nemenyi test over all 7 models jointly.
    Should be called after all models have completed steps 1–2.

    all_model_auroc_results: {model_name: {concept_strategy: {auroc: float}}}
    """
    print(f"\n=== Step 5: Nemenyi strategy significance test ===")
    NEMENYI_DIR.mkdir(parents=True, exist_ok=True)

    concepts = CONCEPT_NAMES if concept_filter is None else [concept_filter]
    models   = list(all_model_auroc_results.keys())

    auroc_matrix = build_nemenyi_auroc_matrix(
        auroc_results_per_model = all_model_auroc_results,
        strategy_ids            = RANKED_STRATEGIES,
        concept_names           = concepts,
        model_names             = models,
    )

    result = nemenyi_strategy_significance(auroc_matrix, RANKED_STRATEGIES)
    # Serialise (nemenyi_pvalues is ndarray — convert to list)
    result_serialisable = {k: v for k, v in result.items() if k != "nemenyi_pvalues"}
    result_serialisable["nemenyi_pvalues"] = result["nemenyi_pvalues"].tolist() \
        if isinstance(result.get("nemenyi_pvalues"), np.ndarray) else None

    out_path = NEMENYI_DIR / "nemenyi_strategy_significance.json"
    with open(out_path, "w") as f:
        json.dump(result_serialisable, f, indent=2)
    print(f"  Friedman p={result.get('friedman_p', 'N/A'):.4e}  CD={result.get('cd', 'N/A'):.3f}")
    print(f"  Significant pairs: {len(result.get('significant_pairs', []))}")
    print(f"  Saved → {out_path}")


# ── Full per-model pipeline ───────────────────────────────────────────────────

def run_model(model_name: str, args: argparse.Namespace) -> dict:
    """
    Execute the full 4-step per-model pipeline.
    Returns the auroc results dict for later use in the cross-model Nemenyi test.
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model '{model_name}'. Valid: {list(MODEL_CONFIGS)}")

    concept_filter = getattr(args, "concept", None)

    if not args.skip_extraction:
        step_extract(model_name, device=args.device, concept_filter=concept_filter)

    auroc_summary = step_pool_and_auroc(
        model_name           = model_name,
        construction_method  = getattr(args, "construction_method", DEFAULT_CONSTRUCTION),
        concept_filter       = concept_filter,
    )
    best_layer = auroc_summary["best_layer"]

    if not args.linearity_only:
        step_linearity(model_name, best_layer,
                       construction_method=getattr(args, "construction_method", DEFAULT_CONSTRUCTION),
                       concept_filter=concept_filter)
        step_icc(model_name,
                 construction_method=getattr(args, "construction_method", DEFAULT_CONSTRUCTION),
                 concept_filter=concept_filter)

    # Load the best-layer auroc dict for Nemenyi aggregation
    best_layer_path = AUROC_DIR / model_name / "best_layer_auroc.json"
    if best_layer_path.exists():
        with open(best_layer_path) as f:
            data = json.load(f)
        best_layer_auroc = data.get("per_layer", {}).get(str(best_layer), {})
    else:
        best_layer_auroc = {}

    return best_layer_auroc


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PoolBench per-model runner")
    parser.add_argument("--model",    type=str, choices=list(MODEL_CONFIGS),
                        help="Model to run (single model)")
    parser.add_argument("--all",      action="store_true",
                        help="Run all 7 models sequentially")
    parser.add_argument("--concept",  type=str, default=None,
                        help="Restrict to a single concept (for testing)")
    parser.add_argument("--device",   type=str, default="cuda:0",
                        help="Torch device, e.g. cuda:0 or cpu")
    parser.add_argument("--skip_extraction", action="store_true",
                        help="Skip activation extraction (assume already done)")
    parser.add_argument("--linearity_only",  action="store_true",
                        help="Only run linearity checks (no extraction)")
    parser.add_argument("--nemenyi_only",    action="store_true",
                        help="Load saved AUROC results and run Nemenyi test only")
    parser.add_argument("--construction_method", type=str, default=DEFAULT_CONSTRUCTION,
                        help="Construction method (C1–C5)")
    args = parser.parse_args()

    if args.nemenyi_only:
        # Load saved best-layer AUROC from all models
        all_results: dict[str, dict] = {}
        for mn in MODEL_CONFIGS:
            p = AUROC_DIR / mn / "best_layer_auroc.json"
            if not p.exists():
                print(f"  [nemenyi] {mn}: no saved results — skip")
                continue
            with open(p) as f:
                data = json.load(f)
            bl = data.get("best_layer")
            all_results[mn] = data.get("per_layer", {}).get(str(bl), {})
        step_nemenyi(all_results, concept_filter=args.concept)
        return

    models_to_run = list(MODEL_CONFIGS) if args.all else ([args.model] if args.model else None)
    if models_to_run is None:
        parser.print_help()
        return

    all_auroc_results: dict[str, dict] = {}
    for model_name in models_to_run:
        print(f"\n{'='*60}")
        print(f"  MODEL: {model_name}")
        print(f"{'='*60}")
        model_results = run_model(model_name, args)
        all_auroc_results[model_name] = model_results

    if args.all and not args.linearity_only:
        step_nemenyi(all_auroc_results, concept_filter=args.concept)

    print("\nAll done.")


if __name__ == "__main__":
    main()
