"""
run_model.py
Per-model orchestration (full 6-step pipeline):
  Step 1: Activation extraction
  Step 2: Pool + AUROC (D1)
  Step 3: Linearity check
  Step 4: ICC computation
  Step 4b: Keyword ablation (seeded concepts only)
  Step 5: Classifier B training (once per run, all concepts)
  Step 6: D2 SCP — Steered Concept Prevalence
  Step 7: D3 Disentanglement
  Step 8: Nemenyi significance test (after all models)

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

# Skip D2/D3 (D1 only)
python run_model.py --model llama3_8b --skip_scp --device cuda:0

# Run linearity checks only (still benefits from GPU for MLP folds)
python run_model.py --model llama3_8b --linearity_only --device cuda:0
"""

from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from poolbench.concepts           import CONCEPTS, CONCEPT_NAMES
from poolbench.pooling_strategies import (STRATEGY_REGISTRY, RANKED_STRATEGIES,
                                    compute_all_pooling_strategies,
                                    build_unigram_probs_from_activations,
                                    build_iti_concept_probes)
from poolbench.construction_methods import DEFAULT_CONSTRUCTION
from poolbench.probe_training       import (compute_all_auroc, check_linearity_assumption,
                                      compute_layer_icc, nemenyi_strategy_significance,
                                      build_nemenyi_auroc_matrix,
                                      compute_all_train_test_auroc)
from poolbench.extract_activations  import extract_activations_for_model, load_activations
from poolbench.logger               import get_logger, gpu_mem_str, free_gpu_memory, log_step, find_free_gpu


# ── Model configs ─────────────────────────────────────────────────────────────

MODEL_CONFIGS: dict[str, dict] = {
    "llama3_8b": {
        "hf_id":            "NousResearch/Meta-Llama-3.1-8B",
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
        "hf_id":            "mistralai/Mistral-7B-v0.1",
        "d_model":          4096,
        "n_layers":         32,
        "candidate_layers": [8, 16, 24],  # aligns with jbloom SAE layers
        "architecture":     "causal_lm",
        "batch_size":       8,
    },
}


def _apply_runtime_model_config_overrides() -> None:
    """Allow safe runtime overrides (for example, bigger H100 extraction batches)."""
    global_override = os.environ.get("POOLBENCH_BATCH_SIZE")
    for model_name, cfg in MODEL_CONFIGS.items():
        specific_key = f"POOLBENCH_BATCH_SIZE_{model_name.upper()}"
        override = os.environ.get(specific_key, global_override)
        if override is None:
            continue
        try:
            batch_size = int(override)
        except ValueError as exc:
            raise ValueError(f"Invalid {specific_key}={override!r}; expected integer") from exc
        if batch_size < 1:
            raise ValueError(f"Invalid {specific_key}={override!r}; must be >= 1")
        cfg["batch_size"] = batch_size


_apply_runtime_model_config_overrides()

BASE_DIR      = Path(__file__).parent.parent
RESULTS_DIR   = Path(os.environ.get("POOLBENCH_RESULTS_DIR", BASE_DIR / "results"))
ACT_DIR       = RESULTS_DIR / "activations"
AUROC_DIR     = RESULTS_DIR / "auroc"
LINEARITY_DIR = RESULTS_DIR / "linearity"
NEMENYI_DIR   = RESULTS_DIR / "nemenyi"
ICC_DIR       = RESULTS_DIR / "icc"
CORPUS_DIR    = BASE_DIR / "data" / "corpora"
ABLATION_DIR  = RESULTS_DIR / "ablation"
CLASSIFIERS_DIR = RESULTS_DIR / "bert_classifiers"
SCP_DIR       = RESULTS_DIR / "scp"
D3_DIR        = RESULTS_DIR / "disentanglement"

LAYER_SELECTION_REPRESENTATIVE_CONCEPTS = [
    "hedging",        # sparse lexical
    "imdb_sentiment", # dense lexical
    "causation",      # syntactic
    "academic_tone",  # register
    "planning",       # semantic abstract
]
LAYER_SELECTION_STRATEGIES = ["A1_mean", "P1_last_token", "A2_max"]

# Initialise the shared logger (writes to stdout + log file)
log = get_logger(
    "poolbench",
    log_file=RESULTS_DIR / "run.log",
)


def _safe_load_json(path: Path) -> dict | None:
    """Load JSON checkpoint if it exists and is readable."""
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception as exc:
        raise RuntimeError(f"Checkpoint exists but could not be read: {path}: {exc}") from exc


def _missing_keys(data: dict, required_keys: list[str]) -> list[str]:
    """Return required keys absent from a checkpoint dictionary."""
    return [k for k in required_keys if k not in data]


def _test_size_for_concept(concept_name: str) -> int:
    """Return the real held-out test size per class for a concept."""
    from poolbench.utils import load_jsonl  # noqa: PLC0415
    pos_path = CORPUS_DIR / concept_name / "test_pos.jsonl"
    neg_path = CORPUS_DIR / concept_name / "test_neg.jsonl"
    if not pos_path.exists() or not neg_path.exists():
        raise FileNotFoundError(f"Missing test split for {concept_name}: {pos_path} / {neg_path}")
    return min(len(load_jsonl(str(pos_path))), len(load_jsonl(str(neg_path))))


def _write_extraction_manifest(model_name: str, candidate_layers: list[int],
                               concepts_to_run: list[str]) -> None:
    """Validate Step 1 activation files and save a resumability manifest."""
    missing: list[str] = []
    files: list[str] = []
    for layer_idx in candidate_layers:
        for concept_name in concepts_to_run:
            for partition in ("train", "test"):
                for split in ("pos", "neg"):
                    path = ACT_DIR / model_name / f"layer_{layer_idx}" / f"{concept_name}_{partition}_{split}.npy"
                    if not path.exists():
                        missing.append(str(path))
                    else:
                        files.append(str(path))
    if missing:
        raise RuntimeError(
            f"[extract] Step 1 incomplete; missing {len(missing)} activation files: {missing[:5]}"
        )

    manifest_path = ACT_DIR / model_name / "extraction_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump({
            "model_name": model_name,
            "candidate_layers": candidate_layers,
            "concepts": concepts_to_run,
            "n_files": len(files),
            "files": files,
        }, f, indent=2)
    log.info(f"  [extract] manifest saved → {manifest_path}")


def _concepts_present_in_auroc(layer_results: dict) -> list[str]:
    """Infer concept names present in an AUROC result dict keyed concept_strategy."""
    concepts: set[str] = set()
    for key in layer_results:
        for strategy_id in RANKED_STRATEGIES:
            suffix = f"_{strategy_id}"
            if key.endswith(suffix):
                concepts.add(key[:-len(suffix)])
                break
    return sorted(concepts)


def _merge_auroc_results(existing: dict | None, new: dict, concepts_to_replace: list[str]) -> dict:
    """Merge per-concept AUROC results while replacing stale entries for the current concepts."""
    merged = dict(existing or {})
    prefixes = tuple(f"{concept}_" for concept in concepts_to_replace)
    for key in list(merged):
        if key.startswith(prefixes):
            merged.pop(key, None)
    merged.update(new)
    return merged


# ── Step 1: Extract activations ───────────────────────────────────────────────

def step_extract(model_name: str, device: str, skip_existing: bool = True,
                 concept_filter: str | None = None) -> None:
    cfg = MODEL_CONFIGS[model_name]
    concepts_to_run = [concept_filter] if concept_filter is not None else sorted(CONCEPTS.keys())
    with log_step(log, f"Step 1 extract  model={model_name}", device):
        log.info(f"  corpus_dir={CORPUS_DIR}  out_dir={ACT_DIR}  GPU: {gpu_mem_str(device)}")
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
        _write_extraction_manifest(model_name, cfg["candidate_layers"], concepts_to_run)
    free_gpu_memory(device)
    log.info(f"  Step 1 done. GPU: {gpu_mem_str(device)}")


# ── Step 2: Pool + compute AUROC per layer, select best ───────────────────────

def step_pool_and_auroc(model_name: str, construction_method: str = DEFAULT_CONSTRUCTION,
                        concept_filter: str | None = None,
                        skip_existing: bool = True) -> dict:
    """
    For each candidate layer, apply all 19 ranked strategies + compute AUROC.
    Select best layer per strategy (modal layer selection across all concepts).

    Returns consolidated auroc_results dict:
        {"best_layer": int, "per_layer": {layer: auroc_results_dict}}
    """
    cfg = MODEL_CONFIGS[model_name]
    log.info(f"\n=== Step 2: Pool + AUROC — {model_name} ===  GPU: {gpu_mem_str()}")
    concepts_to_run = {k: v for k, v in CONCEPTS.items()
                       if concept_filter is None or k == concept_filter}
    expected_keys = [f"{concept}_{strategy}" for concept in concepts_to_run for strategy in RANKED_STRATEGIES]

    primary_out = AUROC_DIR / model_name / "best_layer_auroc.json"
    existing_summary = _safe_load_json(primary_out) if skip_existing else None
    if existing_summary is not None:
        if existing_summary.get("layer_selection_method") != "methodology_representative_mean":
            raise RuntimeError(
                f"Existing Step 2 checkpoint lacks current layer-selection method metadata: {primary_out}. "
                "Rerun with --force_from_step 2 or earlier."
            )
        per_layer_existing = existing_summary.get("per_layer", {})
        layers_complete = True
        for layer_idx in cfg["candidate_layers"]:
            layer_res = per_layer_existing.get(str(layer_idx), {})
            if _missing_keys(layer_res, expected_keys):
                layers_complete = False
                break
        if layers_complete:
            log.info(f"  [checkpoint] Step 2 already complete → {primary_out}; skipping")
            return {
                "best_layer": existing_summary.get("best_layer"),
                "per_layer": {int(k): v for k, v in per_layer_existing.items()},
            }

    per_layer_results: dict[int, dict] = {}
    fallback_accumulator: dict = {"counts": {}, "total": 0}

    for layer_idx in cfg["candidate_layers"]:
        log.info(f"\n  Layer {layer_idx}  GPU: {gpu_mem_str()}")
        layer_act_dir = ACT_DIR / model_name / f"layer_{layer_idx}"
        layer_auroc_dir = AUROC_DIR / model_name / f"layer_{layer_idx}"
        layer_out_path = layer_auroc_dir / f"{model_name}_auroc_results.json"

        existing_layer = _safe_load_json(layer_out_path) if skip_existing else None
        if existing_layer is not None:
            missing = _missing_keys(existing_layer, expected_keys)
            if not missing:
                log.info(f"  [checkpoint] Step 2 layer {layer_idx} already complete → {layer_out_path}; skipping")
                per_layer_results[layer_idx] = existing_layer
                continue
            log.info(f"  [checkpoint] Step 2 layer {layer_idx} missing {len(missing)} AUROC cells — recomputing layer")

        unigram_probs = build_unigram_probs_from_activations(layer_act_dir, concepts_to_run, partition="train")
        concept_probes = build_iti_concept_probes(layer_act_dir, concepts_to_run, partition="train")
        log.info(f"  [pool] S2 unigram vocab={len(unigram_probs)}  S3 ITI probes={len(concept_probes)}")

        # Build pooled_results: {concept_strategy: {pos_pooled, neg_pooled}}
        train_pooled_results: dict = {}
        test_pooled_results: dict = {}
        for concept_name in concepts_to_run:
            # Verify files exist without loading them (compute_all_pooling_strategies reloads from disk)
            for _part in ("train", "test"):
                for _split in ("pos", "neg"):
                    _p = layer_act_dir / f"{concept_name}_{_part}_{_split}.npy"
                    if not _p.exists():
                        raise RuntimeError(f"[pool] {concept_name} L{layer_idx}: missing {_p.name}")
            # Apply all pooling strategies
            concept_dict = {concept_name: concepts_to_run[concept_name]}
            train_layer_pooled = compute_all_pooling_strategies(
                act_dir       = layer_act_dir,
                concepts      = concept_dict,
                tokenizer_name= cfg["hf_id"],
                unigram_probs = unigram_probs,
                concept_probes= concept_probes,
                partition     = "train",
                fallback_accumulator = fallback_accumulator,
            )
            test_layer_pooled = compute_all_pooling_strategies(
                act_dir       = layer_act_dir,
                concepts      = concept_dict,
                tokenizer_name= cfg["hf_id"],
                unigram_probs = unigram_probs,
                concept_probes= concept_probes,
                partition     = "test",
                fallback_accumulator = fallback_accumulator,
            )
            train_pooled_results.update(train_layer_pooled)
            test_pooled_results.update(test_layer_pooled)

        auroc_res = compute_all_train_test_auroc(
            train_pooled_results = train_pooled_results,
            test_pooled_results  = test_pooled_results,
            model_name           = model_name,
            out_dir              = layer_auroc_dir,
            construction_method  = construction_method,
        )
        if concept_filter is not None:
            auroc_res = _merge_auroc_results(existing_layer, auroc_res, list(concepts_to_run))
            with open(layer_out_path, "w") as f:
                json.dump(auroc_res, f, indent=2)
        per_layer_results[layer_idx] = auroc_res

    fallback_out = AUROC_DIR / model_name / "fallback_rates.json"
    fallback_total = max(fallback_accumulator.get("total", 0), 1)
    fallback_rates = {
        s: count / fallback_total for s, count in fallback_accumulator.get("counts", {}).items()
    }
    fallback_out.parent.mkdir(parents=True, exist_ok=True)
    with open(fallback_out, "w") as f:
        json.dump({"rates": fallback_rates, **fallback_accumulator}, f, indent=2)
    log.info(f"  [pool] aggregate fallback rates saved → {fallback_out}")

    if concept_filter is not None and existing_summary is not None:
        existing_per_layer = {
            int(k): v for k, v in existing_summary.get("per_layer", {}).items()
        }
        for layer_idx, layer_res in per_layer_results.items():
            existing_per_layer[layer_idx] = _merge_auroc_results(
                existing_per_layer.get(layer_idx), layer_res, list(concepts_to_run)
            )
        per_layer_results = existing_per_layer

    layer_selection_concepts = sorted({
        concept
        for layer_res in per_layer_results.values()
        for concept in _concepts_present_in_auroc(layer_res)
    }) or list(concepts_to_run)
    best_layer = _select_methodology_layer(per_layer_results, cfg["candidate_layers"], layer_selection_concepts)
    log.info(f"\n  Best layer (methodology mean over representative concepts × mean/last/max): {best_layer}")

    # Save the best-layer results as the primary leaderboard input
    primary_out.parent.mkdir(parents=True, exist_ok=True)
    with open(primary_out, "w") as f:
        json.dump({
            "best_layer":    best_layer,
            "per_layer":     {str(k): v for k, v in per_layer_results.items()},
            "layer_selection_method": "methodology_representative_mean",
            "layer_selection_concepts": LAYER_SELECTION_REPRESENTATIVE_CONCEPTS,
            "layer_selection_strategies": LAYER_SELECTION_STRATEGIES,
        }, f, indent=2)

    return {"best_layer": best_layer, "per_layer": per_layer_results}


def _select_methodology_layer(per_layer_results: dict[int, dict],
                              candidate_layers: list[int],
                              concepts_to_run: list[str]) -> int:
    """
    Select shared layer by highest mean AUROC over representative concepts and
    mean/last/max pooling, matching the methodology.
    """
    if len(concepts_to_run) == 1:
        selected_concepts = concepts_to_run
    else:
        selected_concepts = [c for c in LAYER_SELECTION_REPRESENTATIVE_CONCEPTS if c in concepts_to_run]
    if not selected_concepts:
        raise RuntimeError("No representative concepts available for layer selection")

    layer_scores: dict[int, float] = {}
    for layer in candidate_layers:
        res = per_layer_results.get(layer, {})
        scores = []
        missing = []
        for concept in selected_concepts:
            for strategy in LAYER_SELECTION_STRATEGIES:
                key = f"{concept}_{strategy}"
                auroc = res.get(key, {}).get("auroc")
                if auroc is None:
                    missing.append(key)
                else:
                    scores.append(float(auroc))
        if missing:
            raise RuntimeError(f"Layer {layer} missing layer-selection AUROC cells: {missing}")
        layer_scores[layer] = float(np.mean(scores))
        log.info(f"  [layer-select] L{layer}: mean AUROC={layer_scores[layer]:.4f} "
                 f"over concepts={selected_concepts} strategies={LAYER_SELECTION_STRATEGIES}")

    return max(layer_scores, key=layer_scores.get)


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
                   device: str = "cpu",
                   construction_method: str = DEFAULT_CONSTRUCTION,
                   concept_filter: str | None = None,
                   skip_existing: bool = True) -> None:
    """
    Run the linearity assumption check for each concept at the best layer.
    MLP runs on `device` (GPU accelerated when device=cuda:N).
    Saves results to results/linearity/{model_name}_linearity.json.
    """
    log.info(f"\n=== Step 3: Linearity check — {model_name} L{best_layer} ===  GPU: {gpu_mem_str(device)}")
    LINEARITY_DIR.mkdir(parents=True, exist_ok=True)
    concepts_to_run = CONCEPT_NAMES if concept_filter is None else [concept_filter]
    out_path = LINEARITY_DIR / f"{model_name}_linearity.json"
    existing = _safe_load_json(out_path) if skip_existing else None
    if existing is not None:
        missing = _missing_keys(existing, concepts_to_run)
        if not missing:
            log.info(f"  [checkpoint] Step 3 already complete → {out_path}; skipping")
            return
        log.info(f"  [checkpoint] Step 3 missing {len(missing)} concept(s) — resuming")

    linearity_results = existing or {}
    for concept_name in concepts_to_run:
        if concept_name in linearity_results:
            log.info(f"  [checkpoint] Step 3 {concept_name} already done — skipping")
            continue
        pos_acts = load_activations(ACT_DIR, model_name, best_layer, concept_name, "pos", partition="train")
        neg_acts = load_activations(ACT_DIR, model_name, best_layer, concept_name, "neg", partition="train")
        if pos_acts is None or neg_acts is None:
            raise RuntimeError(f"[linearity] missing train activations for {concept_name} L{best_layer}")

        pos_pooled = np.stack([np.asarray(item["hidden"], dtype=np.float32).mean(0) for item in pos_acts])
        neg_pooled = np.stack([np.asarray(item["hidden"], dtype=np.float32).mean(0) for item in neg_acts])

        result = check_linearity_assumption(pos_pooled, neg_pooled, concept_name,
                                             construction_method, device=device)
        linearity_results[concept_name] = result
        with open(out_path, "w") as f:
            json.dump(linearity_results, f, indent=2)
        status = "✓" if result["passes"] else "✗ FAIL"
        log.info(f"  {status}  {concept_name}: linear={result['linear_auroc']:.3f} "
              f"mlp={result['mlp_auroc']:.3f} gap={result['gap']:.4f}")

    with open(out_path, "w") as f:
        json.dump(linearity_results, f, indent=2)
    log.info(f"  Saved → {out_path}")

    n_fail = sum(1 for r in linearity_results.values() if not r["passes"])
    if n_fail > 0:
        log.warning(f"  {n_fail} concept(s) fail linearity check (gap ≥ 0.03). "
              f"Inspect results and consider using C3_logreg or C4_repe for those concepts.")


# ── Step 4: ICC computation ───────────────────────────────────────────────────

def step_icc(model_name: str, construction_method: str = DEFAULT_CONSTRUCTION,
             concept_filter: str | None = None,
             skip_existing: bool = True) -> None:
    """
    Compute intraclass correlation across candidate layers (for N_eff correction).
    Requires best-layer auroc results already saved by step_pool_and_auroc.
    """
    log.info(f"\n=== Step 4: ICC computation — {model_name} ===  GPU: {gpu_mem_str()}")
    ICC_DIR.mkdir(parents=True, exist_ok=True)
    concepts_to_run = CONCEPT_NAMES if concept_filter is None else [concept_filter]
    out_path = ICC_DIR / f"{model_name}_icc.json"
    existing = _safe_load_json(out_path) if skip_existing else None
    if existing is not None:
        existing_concepts = existing.get("icc_per_concept", {})
        if isinstance(existing_concepts, dict) and not _missing_keys(existing_concepts, concepts_to_run):
            log.info(f"  [checkpoint] Step 4 already complete → {out_path}; skipping")
            return
        log.info("  [checkpoint] Step 4 ICC checkpoint incomplete — recomputing")

    best_layer_path = AUROC_DIR / model_name / "best_layer_auroc.json"
    if not best_layer_path.exists():
        raise FileNotFoundError(f"[icc] {best_layer_path} not found — run step 2 first")

    with open(best_layer_path) as f:
        data = json.load(f)

    # Build per-concept strategy × layer AUROC matrices for ICC(2,1).  Layers are
    # the raters; pooling strategies are the targets for each model-concept pair.
    per_layer = data.get("per_layer", {})
    layer_aurocs_per_concept: dict[str, list[list[float]]] = {}
    for concept_name in concepts_to_run:
        concept_matrix: list[list[float]] = []
        for strategy_id in RANKED_STRATEGIES:
            strategy_aurocs: list[float] = []
            for layer_str in sorted(per_layer.keys(), key=int):
                key = f"{concept_name}_{strategy_id}"
                auroc = per_layer[layer_str].get(key, {}).get("auroc")
                if auroc is None:
                    raise RuntimeError(f"[icc] Missing AUROC for {key} layer={layer_str}")
                strategy_aurocs.append(float(auroc))
            concept_matrix.append(strategy_aurocs)
        layer_aurocs_per_concept[concept_name] = concept_matrix

    cfg     = MODEL_CONFIGS[model_name]
    n_base  = {concept_name: _test_size_for_concept(concept_name) for concept_name in layer_aurocs_per_concept}
    icc_res = compute_layer_icc(layer_aurocs_per_concept, n_base=n_base)
    icc_res["model_name"] = model_name

    with open(out_path, "w") as f:
        json.dump(icc_res, f, indent=2)
    log.info(f"  mean_icc={icc_res['mean_icc']:.3f}  N_eff={icc_res['N_eff']}")
    log.info(f"  Saved → {out_path}")


# ── Step 5: Nemenyi strategy significance test ────────────────────────────────

def step_nemenyi(all_model_auroc_results: dict[str, dict],
                 concept_filter: str | None = None) -> None:
    """
    Run the Nemenyi test over all 3 models jointly.
    Should be called after all models have completed steps 1–2.

    all_model_auroc_results: {model_name: {concept_strategy: {auroc: float}}}
    """
    log.info(f"\n=== Step 8: Nemenyi strategy significance test ===  GPU: {gpu_mem_str()}")
    NEMENYI_DIR.mkdir(parents=True, exist_ok=True)

    concepts = CONCEPT_NAMES if concept_filter is None else [concept_filter]
    models   = list(all_model_auroc_results.keys())
    effective_n = 0.0
    for model_name in models:
        icc_path = ICC_DIR / f"{model_name}_icc.json"
        icc_data = _safe_load_json(icc_path)
        if not icc_data:
            raise RuntimeError(f"Missing ICC file required for ICC-adjusted Nemenyi: {icc_path}")
        n_base_map = icc_data.get("n_base_per_concept", {})
        n_eff_map = icc_data.get("n_eff_per_concept", {})
        missing = [c for c in concepts if c not in n_base_map or c not in n_eff_map]
        if missing:
            raise RuntimeError(f"ICC file {icc_path} missing per-concept N values for: {missing}")
        effective_n += sum(float(n_eff_map[c]) / max(float(n_base_map[c]), 1.0) for c in concepts)

    auroc_matrix = build_nemenyi_auroc_matrix(
        auroc_results_per_model = all_model_auroc_results,
        strategy_ids            = RANKED_STRATEGIES,
        concept_names           = concepts,
        model_names             = models,
    )

    result = nemenyi_strategy_significance(auroc_matrix, RANKED_STRATEGIES,
                                           effective_n=effective_n)
    # Serialise (nemenyi_pvalues is ndarray — convert to list)
    result_serialisable = {k: v for k, v in result.items() if k != "nemenyi_pvalues"}
    result_serialisable["nemenyi_pvalues"] = result["nemenyi_pvalues"].tolist() \
        if isinstance(result.get("nemenyi_pvalues"), np.ndarray) else None

    out_path = NEMENYI_DIR / "nemenyi_strategy_significance.json"
    with open(out_path, "w") as f:
        json.dump(result_serialisable, f, indent=2)
    log.info(f"  Friedman p={result.get('friedman_p', 'N/A'):.4e}  CD={result.get('cd', 'N/A'):.3f}  "
             f"N_eff={result.get('N_effective', 'N/A')}")
    log.info(f"  Significant pairs: {len(result.get('significant_pairs', []))}")
    log.info(f"  Saved → {out_path}")


# ── Step 4b: Keyword ablation check ──────────────────────────────────────────

def step_keyword_ablation(
    model_name: str,
    best_layer: int,
    device: str,
    concept_filter: str | None = None,
    skip_existing: bool = True,
) -> None:
    """
    For each seeded concept, mask seed words in positive test passages, re-extract
    activations, and compare AUROC. Flags concepts that rely on surface keywords.
    Saves results to results/ablation/{model_name}_keyword_ablation.json.
    (§40 of methodology)
    """
    import re as _re  # noqa: PLC0415
    from poolbench.extract_activations import load_model as _load_model  # noqa: PLC0415
    from poolbench.extract_activations import _extract_batch  # noqa: PLC0415
    from poolbench.evaluation.probe import keyword_ablation_check  # noqa: PLC0415
    from poolbench.utils import load_jsonl  # noqa: PLC0415

    log.info(f"\n=== Step 4b: Keyword ablation — {model_name} L{best_layer} ===  GPU: {gpu_mem_str(device)}")
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)

    seeded_concepts = [
        (name, meta) for name, meta in CONCEPTS.items()
        if meta.get("seed_words") and (concept_filter is None or name == concept_filter)
    ]

    if not seeded_concepts:
        log.info("  No seeded concepts to ablate.")
        return

    out_path = ABLATION_DIR / f"{model_name}_keyword_ablation.json"
    required_concepts = [name for name, _ in seeded_concepts]
    existing = _safe_load_json(out_path) if skip_existing else None
    if existing is not None:
        missing = _missing_keys(existing, required_concepts)
        if not missing:
            log.info(f"  [checkpoint] Step 4b already complete → {out_path}; skipping")
            return
        log.info(f"  [checkpoint] Step 4b missing {len(missing)} concept(s) — resuming")

    cfg = MODEL_CONFIGS[model_name]
    model, tokenizer = _load_model(model_name, cfg["hf_id"], device)
    log.info(f"  Model loaded for ablation  GPU: {gpu_mem_str(device)}")

    results: dict = existing or {}

    for concept_name, concept_meta in seeded_concepts:
        if concept_name in results:
            log.info(f"  [checkpoint] Step 4b {concept_name} already done — skipping")
            continue
        seed_words = concept_meta.get("seed_words", [])
        if not seed_words:
            continue

        # Build regex that matches seed words (case-insensitive, whole-word)
        pattern = _re.compile(
            r"\b(" + "|".join(_re.escape(w) for w in seed_words) + r")\b",
            flags=_re.IGNORECASE,
        )

        jsonl_path = CORPUS_DIR / concept_name / "test_pos.jsonl"
        if not jsonl_path.exists():
            raise FileNotFoundError(f"[ablation] {concept_name}: test_pos.jsonl not found")

        records   = load_jsonl(str(jsonl_path))
        ablated_texts = [pattern.sub("", r["text"]).strip() for r in records]

        # Extract ablated activations
        abl_items: list[dict] = []
        for i in range(0, len(ablated_texts), cfg["batch_size"]):
            batch = ablated_texts[i: i + cfg["batch_size"]]
            items = _extract_batch(model, tokenizer, model_name, batch, best_layer, device)
            abl_items.extend(items)

        if not abl_items:
            raise RuntimeError(f"[ablation] {concept_name}: no ablated activations extracted")

        # Full activations already extracted in Step 1
        full_pos = load_activations(ACT_DIR, model_name, best_layer, concept_name, "pos", partition="test")
        full_neg = load_activations(ACT_DIR, model_name, best_layer, concept_name, "neg", partition="test")
        if full_pos is None or full_neg is None:
            raise RuntimeError(f"[ablation] {concept_name}: full test activations not found — run Step 1 first")

        pos_full_pooled = np.stack([np.asarray(item["hidden"], dtype=np.float32).mean(0) for item in full_pos])
        neg_full_pooled = np.stack([np.asarray(item["hidden"], dtype=np.float32).mean(0) for item in full_neg])
        pos_abl_pooled  = np.stack([np.asarray(item["hidden"], dtype=np.float32).mean(0) for item in abl_items])
        neg_abl_pooled  = neg_full_pooled  # negatives unchanged

        res = keyword_ablation_check(
            pos_full_pooled, neg_full_pooled,
            pos_abl_pooled,  neg_abl_pooled,
            concept_name=concept_name,
        )
        results[concept_name] = res
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        flag = "♦ SURFACE KEYWORD" if res["signal_in_surface"] else "✓"
        log.info(f"  {flag}  {concept_name}: full={res['full_auroc']:.3f} "
                 f"ablated={res['ablated_auroc']:.3f} drop={res['drop']:.4f}")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"  Ablation results saved → {out_path}")

    del model, tokenizer
    free_gpu_memory(device)
    log.info(f"  [ablation] GPU freed  {gpu_mem_str(device)}")


# ── Step 5: Classifier B training (run once per experiment) ──────────────────

def step_train_classifiers(
    device: str = "cuda:0",   # GPU strongly preferred — bs=64 on A100 is ~3× faster than CPU
    force_retrain: bool = False,
) -> None:
    """
    Train Classifier B for all non-LLM-scored concepts.
    Saved to results/bert_classifiers/{concept}/.
    Idempotent: skips already-trained classifiers unless force_retrain=True.
    """
    from poolbench.evaluation.classifier_b import train_all_classifiers_b  # noqa: PLC0415
    log.info(f"\n=== Step 5: Training Classifier B (all concepts)  GPU: {gpu_mem_str(device)} ===")
    results = train_all_classifiers_b(
        classifiers_dir = CLASSIFIERS_DIR,
        device          = device,
        force_retrain   = force_retrain,
    )
    trained   = sum(1 for v in results.values() if v is not None)
    llm_scored = sum(1 for v in results.values() if v is None)
    CLASSIFIERS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = CLASSIFIERS_DIR / "classifier_b_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "trained": {k: str(v) for k, v in results.items() if v is not None},
            "llm_scored": [k for k, v in results.items() if v is None],
        }, f, indent=2)
    log.info(f"  Classifier B manifest saved → {manifest_path}")
    log.info(f"  Classifier B: {trained} trained, {llm_scored} using LLM scoring  GPU: {gpu_mem_str(device)}")


# ── Step 6: D2 SCP ───────────────────────────────────────────────────────────

def step_scp(
    model_name: str,
    best_layer: int,
    device: str,
    concept_filter: str | None = None,
    skip_existing: bool = True,
    model=None,
    tokenizer=None,
) -> dict:
    """
    Compute D2 Steered Concept Prevalence for all (concept × strategy) pairs.
    Requires Classifier B to already be trained (Step 5).
    Saves to results/scp/{model_name}_scp.json.
    If `model`/`tokenizer` are passed in, they are reused (no redundant load).
    """
    from poolbench.evaluation.scp_eval import compute_scp_for_model  # noqa: PLC0415

    concepts = CONCEPT_NAMES if concept_filter is None else [concept_filter]
    cfg      = MODEL_CONFIGS[model_name]

    with log_step(log, f"Step 6 SCP  model={model_name}", device):
        scp_results = compute_scp_for_model(
            model_name      = model_name,
            hf_id           = cfg["hf_id"],
            device          = device,
            best_layer      = best_layer,
            concepts        = concepts,
            strategy_ids    = RANKED_STRATEGIES,
            act_dir         = ACT_DIR,
            classifiers_dir = CLASSIFIERS_DIR,
            out_dir         = SCP_DIR,
            skip_existing   = skip_existing,
            model           = model,
            tokenizer       = tokenizer,
        )

    log.info(f"  Step 6 done. GPU: {gpu_mem_str(device)}")
    return scp_results


def step_prompted_baseline(
    model_name: str,
    device: str,
    concept_filter: str | None = None,
    skip_existing: bool = True,
    model=None,
    tokenizer=None,
) -> dict:
    """Compute the unranked prompted-baseline SCP row required by the methodology."""
    from poolbench.evaluation.scp_eval import (  # noqa: PLC0415
        CURATED_CONCEPT_PROMPTS,
        PROMPTED_BASELINE_PROMPT_SET,
        compute_prompted_baseline,
    )

    concepts = CONCEPT_NAMES if concept_filter is None else [concept_filter]
    cfg = MODEL_CONFIGS[model_name]
    out_path = SCP_DIR / f"{model_name}_prompted_baseline.json"
    if skip_existing and out_path.exists():
        existing = _safe_load_json(out_path)
        prompt_set = (existing or {}).get("_metadata", {}).get("prompt_set")
        if prompt_set != PROMPTED_BASELINE_PROMPT_SET:
            log.info(
                f"  [checkpoint] Prompted baseline uses stale prompt set ({prompt_set}); recomputing"
            )
        else:
            missing = _missing_keys(existing or {}, concepts)
            if not missing:
                log.info(f"  [checkpoint] Prompted baseline already complete → {out_path}; skipping")
                return existing or {}
            log.info(f"  [checkpoint] Prompted baseline missing {len(missing)} concept(s) — resuming")

    with log_step(log, f"Step 6b prompted baseline  model={model_name}", device):
        return compute_prompted_baseline(
            model_name      = model_name,
            hf_id           = cfg["hf_id"],
            device          = device,
            concepts        = concepts,
            classifiers_dir = CLASSIFIERS_DIR,
            out_dir         = SCP_DIR,
            concept_prompts = CURATED_CONCEPT_PROMPTS,
            model           = model,
            tokenizer       = tokenizer,
            skip_existing   = skip_existing,
        )


# ── Step 7: D3 Disentanglement ───────────────────────────────────────────────

def step_disentanglement(
    model_name: str,
    best_layer: int,
    device: str,
    concept_filter: str | None = None,
    skip_existing: bool = True,
    model=None,
    tokenizer=None,
) -> dict:
    """
    Compute D3 disentanglement metrics.
    Requires D2 SCP results (Step 6) to be saved.
    Saves to results/disentanglement/{model_name}_d3.json.
    If `model`/`tokenizer` are passed in, they are reused (no redundant load).
    """
    from poolbench.evaluation.disentanglement import compute_disentanglement_for_model  # noqa: PLC0415

    concepts      = CONCEPT_NAMES if concept_filter is None else [concept_filter]
    cfg           = MODEL_CONFIGS[model_name]
    scp_path      = SCP_DIR / f"{model_name}_scp.json"

    with log_step(log, f"Step 7 D3  model={model_name}", device):
        d3_results = compute_disentanglement_for_model(
            model_name        = model_name,
            hf_id             = cfg["hf_id"],
            device            = device,
            best_layer        = best_layer,
            concepts          = concepts,
            strategy_ids      = RANKED_STRATEGIES,
            scp_results_path  = scp_path,
            act_dir           = ACT_DIR,
            classifiers_dir   = CLASSIFIERS_DIR,
            out_dir           = D3_DIR,
            skip_existing     = skip_existing,
            model             = model,
            tokenizer         = tokenizer,
        )

    free_gpu_memory(device)
    log.info(f"  Step 7 done. GPU: {gpu_mem_str(device)}")
    return d3_results


# ── Full per-model pipeline ───────────────────────────────────────────────────

def _steps_6_7_checkpoints_complete(
    model_name: str,
    concepts: list[str],
    force_step6: bool,
    force_step7: bool,
) -> bool:
    """
    Return True only when *all three* of Steps 6, 6b, and 7 can be skipped due to
    complete on-disk checkpoints.  Returns False immediately if any step is forced.
    Used to avoid loading the 7–9 B LLM when a partial rerun has all three already done.
    """
    from poolbench.evaluation.scp_eval import PROMPTED_BASELINE_PROMPT_SET  # noqa: PLC0415
    from poolbench.evaluation.disentanglement import NEIGHBOUR_PAIRS  # noqa: PLC0415

    # Any forced step means we must rerun — don't bother checking files
    if force_step6 or force_step7:
        return False

    strategy_ids = RANKED_STRATEGIES

    # Step 6 — SCP
    scp_path = SCP_DIR / f"{model_name}_scp.json"
    if not scp_path.exists():
        return False
    scp_data = _safe_load_json(scp_path) or {}
    if any(
        c not in scp_data or any(sid not in scp_data.get(c, {}) for sid in strategy_ids)
        for c in concepts
    ):
        return False

    # Step 6b — Prompted baseline
    pb_path = SCP_DIR / f"{model_name}_prompted_baseline.json"
    if not pb_path.exists():
        return False
    pb_data = _safe_load_json(pb_path) or {}
    if pb_data.get("_metadata", {}).get("prompt_set") != PROMPTED_BASELINE_PROMPT_SET:
        return False
    if _missing_keys(pb_data, concepts):
        return False

    # Step 7 — D3
    d3_path = D3_DIR / f"{model_name}_d3.json"
    if not d3_path.exists():
        return False
    d3_data = _safe_load_json(d3_path) or {}
    d3_concepts = [c for c in concepts if c in NEIGHBOUR_PAIRS]
    if any(
        c not in d3_data or any(sid not in d3_data.get(c, {}) for sid in strategy_ids)
        for c in d3_concepts
    ):
        return False

    return True


def run_model(model_name: str, args: argparse.Namespace) -> dict:
    """
    Execute the full per-model pipeline (Steps 1–7).
    Returns the AUROC results dict for later use in the cross-model Nemenyi test.
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model '{model_name}'. Valid: {list(MODEL_CONFIGS)}")

    concept_filter = getattr(args, "concept", None)
    skip_scp       = getattr(args, "skip_scp", False)
    force_from_step = getattr(args, "force_from_step", None)
    t_start        = time.perf_counter()

    def force_step(step: int) -> bool:
        return force_from_step is not None and step >= force_from_step

    log.info(f"\n{'='*60}")
    log.info(f"  PoolBench pipeline starting: model={model_name}  device={args.device}")
    log.info(f"  GPU on start: {gpu_mem_str(args.device)}")
    if force_from_step is not None:
        log.info(f"  FORCE MODE: recomputing from Step {force_from_step} onward")
    log.info(f"{'='*60}")

    if not args.skip_extraction:
        step_extract(model_name, device=args.device, concept_filter=concept_filter,
                     skip_existing=not force_step(1))

    # Step 3 runs BEFORE the main pooling sweep (§38 pre-check requirement).
    # Linearity is assessed on the middle candidate layer under mean pooling before
    # any strategy ranking begins, so the check is not biased by layer selection.
    _pre_check_layer = MODEL_CONFIGS[model_name]["candidate_layers"][
        len(MODEL_CONFIGS[model_name]["candidate_layers"]) // 2
    ]
    step_linearity(model_name, _pre_check_layer,
                   device=args.device,
                   construction_method=getattr(args, "construction_method", DEFAULT_CONSTRUCTION),
                   concept_filter=concept_filter,
                   skip_existing=not force_step(3))

    auroc_summary = step_pool_and_auroc(
        model_name           = model_name,
        construction_method  = getattr(args, "construction_method", DEFAULT_CONSTRUCTION),
        concept_filter       = concept_filter,
        skip_existing        = not force_step(2),
    )
    best_layer = auroc_summary["best_layer"]
    log.info(f"  Best layer selected: {best_layer}")

    if not args.linearity_only:
        step_icc(model_name,
                 construction_method=getattr(args, "construction_method", DEFAULT_CONSTRUCTION),
                 concept_filter=concept_filter,
                 skip_existing=not force_step(4))

        # Step 4b — keyword ablation check for seeded concepts
        step_keyword_ablation(model_name, best_layer, args.device, concept_filter,
                              skip_existing=not force_step(4))

        if not skip_scp:
            # Step 5 — Train Classifier B on GPU (BERT is small ~500 MB, releases before Step 6)
            step_train_classifiers(device=args.device, force_retrain=force_step(5))

            # Steps 6, 6b, 7 share the same base LLM — load once, pass through all three.
            # But skip the load entirely when all three steps are already checkpointed.
            _concepts_for_check = [concept_filter] if concept_filter else list(CONCEPT_NAMES)
            _need_model = not _steps_6_7_checkpoints_complete(
                model_name,
                _concepts_for_check,
                force_step6=force_step(6),
                force_step7=force_step(7),
            )
            if _need_model:
                from poolbench.extract_activations import load_model as _load_model  # noqa: PLC0415
                cfg = MODEL_CONFIGS[model_name]
                log.info(f"  [shared model] Loading {model_name} for Steps 6/6b/7  GPU: {gpu_mem_str(args.device)}")
                _shared_model, _shared_tok = _load_model(model_name, cfg["hf_id"], args.device)
                log.info(f"  [shared model] Loaded  GPU: {gpu_mem_str(args.device)}")
            else:
                log.info(f"  [shared model] Steps 6/6b/7 fully checkpointed — skipping model load")
                _shared_model, _shared_tok = None, None

            # Step 6 — D2 SCP
            step_scp(model_name, best_layer, args.device, concept_filter,
                     skip_existing=not force_step(6),
                     model=_shared_model, tokenizer=_shared_tok)

            # Step 6b — Prompted baseline row (not ranked)
            step_prompted_baseline(model_name, args.device, concept_filter,
                                   skip_existing=not force_step(6),
                                   model=_shared_model, tokenizer=_shared_tok)

            # Step 7 — D3 Disentanglement
            step_disentanglement(model_name, best_layer, args.device, concept_filter,
                                 skip_existing=not force_step(7),
                                 model=_shared_model, tokenizer=_shared_tok)

            # Release the shared LLM now that all three steps are done
            if _need_model:
                del _shared_model, _shared_tok
                free_gpu_memory(args.device)
                log.info(f"  [shared model] Released  GPU: {gpu_mem_str(args.device)}")

    # Load the best-layer auroc dict for Nemenyi aggregation
    best_layer_path = AUROC_DIR / model_name / "best_layer_auroc.json"
    if best_layer_path.exists():
        with open(best_layer_path) as f:
            data = json.load(f)
        best_layer_auroc = data.get("per_layer", {}).get(str(best_layer), {})
    else:
        raise FileNotFoundError(f"Missing best-layer AUROC file: {best_layer_path}")
    if not best_layer_auroc:
        raise RuntimeError(f"Best-layer AUROC results are empty for {model_name} layer {best_layer}")

    elapsed = time.perf_counter() - t_start
    log.info(f"\n  ✓ Pipeline complete: {model_name}  total={elapsed:.0f}s  GPU: {gpu_mem_str(args.device)}")
    return best_layer_auroc


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PoolBench per-model runner")
    parser.add_argument("--model",    type=str, choices=list(MODEL_CONFIGS),
                        help="Model to run (single model)")
    parser.add_argument("--all",      action="store_true",
                        help="Run all 3 models sequentially")
    parser.add_argument("--concept",  type=str, default=None,
                        help="Restrict to a single concept (for testing)")
    parser.add_argument("--device",   type=str, default="auto",
                        help="Torch device, e.g. cuda:0, cpu, or 'auto' (scan and pick freest GPU)")
    parser.add_argument("--min_free_gb", type=float, default=20.0,
                        help="When --device auto: minimum free VRAM (GB) to consider a GPU usable (default 20)")
    parser.add_argument("--skip_extraction", action="store_true",
                        help="Skip activation extraction (assume already done)")
    parser.add_argument("--skip_scp", action="store_true",
                        help="Skip D2 SCP and D3 disentanglement steps")
    parser.add_argument("--force_from_step", type=int, choices=range(1, 8), default=None,
                        metavar="{1,2,3,4,5,6,7}",
                        help="Recompute from this pipeline step onward, ignoring checkpoints for those steps")
    parser.add_argument("--linearity_only",  action="store_true",
                        help="Only run linearity checks (no extraction)")
    parser.add_argument("--nemenyi_only",    action="store_true",
                        help="Load saved AUROC results and run Nemenyi test only")
    parser.add_argument("--construction_method", type=str, default=DEFAULT_CONSTRUCTION,
                        help="Construction method (C1–C5)")
    args = parser.parse_args()

    # ── Resolve device ──────────────────────────────────────────────────────
    if args.device == "auto":
        log.info("\n[device] --device auto: scanning all GPUs for the freest one...")
        try:
            args.device = find_free_gpu(min_free_gb=args.min_free_gb, logger=log)
            log.info(f"[device] AUTO-SELECTED: {args.device}")
        except RuntimeError as exc:
            log.error(f"[device] {exc}")
            raise SystemExit(1) from exc
    else:
        log.info(f"[device] Using explicitly requested device: {args.device}")

    if args.nemenyi_only:
        # Load saved best-layer AUROC from all models
        all_results: dict[str, dict] = {}
        for mn in MODEL_CONFIGS:
            p = AUROC_DIR / mn / "best_layer_auroc.json"
            if not p.exists():
                log.warning(f"  [nemenyi] {mn}: no saved results — skip")
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
        log.info(f"\n{'='*60}")
        log.info(f"  MODEL: {model_name}")
        log.info(f"{'='*60}")
        model_results = run_model(model_name, args)
        all_auroc_results[model_name] = model_results

    if args.all and not args.linearity_only:
        step_nemenyi(all_auroc_results, concept_filter=args.concept)

    log.info("\nAll models done.")


if __name__ == "__main__":
    main()
