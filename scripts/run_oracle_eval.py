"""
run_oracle_eval.py
Oracle evaluation for G1_IxG (Input × Gradient) pooling.

G1_IxG is excluded from the ranked leaderboard because it requires a full backward
pass per passage (2× wall-clock cost vs inference-only strategies). It is reported as
a one-pass calibration ceiling in the appendix (§45 / Table 7).

This script mirrors the pool → DiffMean → AUROC pipeline of run_model.py but replaces
the activation extraction step with on-the-fly IxG computation, writing oracle
activations to:

    results/activations/{model}/layer_{L}/{concept}_train_pos_oracle.npy
    results/activations/{model}/layer_{L}/{concept}_train_neg_oracle.npy
    results/activations/{model}/layer_{L}/{concept}_test_pos_oracle.npy
    results/activations/{model}/layer_{L}/{concept}_test_neg_oracle.npy

and AUROC results to:

    results/oracle_auroc/{model_name}_oracle_auroc.json

Usage
-----
# Single model, all concepts (best layer only)
python run_oracle_eval.py --model llama3_8b --device cuda:0

# All three models
python run_oracle_eval.py --all --device cuda:0

# Single concept for quick iteration
python run_oracle_eval.py --model gemma2_9b --concept hedging --device cuda:0

# Force recompute even when checkpoints exist
python run_oracle_eval.py --model llama3_8b --force --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from poolbench.concepts           import CONCEPTS, CONCEPT_NAMES
from poolbench.pooling_strategies import pool_IxG
from poolbench.logger             import get_logger, gpu_mem_str, free_gpu_memory, log_step, find_free_gpu
from poolbench.utils              import load_jsonl

# ── Path constants (mirrors run_model.py) ──────────────────────────────────────

BASE_DIR          = Path(__file__).parent.parent
RESULTS_DIR       = Path(os.environ.get("POOLBENCH_RESULTS_DIR", BASE_DIR / "results"))
ACT_DIR           = RESULTS_DIR / "activations"
ORACLE_AUROC_DIR  = RESULTS_DIR / "oracle_auroc"
CORPUS_DIR        = BASE_DIR / "data" / "corpora"

# Mirrors MODEL_CONFIGS in run_model.py
MODEL_CONFIGS: dict[str, dict] = {
    "llama3_8b": {
        "hf_id":            "NousResearch/Meta-Llama-3.1-8B",
        "d_model":          4096,
        "n_layers":         32,
        "candidate_layers": [16, 24, 31],
    },
    "gemma2_9b": {
        "hf_id":            "google/gemma-2-9b",
        "d_model":          3584,
        "n_layers":         42,
        "candidate_layers": [14, 28, 41],
    },
    "mistral_7b": {
        "hf_id":            "mistralai/Mistral-7B-v0.1",
        "d_model":          4096,
        "n_layers":         32,
        "candidate_layers": [8, 16, 24],
    },
}

log = get_logger("poolbench.oracle_eval", log_file=RESULTS_DIR / "oracle_eval.log")

# ── Helpers ───────────────────────────────────────────────────────────────────

def _unit(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / (norm + 1e-9)


def _run_ixg_for_split(
    model,
    tokenizer,
    texts: list[str],
    layer: int,
    device: str,
    desc: str,
) -> np.ndarray:
    """
    Run pool_IxG on every text in ``texts`` for the given layer.
    Falls back to mean-pool for passages where backward fails.
    Returns float32 array of shape (n, d_model).
    """
    from poolbench.pooling_strategies import pool_mean  # noqa: PLC0415
    from poolbench.extract_activations import _get_token_hidden_states  # type: ignore[attr-defined]  # noqa: PLC0415

    vectors = []
    n = len(texts)
    for i, text in enumerate(texts, 1):
        if i % 50 == 0 or i == n:
            log.info(f"    [{desc}] {i}/{n}  GPU: {gpu_mem_str(device)}")
        vec = pool_IxG(model, text, tokenizer, layer, device=device)
        if vec is None:
            # Backward failed — fall back to mean-pool hidden states at this layer
            log.warning(f"    [{desc}] IxG backward failed at idx {i}; falling back to mean-pool")
            import torch  # noqa: PLC0415
            enc = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=200, add_special_tokens=True).to(device)
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True)
            h = out.hidden_states[layer + 1][0].float().cpu().numpy()  # (seq, d)
            vec = h.mean(0)
        vectors.append(vec.astype(np.float32))
    return np.stack(vectors, axis=0)


def _diff_mean_auroc(
    train_pos: np.ndarray,
    train_neg: np.ndarray,
    test_pos: np.ndarray,
    test_neg: np.ndarray,
) -> float:
    """
    DiffMean steering vector + cosine-similarity probe AUROC.
    Matches the C1 + D1 approach used in the main pipeline.
    """
    from sklearn.metrics import roc_auc_score  # noqa: PLC0415

    sv = _unit(train_pos.mean(0) - train_neg.mean(0))   # (d_model,)
    all_vecs   = np.concatenate([test_pos, test_neg], axis=0)
    scores     = all_vecs @ sv                            # cosine-like (unit sv)
    labels     = np.array([1] * len(test_pos) + [0] * len(test_neg), dtype=int)
    try:
        return float(roc_auc_score(labels, scores))
    except ValueError:
        return float("nan")


def run_oracle_for_model(model_name: str, args: argparse.Namespace) -> dict:
    """
    Full oracle pipeline for one model.
    Returns dict mapping concept → {layer → auroc}.
    """
    from poolbench.extract_activations import load_model  # noqa: PLC0415

    cfg = MODEL_CONFIGS[model_name]
    device = args.device

    ORACLE_AUROC_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ORACLE_AUROC_DIR / f"{model_name}_oracle_auroc.json"

    # Load existing checkpoint if present
    existing: dict = {}
    if out_path.exists() and not args.force:
        try:
            with open(out_path) as f:
                existing = json.load(f)
            log.info(f"  [oracle] Loaded checkpoint: {out_path}")
        except Exception:
            existing = {}

    concepts = CONCEPT_NAMES if args.concept is None else [args.concept]

    # Determine which concepts still need computation across any layer
    pending_concepts = [
        c for c in concepts
        if args.force or any(
            str(lay) not in existing.get(c, {})
            for lay in cfg["candidate_layers"]
        )
    ]

    if not pending_concepts:
        log.info(f"  [oracle] {model_name}: all concepts cached — skipping model load")
        return existing

    log.info(f"  [oracle] Loading {model_name} ({cfg['hf_id']}) on {device}  ...")
    model, tokenizer = load_model(model_name, cfg["hf_id"], device)
    model.eval()
    log.info(f"  [oracle] Model loaded  GPU: {gpu_mem_str(device)}")

    results = dict(existing)  # carry over cached results

    for concept in pending_concepts:
        concept_cfg = CONCEPTS[concept]
        concept_dir = CORPUS_DIR / concept

        # Load corpus splits
        train_pos_texts = [item["text"] for item in load_jsonl(concept_dir / "train_pos.jsonl")]
        train_neg_texts = [item["text"] for item in load_jsonl(concept_dir / "train_neg.jsonl")]
        test_pos_texts  = [item["text"] for item in load_jsonl(concept_dir / "test_pos.jsonl")]
        test_neg_texts  = [item["text"] for item in load_jsonl(concept_dir / "test_neg.jsonl")]

        if not results.get(concept):
            results[concept] = {}

        for layer in cfg["candidate_layers"]:
            layer_key = str(layer)
            if not args.force and layer_key in results.get(concept, {}):
                log.info(f"  [oracle] {concept} layer {layer}: cached — skip")
                continue

            log.info(f"\n  [oracle] {concept}  layer={layer}  model={model_name}")
            t0 = time.perf_counter()

            # IxG pooling for all four splits
            oracle_dir = ACT_DIR / model_name / f"layer_{layer}"
            oracle_dir.mkdir(parents=True, exist_ok=True)

            train_pos = _run_ixg_for_split(model, tokenizer, train_pos_texts, layer, device,
                                           f"{concept} train_pos L{layer}")
            train_neg = _run_ixg_for_split(model, tokenizer, train_neg_texts, layer, device,
                                           f"{concept} train_neg L{layer}")
            test_pos  = _run_ixg_for_split(model, tokenizer, test_pos_texts,  layer, device,
                                           f"{concept} test_pos  L{layer}")
            test_neg  = _run_ixg_for_split(model, tokenizer, test_neg_texts,  layer, device,
                                           f"{concept} test_neg  L{layer}")

            # Save oracle activations alongside the regular ones
            np.save(str(oracle_dir / f"{concept}_train_pos_oracle.npy"), train_pos)
            np.save(str(oracle_dir / f"{concept}_train_neg_oracle.npy"), train_neg)
            np.save(str(oracle_dir / f"{concept}_test_pos_oracle.npy"),  test_pos)
            np.save(str(oracle_dir / f"{concept}_test_neg_oracle.npy"),  test_neg)

            auroc = _diff_mean_auroc(train_pos, train_neg, test_pos, test_neg)
            results[concept][layer_key] = {"auroc": round(auroc, 5), "strategy": "G1_IxG"}

            elapsed = time.perf_counter() - t0
            log.info(f"  [oracle] {concept} layer={layer}  AUROC={auroc:.4f}  ({elapsed:.0f}s)")

            # Checkpoint after every (concept, layer) to tolerate preemption
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)

    del model, tokenizer
    free_gpu_memory(device)
    log.info(f"  [oracle] {model_name} done  GPU: {gpu_mem_str(device)}")
    return results


def _best_layer_oracle_auroc(model_results: dict, candidate_layers: list[int]) -> dict:
    """
    For each concept, pick the layer with the highest oracle AUROC.
    Returns dict: concept → {best_layer, auroc}.
    """
    summary = {}
    for concept, layer_dict in model_results.items():
        best_layer, best_auroc = None, -1.0
        for layer in candidate_layers:
            entry = layer_dict.get(str(layer), {})
            a = entry.get("auroc", float("nan"))
            if not (a != a) and a > best_auroc:  # nan check
                best_auroc, best_layer = a, layer
        if best_layer is not None:
            summary[concept] = {"best_layer": best_layer, "auroc": best_auroc}
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="PoolBench oracle evaluation: G1_IxG pooling (§45 / Table 7)"
    )
    parser.add_argument("--model", type=str, choices=list(MODEL_CONFIGS),
                        help="Single model to run")
    parser.add_argument("--all",   action="store_true",
                        help="Run all 3 models sequentially")
    parser.add_argument("--concept", type=str, default=None,
                        help="Restrict to a single concept (for testing)")
    parser.add_argument("--device",  type=str, default="auto",
                        help="Torch device, e.g. cuda:0, cpu, or 'auto'")
    parser.add_argument("--min_free_gb", type=float, default=20.0,
                        help="When --device auto: minimum free VRAM (GB) to accept a GPU")
    parser.add_argument("--force",   action="store_true",
                        help="Ignore existing checkpoints and recompute from scratch")
    parser.add_argument("--best_layer_only", action="store_true",
                        help="Only run the single best layer per model (reads from "
                             "results/auroc/{model}/best_layer_auroc.json). "
                             "Gives 3× speedup; falls back to all candidate layers if "
                             "no checkpoint is found.")
    args = parser.parse_args()

    # ── Device resolution ───────────────────────────────────────────────────
    if args.device == "auto":
        log.info("[device] --device auto: scanning GPUs...")
        try:
            args.device = find_free_gpu(min_free_gb=args.min_free_gb, logger=log)
            log.info(f"[device] AUTO-SELECTED: {args.device}")
        except RuntimeError as exc:
            log.error(f"[device] {exc}")
            raise SystemExit(1) from exc
    else:
        log.info(f"[device] Using: {args.device}")

    models_to_run = list(MODEL_CONFIGS) if args.all else ([args.model] if args.model else None)
    if models_to_run is None:
        parser.print_help()
        return

    all_summaries: dict[str, dict] = {}
    for model_name in models_to_run:
        log.info(f"\n{'='*60}\n  ORACLE MODEL: {model_name}\n{'='*60}")

        # ── --best_layer_only: restrict candidate_layers to the single best layer ──
        if args.best_layer_only:
            best_layer_path = RESULTS_DIR / "auroc" / model_name / "best_layer_auroc.json"
            if best_layer_path.exists():
                try:
                    with open(best_layer_path) as _f:
                        _bl = json.load(_f).get("best_layer")
                    if _bl is not None:
                        MODEL_CONFIGS[model_name]["candidate_layers"] = [int(_bl)]
                        log.info(f"  [best_layer_only] {model_name}: restricting to layer {_bl} "
                                 f"(read from {best_layer_path})")
                    else:
                        log.warning(f"  [best_layer_only] {model_name}: 'best_layer' key missing in "
                                    f"{best_layer_path} — using all candidate layers")
                except Exception as exc:
                    log.warning(f"  [best_layer_only] {model_name}: failed to read {best_layer_path} "
                                f"({exc}) — using all candidate layers")
            else:
                log.warning(f"  [best_layer_only] {model_name}: no checkpoint at {best_layer_path} "
                            f"— using all candidate layers {MODEL_CONFIGS[model_name]['candidate_layers']}")

        model_results = run_oracle_for_model(model_name, args)
        cfg = MODEL_CONFIGS[model_name]
        all_summaries[model_name] = _best_layer_oracle_auroc(model_results, cfg["candidate_layers"])

    # Print summary table
    log.info("\n\n=== G1_IxG Oracle AUROC Summary ===")
    for model_name, summary in all_summaries.items():
        log.info(f"\n  {model_name}:")
        for concept, info in sorted(summary.items()):
            log.info(f"    {concept:<25s}  best_layer={info['best_layer']}  AUROC={info['auroc']:.4f}")

    log.info("\nOracle evaluation complete.")


if __name__ == "__main__":
    main()
