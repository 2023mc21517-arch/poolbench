#!/usr/bin/env python3
"""
TEMPORARY SCRIPT — save_steering_vectors.py
Computes and saves DiffMean steering vectors for all concepts/strategies to .npy files.
Delete after uploading to HF.

Usage (on Lightning, run in a separate terminal):
    python scripts/save_steering_vectors.py --model mistral_7b --device cuda:0
    python scripts/save_steering_vectors.py --model llama3_8b  --device cuda:0
    python scripts/save_steering_vectors.py --model gemma2_9b  --device cuda:0

Output:
    results/steering_vectors/{model_name}/layer_{N}/{concept}_{strategy}.npy
    Each file is a normalised float32 array of shape (d_model,).

To upload after all models are done:
    python scripts/upload_to_hf.py --token hf_xxx --user nips234678 --only steering-vectors
    (after adding steering-vectors support to upload_to_hf.py)
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import numpy as np

BASE_DIR    = Path(__file__).parent.parent
RESULTS_DIR = Path(os.environ.get("POOLBENCH_RESULTS_DIR", BASE_DIR / "results"))
ACT_DIR     = RESULTS_DIR / "activations"
OUT_DIR     = RESULTS_DIR / "steering_vectors"

MODEL_BEST_LAYERS = {
    "mistral_7b": 16,
    "llama3_8b":  None,   # update once D1 finishes
    "gemma2_9b":  28,
}

sys.path.insert(0, str(BASE_DIR))


def _compute_sv(act_dir: Path, model_name: str, layer: int,
                concept_name: str, strategy_id: str,
                unigram_probs: dict | None,
                concept_probe) -> np.ndarray | None:
    """Import and call the same _compute_steering_vector used by D3."""
    from poolbench.evaluation.disentanglement import _compute_steering_vector
    return _compute_steering_vector(
        act_dir, model_name, layer, concept_name, strategy_id,
        unigram_probs=unigram_probs,
        concept_probe=concept_probe,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  required=True, choices=list(MODEL_BEST_LAYERS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layer",  type=int, default=None,
                        help="Override best layer (required if not hardcoded above)")
    args = parser.parse_args()

    best_layer = args.layer or MODEL_BEST_LAYERS[args.model]
    if best_layer is None:
        print(f"[ERROR] Best layer not known for {args.model}. Pass --layer N.")
        sys.exit(1)

    from poolbench.data.concepts import CONCEPTS
    from poolbench.pooling_strategies import (
        STRATEGY_REGISTRY,
        build_iti_concept_probes,
        build_unigram_probs_from_activations,
    )

    layer_act_dir = ACT_DIR / args.model / f"layer_{best_layer}"
    if not layer_act_dir.exists():
        print(f"[ERROR] Activation dir not found: {layer_act_dir}")
        sys.exit(1)

    out_layer_dir = OUT_DIR / args.model / f"layer_{best_layer}"
    out_layer_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building unigram probs for {args.model} ...")
    unigram_probs = build_unigram_probs_from_activations(layer_act_dir)

    print(f"Building ITI probes for {args.model} ...")
    concept_probes = build_iti_concept_probes(
        layer_act_dir, CONCEPTS, partition="train", device=args.device
    )

    strategy_ids = list(STRATEGY_REGISTRY.keys())
    total = len(CONCEPTS) * len(strategy_ids)
    done = skipped = 0

    for concept_name in CONCEPTS:
        for strategy_id in strategy_ids:
            out_file = out_layer_dir / f"{concept_name}__{strategy_id}.npy"
            if out_file.exists():
                done += 1
                continue

            sv = _compute_sv(
                ACT_DIR, args.model, best_layer,
                concept_name, strategy_id,
                unigram_probs=unigram_probs,
                concept_probe=concept_probes.get(concept_name),
            )

            if sv is None:
                print(f"  [SKIP] {concept_name}/{strategy_id} — zero-norm or missing activations")
                skipped += 1
                # Write a sentinel so we don't recompute on re-run
                np.save(out_file.with_suffix(".skip"), np.array([]))
                continue

            np.save(out_file, sv)
            done += 1
            print(f"  [{done}/{total}] saved {concept_name}/{strategy_id}  shape={sv.shape}")

    print(f"\n[DONE] {done} saved, {skipped} skipped → {out_layer_dir}")
    print(f"  Upload with: python scripts/upload_to_hf.py --token hf_xxx "
          f"--user nips234678 --only steering-vectors  (after adding that --only option)")


if __name__ == "__main__":
    main()
