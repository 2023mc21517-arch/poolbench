#!/usr/bin/env python3
"""
TEMPORARY SCRIPT — save_iti_scores.py
Run once to save ITI head scores to disk. Delete after use.

Usage (on Lightning, run in a separate terminal):
    python scripts/save_iti_scores.py --model mistral_7b --device cuda:0
    python scripts/save_iti_scores.py --model llama3_8b  --device cuda:0
    python scripts/save_iti_scores.py --model gemma2_9b  --device cuda:0

Output:
    results/iti_head_scores/{model_name}_iti_head_scores.json
    One entry per concept: {"best_head_auroc": float, "all_head_aurocs": [float, ...]}
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

BASE_DIR    = Path(__file__).parent.parent
RESULTS_DIR = Path(os.environ.get("POOLBENCH_RESULTS_DIR", BASE_DIR / "results"))
ACT_DIR     = RESULTS_DIR / "activations"
OUT_DIR     = RESULTS_DIR / "iti_head_scores"

MODEL_BEST_LAYERS = {
    "mistral_7b": 16,
    "llama3_8b":  31,
    "gemma2_9b":  28,
}

sys.path.insert(0, str(BASE_DIR))


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
    from poolbench.pooling_strategies import build_iti_concept_probes

    layer_act_dir = ACT_DIR / args.model / f"layer_{best_layer}"
    if not layer_act_dir.exists():
        print(f"[ERROR] Activation dir not found: {layer_act_dir}")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{args.model}_iti_head_scores.json"

    if out_path.exists():
        print(f"[SKIP] {out_path} already exists — delete it to recompute.")
        return

    print(f"Building ITI probes for {args.model} at layer {best_layer} ...")
    probes = build_iti_concept_probes(
        layer_act_dir, CONCEPTS, partition="train", device=args.device
    )

    results = {}
    for concept_name, probe in probes.items():
        scores = probe.head_scores.tolist()
        results[concept_name] = {
            "best_head_auroc": float(max(scores)),
            "best_head_index": int(probe.head_scores.argmax()),
            "all_head_aurocs": scores,
        }

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[DONE] Saved {len(results)} concepts → {out_path}")


if __name__ == "__main__":
    main()
