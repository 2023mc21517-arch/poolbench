"""
examples/evaluate.py
~~~~~~~~~~~~~~~~~~~~
No-GPU evaluation harness for community strategy submissions.

Uses pre-extracted PoolBench activations from HuggingFace
(agarwalayushi/poolbench) so no model GPU is required.

Usage
-----
    python examples/evaluate.py \\
        --strategy_file examples/my_strategy.py \\
        --strategy_fn   pool_my_strategy \\
        --model         llama3_8b \\
        --output        my_results.json

Prerequisites
-------------
    pip install poolbench[dev] huggingface_hub
    huggingface-cli login   # only needed if activations repo is private
"""

from __future__ import annotations
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def load_strategy(strategy_file: str, strategy_fn: str):
    """Dynamically import a strategy function from a .py file."""
    spec = importlib.util.spec_from_file_location("custom_strategy", strategy_file)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, strategy_fn, None)
    if fn is None:
        raise AttributeError(f"Function '{strategy_fn}' not found in {strategy_file}")
    return fn


def download_activations(model: str, concept: str, cache_dir: Path) -> tuple:
    """Download pre-extracted .npy activation files for one concept/model."""
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    repo_id = "agarwalayushi/poolbench"
    pos_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"activations/{model}/{concept}_pos.npy",
        repo_type="dataset",
        local_dir=str(cache_dir),
    )
    neg_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"activations/{model}/{concept}_neg.npy",
        repo_type="dataset",
        local_dir=str(cache_dir),
    )
    return pos_path, neg_path


def pool_corpus(acts: np.ndarray, pool_fn) -> np.ndarray:
    """Apply pool_fn to every item in a numpy object array of activation dicts."""
    pooled = []
    for item in acts:
        h = item["hidden"]
        try:
            vec = pool_fn(
                h,
                text=item.get("text", ""),
                token_ids=item.get("token_ids", []),
                offset_mapping=item.get("offset_mapping", []),
                attn_weights=item.get("attn_weights"),
            )
        except TypeError:
            # Strategy doesn't accept **kwargs — call with h only
            vec = pool_fn(h)
        pooled.append(vec)
    return np.stack(pooled).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a custom pooling strategy")
    parser.add_argument("--strategy_file", required=True)
    parser.add_argument("--strategy_fn",   required=True)
    parser.add_argument("--model",         default="llama3_8b",
                        choices=["llama3_8b", "gemma2_9b", "mistral_7b"])
    parser.add_argument("--output",        default="community_results.json")
    parser.add_argument("--cache_dir",     default=".hf_cache")
    args = parser.parse_args()

    pool_fn   = load_strategy(args.strategy_file, args.strategy_fn)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(exist_ok=True)

    from poolbench.data.concepts import CONCEPT_NAMES          # noqa: PLC0415
    from poolbench.evaluation.probe import compute_auroc_for_strategy  # noqa: PLC0415

    results = {}
    for concept in CONCEPT_NAMES:
        print(f"  [{concept}] downloading activations ...", end=" ", flush=True)
        try:
            pos_path, neg_path = download_activations(args.model, concept, cache_dir)
        except Exception as exc:
            print(f"SKIP ({exc})")
            continue

        pos_acts = np.load(pos_path, allow_pickle=True)
        neg_acts = np.load(neg_path, allow_pickle=True)

        pos_pooled = pool_corpus(pos_acts, pool_fn)
        neg_pooled = pool_corpus(neg_acts, pool_fn)

        res = compute_auroc_for_strategy(pos_pooled, neg_pooled)
        results[concept] = res
        print(f"AUROC={res['auroc']:.3f} CI=[{res['ci_low']:.3f},{res['ci_high']:.3f}]")

    mean_auroc = float(np.mean([r["auroc"] for r in results.values()]))
    print(f"\nMean AUROC across {len(results)} concepts: {mean_auroc:.4f}")

    output = {
        "model":      args.model,
        "strategy":   args.strategy_fn,
        "mean_auroc": mean_auroc,
        "per_concept": results,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved → {args.output}")


if __name__ == "__main__":
    main()
