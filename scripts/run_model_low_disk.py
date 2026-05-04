"""
Low-disk per-model PoolBench runner.

This runner keeps disk usage low by extracting activations for one concept at a
 time, immediately pooling/scoring the concept, saving JSON results, and deleting
that concept's activation .npy files before moving on.

It is intended for machines with < 1 TB local storage. The normal run_model.py
is faster, but stores all concept activations at once.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from poolbench.concepts import CONCEPT_NAMES
from poolbench.extract_activations import extract_activations_for_model
from poolbench.logger import find_free_gpu, free_gpu_memory, get_logger, gpu_mem_str
from poolbench.evaluation.disentanglement import NEIGHBOUR_PAIRS

from scripts.run_model import (  # noqa: E402
    ACT_DIR,
    ABLATION_DIR,
    AUROC_DIR,
    CLASSIFIERS_DIR,
    CORPUS_DIR,
    D3_DIR,
    ICC_DIR,
    LINEARITY_DIR,
    MODEL_CONFIGS,
    RANKED_STRATEGIES,
    RESULTS_DIR,
    SCP_DIR,
    LAYER_SELECTION_REPRESENTATIVE_CONCEPTS,
    LAYER_SELECTION_STRATEGIES,
    _select_methodology_layer,
    step_icc,
    step_keyword_ablation,
    step_linearity,
    step_pool_and_auroc,
    step_prompted_baseline,
    step_scp,
    step_disentanglement,
    step_train_classifiers,
)

log = get_logger("poolbench.low_disk", log_file=RESULTS_DIR / "run_low_disk.log")


def _remove_file(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def cleanup_concept_activations(model_name: str, layers: list[int], concepts: set[str]) -> None:
    """Delete activation files for selected concepts/layers, preserving JSON results."""
    for layer_idx in layers:
        layer_dir = ACT_DIR / model_name / f"layer_{layer_idx}"
        for concept in concepts:
            for partition in ("train", "test"):
                for split in ("pos", "neg"):
                    _remove_file(layer_dir / f"{concept}_{partition}_{split}.npy")
        try:
            if layer_dir.exists() and not any(layer_dir.iterdir()):
                layer_dir.rmdir()
        except OSError:
            pass
    log.info(f"  [cleanup] removed activations for {sorted(concepts)} layers={layers}")


def extract_concept_layers(model_name: str, concept: str, layers: list[int], device: str,
                           skip_existing: bool, batch_size: int | None = None,
                           activation_save_dtype: str = "float16") -> None:
    cfg = MODEL_CONFIGS[model_name]
    extract_activations_for_model(
        model_name=model_name,
        hf_id=cfg["hf_id"],
        concept_corpus_dir=CORPUS_DIR / concept,
        out_dir=ACT_DIR,
        candidate_layers=layers,
        batch_size=batch_size or cfg["batch_size"],
        device=device,
        skip_existing=skip_existing,
        activation_save_dtype=activation_save_dtype,
    )


def best_layer_for_model(model_name: str) -> int:
    path = AUROC_DIR / model_name / "best_layer_auroc.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing best-layer file after D1 pass: {path}")
    with open(path) as f:
        data = json.load(f)
    best_layer = data.get("best_layer")
    if best_layer is None:
        raise RuntimeError(f"No best_layer in {path}")
    return int(best_layer)


def pool_one_concept_layer(model_name: str, concept: str, layer_idx: int, force: bool) -> None:
    """Run Step 2 for exactly one concept and one layer, preserving low disk use."""
    cfg = MODEL_CONFIGS[model_name]
    original_layers = list(cfg["candidate_layers"])
    cfg["candidate_layers"] = [layer_idx]
    try:
        step_pool_and_auroc(model_name, concept_filter=concept, skip_existing=not force)
    finally:
        cfg["candidate_layers"] = original_layers


def rebuild_best_layer_summary(model_name: str, candidate_layers: list[int], concepts: list[str] | None = None) -> int:
    """Rebuild best_layer_auroc.json from completed per-layer AUROC files."""
    concepts_to_check = list(concepts or CONCEPT_NAMES)
    per_layer: dict[int, dict] = {}
    for layer_idx in candidate_layers:
        path = AUROC_DIR / model_name / f"layer_{layer_idx}" / f"{model_name}_auroc_results.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing per-layer AUROC file: {path}")
        with open(path) as f:
            layer_results = json.load(f)
        missing = [
            f"{concept}_{strategy}"
            for concept in concepts_to_check
            for strategy in RANKED_STRATEGIES
            if f"{concept}_{strategy}" not in layer_results
        ]
        if missing:
            raise RuntimeError(f"Layer {layer_idx} AUROC file incomplete; missing examples: {missing[:10]}")
        per_layer[layer_idx] = layer_results

    best_layer = _select_methodology_layer(per_layer, candidate_layers, concepts_to_check)
    out_path = AUROC_DIR / model_name / "best_layer_auroc.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "best_layer": best_layer,
            "per_layer": {str(k): v for k, v in per_layer.items()},
            "layer_selection_method": "methodology_representative_mean",
            "layer_selection_concepts": LAYER_SELECTION_REPRESENTATIVE_CONCEPTS,
            "layer_selection_strategies": LAYER_SELECTION_STRATEGIES,
        }, f, indent=2)
    log.info(f"[low-disk] rebuilt best-layer summary → {out_path}")
    return int(best_layer)


def reset_model_outputs(model_name: str) -> None:
    """Remove model-specific generated outputs for a clean low-disk rerun."""
    paths = [
        ACT_DIR / model_name,
        AUROC_DIR / model_name,
        LINEARITY_DIR / f"{model_name}_linearity.json",
        ICC_DIR / f"{model_name}_icc.json",
        ABLATION_DIR / f"{model_name}_keyword_ablation.json",
        SCP_DIR / f"{model_name}_scp.json",
        SCP_DIR / f"{model_name}_prompted_baseline.json",
        D3_DIR / f"{model_name}_d3.json",
        D3_DIR / f"{model_name}_d3_rep.json",
    ]
    for path in paths:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            _remove_file(path)
    log.info(f"[reset] removed old outputs for {model_name}")


def run_low_disk(model_name: str, device: str, skip_scp: bool, force: bool,
                 batch_size: int | None = None,
                 stream_granularity: str = "layer",
                 concepts: list[str] | None = None,
                 stage: str = "all",
                 train_classifiers: bool = True,
                 activation_save_dtype: str = "float16") -> None:
    cfg = MODEL_CONFIGS[model_name]
    candidate_layers = list(cfg["candidate_layers"])
    effective_batch_size = batch_size or cfg["batch_size"]
    concepts_to_run = list(concepts or CONCEPT_NAMES)
    unknown = sorted(set(concepts_to_run) - set(CONCEPT_NAMES))
    if unknown:
        raise ValueError(f"Unknown concept(s): {unknown}")
    if not concepts_to_run:
        raise ValueError("No concepts selected")

    if force and stage != "metrics":
        reset_model_outputs(model_name)

    log.info("=" * 60)
    log.info(f"LOW-DISK PoolBench run: model={model_name} device={device}")
    log.info(f"candidate_layers={candidate_layers} batch_size={effective_batch_size} "
             f"stream_granularity={stream_granularity} stage={stage} "
             f"activation_save_dtype={activation_save_dtype} concepts={concepts_to_run} "
             f"GPU={gpu_mem_str(device)}")
    log.info("=" * 60)

    # Pass A: extract one concept at one candidate layer, compute that layer's
    # D1 AUROC, save JSON, then delete those activation files before extracting
    # the next layer. This keeps peak storage low even for concepts with long text.
    if stage in {"all", "d1"}:
        for concept in concepts_to_run:
            log.info(f"\n=== Low-disk D1 pass: {concept} ===")
            if stream_granularity == "concept":
                log.info(f"\n--- Low-disk D1 concept pass: concept={concept} layers={candidate_layers} ---")
                extract_concept_layers(model_name, concept, candidate_layers, device,
                                       skip_existing=not force,
                                       batch_size=effective_batch_size,
                                       activation_save_dtype=activation_save_dtype)
                step_pool_and_auroc(model_name, concept_filter=concept, skip_existing=not force)
                cleanup_concept_activations(model_name, candidate_layers, {concept})
                free_gpu_memory(device)
            else:
                for layer_idx in candidate_layers:
                    log.info(f"\n--- Low-disk D1 layer pass: concept={concept} layer={layer_idx} ---")
                    extract_concept_layers(model_name, concept, [layer_idx], device,
                                           skip_existing=not force,
                                           batch_size=effective_batch_size,
                                           activation_save_dtype=activation_save_dtype)
                    pool_one_concept_layer(model_name, concept, layer_idx, force=force)
                    cleanup_concept_activations(model_name, [layer_idx], {concept})
                    free_gpu_memory(device)

        if stage == "d1" and set(concepts_to_run) != set(CONCEPT_NAMES):
            log.info("\n✓ Low-disk D1 shard complete; merge all shards before metrics")
            return

        best_layer = rebuild_best_layer_summary(model_name, candidate_layers, concepts_to_run)
        log.info(f"\n[low-disk] Shared best layer selected from saved D1 results: {best_layer}")
        if set(concepts_to_run) == set(CONCEPT_NAMES):
            step_icc(model_name, skip_existing=not force)
    else:
        best_layer = best_layer_for_model(model_name)

    if stage == "d1":
        log.info(f"\n✓ Low-disk D1 run complete for {model_name}")
        return

    if not skip_scp and train_classifiers:
        step_train_classifiers(device=device, force_retrain=force)

    # Pass B: re-extract only the shared best layer for each target concept.
    # D3 needs neighbour steering vectors, so temporarily extract target + LD/LC
    # neighbours at that one layer, then delete them after the target finishes.
    for concept in concepts_to_run:
        needed = {concept}
        if not skip_scp:
            neighbours = NEIGHBOUR_PAIRS.get(concept, {})
            needed.update(neighbours.values())
        log.info(f"\n=== Low-disk metric pass: target={concept} temp_activations={sorted(needed)} ===")
        for needed_concept in sorted(needed):
            extract_concept_layers(model_name, needed_concept, [best_layer], device,
                                   skip_existing=True,
                                   batch_size=effective_batch_size,
                                   activation_save_dtype=activation_save_dtype)

        step_linearity(model_name, best_layer, concept_filter=concept, skip_existing=not force)
        step_keyword_ablation(model_name, best_layer, device, concept_filter=concept, skip_existing=not force)

        if not skip_scp:
            step_scp(model_name, best_layer, device, concept_filter=concept, skip_existing=not force)
            step_prompted_baseline(model_name, device, concept_filter=concept, skip_existing=not force)
            step_disentanglement(model_name, best_layer, device, concept_filter=concept, skip_existing=not force)

        cleanup_concept_activations(model_name, [best_layer], needed)
        free_gpu_memory(device)

    log.info(f"\n✓ Low-disk run complete for {model_name}")


def _parse_concepts(raw: str | None) -> list[str] | None:
    if raw is None or not raw.strip():
        return None
    return [part.strip() for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Low-disk PoolBench per-model runner")
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--min_free_gb", type=float, default=35.0)
    parser.add_argument("--skip_scp", action="store_true", help="Skip D2 SCP, prompted baseline, and D3")
    parser.add_argument("--force", action="store_true", help="Delete old model outputs before running")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override model extraction batch size; use 1-2 on A100 40GB")
    parser.add_argument("--stream_granularity", choices=["layer", "concept"], default="layer",
                        help="'layer' minimizes disk; 'concept' is faster but stores all candidate layers for one concept")
    parser.add_argument("--concepts", default=None,
                        help="Comma-separated concept subset for sharded runs")
    parser.add_argument("--stage", choices=["all", "d1", "metrics"], default="all",
                        help="Run full low-disk pipeline, only D1, or only post-D1 metrics")
    parser.add_argument("--skip_train_classifiers", action="store_true",
                        help="For metric shards: reuse an existing/symlinked classifier directory")
    parser.add_argument("--activation_save_dtype", choices=["float32", "float16"], default="float16",
                        help="Activation storage dtype; float32 preserves full runner fidelity, float16 saves disk")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = find_free_gpu(min_free_gb=args.min_free_gb, logger=log)
        if device == "cpu":
            raise RuntimeError(
                "CUDA is not available in this Python environment. Refusing to run an 8B model on CPU. "
                "Install a PyTorch wheel compatible with the cluster CUDA driver, then rerun."
            )
    elif device == "cpu" and args.model not in {"bert_base_uncased"}:
        raise RuntimeError("Refusing to run a large PoolBench model on CPU; choose a CUDA device.")
    run_low_disk(args.model, device, skip_scp=args.skip_scp, force=args.force,
                 batch_size=args.batch_size,
                 stream_granularity=args.stream_granularity,
                 concepts=_parse_concepts(args.concepts),
                 stage=args.stage,
                 train_classifiers=not args.skip_train_classifiers,
                 activation_save_dtype=args.activation_save_dtype)


if __name__ == "__main__":
    main()
