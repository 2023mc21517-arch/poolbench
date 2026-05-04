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
    SCP_DIR,
    step_icc,
    step_keyword_ablation,
    step_linearity,
    step_pool_and_auroc,
    step_prompted_baseline,
    step_scp,
    step_train_classifiers,
)

log = get_logger("poolbench.low_disk", log_file=Path(__file__).parent.parent / "results" / "run_low_disk.log")


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
                           skip_existing: bool) -> None:
    cfg = MODEL_CONFIGS[model_name]
    extract_activations_for_model(
        model_name=model_name,
        hf_id=cfg["hf_id"],
        concept_corpus_dir=CORPUS_DIR / concept,
        out_dir=ACT_DIR,
        candidate_layers=layers,
        batch_size=cfg["batch_size"],
        device=device,
        skip_existing=skip_existing,
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


def run_low_disk(model_name: str, device: str, skip_scp: bool, force: bool) -> None:
    cfg = MODEL_CONFIGS[model_name]
    candidate_layers = list(cfg["candidate_layers"])

    if force:
        reset_model_outputs(model_name)

    log.info("=" * 60)
    log.info(f"LOW-DISK PoolBench run: model={model_name} device={device}")
    log.info(f"candidate_layers={candidate_layers} GPU={gpu_mem_str(device)}")
    log.info("=" * 60)

    # Pass A: extract one concept across all candidate layers, compute D1 AUROC,
    # save JSON, then delete that concept's activations.
    for concept in CONCEPT_NAMES:
        log.info(f"\n=== Low-disk D1 pass: {concept} ===")
        extract_concept_layers(model_name, concept, candidate_layers, device, skip_existing=not force)
        step_pool_and_auroc(model_name, concept_filter=concept, skip_existing=not force)
        cleanup_concept_activations(model_name, candidate_layers, {concept})
        free_gpu_memory(device)

    best_layer = best_layer_for_model(model_name)
    log.info(f"\n[low-disk] Shared best layer selected from saved D1 results: {best_layer}")
    step_icc(model_name, skip_existing=not force)

    if not skip_scp:
        step_train_classifiers(device=device, force_retrain=force)

    # Pass B: re-extract only the shared best layer for each target concept.
    # D3 needs neighbour steering vectors, so temporarily extract target + LD/LC
    # neighbours at that one layer, then delete them after the target finishes.
    for concept in CONCEPT_NAMES:
        needed = {concept}
        if not skip_scp:
            neighbours = NEIGHBOUR_PAIRS.get(concept, {})
            needed.update(neighbours.values())
        log.info(f"\n=== Low-disk metric pass: target={concept} temp_activations={sorted(needed)} ===")
        for needed_concept in sorted(needed):
            extract_concept_layers(model_name, needed_concept, [best_layer], device, skip_existing=True)

        step_linearity(model_name, best_layer, concept_filter=concept, skip_existing=not force)
        step_keyword_ablation(model_name, best_layer, device, concept_filter=concept, skip_existing=not force)

        if not skip_scp:
            step_scp(model_name, best_layer, device, concept_filter=concept, skip_existing=not force)
            step_prompted_baseline(model_name, device, concept_filter=concept, skip_existing=not force)
            step_disentanglement(model_name, best_layer, device, concept_filter=concept, skip_existing=not force)

        cleanup_concept_activations(model_name, [best_layer], needed)
        free_gpu_memory(device)

    log.info(f"\n✓ Low-disk run complete for {model_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Low-disk PoolBench per-model runner")
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--min_free_gb", type=float, default=35.0)
    parser.add_argument("--skip_scp", action="store_true", help="Skip D2 SCP, prompted baseline, and D3")
    parser.add_argument("--force", action="store_true", help="Delete old model outputs before running")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = find_free_gpu(min_free_gb=args.min_free_gb, logger=log)
    run_low_disk(args.model, device, skip_scp=args.skip_scp, force=args.force)


if __name__ == "__main__":
    main()
