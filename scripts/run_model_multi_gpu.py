"""
Shard-safe multi-GPU PoolBench runner for one model.

This orchestrates the low-disk runner in isolated per-shard result directories so
parallel workers do not race while writing JSON checkpoints. It preserves the
research protocol: same corpora, same candidate layers, same pooling strategies,
same greedy SCP/D3 generation, and the same global methodology layer selection.

Typical 8×A100 run:
    python scripts/run_model_multi_gpu.py --model llama3_8b --gpus 0,1,2,3,4,5,6,7 \
        --batch_size 2 --stream_granularity concept --force
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from poolbench.concepts import CONCEPT_NAMES

from scripts.run_model import (  # noqa: E402
    ABLATION_DIR,
    AUROC_DIR,
    BASE_DIR,
    CLASSIFIERS_DIR,
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
    step_train_classifiers,
)


def _chunks_round_robin(items: list[str], n: int) -> list[list[str]]:
    chunks = [[] for _ in range(n)]
    for idx, item in enumerate(items):
        chunks[idx % n].append(item)
    return [chunk for chunk in chunks if chunk]


def _load_json(path: Path) -> dict:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected JSON object at {path}")
    return data


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _shard_dir(run_id: str, shard_idx: int) -> Path:
    return RESULTS_DIR / "shards" / run_id / f"shard_{shard_idx}"


def _launch_stage(
    *,
    model: str,
    gpus: list[str],
    concept_shards: list[list[str]],
    run_id: str,
    stage: str,
    batch_size: int | None,
    stream_granularity: str,
    skip_scp: bool,
    skip_train_classifiers: bool,
    force: bool,
    activation_save_dtype: str,
) -> None:
    logs_dir = RESULTS_DIR / "shards" / run_id / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    procs: list[tuple[int, subprocess.Popen, object, Path]] = []

    for shard_idx, concepts in enumerate(concept_shards):
        gpu = gpus[shard_idx % len(gpus)]
        shard_results = _shard_dir(run_id, shard_idx)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["POOLBENCH_RESULTS_DIR"] = str(shard_results)
        env["PYTHONPATH"] = str(BASE_DIR) + os.pathsep + env.get("PYTHONPATH", "")

        cmd = [
            sys.executable,
            str(BASE_DIR / "scripts" / "run_model_low_disk.py"),
            "--model", model,
            "--device", "cuda:0",
            "--stage", stage,
            "--concepts", ",".join(concepts),
            "--stream_granularity", stream_granularity,
            "--activation_save_dtype", activation_save_dtype,
        ]
        if batch_size is not None:
            cmd += ["--batch_size", str(batch_size)]
        if skip_scp:
            cmd.append("--skip_scp")
        if skip_train_classifiers:
            cmd.append("--skip_train_classifiers")
        if force:
            cmd.append("--force")

        log_path = logs_dir / f"{stage}_shard_{shard_idx}_gpu_{gpu}.log"
        log_f = open(log_path, "w")
        print(f"[multi-gpu] launching {stage} shard={shard_idx} gpu={gpu} concepts={concepts} log={log_path}")
        procs.append((shard_idx, subprocess.Popen(cmd, cwd=BASE_DIR, env=env, stdout=log_f, stderr=subprocess.STDOUT), log_f, log_path))

    failed: list[str] = []
    for shard_idx, proc, log_f, log_path in procs:
        code = proc.wait()
        log_f.close()
        if code != 0:
            failed.append(f"shard {shard_idx} exited {code}; see {log_path}")
    if failed:
        raise RuntimeError("; ".join(failed))


def merge_d1(model: str, run_id: str, n_shards: int) -> int:
    cfg = MODEL_CONFIGS[model]
    candidate_layers = list(cfg["candidate_layers"])
    per_layer: dict[int, dict] = {layer: {} for layer in candidate_layers}

    for shard_idx in range(n_shards):
        shard_results = _shard_dir(run_id, shard_idx)
        for layer in candidate_layers:
            path = shard_results / "auroc" / model / f"layer_{layer}" / f"{model}_auroc_results.json"
            if not path.exists():
                raise FileNotFoundError(f"Missing shard AUROC file: {path}")
            shard_layer = _load_json(path)
            overlap = set(per_layer[layer]) & set(shard_layer)
            if overlap:
                raise RuntimeError(f"Duplicate AUROC keys while merging layer {layer}: {sorted(overlap)[:10]}")
            per_layer[layer].update(shard_layer)

    missing = [
        f"{concept}_{strategy}"
        for layer in candidate_layers
        for concept in CONCEPT_NAMES
        for strategy in RANKED_STRATEGIES
        if f"{concept}_{strategy}" not in per_layer[layer]
    ]
    if missing:
        raise RuntimeError(f"Merged D1 is incomplete; missing examples: {missing[:20]}")

    for layer, layer_results in per_layer.items():
        _write_json(AUROC_DIR / model / f"layer_{layer}" / f"{model}_auroc_results.json", layer_results)

    best_layer = _select_methodology_layer(per_layer, candidate_layers, list(CONCEPT_NAMES))
    _write_json(AUROC_DIR / model / "best_layer_auroc.json", {
        "best_layer": best_layer,
        "per_layer": {str(k): v for k, v in per_layer.items()},
        "layer_selection_method": "methodology_representative_mean",
        "layer_selection_concepts": LAYER_SELECTION_REPRESENTATIVE_CONCEPTS,
        "layer_selection_strategies": LAYER_SELECTION_STRATEGIES,
    })
    print(f"[multi-gpu] merged D1; best_layer={best_layer}")
    step_icc(model, skip_existing=False)
    return int(best_layer)


def _prepare_metric_shards(model: str, run_id: str, n_shards: int) -> None:
    best_layer_path = AUROC_DIR / model / "best_layer_auroc.json"
    if not best_layer_path.exists():
        raise FileNotFoundError(f"Missing final best-layer file: {best_layer_path}")
    for shard_idx in range(n_shards):
        shard_results = _shard_dir(run_id, shard_idx)
        dst = shard_results / "auroc" / model / "best_layer_auroc.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_layer_path, dst)

        shard_cls = shard_results / "bert_classifiers"
        if shard_cls.exists() or shard_cls.is_symlink():
            if shard_cls.is_dir() and not shard_cls.is_symlink():
                shutil.rmtree(shard_cls)
            else:
                shard_cls.unlink()
        if CLASSIFIERS_DIR.exists():
            shard_cls.parent.mkdir(parents=True, exist_ok=True)
            try:
                shard_cls.symlink_to(CLASSIFIERS_DIR, target_is_directory=True)
            except OSError:
                shutil.copytree(CLASSIFIERS_DIR, shard_cls)


def _merge_concept_json(model: str, run_id: str, n_shards: int, rel_path: str, final_path: Path, metadata_key: str | None = None) -> None:
    merged: dict = {}
    for shard_idx in range(n_shards):
        path = _shard_dir(run_id, shard_idx) / rel_path.format(model=model)
        if not path.exists():
            continue
        data = _load_json(path)
        if metadata_key and metadata_key in data:
            merged[metadata_key] = data[metadata_key]
            data = {k: v for k, v in data.items() if k != metadata_key}
        overlap = set(merged) & set(data)
        if overlap:
            raise RuntimeError(f"Duplicate keys while merging {final_path}: {sorted(overlap)}")
        merged.update(data)
    if merged:
        _write_json(final_path, merged)
        print(f"[multi-gpu] merged {final_path}")


def merge_metrics(model: str, run_id: str, n_shards: int, skip_scp: bool) -> None:
    _merge_concept_json(model, run_id, n_shards, "linearity/{model}_linearity.json", LINEARITY_DIR / f"{model}_linearity.json")
    _merge_concept_json(model, run_id, n_shards, "ablation/{model}_keyword_ablation.json", ABLATION_DIR / f"{model}_keyword_ablation.json")
    if not skip_scp:
        _merge_concept_json(model, run_id, n_shards, "scp/{model}_scp.json", SCP_DIR / f"{model}_scp.json")
        _merge_concept_json(model, run_id, n_shards, "scp/{model}_prompted_baseline.json", SCP_DIR / f"{model}_prompted_baseline.json", metadata_key="_metadata")
        _merge_concept_json(model, run_id, n_shards, "disentanglement/{model}_d3.json", D3_DIR / f"{model}_d3.json")
        _merge_concept_json(model, run_id, n_shards, "disentanglement/{model}_d3_rep.json", D3_DIR / f"{model}_d3_rep.json")


def reset_final_outputs(model: str, run_id: str) -> None:
    for path in [
        AUROC_DIR / model,
        LINEARITY_DIR / f"{model}_linearity.json",
        ICC_DIR / f"{model}_icc.json",
        ABLATION_DIR / f"{model}_keyword_ablation.json",
        SCP_DIR / f"{model}_scp.json",
        SCP_DIR / f"{model}_prompted_baseline.json",
        D3_DIR / f"{model}_d3.json",
        D3_DIR / f"{model}_d3_rep.json",
        RESULTS_DIR / "shards" / run_id,
    ]:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists() or path.is_symlink():
            path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Shard-safe multi-GPU PoolBench runner")
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS))
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7", help="Comma-separated physical GPU ids")
    parser.add_argument("--batch_size", type=int, default=None, help="Extraction batch size per GPU; use 1-2 on A100 40GB")
    parser.add_argument("--stream_granularity", choices=["layer", "concept"], default="concept")
    parser.add_argument("--skip_scp", action="store_true", help="Skip D2 SCP, prompted baseline, and D3")
    parser.add_argument("--activation_save_dtype", choices=["float32", "float16"], default="float32",
                        help="Activation storage dtype; default float32 avoids a precision tradeoff")
    parser.add_argument("--force", action="store_true", help="Delete this run's final and shard outputs first")
    parser.add_argument("--run_id", default=None, help="Shard run id; defaults to model name")
    parser.add_argument("--stage", choices=["all", "d1", "merge_d1", "classifiers", "metrics", "merge_metrics"], default="all")
    args = parser.parse_args()

    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise ValueError("At least one GPU id is required")
    concept_shards = _chunks_round_robin(list(CONCEPT_NAMES), len(gpus))
    run_id = args.run_id or args.model

    if args.force:
        reset_final_outputs(args.model, run_id)

    if args.stage in {"all", "d1"}:
        _launch_stage(
            model=args.model,
            gpus=gpus,
            concept_shards=concept_shards,
            run_id=run_id,
            stage="d1",
            batch_size=args.batch_size,
            stream_granularity=args.stream_granularity,
            skip_scp=args.skip_scp,
            skip_train_classifiers=True,
            force=args.force,
            activation_save_dtype=args.activation_save_dtype,
        )
    if args.stage == "d1":
        return

    if args.stage in {"all", "merge_d1"}:
        merge_d1(args.model, run_id, len(concept_shards))
    if args.stage == "merge_d1":
        return

    if args.stage in {"all", "classifiers"} and not args.skip_scp:
        step_train_classifiers(device="cuda:0", force_retrain=args.force)
    if args.stage == "classifiers":
        return

    if args.stage in {"all", "metrics"}:
        _prepare_metric_shards(args.model, run_id, len(concept_shards))
        _launch_stage(
            model=args.model,
            gpus=gpus,
            concept_shards=concept_shards,
            run_id=run_id,
            stage="metrics",
            batch_size=args.batch_size,
            stream_granularity=args.stream_granularity,
            skip_scp=args.skip_scp,
            skip_train_classifiers=True,
            force=args.force,
            activation_save_dtype=args.activation_save_dtype,
        )
    if args.stage == "metrics":
        return

    if args.stage in {"all", "merge_metrics"}:
        merge_metrics(args.model, run_id, len(concept_shards), skip_scp=args.skip_scp)

    print(f"[multi-gpu] complete for {args.model}; final outputs are under {RESULTS_DIR}")


if __name__ == "__main__":
    main()
