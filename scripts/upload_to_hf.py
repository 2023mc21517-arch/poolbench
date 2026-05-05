#!/usr/bin/env python3
"""
upload_to_hf.py  — Upload PoolBench camera-ready artifacts to HuggingFace.

Usage:
    python scripts/upload_to_hf.py --token hf_xxx --org your-anon-org

What it uploads:
    poolbench-corpus          data/corpora/          (17 concepts × 4 JSONL splits)
    poolbench-activations     results/activations/   (~390 GB, per-model .npy files)
    poolbench-bert-scorers    results/bert_classifiers/d2/  (17 BERT classifiers)
    poolbench-steered-outputs results/steered_outputs/      (15 non-sensitive concepts)

Steering vectors are NOT uploaded here — run scripts/upload_steering_vectors.py
after you have saved them via the separate compute step.

Prerequisites on HuggingFace (do this once manually before running):
    Create the following repos under your org as PRIVATE:
        {org}/poolbench-corpus          (Dataset)
        {org}/poolbench-activations     (Dataset)
        {org}/poolbench-bert-scorers    (Model)
        {org}/poolbench-steered-outputs (Dataset)

Options:
    --token          HuggingFace write token (required)
    --org            HuggingFace org or username to upload to (required)
    --skip           Comma-separated list of artifact names to skip.
                     Valid names: corpus, activations, bert-scorers, steered-outputs
                     e.g. --skip activations   (to skip the big 390 GB upload)
    --repo-base-dir  Root of poolbench repo (default: directory of this script's parent)
    --commit-msg     Commit message for all uploads (default: timestamped)
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path


# ── Helpers ───────────────────────────────────────────────────────────────────

def _login(token: str) -> None:
    from huggingface_hub import login  # noqa: PLC0415
    login(token=token)


def _check_repo(api, org: str, repo_id: str, repo_type: str) -> None:
    """Verify the repo exists and is accessible; give a clear error if not."""
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
    except Exception as exc:
        print(f"\n[ERROR] Repo '{repo_id}' not found or not accessible.")
        print(f"        Create it on HuggingFace as a private {repo_type} repo first.")
        print(f"        Details: {exc}")
        sys.exit(1)


def _upload_folder(
    api,
    local_dir: Path,
    repo_id: str,
    repo_type: str,
    path_in_repo: str = "",
    commit_message: str = "upload",
) -> None:
    if not local_dir.exists():
        print(f"  [SKIP] {local_dir} does not exist — skipping.")
        return
    file_count = sum(1 for _ in local_dir.rglob("*") if _.is_file())
    if file_count == 0:
        print(f"  [SKIP] {local_dir} is empty — skipping.")
        return
    print(f"  Uploading {file_count} files from {local_dir} → {repo_id}/{path_in_repo or '(root)'}")
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type=repo_type,
        path_in_repo=path_in_repo,
        commit_message=commit_message,
        ignore_patterns=["*.gitkeep", ".DS_Store", "__pycache__"],
    )
    print(f"  [DONE] {repo_id}")


# ── Artifact uploaders ────────────────────────────────────────────────────────

def upload_corpus(api, base_dir: Path, org: str, commit_msg: str) -> None:
    repo_id = f"{org}/poolbench-corpus"
    print(f"\n[1/4] Corpus → {repo_id}")
    _check_repo(api, org, repo_id, "dataset")
    _upload_folder(
        api, base_dir / "data" / "corpora",
        repo_id=repo_id, repo_type="dataset",
        path_in_repo="",
        commit_message=commit_msg,
    )


def upload_activations(api, base_dir: Path, org: str, commit_msg: str) -> None:
    repo_id = f"{org}/poolbench-activations"
    print(f"\n[2/4] Activations → {repo_id}")
    print("  WARNING: This is ~390 GB. Make sure you have time and bandwidth.")
    _check_repo(api, org, repo_id, "dataset")
    act_dir = base_dir / "results" / "activations"
    if not act_dir.exists():
        print(f"  [SKIP] {act_dir} does not exist.")
        return
    # Upload per-model to keep individual commits manageable
    model_dirs = [d for d in sorted(act_dir.iterdir()) if d.is_dir()]
    if not model_dirs:
        print(f"  [SKIP] No model subdirectories found in {act_dir}.")
        return
    for model_dir in model_dirs:
        print(f"  Uploading model: {model_dir.name}")
        _upload_folder(
            api, model_dir,
            repo_id=repo_id, repo_type="dataset",
            path_in_repo=model_dir.name,
            commit_message=f"{commit_msg} — {model_dir.name}",
        )


def upload_bert_scorers(api, base_dir: Path, org: str, commit_msg: str) -> None:
    repo_id = f"{org}/poolbench-bert-scorers"
    print(f"\n[3/4] BERT Scorers (D2 Classifier B) → {repo_id}")
    _check_repo(api, org, repo_id, "model")
    clf_dir = base_dir / "results" / "bert_classifiers" / "d2"
    if not clf_dir.exists():
        print(f"  [SKIP] {clf_dir} does not exist.")
        return
    concept_dirs = [d for d in sorted(clf_dir.iterdir()) if d.is_dir()]
    non_empty = [d for d in concept_dirs if any(d.rglob("*"))]
    if not non_empty:
        print(f"  [SKIP] All classifier directories are empty — Step 5 may not have run yet.")
        return
    print(f"  Found {len(non_empty)}/{len(concept_dirs)} non-empty classifiers.")
    _upload_folder(
        api, clf_dir,
        repo_id=repo_id, repo_type="model",
        path_in_repo="",
        commit_message=commit_msg,
    )


def upload_steered_outputs(api, base_dir: Path, org: str, commit_msg: str) -> None:
    repo_id = f"{org}/poolbench-steered-outputs"
    print(f"\n[4/4] Steered Outputs → {repo_id}")
    _check_repo(api, org, repo_id, "dataset")
    outputs_dir = base_dir / "results" / "steered_outputs"
    if not outputs_dir.exists():
        print(f"  [SKIP] {outputs_dir} does not exist — Step 6 may not have run yet.")
        return
    # Upload per-model
    model_dirs = [d for d in sorted(outputs_dir.iterdir()) if d.is_dir()]
    if not model_dirs:
        print(f"  [SKIP] No model output directories found.")
        return
    SENSITIVE = {"toxicity", "depression"}
    for model_dir in model_dirs:
        # Filter out sensitive concept files before uploading
        jsonl_files = list(model_dir.glob("*.jsonl"))
        safe_files = [f for f in jsonl_files if f.stem not in SENSITIVE]
        sensitive_files = [f for f in jsonl_files if f.stem in SENSITIVE]
        if sensitive_files:
            print(f"  [SENSITIVE] Skipping {[f.name for f in sensitive_files]} — not uploaded per §65")
        if not safe_files:
            print(f"  [SKIP] {model_dir.name}: no safe output files yet.")
            continue
        print(f"  Uploading {len(safe_files)} concepts for {model_dir.name}")
        for jsonl_file in safe_files:
            api.upload_file(
                path_or_fileobj=str(jsonl_file),
                path_in_repo=f"{model_dir.name}/{jsonl_file.name}",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"{commit_msg} — {model_dir.name}/{jsonl_file.name}",
            )
    print(f"  [DONE] {repo_id}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload PoolBench artifacts to HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--token",  required=True, help="HuggingFace write token (hf_xxx...)")
    parser.add_argument("--org",    required=True, help="HuggingFace org or username to upload to")
    parser.add_argument(
        "--skip",
        default="",
        help="Comma-separated artifact names to skip: corpus, activations, bert-scorers, steered-outputs",
    )
    parser.add_argument(
        "--repo-base-dir",
        default=None,
        help="Root of poolbench repo (default: parent of this script's directory)",
    )
    parser.add_argument(
        "--commit-msg",
        default=None,
        help="Commit message for all uploads (default: timestamped)",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi  # noqa: PLC0415
    except ImportError:
        print("[ERROR] huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    base_dir = Path(args.repo_base_dir) if args.repo_base_dir else Path(__file__).parent.parent
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    commit_msg = args.commit_msg or f"PoolBench upload {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"

    print(f"PoolBench HuggingFace Upload")
    print(f"  org:        {args.org}")
    print(f"  base_dir:   {base_dir}")
    print(f"  commit_msg: {commit_msg}")
    print(f"  skipping:   {skip or 'nothing'}")

    _login(args.token)
    api = HfApi()

    if "corpus" not in skip:
        upload_corpus(api, base_dir, args.org, commit_msg)

    if "activations" not in skip:
        upload_activations(api, base_dir, args.org, commit_msg)

    if "bert-scorers" not in skip:
        upload_bert_scorers(api, base_dir, args.org, commit_msg)

    if "steered-outputs" not in skip:
        upload_steered_outputs(api, base_dir, args.org, commit_msg)

    print("\n[ALL DONE] Upload complete.")
    print(f"  Steering vectors: NOT uploaded — run scripts/upload_steering_vectors.py after saving them.")


if __name__ == "__main__":
    main()
