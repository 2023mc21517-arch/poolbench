#!/usr/bin/env python3
"""
upload_to_hf.py  — Upload PoolBench camera-ready artifacts to HuggingFace.

Usage:
    python scripts/upload_to_hf.py --token hf_xxx --user nips234678 --only activations
    python scripts/upload_to_hf.py --token hf_xxx --user nips234678 --only bert-scorers
    python scripts/upload_to_hf.py --token hf_xxx --user nips234678 --only steered-outputs

What it uploads (repo names are {user}/poolbench-{artifact}):
    activations      results/activations/               (~390 GB, per-model .npy files)
    bert-scorers     results/bert_classifiers/           (17 BERT Classifier B models, one dir per concept)
    steered-outputs  results/steered_outputs/           (15 non-sensitive concepts)

Corpus is already uploaded at nips234678/poolbench — not re-uploaded here.
Steering vectors are NOT uploaded here — separate task after compute step.

Prerequisites on HuggingFace (create once manually as PRIVATE before running):
    {user}/poolbench-activations     (Dataset)
    {user}/poolbench-bert-scorers    (Model)
    {user}/poolbench-steered-outputs (Dataset)

Options:
    --token          HuggingFace write token (required)
    --user           HuggingFace username (required, e.g. nips234678)
    --only           Which artifact to upload: activations | bert-scorers | steered-outputs
    --repo-base-dir  Root of poolbench repo (default: parent of this script)
    --commit-msg     Commit message (default: timestamped)
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from datetime import datetime, timezone
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
    """Small folder upload (< ~50 files / < a few GB). For large dirs use _upload_large_folder."""
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


def _upload_large_folder(
    api,
    local_dir: Path,
    repo_id: str,
    repo_type: str,
) -> None:
    """Resumable chunked upload for large folders (uses HF upload_large_folder).

    Automatically resumes if the script is re-run — already-uploaded chunks are
    skipped.  No commit message is supported by this API; HF auto-commits.
    """
    if not local_dir.exists():
        print(f"  [SKIP] {local_dir} does not exist — skipping.")
        return
    file_count = sum(1 for _ in local_dir.rglob("*") if _.is_file())
    if file_count == 0:
        print(f"  [SKIP] {local_dir} is empty — skipping.")
        return
    print(f"  upload_large_folder: {file_count} files from {local_dir} → {repo_id}")
    print(f"  (resumable — safe to Ctrl-C and re-run; already-uploaded chunks are skipped)")
    api.upload_large_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type=repo_type,
        ignore_patterns=["*.gitkeep", ".DS_Store", "__pycache__"],
    )
    print(f"  [DONE] {repo_id}")


# ── Artifact uploaders ────────────────────────────────────────────────────────

def upload_activations(api, base_dir: Path, user: str, commit_msg: str) -> None:
    repo_id = f"{user}/poolbench-activations"
    print(f"\n[activations] → {repo_id}")
    print("  ~390 GB total — using resumable chunked upload (safe to Ctrl-C and re-run).")
    _check_repo(api, user, repo_id, "dataset")
    act_dir = base_dir / "results" / "activations"
    if not act_dir.exists():
        print(f"  [SKIP] {act_dir} does not exist.")
        return
    # upload_large_folder uploads the whole tree at once and handles chunking/resuming
    _upload_large_folder(api, act_dir, repo_id=repo_id, repo_type="dataset")


def upload_bert_scorers(api, base_dir: Path, user: str, commit_msg: str) -> None:
    repo_id = f"{user}/poolbench-bert-scorers"
    print(f"\n[bert-scorers] → {repo_id}")
    _check_repo(api, user, repo_id, "model")
    clf_dir = base_dir / "results" / "bert_classifiers"
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


def upload_steered_outputs(api, base_dir: Path, user: str, commit_msg: str) -> None:
    repo_id = f"{user}/poolbench-steered-outputs"
    print(f"\n[steered-outputs] → {repo_id}")
    _check_repo(api, user, repo_id, "dataset")
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
    # Build a filtered temp-view: symlink only safe files into a staging dir,
    # then use upload_large_folder so we get resumable chunked upload.
    with tempfile.TemporaryDirectory(prefix="poolbench_steered_") as staging_root:
        staging = Path(staging_root)
        total_staged = 0
        for model_dir in model_dirs:
            jsonl_files = list(model_dir.glob("*.jsonl"))
            safe_files = [f for f in jsonl_files if f.stem not in SENSITIVE]
            sensitive_files = [f for f in jsonl_files if f.stem in SENSITIVE]
            if sensitive_files:
                print(f"  [SENSITIVE] Skipping {[f.name for f in sensitive_files]} — not uploaded per §65")
            if not safe_files:
                print(f"  [SKIP] {model_dir.name}: no safe output files yet.")
                continue
            model_staging = staging / model_dir.name
            model_staging.mkdir()
            for f in safe_files:
                (model_staging / f.name).symlink_to(f.resolve())
            total_staged += len(safe_files)
            print(f"  Staged {len(safe_files)} concepts for {model_dir.name}")
        if total_staged == 0:
            print(f"  [SKIP] No safe output files found yet.")
            return
        _upload_large_folder(api, staging, repo_id=repo_id, repo_type="dataset")
    print(f"  [DONE] {repo_id}")


def upload_steering_vectors(api, base_dir: Path, user: str, commit_msg: str) -> None:
    repo_id = f"{user}/poolbench-steering-vectors"
    print(f"\n[steering-vectors] → {repo_id}")
    _check_repo(api, user, repo_id, "dataset")
    sv_dir = base_dir / "results" / "steering_vectors"
    if not sv_dir.exists():
        print(f"  [SKIP] {sv_dir} does not exist — run save_steering_vectors.py first.")
        return
    _upload_large_folder(api, sv_dir, repo_id=repo_id, repo_type="dataset")
    print(f"  [DONE] {repo_id}")


# ── Main ──────────────────────────────────────────────────────────────────────

VALID_ARTIFACTS = {"activations", "bert-scorers", "steered-outputs", "steering-vectors"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload one PoolBench artifact to HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--token",  required=True, help="HuggingFace write token (hf_xxx...)")
    parser.add_argument("--user",   required=True, help="HuggingFace username (e.g. nips234678)")
    parser.add_argument(
        "--only",
        required=True,
        choices=sorted(VALID_ARTIFACTS),
        help="Which artifact to upload: activations | bert-scorers | steered-outputs",
    )
    parser.add_argument(
        "--repo-base-dir",
        default=None,
        help="Root of poolbench repo (default: parent of this script's directory)",
    )
    parser.add_argument(
        "--commit-msg",
        default=None,
        help="Commit message (default: timestamped)",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi  # noqa: PLC0415
    except ImportError:
        print("[ERROR] huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    base_dir = Path(args.repo_base_dir) if args.repo_base_dir else Path(__file__).parent.parent
    commit_msg = args.commit_msg or f"PoolBench upload {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"

    print(f"PoolBench HuggingFace Upload")
    print(f"  user:       {args.user}")
    print(f"  artifact:   {args.only}")
    print(f"  base_dir:   {base_dir}")
    print(f"  commit_msg: {commit_msg}")

    _login(args.token)
    api = HfApi()

    if args.only == "activations":
        upload_activations(api, base_dir, args.user, commit_msg)
    elif args.only == "bert-scorers":
        upload_bert_scorers(api, base_dir, args.user, commit_msg)
    elif args.only == "steered-outputs":
        upload_steered_outputs(api, base_dir, args.user, commit_msg)
    elif args.only == "steering-vectors":
        upload_steering_vectors(api, base_dir, args.user, commit_msg)

    print("\n[DONE]")


if __name__ == "__main__":
    main()
