"""
Push PoolBench corpus + Croissant file to HuggingFace.
Usage: HF_TOKEN=<token> python scripts/push_to_hf.py
"""
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd

REPO_ID = "nips234678/poolbench"
CORPORA_DIR = Path(__file__).parent.parent / "data" / "corpora"
CROISSANT_FILE = Path(__file__).parent.parent.parent / "Poolbench - nips d&b - use this" / "poolbench_croissant.json"

token = os.environ.get("HF_TOKEN")
if not token:
    sys.exit("Set HF_TOKEN env var before running")

api = HfApi()

# Ensure the repo exists as a dataset repo
try:
    api.repo_info(repo_id=REPO_ID, repo_type="dataset", token=token)
    print(f"Repo {REPO_ID} exists.")
except Exception:
    print(f"Creating repo {REPO_ID} ...")
    api.create_repo(repo_id=REPO_ID, repo_type="dataset", private=True, token=token)

# Collect all JSONL files
operations: list[CommitOperationAdd] = []

for concept_dir in sorted(CORPORA_DIR.iterdir()):
    if not concept_dir.is_dir():
        continue
    concept = concept_dir.name
    for jsonl_file in sorted(concept_dir.glob("*.jsonl")):
        path_in_repo = f"data/corpora/{concept}/{jsonl_file.name}"
        operations.append(
            CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=str(jsonl_file))
        )
        print(f"  + {path_in_repo}")

# Add Croissant metadata file
if CROISSANT_FILE.exists():
    operations.append(
        CommitOperationAdd(
            path_in_repo="poolbench_croissant.json",
            path_or_fileobj=str(CROISSANT_FILE),
        )
    )
    print("  + poolbench_croissant.json")
else:
    print(f"WARNING: Croissant file not found at {CROISSANT_FILE}", file=sys.stderr)

print(f"\nCommitting {len(operations)} file(s) to {REPO_ID} ...")
api.create_commit(
    repo_id=REPO_ID,
    repo_type="dataset",
    operations=operations,
    commit_message="Add PoolBench corpus (17 concepts, 700/300 per class) + Croissant metadata",
    token=token,
)
print("Done.")
