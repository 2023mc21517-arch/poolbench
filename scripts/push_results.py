"""
push_results.py
Force-add all computed JSON/PNG result files (gitignored by default to exclude
large .npy activations) and push to remote.

What gets pushed:
    results/auroc/**/best_layer_auroc.json
    results/auroc/**/fallback_rates.json
    results/nemenyi/*.json
    results/icc/*.json
    results/linearity/*.json
    results/scp/*.json
    results/disentanglement/*.json
    results/oracle_auroc/*.json
    results/layer_rank_correlation.json
    results/layer_rank_correlation.png
    results/power_analysis.json
    leaderboard/official/poolbench_v1.json

What does NOT get pushed (too large or already on HuggingFace):
    results/activations/**  (.npy files)
    results/bert_classifiers/**  (sklearn model pickles)
    data/corpora/**  (on HF Hub)

Usage
-----
    python scripts/push_results.py
    python scripts/push_results.py --message "chore: update D1 AUROC results"
    python scripts/push_results.py --dry_run   # show what would be added, no commit
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

BASE_DIR    = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"

# Glob patterns relative to BASE_DIR that we want to force-add.
# These are all JSON/PNG result files — no binaries, no large arrays.
INCLUDE_PATTERNS: list[str] = [
    "results/auroc/**/best_layer_auroc.json",
    "results/auroc/**/fallback_rates.json",
    "results/nemenyi/*.json",
    "results/icc/*.json",
    "results/linearity/*.json",
    "results/scp/*.json",
    "results/disentanglement/*.json",
    "results/oracle_auroc/*.json",
    "results/layer_rank_correlation.json",
    "results/layer_rank_correlation.png",
    "results/power_analysis.json",
    "leaderboard/official/poolbench_v1.json",
]


def _run(cmd: list[str], dry_run: bool, capture: bool = False) -> str:
    """Run a shell command, printing it first. In dry_run mode just print."""
    print(f"  $ {' '.join(cmd)}")
    if dry_run:
        return ""
    result = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=capture, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(result.returncode)
    return result.stdout.strip() if capture else ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Force-add result JSONs and push to GitHub"
    )
    parser.add_argument("--message", "-m", type=str,
                        default="chore: push computed results",
                        help="Git commit message")
    parser.add_argument("--dry_run", action="store_true",
                        help="Show what would happen without modifying git state")
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print("  push_results.py")
    if args.dry_run:
        print("  DRY RUN — no files will be staged or pushed")
    print(f"{'='*55}\n")

    # Collect files that actually exist on disk
    files_to_add: list[str] = []
    for pattern in INCLUDE_PATTERNS:
        matched = sorted(BASE_DIR.glob(pattern))
        for f in matched:
            rel = str(f.relative_to(BASE_DIR))
            files_to_add.append(rel)

    if not files_to_add:
        print("No result files found yet — nothing to push.")
        print("Run run_model.py (Steps 1–7) first to generate results.")
        return

    print(f"Found {len(files_to_add)} result file(s) to stage:\n")
    for f in files_to_add:
        print(f"  {f}")

    print()

    # git add --force (bypasses .gitignore for these specific files)
    _run(["git", "add", "--force"] + files_to_add, dry_run=args.dry_run)

    # Check if anything is actually staged
    if not args.dry_run:
        staged = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            cwd=str(BASE_DIR), capture_output=True, text=True
        ).stdout.strip()
        if not staged:
            print("Nothing new to commit — all result files already up to date.")
            return
        print(f"\nStaged {len(staged.splitlines())} file(s).")

    # git commit
    _run(["git", "commit", "-m", args.message], dry_run=args.dry_run)

    # git push
    _run(["git", "push"], dry_run=args.dry_run)

    if not args.dry_run:
        print("\nDone. Results pushed to GitHub.")


if __name__ == "__main__":
    main()
