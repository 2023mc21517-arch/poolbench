"""
scripts/audit_corpus.py
=======================
Full corpus quality audit. Checks:
  1.  Same-label train/test leakage (pos train text in pos test)
  2.  Cross-label train/test leakage (pos train text in neg test or vice-versa)
  3.  Cross-label contamination within any split (same text as pos AND neg)
  4.  Seed-word contamination in negatives
  5.  Token length out of [300, 500] range
  6.  Matched-pair token diff > 25 (for concepts with needs_matched_pairs)
  7.  PubMed / structured-abstract section headers in academic_tone positives
  8.  Hostile vocabulary in toxicity negatives
  9.  Domain diversity < 3 per concept
  10. Within-split duplicates (same text twice in same file — shouldn't happen)
  11. Label field sanity (pos label==1, neg label==0)

Prints a summary with [OK] / [ISSUE] / [FIX] per concept.
Pass --fix to apply in-place cleaning for items 7 and 8 (and leakage removal).

Usage:
    python scripts/audit_corpus.py
    python scripts/audit_corpus.py --fix
    python scripts/audit_corpus.py --concept toxicity --fix
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from poolbench.concepts import CONCEPTS

CORPORA_DIR = REPO_ROOT / "data" / "corpora"


# ── helpers ────────────────────────────────────────────────────────────────────

def _md5(t: str) -> str:
    return hashlib.md5(t.encode()).hexdigest()


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _save(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n")


# ── PubMed section-header regex ────────────────────────────────────────────────
# Matches "Background:", "Methods:", etc. at the *start* of the text (first 150 chars).
_PUBMED_HEADER_RE = re.compile(
    r"^(background|methods?|results?|conclusions?|objective|purpose|aims?|"
    r"introduction|discussion|significance|summary)\s*:",
    re.IGNORECASE | re.MULTILINE,
)

def _strip_pubmed_header(text: str) -> str:
    """Remove a leading PubMed section label and trim."""
    return _PUBMED_HEADER_RE.sub("", text, count=1).lstrip(". \n").strip()


# ── Toxicity hostile-vocabulary regex ─────────────────────────────────────────
_HOSTILE_RE = re.compile(
    r"\b(horrible|disgusting|pathetic|awful|atrocious|worthless|"
    r"incompetent|rude|nasty|scam|fraud|cheated|ripped off|"
    r"disgraceful|unacceptable|offensive|abysmal|deplorable|"
    r"unprofessional|absolutely terrible|stay away)\b",
    re.IGNORECASE,
)
_POSITIVE_RE = re.compile(
    r"\b(excellent|wonderful|amazing|fantastic|outstanding|love|"
    r"perfect|best|recommend|delicious|friendly|professional|"
    r"superb|brilliant|phenomenal|incredible|awesome)\b",
    re.IGNORECASE,
)


# ── main audit ────────────────────────────────────────────────────────────────

def audit_concept(concept: str, fix: bool = False) -> dict:
    """Audit a single concept. Returns a results dict."""
    cd = CORPORA_DIR / concept
    meta = CONCEPTS.get(concept, {})
    seed_words = [s.lower() for s in meta.get("seed_words", [])]
    is_matched  = meta.get("needs_matched_pairs", False)

    files: dict[str, list[dict]] = {
        stem: _load(cd / f"{stem}.jsonl")
        for stem in ("train_pos", "train_neg", "test_pos", "test_neg")
    }

    issues: list[str] = []
    fixes_applied: list[str] = []

    # ── 10. Within-split duplicates ──────────────────────────────────────────
    for stem, recs in files.items():
        hashes = [_md5(r["text"]) for r in recs]
        seen: set = set()
        dupes = sum(1 for h in hashes if h in seen or seen.add(h))  # type: ignore[func-returns-value]
        # seen.add returns None so the `or` branch never fires; this counts dupes correctly
        seen2: set = set()
        dupes2 = sum(1 for h in hashes if h in seen2 or not seen2.add(h))
        in_seen_count = sum(1 for h in hashes if h in seen)
        seen3: set = set()
        dup_count = 0
        for h in hashes:
            if h in seen3:
                dup_count += 1
            seen3.add(h)
        if dup_count:
            issues.append(f"within-split dupes in {stem}: {dup_count}")
            if fix:
                seen4: set = set()
                deduped = []
                for r in files[stem]:
                    h = _md5(r["text"])
                    if h not in seen4:
                        seen4.add(h)
                        deduped.append(r)
                files[stem] = deduped
                fixes_applied.append(f"deduped {stem}: removed {dup_count}")
                _save(cd / f"{stem}.jsonl", deduped)

    # ── 1. Same-label train→test leakage ────────────────────────────────────
    for label in ("pos", "neg"):
        tr_hashes = {_md5(r["text"]) for r in files[f"train_{label}"]}
        te_recs   = files[f"test_{label}"]
        leaked    = [r for r in te_recs if _md5(r["text"]) in tr_hashes]
        if leaked:
            issues.append(f"same-label train→test leakage: {len(leaked)} {label} records")
            if fix:
                cleaned = [r for r in te_recs if _md5(r["text"]) not in tr_hashes]
                files[f"test_{label}"] = cleaned
                _save(cd / f"test_{label}.jsonl", cleaned)
                fixes_applied.append(f"removed {len(leaked)} same-label-leaking {label} from test")

    # ── 2+3. Cross-label contamination ──────────────────────────────────────
    all_pos_h = {_md5(r["text"]) for stem in ("train_pos", "test_pos") for r in files[stem]}
    all_neg_h = {_md5(r["text"]) for stem in ("train_neg", "test_neg") for r in files[stem]}
    cross     = all_pos_h & all_neg_h
    if cross:
        issues.append(f"cross-label contamination: {len(cross)} texts appear in both pos and neg")
        if fix:
            # Remove the cross-contaminated records from NEGATIVES (pos is canonical ground truth)
            for stem in ("train_neg", "test_neg"):
                before = len(files[stem])
                files[stem] = [r for r in files[stem] if _md5(r["text"]) not in cross]
                removed = before - len(files[stem])
                if removed:
                    _save(cd / f"{stem}.jsonl", files[stem])
                    fixes_applied.append(f"removed {removed} cross-label records from {stem}")

    # Cross-split cross-label (train_pos ↔ test_neg and test_pos ↔ train_neg)
    tr_pos_h = {_md5(r["text"]) for r in files["train_pos"]}
    te_neg_h = {_md5(r["text"]) for r in files["test_neg"]}
    te_pos_h = {_md5(r["text"]) for r in files["test_pos"]}
    tr_neg_h = {_md5(r["text"]) for r in files["train_neg"]}
    cs1 = tr_pos_h & te_neg_h
    cs2 = te_pos_h & tr_neg_h
    if cs1:
        issues.append(f"cross-split cross-label: {len(cs1)} train_pos texts appear in test_neg")
        if fix:
            before = len(files["test_neg"])
            files["test_neg"] = [r for r in files["test_neg"] if _md5(r["text"]) not in cs1]
            _save(cd / "test_neg.jsonl", files["test_neg"])
            fixes_applied.append(f"removed {before - len(files['test_neg'])} from test_neg (train_pos leak)")
    if cs2:
        issues.append(f"cross-split cross-label: {len(cs2)} test_pos texts appear in train_neg")
        if fix:
            before = len(files["train_neg"])
            files["train_neg"] = [r for r in files["train_neg"] if _md5(r["text"]) not in cs2]
            _save(cd / "train_neg.jsonl", files["train_neg"])
            fixes_applied.append(f"removed {before - len(files['train_neg'])} from train_neg (test_pos leak)")

    # ── 4. Seed-word contamination in negatives ──────────────────────────────
    # Use contamination_markers if defined; otherwise fall back to seed_words.
    # Always match at word boundaries to avoid substring false-positives
    # (e.g. "ugh" matching "through", "if" matching "knife").
    raw_contamination = [
        m.lower()
        for m in meta.get("contamination_markers", meta.get("seed_words", []))
    ]
    # Build word-boundary regex patterns; multi-word phrases use literal match
    def _make_wb_pattern(phrase: str) -> re.Pattern:
        return re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE)

    contamination_patterns = [_make_wb_pattern(m) for m in raw_contamination]

    if contamination_patterns:
        for stem in ("train_neg", "test_neg"):
            contaminated = [
                r for r in files[stem]
                if any(pat.search(r["text"]) for pat in contamination_patterns)
            ]
            if contaminated:
                issues.append(f"seed-word contamination in {stem}: {len(contaminated)} records")
                if fix:
                    cleaned = [
                        r for r in files[stem]
                        if not any(pat.search(r["text"]) for pat in contamination_patterns)
                    ]
                    files[stem] = cleaned
                    _save(cd / f"{stem}.jsonl", cleaned)
                    fixes_applied.append(f"removed {len(contaminated)} seed-contaminated records from {stem}")

    # ── 5. Token length out of range ─────────────────────────────────────────
    for stem, recs in files.items():
        bad = [r for r in recs if not (300 <= r.get("token_count", 0) <= 500)]
        if bad:
            issues.append(f"token length out of [300,500] in {stem}: {len(bad)} records")

    # ── 6. Matched-pair token diff > 25 ─────────────────────────────────────
    if is_matched:
        for split in ("train", "test"):
            pos_by_id = {r["matched_pair_id"]: r for r in files[f"{split}_pos"] if r.get("matched_pair_id")}
            neg_by_id = {r["matched_pair_id"]: r for r in files[f"{split}_neg"] if r.get("matched_pair_id")}
            bad_pairs = {
                pid for pid in pos_by_id
                if pid in neg_by_id and abs(pos_by_id[pid]["token_count"] - neg_by_id[pid]["token_count"]) > 25
            }
            if bad_pairs:
                issues.append(f"matched-pair token diff >25 in {split}: {len(bad_pairs)} pairs")

    # ── 7. PubMed section headers in academic_tone positives ─────────────────
    if concept == "academic_tone":
        for stem in ("train_pos", "test_pos"):
            flagged = [r for r in files[stem] if _PUBMED_HEADER_RE.search(r["text"][:150])]
            if flagged:
                issues.append(f"PubMed section headers in {stem}: {len(flagged)} records")
                if fix:
                    for r in files[stem]:
                        if _PUBMED_HEADER_RE.search(r["text"][:150]):
                            r["text"] = _strip_pubmed_header(r["text"])
                    _save(cd / f"{stem}.jsonl", files[stem])
                    fixes_applied.append(f"stripped PubMed headers from {len(flagged)} records in {stem}")

    # ── 8. Hostile vocabulary in toxicity negatives ───────────────────────────
    if concept == "toxicity":
        for stem in ("train_neg", "test_neg"):
            flagged = [r for r in files[stem] if _HOSTILE_RE.search(r["text"])]
            if flagged:
                issues.append(f"hostile vocab in {stem}: {len(flagged)} records")
                if fix:
                    # Keep only records that have a positive sentiment marker AND no hostile vocabulary
                    cleaned = [r for r in files[stem]
                               if _POSITIVE_RE.search(r["text"]) and not _HOSTILE_RE.search(r["text"])]
                    removed = len(files[stem]) - len(cleaned)
                    files[stem] = cleaned
                    _save(cd / f"{stem}.jsonl", cleaned)
                    fixes_applied.append(f"removed {removed} hostile-vocab records from {stem}")

    # ── 9. Domain diversity ───────────────────────────────────────────────────
    all_recs = [r for recs in files.values() for r in recs]
    domains  = {r.get("domain", "unknown") for r in all_recs}
    if len(domains) < 3:
        issues.append(f"domain diversity: only {len(domains)} domain(s) — {sorted(domains)}")

    # ── 11. Label field sanity ─────────────────────────────────────────────────
    for stem, expected_label in [("train_pos", 1), ("test_pos", 1), ("train_neg", 0), ("test_neg", 0)]:
        bad_label = [r for r in files[stem] if r.get("label") != expected_label]
        if bad_label:
            issues.append(f"wrong label field in {stem}: {len(bad_label)} records have label != {expected_label}")

    return {"concept": concept, "issues": issues, "fixes": fixes_applied}


def main():
    parser = argparse.ArgumentParser(description="PoolBench corpus quality audit")
    parser.add_argument("--concept", type=str, default=None,
                        help="Audit a single concept (default: all)")
    parser.add_argument("--fix", action="store_true",
                        help="Apply in-place fixes for items 1-4, 7, 8")
    args = parser.parse_args()

    concepts_to_audit = [args.concept] if args.concept else sorted(CONCEPTS.keys())

    total_issues = 0
    total_fixes  = 0

    for concept in concepts_to_audit:
        result = audit_concept(concept, fix=args.fix)
        issues  = result["issues"]
        fixes   = result["fixes"]
        total_issues += len(issues)
        total_fixes  += len(fixes)

        if not issues:
            print(f"[OK]    {concept}")
        else:
            for iss in issues:
                print(f"[ISSUE] {concept}: {iss}")
        if fixes:
            for fx in fixes:
                print(f"[FIX]   {concept}: {fx}")

    print(f"\n{'='*60}")
    print(f"Total issues found: {total_issues}")
    if args.fix:
        print(f"Total fixes applied: {total_fixes}")
    elif total_issues > 0:
        print("Re-run with --fix to apply in-place corrections.")
    print("All checks complete.")


if __name__ == "__main__":
    main()
