"""
src/utils.py
Shared helpers: length validation, JSONL I/O, passage record construction.
"""

from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Optional


# ── Length constraints ────────────────────────────────────────────────────────

MIN_TOKENS = 300
MAX_TOKENS = 500
MAX_PAIR_DIFF = 25   # Max token-count difference between matched-pair pos/neg


def token_count(text: str, tokenizer) -> int:
    """Return the number of tokens produced by `tokenizer` (no special tokens)."""
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def is_valid_length(text: str, tokenizer) -> bool:
    """True if the passage is within [MIN_TOKENS, MAX_TOKENS] tokens."""
    n = token_count(text, tokenizer)
    return MIN_TOKENS <= n <= MAX_TOKENS


def is_length_matched(pos_text: str, neg_text: str, tokenizer) -> bool:
    """
    True if the |pos_tokens - neg_tokens| ≤ MAX_PAIR_DIFF.
    Only called for matched-pair concepts.
    """
    diff = abs(token_count(pos_text, tokenizer) - token_count(neg_text, tokenizer))
    return diff <= MAX_PAIR_DIFF


# ── Passage record ────────────────────────────────────────────────────────────

def make_record(
    idx: int,
    concept: str,
    split: str,          # "train" or "test"
    label: int,          # 1 = positive, 0 = negative
    text: str,
    domain: str,
    tokenizer,
    matched_pair_id: Optional[str] = None,
) -> dict:
    """Build a JSONL record for one passage."""
    return {
        "id": f"{concept}_{split}_{'pos' if label == 1 else 'neg'}_{idx:04d}",
        "text": text,
        "label": label,
        "domain": domain,
        "token_count": token_count(text, tokenizer),
        "matched_pair_id": matched_pair_id,
        "split": split,
    }


# ── JSONL I/O ─────────────────────────────────────────────────────────────────

def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records):,} records → {path}")


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Basic whitespace normalisation shared across all concept builders."""
    text = re.sub(r"(?i)<br\s*/?>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ── Seed-word contamination check ────────────────────────────────────────────

def has_seed_words(text: str, seed_words: list[str]) -> bool:
    """
    Return True if the text contains any seed word at a word boundary
    (case-insensitive).  Word-boundary matching prevents substring false
    positives such as ``ugh`` matching ``through`` or ``enough``.
    Used to verify that concept-negative passages are clean.
    """
    if not seed_words:
        return False
    import re
    for sw in seed_words:
        pattern = re.compile(r"\b" + re.escape(sw.lower()) + r"\b", re.IGNORECASE)
        if pattern.search(text):
            return True
    return False


# ── Domain tagging helpers ────────────────────────────────────────────────────

# Coarse mapping from HuggingFace dataset names to domain labels used in metadata.
DOMAIN_MAP: dict[str, str] = {
    "scientific_papers": "academic",
    "arxiv": "academic",
    "pile-of-law": "legal",
    "us_courts": "legal",
    "us_bills": "legal",
    "edgar": "financial",
    "cc_news": "news",
    "wikipedia": "wiki",
    "reddit": "social",
    "go_emotions": "social",
    "sst2": "review",
    "yelp": "review",
    "jigsaw": "social",
    "daily_dialog": "conversation",
    "multi_nli": "mixed",
    "altlex": "news",
    "discogem": "wiki",
    "conj_nli": "academic",
    "bigbio": "biomedical",
    "bio_scope": "biomedical",
    "scicite": "academic",
    "bigbench": "howto",
    "code_search_net": "code",
    "the-stack": "code",
    "depression": "social",
    "math": "math",
    "lighteval": "math",
}


def infer_domain(dataset_name: str) -> str:
    lowered = dataset_name.lower()
    for key, domain in DOMAIN_MAP.items():
        if key in lowered:
            return domain
    return "other"
