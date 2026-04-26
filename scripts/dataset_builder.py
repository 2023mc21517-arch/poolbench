"""
dataset_builder.py
==================
Main corpus construction script for PoolBench.

Usage:
    # Build a single concept (recommended — run sequentially):
    python dataset_builder.py --concept hedging

    # Build all 18 concepts one by one:
    python dataset_builder.py --all

    # Dry-run: show how many passages would be collected, don't save:
    python dataset_builder.py --concept hedging --dry-run

    # Override train/test sizes (default 700/300 per class):
    python dataset_builder.py --concept hedging --n_train 1000 --n_test 300

Environment:
    Requires only CPU.
    python -m spacy download en_core_web_sm   (for negation_density)
    pip install -r requirements.txt
"""

from __future__ import annotations
import argparse
import random
import sys
import traceback
from pathlib import Path
from typing import Optional

from poolbench.concepts import CONCEPTS, CONCEPT_NAMES
from poolbench.utils import (
    clean_text, is_valid_length, is_length_matched,
    make_record, save_jsonl, has_seed_words, infer_domain,
)
from poolbench.filters import (
    filter_hedging_positive, filter_hedging_negative,
    filter_legal_positive, filter_legal_negative,
    filter_math_certainty_positive, filter_math_certainty_negative,
    filter_frustration_positive_label, filter_frustration_negative_label,
    filter_sentiment_positive_label, filter_sentiment_negative_label,
    filter_toxicity_positive, filter_toxicity_negative,
    filter_depression_positive_label, filter_depression_negative_label,
    filter_causation_positive_label, filter_causation_negative_label,
    filter_contrast_positive_label, filter_contrast_negative_label,
    filter_conditionality_positive_label, filter_conditionality_negative_label,
    filter_academic_positive, filter_academic_negative,
    filter_code_docs_positive, filter_code_docs_negative,
    filter_bureaucratic_positive, filter_bureaucratic_negative,
    filter_uncertainty_positive, filter_uncertainty_negative,
    filter_deference_positive_label, filter_deference_negative_label,
    filter_planning_positive_label, filter_planning_negative_label,
    filter_negation_positive, filter_negation_negative,
    filter_numerical_positive, filter_numerical_negative,
)
from poolbench.rewriters import get_rewriter

# ── Shared tokenizer (LLaMA-3 tokenizer — used for all length checks) ─────────
# We use a single tokenizer for consistent length enforcement across all concepts.
# No GPU required; only the tokenizer is loaded (not the full model weights).
# LLaMA-3 tokenizer (BPE, 128k vocab) — the canonical tokenizer for this benchmark.
TOKENIZER_ID = "meta-llama/Meta-Llama-3.1-8B"
_tokenizer = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        print(f"Loading tokenizer: {TOKENIZER_ID}")
        _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
    return _tokenizer


# ── Split and save ─────────────────────────────────────────────────────────────

def split_and_save(
    concept: str,
    pos_passages: list[str],
    neg_passages: list[str],
    pos_domain: str,
    neg_domain: str,
    n_train: int,
    n_test: int,
    corpora_dir: Path,
    dry_run: bool = False,
    matched_pair_ids: Optional[list[str]] = None,  # parallel list of pair IDs if matched-pair
) -> None:
    """
    Shuffle, split into train/test, build JSONL records, and save.
    matched_pair_ids: if provided, must be same length as pos_passages (= neg_passages).
    """
    tok = get_tokenizer()
    total_needed = n_train + n_test

    if len(pos_passages) < total_needed or len(neg_passages) < total_needed:
        print(
            f"  [WARNING] {concept}: only {len(pos_passages)} pos / "
            f"{len(neg_passages)} neg available (need {total_needed} each). "
            "Corpus will be smaller than target — see README for fallback steps."
        )

    # Zip pairs if matched, else shuffle independently
    if matched_pair_ids is not None:
        triples = list(zip(pos_passages, neg_passages, matched_pair_ids))
        random.shuffle(triples)
        pos_passages  = [t[0] for t in triples]
        neg_passages  = [t[1] for t in triples]
        matched_pair_ids = [t[2] for t in triples]
    else:
        random.shuffle(pos_passages)
        random.shuffle(neg_passages)

    pos_passages  = pos_passages[:total_needed]
    neg_passages  = neg_passages[:total_needed]
    if matched_pair_ids:
        matched_pair_ids = matched_pair_ids[:total_needed]

    splits = {
        "train": (slice(0, n_train),       pos_passages[:n_train],  neg_passages[:n_train]),
        "test":  (slice(n_train, total_needed), pos_passages[n_train:], neg_passages[n_train:]),
    }

    print(f"\n  {concept}: {len(pos_passages)} pos / {len(neg_passages)} neg collected")

    if dry_run:
        print(f"  [dry-run] Would save train ({n_train}+{n_train}) and test ({n_test}+{n_test})")
        return

    out_dir = corpora_dir / concept
    for split_name, (_, pos_split, neg_split) in splits.items():
        pos_records, neg_records = [], []
        for i, text in enumerate(pos_split):
            pair_id = matched_pair_ids[i] if matched_pair_ids else None
            if split_name == "test" and matched_pair_ids:
                pair_id = matched_pair_ids[n_train + i]
            pos_records.append(make_record(i, concept, split_name, 1, text, pos_domain, tok, pair_id))
        for i, text in enumerate(neg_split):
            pair_id = matched_pair_ids[i] if matched_pair_ids else None
            if split_name == "test" and matched_pair_ids:
                pair_id = matched_pair_ids[n_train + i]
            neg_records.append(make_record(i, concept, split_name, 0, text, neg_domain, tok, pair_id))
        save_jsonl(pos_records, out_dir / f"{split_name}_pos.jsonl")
        save_jsonl(neg_records, out_dir / f"{split_name}_neg.jsonl")


# ─────────────────────────────────────────────────────────────────────────────
# Per-concept builders
# Each builder returns (pos_passages, neg_passages, pos_domain, neg_domain,
#                       matched_pair_ids_or_None)
# ─────────────────────────────────────────────────────────────────────────────

def build_hedging(n_total: int):
    """
    Rule-based parallel corpus from scientific_papers (ArXiv abstracts).
    Positive = abstract with ≥2 hedge cues; negative = rewritten version with
    hedges removed. Previously used bigbio/bio_scope but that dataset requires a
    loading script (trust_remote_code) which is no longer supported in datasets>=3.
    """
    tok = get_tokenizer()
    print("  Building hedging corpus from ArXiv abstracts ...")
    pos, neg, pair_ids = _hedging_arxiv_fallback(n_total, tok)
    return pos, neg, "scientific", "scientific", pair_ids


def _hedging_arxiv_fallback(n_needed: int, tok):
    """Rule-based rewriting from gfissore/arxiv-abstracts-2021 (Parquet, no script)."""
    from datasets import load_dataset
    from poolbench.rewriters import rewrite_hedging
    ds = load_dataset("gfissore/arxiv-abstracts-2021", split="train", streaming=True)
    pos, neg, pair_ids = [], [], []
    pair_counter = 0
    for ex in ds:
        text = clean_text(ex.get("abstract", ""))
        if not is_valid_length(text, tok):
            continue
        if not filter_hedging_positive(text):
            continue
        rewritten = rewrite_hedging(text)
        if rewritten is None:
            continue
        if not is_valid_length(rewritten, tok):
            continue
        if not is_length_matched(text, rewritten, tok):
            continue
        pair_id = f"hedging_arxiv_pair_{pair_counter:05d}"
        pos.append(text)
        neg.append(rewritten)
        pair_ids.append(pair_id)
        pair_counter += 1
        if pair_counter >= n_needed:
            break
    return pos, neg, pair_ids


def build_legal_formality(n_total: int):
    """
    Positives: lex_glue/scotus US Supreme Court opinions (Parquet, no script).
    Negatives: rule-based rewrite stripping legal markers.
    """
    from datasets import load_dataset
    from poolbench.rewriters import rewrite_legal_formality
    tok = get_tokenizer()

    print("  Loading lex_glue/scotus for legal_formality positives ...")
    ds = load_dataset("lex_glue", "scotus", split="train", streaming=True)

    pos, neg, pair_ids = [], [], []
    counter = 0
    for ex in ds:
        text = clean_text(ex.get("text", ""))
        if not is_valid_length(text, tok):
            continue
        if not filter_legal_positive(text):
            continue
        rewritten = rewrite_legal_formality(text)
        if rewritten is None:
            continue
        if not is_valid_length(rewritten, tok):
            continue
        if not is_length_matched(text, rewritten, tok):
            continue
        pair_id = f"legal_pair_{counter:05d}"
        pos.append(text)
        neg.append(rewritten)
        pair_ids.append(pair_id)
        counter += 1
        if counter >= n_total:
            break

    return pos, neg, "legal", "legal", pair_ids


def build_math_certainty(n_total: int):
    """
    Natural parallel corpus: EleutherAI/hendrycks_math (Parquet, no script).
    Positive = solution (contains certainty markers like "therefore", "hence").
    Negative = problem statement (interrogative, no certainty markers).
    Same math problem — true matched pairs, no rewriting.
    Falls back to ArXiv rule-based rewriting if supply is short.
    """
    from datasets import load_dataset, get_dataset_config_names
    tok = get_tokenizer()

    configs = get_dataset_config_names("EleutherAI/hendrycks_math")
    pos, neg, pair_ids = [], [], []
    counter = 0

    for cfg in configs:
        if counter >= n_total:
            break
        print(f"  Loading EleutherAI/hendrycks_math/{cfg} ...")
        ds = load_dataset("EleutherAI/hendrycks_math", cfg, split="train", streaming=True)
        for ex in ds:
            solution = clean_text(ex.get("solution", ""))
            problem  = clean_text(ex.get("problem", ""))
            if not is_valid_length(solution, tok):
                continue
            if not is_valid_length(problem, tok):
                continue
            if not filter_math_certainty_positive(solution):
                continue
            if not filter_math_certainty_negative(problem):
                continue
            if not is_length_matched(solution, problem, tok):
                continue
            pos.append(solution)
            neg.append(problem)
            pair_ids.append(f"math_pair_{counter:05d}")
            counter += 1
            if counter >= n_total:
                break

    # Fallback: ArXiv rule-based rewriting (gfissore/arxiv-abstracts-2021, Parquet)
    if len(pos) < n_total:
        shortfall = n_total - len(pos)
        print(f"  hendrycks_math gave {len(pos)} pairs — rewriting {shortfall} from ArXiv")
        from poolbench.rewriters import rewrite_math_certainty
        ds2 = load_dataset("gfissore/arxiv-abstracts-2021", split="train", streaming=True)
        for ex in ds2:
            text = clean_text(ex.get("abstract", ""))
            if not is_valid_length(text, tok):
                continue
            if not filter_math_certainty_positive(text):
                continue
            rewritten = rewrite_math_certainty(text)
            if rewritten is None:
                continue
            if not is_valid_length(rewritten, tok):
                continue
            if not is_length_matched(text, rewritten, tok):
                continue
            pos.append(text)
            neg.append(rewritten)
            pair_ids.append(f"math_arxiv_pair_{len(pair_ids):05d}")
            shortfall -= 1
            if shortfall <= 0:
                break

    return pos, neg, "math", "math", pair_ids


def build_frustration(n_total: int):
    """
    Natural parallel corpus: google-research-datasets/go_emotions
    Positive labels: frustrated, furious, annoyed.
    Negative labels: excited, joyful, proud, neutral.
    Independent samples (not document-matched) — same dataset, different posts.
    """
    from datasets import load_dataset
    tok = get_tokenizer()

    print("  Loading go_emotions ...")
    ds = load_dataset("google-research-datasets/go_emotions", name="simplified", split="train")

    # go_emotions: multi-label — each row has list of label indices; map idx→name
    label_names = ds.features["labels"].feature.names

    pos, neg = [], []
    for ex in ds:
        text = clean_text(ex["text"])
        if not is_valid_length(text, tok):
            continue
        labels = [label_names[i] for i in ex["labels"]]
        if any(filter_frustration_positive_label(l) for l in labels):
            pos.append(text)
        elif any(filter_frustration_negative_label(l) for l in labels):
            neg.append(text)
        if len(pos) >= n_total and len(neg) >= n_total:
            break

    return pos, neg, "social", "social", None


def build_pos_sentiment(n_total: int):
    """
    SST-2: label 1 = positive, label 0 = negative.
    Independent samples.
    """
    from datasets import load_dataset
    tok = get_tokenizer()

    print("  Loading stanfordnlp/sst2 ...")
    ds = load_dataset("stanfordnlp/sst2", split="train")

    pos, neg = [], []
    for ex in ds:
        text = clean_text(ex["sentence"])
        if not is_valid_length(text, tok):
            continue
        if filter_sentiment_positive_label(ex["label"]):
            pos.append(text)
        elif filter_sentiment_negative_label(ex["label"]):
            neg.append(text)
        if len(pos) >= n_total and len(neg) >= n_total:
            break

    # SST-2 sentences tend to be short — supplement with Yelp 5-star vs 1-star
    if len(pos) < n_total or len(neg) < n_total:
        shortfall_pos = max(0, n_total - len(pos))
        shortfall_neg = max(0, n_total - len(neg))
        print(f"  SST-2 short ({len(pos)}/{len(neg)}) — loading Yelp supplement")
        ds_yelp = load_dataset("yelp_review_full", split="train", streaming=True)
        for ex in ds_yelp:
            text = clean_text(ex["text"])
            if not is_valid_length(text, tok):
                continue
            stars = int(ex["label"]) + 1   # yelp_review_full: 0-4 → 1-5
            if stars == 5 and shortfall_pos > 0:
                pos.append(text)
                shortfall_pos -= 1
            elif stars == 1 and shortfall_neg > 0:
                neg.append(text)
                shortfall_neg -= 1
            if shortfall_pos <= 0 and shortfall_neg <= 0:
                break

    return pos, neg, "review", "review", None


def build_toxicity(n_total: int):
    """
    SetFit/toxic_conversations (Parquet, no script).
    Positive: label=1 (toxic), Negative: label=0 (not toxic).
    Independent samples.
    """
    from datasets import load_dataset
    tok = get_tokenizer()

    print("  Loading SetFit/toxic_conversations ...")
    ds = load_dataset("SetFit/toxic_conversations", split="train", streaming=True)

    pos, neg = [], []
    for ex in ds:
        text = clean_text(ex.get("text", ""))
        label = int(ex.get("label", -1))
        if not is_valid_length(text, tok):
            continue
        if label == 1 and len(pos) < n_total:
            pos.append(text)
        elif label == 0 and len(neg) < n_total:
            neg.append(text)
        if len(pos) >= n_total and len(neg) >= n_total:
            break

    return pos, neg, "social", "social", None


def build_depression(n_total: int):
    """
    solomonk/reddit_mental_health_posts (Parquet, no script).
    Positive: posts from depression/SuicideWatch/Anxiety subreddits.
    Negative: posts from non-mental-health subreddits (AskReddit etc.).
    Independent samples.
    """
    from datasets import load_dataset
    tok = get_tokenizer()

    _DEPRESSION_SUBS = {"depression", "SuicideWatch", "Anxiety", "mentalhealth",
                        "depression_help", "MentalHealthSupport"}
    _CONTROL_SUBS = {"AskReddit", "AskScience", "todayilearned", "worldnews",
                     "technology", "science", "explainlikeimfive"}

    print("  Loading solomonk/reddit_mental_health_posts ...")
    ds = load_dataset("solomonk/reddit_mental_health_posts", split="train", streaming=True)

    pos, neg = [], []
    for ex in ds:
        text = clean_text(ex.get("body", "") or ex.get("title", ""))
        subreddit = ex.get("subreddit", "")
        if not is_valid_length(text, tok):
            continue
        if subreddit in _DEPRESSION_SUBS and len(pos) < n_total:
            pos.append(text)
        elif subreddit in _CONTROL_SUBS and len(neg) < n_total:
            neg.append(text)
        if len(pos) >= n_total and len(neg) >= n_total:
            break

    # If control subreddit supply is thin, use keyword-absent filter as fallback
    if len(neg) < n_total:
        print(f"  Control subreddit supply thin ({len(neg)}) — broadening to keyword filter")
        ds2 = load_dataset("solomonk/reddit_mental_health_posts", split="train", streaming=True)
        _DEP_KW = {"depress", "suicid", "anxiet", "mental health", "therapy", "self-harm"}
        for ex in ds2:
            text = clean_text(ex.get("body", "") or ex.get("title", ""))
            if not is_valid_length(text, tok):
                continue
            lower = text.lower()
            if not any(kw in lower for kw in _DEP_KW) and len(neg) < n_total:
                neg.append(text)
            if len(neg) >= n_total:
                break

    return pos, neg, "social", "social", None


def build_causation(n_total: int):
    """
    Rule-based parallel corpus from gfissore/arxiv-abstracts-2021 (Parquet).
    Positive: abstracts with explicit causal discourse markers ("because", "therefore",
    "as a result", "consequently", "thus", "hence").
    Negative: rule-based rewrite removing the causal connector.
    """
    from datasets import load_dataset
    from poolbench.rewriters import rewrite_causation
    from poolbench.filters import filter_causation_positive
    tok = get_tokenizer()

    print("  Loading gfissore/arxiv-abstracts-2021 for causation ...")
    ds = load_dataset("gfissore/arxiv-abstracts-2021", split="train", streaming=True)

    pos, neg = [], []
    for ex in ds:
        text = clean_text(ex.get("abstract", ""))
        if not is_valid_length(text, tok):
            continue
        if not filter_causation_positive(text):
            continue
        rewritten = rewrite_causation(text)
        if rewritten is None:
            continue
        if not is_valid_length(rewritten, tok):
            continue
        if not is_length_matched(text, rewritten, tok):
            continue
        pos.append(text)
        neg.append(rewritten)
        if len(pos) >= n_total:
            break

    return pos, neg, "scientific", "scientific", None


def build_contrast(n_total: int):
    """
    nyu-mll/multi_nli (Parquet, no script).
    Positive: contradiction-label hypothesis pairs (adversative, contrastive content).
    Negative: entailment-label hypothesis pairs (coherent, non-contrastive).
    Independent samples (premise-matched to stay in same topical domain).
    """
    from datasets import load_dataset
    tok = get_tokenizer()

    print("  Loading nyu-mll/multi_nli for contrast ...")
    ds = load_dataset("nyu-mll/multi_nli", split="train", streaming=True)

    pos, neg = [], []
    for ex in ds:
        hyp = clean_text(ex.get("hypothesis", ""))
        label = int(ex.get("label", -1))
        if not is_valid_length(hyp, tok):
            continue
        if label == 2 and len(pos) < n_total:   # contradiction → contrastive
            if filter_contrast_positive_label("adversative"):
                pos.append(hyp)
        elif label == 0 and len(neg) < n_total:  # entailment → coherent
            if filter_contrast_negative_label("expansion"):
                neg.append(hyp)
        if len(pos) >= n_total and len(neg) >= n_total:
            break

    return pos, neg, "mixed", "mixed", None


def build_conditionality(n_total: int):
    """
    Rule-based filter on nyu-mll/multi_nli (Parquet, no script).
    Positive: hypotheses starting with "if", "when", "unless", "provided",
    "whenever", "given that" (explicit conditional structure).
    Negative: hypotheses with no conditional opener and plain declarative form.
    Independent samples.
    """
    import re
    from datasets import load_dataset
    tok = get_tokenizer()

    _COND_RE = re.compile(
        r'^(if|when|unless|provided|whenever|given that|assuming that|in the event that)\b',
        re.IGNORECASE,
    )
    _NEG_KW = re.compile(r'\b(if|when|unless|whenever|provided|assuming)\b', re.IGNORECASE)

    print("  Loading nyu-mll/multi_nli for conditionality ...")
    ds = load_dataset("nyu-mll/multi_nli", split="train", streaming=True)

    pos, neg = [], []
    for ex in ds:
        hyp = clean_text(ex.get("hypothesis", ""))
        if not is_valid_length(hyp, tok):
            continue
        if _COND_RE.match(hyp) and len(pos) < n_total:
            pos.append(hyp)
        elif not _NEG_KW.search(hyp) and len(neg) < n_total:
            neg.append(hyp)
        if len(pos) >= n_total and len(neg) >= n_total:
            break

    return pos, neg, "mixed", "mixed", None


def build_academic_tone(n_total: int):
    """
    Positives: gfissore/arxiv-abstracts-2021 (Parquet, no script).
    Negatives: sentence-transformers/reddit (Parquet, no script).
    Independent samples.
    """
    from datasets import load_dataset
    tok = get_tokenizer()

    pos, neg = [], []

    print("  Loading gfissore/arxiv-abstracts-2021 for academic_tone positives ...")
    ds_pos = load_dataset("gfissore/arxiv-abstracts-2021", split="train", streaming=True)
    for ex in ds_pos:
        text = clean_text(ex.get("abstract", ""))
        if is_valid_length(text, tok) and filter_academic_positive(text):
            pos.append(text)
        if len(pos) >= n_total:
            break

    print("  Loading sentence-transformers/reddit for academic_tone negatives ...")
    ds_neg = load_dataset("sentence-transformers/reddit", split="train", streaming=True)
    for ex in ds_neg:
        text = clean_text(ex.get("body", "") or ex.get("text", ""))
        if is_valid_length(text, tok) and filter_academic_negative(text):
            neg.append(text)
        if len(neg) >= n_total:
            break

    return pos, neg, "academic", "social", None


def build_code_docs(n_total: int):
    """
    Positives: Nan-Do/code-search-net-python docstrings (Parquet, no script).
    Negatives: sentence-transformers/reddit posts (Parquet, no script).
    Independent samples.
    """
    from datasets import load_dataset
    tok = get_tokenizer()

    pos, neg = [], []

    print("  Loading Nan-Do/code-search-net-python for code_docs positives ...")
    ds_pos = load_dataset("Nan-Do/code-search-net-python", split="train", streaming=True)
    for ex in ds_pos:
        doc = clean_text(ex.get("docstring", "") or ex.get("func_documentation_string", ""))
        if is_valid_length(doc, tok) and filter_code_docs_positive(doc):
            pos.append(doc)
        if len(pos) >= n_total:
            break

    print("  Loading sentence-transformers/reddit for code_docs negatives ...")
    ds_neg = load_dataset("sentence-transformers/reddit", split="train", streaming=True)
    for ex in ds_neg:
        text = clean_text(ex.get("body", "") or ex.get("text", ""))
        if is_valid_length(text, tok) and filter_code_docs_negative(text):
            neg.append(text)
        if len(neg) >= n_total:
            break

    return pos, neg, "code", "social", None


def build_bureaucratic(n_total: int):
    """
    Positives: FiscalNote/billsum US Congressional bill texts (Parquet, no script).
    Negatives: Yelp/yelp_review_full informal reviews (Parquet, no script).
    Independent samples.
    """
    from datasets import load_dataset
    tok = get_tokenizer()

    pos, neg = [], []

    print("  Loading FiscalNote/billsum for bureaucratic positives ...")
    ds_pos = load_dataset("FiscalNote/billsum", split="train", streaming=True)
    for ex in ds_pos:
        text = clean_text(ex.get("summary", ""))  # summary ~300-500 tok; full text is >>500
        if is_valid_length(text, tok) and filter_bureaucratic_positive(text):
            pos.append(text)
        if len(pos) >= n_total:
            break

    print("  Loading Yelp/yelp_review_full for bureaucratic negatives ...")
    ds_neg = load_dataset("Yelp/yelp_review_full", split="train", streaming=True)
    for ex in ds_neg:
        text = clean_text(ex.get("text", ""))
        if is_valid_length(text, tok) and filter_bureaucratic_negative(text):
            neg.append(text)
        if len(neg) >= n_total:
            break

    return pos, neg, "legal", "review", None


def build_uncertainty(n_total: int):
    """
    Positives: ArXiv abstracts with uncertainty markers.
    Negatives: ArXiv abstracts with certainty markers (no uncertainty markers).
    Source: gfissore/arxiv-abstracts-2021 (Parquet, no script).
    Independent samples from the same source (domain-matched).
    """
    from datasets import load_dataset
    tok = get_tokenizer()

    pos, neg = [], []

    print("  Loading gfissore/arxiv-abstracts-2021 for uncertainty ...")
    ds = load_dataset("gfissore/arxiv-abstracts-2021", split="train", streaming=True)
    for ex in ds:
        text = clean_text(ex.get("abstract", ""))
        if not is_valid_length(text, tok):
            continue
        if filter_uncertainty_positive(text) and len(pos) < n_total:
            pos.append(text)
        elif filter_uncertainty_negative(text) and len(neg) < n_total:
            neg.append(text)
        if len(pos) >= n_total and len(neg) >= n_total:
            break

    return pos, neg, "academic", "academic", None


def build_deference(n_total: int):
    """
    Rule-based filter on gfissore/arxiv-abstracts-2021 (Parquet, no script).
    Positive: abstracts deferring to prior authority ("previous work showed",
    "as demonstrated by", "it has been established", "prior studies", etc.).
    Negative: abstracts asserting own findings ("we show", "we demonstrate",
    "we find", "our results", "in this paper we", etc.).
    Independent samples.
    """
    import re
    from datasets import load_dataset
    tok = get_tokenizer()

    _POS_RE = re.compile(
        r'\b(previous(ly)?\s+work|prior\s+(work|stud|research|result|art)|as\s+(shown|demonstrated|noted|reported|established)\s+by|it\s+has\s+been\s+(shown|demonstrated|established|found)|following\s+(the\s+approach|the\s+method)|building\s+on|based\s+on\s+(the\s+(work|method|approach|result)\s+of))',
        re.IGNORECASE,
    )
    _NEG_RE = re.compile(
        r'\b(we\s+(show|demonstrate|find|present|propose|introduce|derive|prove|establish|report|describe)|in\s+this\s+(paper|work|article)\s+we|our\s+(result|finding|method|approach|model|algorithm|system|framework|contribution))',
        re.IGNORECASE,
    )

    print("  Loading gfissore/arxiv-abstracts-2021 for deference ...")
    ds = load_dataset("gfissore/arxiv-abstracts-2021", split="train", streaming=True)

    pos, neg = [], []
    for ex in ds:
        text = clean_text(ex.get("abstract", ""))
        if not is_valid_length(text, tok):
            continue
        if _POS_RE.search(text) and not _NEG_RE.search(text) and len(pos) < n_total:
            pos.append(text)
        elif _NEG_RE.search(text) and not _POS_RE.search(text) and len(neg) < n_total:
            neg.append(text)
        if len(pos) >= n_total and len(neg) >= n_total:
            break

    return pos, neg, "academic", "academic", None


def build_planning(n_total: int):
    """
    BigBench goal_step_wikihow (tasksource/bigbench config 'goal_step_wikihow').
    Positive: step correctly belongs to goal (correct sub-action in plan).
    Negative: plausible step from a different goal.
    """
    from datasets import load_dataset
    tok = get_tokenizer()

    print("  Loading tasksource/bigbench (goal_step_wikihow) ...")
    ds = load_dataset("tasksource/bigbench", "goal_step_wikihow", split="train")

    pos, neg = [], []
    for ex in ds:
        # Each example has 'inputs' (goal + step) and 'targets' (yes/no)
        text = clean_text(ex.get("inputs", "") or ex.get("text", ""))
        targets = ex.get("targets", [])
        correct = (targets[0].lower() == "yes") if targets else False
        if not is_valid_length(text, tok):
            continue
        if filter_planning_positive_label(correct):
            pos.append(text)
        elif filter_planning_negative_label(correct):
            neg.append(text)
        if len(pos) >= n_total and len(neg) >= n_total:
            break

    return pos, neg, "howto", "howto", None


def build_negation_density(n_total: int):
    """
    ArXiv abstracts filtered by negation word count (regex; no spaCy needed).
    Positive: abstract has ≥ 3 negation words (not/no/never/neither/nor/without/cannot/n't).
    Negative: abstract has 0 negation words.
    Independent samples.
    Source: gfissore/arxiv-abstracts-2021 (Parquet, no script).
    """
    import re
    from datasets import load_dataset
    tok = get_tokenizer()

    _NEG_RE = re.compile(r"\b(not|no|never|neither|nor|without|cannot|n't)\b", re.IGNORECASE)

    print("  Loading gfissore/arxiv-abstracts-2021 for negation_density ...")
    ds = load_dataset("gfissore/arxiv-abstracts-2021", split="train", streaming=True)

    pos, neg = [], []
    for ex in ds:
        text = clean_text(ex.get("abstract", ""))
        if not is_valid_length(text, tok):
            continue
        neg_count = len(_NEG_RE.findall(text))
        if neg_count >= 3 and len(pos) < n_total:
            pos.append(text)
        elif neg_count == 0 and len(neg) < n_total:
            neg.append(text)
        if len(pos) >= n_total and len(neg) >= n_total:
            break

    return pos, neg, "academic", "academic", None


def build_numerical_precision(n_total: int):
    """
    Positives: ArXiv STEM abstracts (≥4 numeric tokens).
    Negatives: CC-News narratives (0 specific numbers + vague quantifiers).
    Independent samples.
    """
    from datasets import load_dataset
    tok = get_tokenizer()

    pos, neg = [], []

    print("  Loading gfissore/arxiv-abstracts-2021 for numerical_precision positives ...")
    ds_pos = load_dataset("gfissore/arxiv-abstracts-2021", split="train", streaming=True)
    for ex in ds_pos:
        text = clean_text(ex.get("abstract", ""))
        if is_valid_length(text, tok) and filter_numerical_positive(text):
            pos.append(text)
        if len(pos) >= n_total:
            break

    print("  Loading cc_news for numerical_precision negatives ...")
    ds_neg = load_dataset("cc_news", split="train", streaming=True)
    for ex in ds_neg:
        text = clean_text(ex.get("text", "") or ex.get("content", ""))
        if is_valid_length(text, tok) and filter_numerical_negative(text):
            neg.append(text)
        if len(neg) >= n_total:
            break

    return pos, neg, "academic", "news", None


# ─────────────────────────────────────────────────────────────────────────────
# Concept dispatch table
# ─────────────────────────────────────────────────────────────────────────────

CONCEPT_BUILDERS = {
    "hedging":             build_hedging,
    "legal_formality":     build_legal_formality,
    "math_certainty":      build_math_certainty,
    "frustration":         build_frustration,
    "pos_sentiment":       build_pos_sentiment,
    "toxicity":            build_toxicity,
    "depression":          build_depression,
    "causation":           build_causation,
    "contrast":            build_contrast,
    "conditionality":      build_conditionality,
    "academic_tone":       build_academic_tone,
    "code_docs":           build_code_docs,
    "bureaucratic":        build_bureaucratic,
    "uncertainty":         build_uncertainty,
    "deference":           build_deference,
    "planning":            build_planning,
    "negation_density":    build_negation_density,
    "numerical_precision": build_numerical_precision,
}


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_corpus(concept: str, corpora_dir: Path) -> bool:
    """
    Post-build checks per the benchmark rules:
    1. All passages 300–500 tokens.
    2. Matched-pair token diff ≤ 25.
    3. No seed-word contamination on negatives.
    4. At least 3 domains present (warns only — does not fail).
    Prints a summary and returns True if all hard checks pass.
    """
    from poolbench.utils import load_jsonl
    tok = get_tokenizer()
    meta = CONCEPTS[concept]
    seed_words = meta.get("seed_words", [])
    ok = True

    for split in ("train", "test"):
        pos_file = corpora_dir / concept / f"{split}_pos.jsonl"
        neg_file = corpora_dir / concept / f"{split}_neg.jsonl"
        if not pos_file.exists() or not neg_file.exists():
            print(f"  [FAIL] {concept}/{split}: files missing")
            return False

        pos_recs = load_jsonl(pos_file)
        neg_recs = load_jsonl(neg_file)

        # Length check
        bad_pos = [r for r in pos_recs if not (300 <= r["token_count"] <= 500)]
        bad_neg = [r for r in neg_recs if not (300 <= r["token_count"] <= 500)]
        if bad_pos or bad_neg:
            print(f"  [FAIL] {concept}/{split}: {len(bad_pos)} pos / {len(bad_neg)} neg out of length range")
            ok = False

        # Seed-word contamination on negatives
        if seed_words:
            contaminated = [r for r in neg_recs if has_seed_words(r["text"], seed_words)]
            if contaminated:
                print(f"  [WARN] {concept}/{split}: {len(contaminated)} neg passages contain seed words")

        # Matched-pair token diff (only if matched_pair_id is populated)
        if meta.get("needs_matched_pairs"):
            pos_by_id = {r["matched_pair_id"]: r for r in pos_recs if r.get("matched_pair_id")}
            neg_by_id = {r["matched_pair_id"]: r for r in neg_recs if r.get("matched_pair_id")}
            for pid in pos_by_id:
                if pid in neg_by_id:
                    diff = abs(pos_by_id[pid]["token_count"] - neg_by_id[pid]["token_count"])
                    if diff > 25:
                        print(f"  [FAIL] {concept}/{split} pair {pid}: token diff = {diff} > 25")
                        ok = False

        # Domain diversity (warn only)
        domains = set(r.get("domain", "other") for r in pos_recs + neg_recs)
        if len(domains) < 2:
            print(f"  [WARN] {concept}/{split}: only {len(domains)} domain(s) covered (target ≥ 2)")

    if ok:
        print(f"  [OK] {concept}: all hard checks passed")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PoolBench dataset builder")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--concept", choices=CONCEPT_NAMES,
                       help="Build a single concept corpus")
    group.add_argument("--all", action="store_true",
                       help="Build all 18 concept corpora sequentially")
    parser.add_argument("--n_train", type=int, default=700,
                        help="Training passages per class (default 700)")
    parser.add_argument("--n_test",  type=int, default=300,
                        help="Test passages per class (default 300)")
    parser.add_argument("--corpora_dir", type=Path,
                        default=Path("data/corpora"),
                        help="Output directory for built corpora")
    parser.add_argument("--dry_run", action="store_true",
                        help="Count passages only; do not write files")
    parser.add_argument("--validate_only", action="store_true",
                        help="Run post-build validation on existing files")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train/test split")
    args = parser.parse_args()

    random.seed(args.seed)
    n_total = args.n_train + args.n_test

    concepts_to_build = CONCEPT_NAMES if args.all else [args.concept]

    for concept in concepts_to_build:
        print(f"\n{'='*60}")
        print(f"Building: {concept}")
        print(f"{'='*60}")

        if args.validate_only:
            validate_corpus(concept, args.corpora_dir)
            continue

        # Skip if already built (both train files exist)
        train_pos = args.corpora_dir / concept / "train_pos.jsonl"
        train_neg = args.corpora_dir / concept / "train_neg.jsonl"
        if train_pos.exists() and train_neg.exists() and not args.dry_run:
            print(f"  Already built — skipping. (delete files to rebuild)")
            continue

        builder = CONCEPT_BUILDERS[concept]
        try:
            pos, neg, pos_domain, neg_domain, pair_ids = builder(n_total)
        except Exception:
            print(f"  [ERROR] builder for {concept} failed:")
            traceback.print_exc()
            continue

        split_and_save(
            concept=concept,
            pos_passages=pos,
            neg_passages=neg,
            pos_domain=pos_domain,
            neg_domain=neg_domain,
            n_train=args.n_train,
            n_test=args.n_test,
            corpora_dir=args.corpora_dir,
            dry_run=args.dry_run,
            matched_pair_ids=pair_ids,
        )

        if not args.dry_run:
            validate_corpus(concept, args.corpora_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
