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


def _chunk_tokens(text: str, tokenizer, target: int = 400, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks of ~target tokens (for long source documents)."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= target + 100:
        return [text]
    step = max(1, target - overlap)
    chunks = []
    for start in range(0, len(tokens), step):
        end = min(start + target + 100, len(tokens))
        chunk_text = tokenizer.decode(tokens[start:end], skip_special_tokens=True).strip()
        if chunk_text:
            chunks.append(chunk_text)
        if end >= len(tokens):
            break
    return chunks


# ── Split and save ─────────────────────────────────────────────────────────────

def split_and_save(
    concept: str,
    pos_passages: list[str],
    neg_passages: list[str],
    pos_domain: "str | list[str]",
    neg_domain: "str | list[str]",
    n_train: int,
    n_test: int,
    corpora_dir: Path,
    dry_run: bool = False,
    matched_pair_ids: Optional[list[str]] = None,  # parallel list of pair IDs if matched-pair
    seed_words: Optional[list[str]] = None,          # concept seed words — filter neg contamination
) -> None:
    """
    Shuffle, split into train/test, build JSONL records, and save.
    pos_domain / neg_domain: single string (broadcast) OR per-record list.
    matched_pair_ids: if provided, must be same length as pos_passages (= neg_passages).
    """
    tok = get_tokenizer()
    total_needed = n_train + n_test

    # ── Deduplication + integrity checks (integrated build-time audit) ────────
    import hashlib as _hs
    def _md5(t: str) -> str:
        return _hs.md5(t.encode()).hexdigest()

    pos_domain_list_raw = (pos_domain if isinstance(pos_domain, list)
                           else [pos_domain] * len(pos_passages))
    neg_domain_list_raw = (neg_domain if isinstance(neg_domain, list)
                           else [neg_domain] * len(neg_passages))

    if matched_pair_ids is not None:
        # Matched-pair: dedup full pairs jointly by positive text hash.
        # NEVER dedup neg independently — that breaks pair alignment because neg
        # is a deterministic rewrite of pos and index correspondence must be preserved.
        seen_hashes: set = set()
        _np, _nn, _dp, _dn, _ids = [], [], [], [], []
        for p, n, dp, dn, pid in zip(
            pos_passages, neg_passages,
            pos_domain_list_raw, neg_domain_list_raw, matched_pair_ids
        ):
            h = _md5(p)
            if h not in seen_hashes:
                seen_hashes.add(h)
                _np.append(p); _nn.append(n); _dp.append(dp); _dn.append(dn); _ids.append(pid)
        pos_passages, neg_passages = _np, _nn
        pos_domain_list_raw, neg_domain_list_raw = _dp, _dn
        matched_pair_ids = _ids
    else:
        # Independent sampling: dedup each class, then cross-class dedup to
        # remove label-ambiguous passages (same text in both pos and neg).
        def _dedup_list(passages, domains):
            seen: set = set()
            out_p, out_d = [], []
            for p, d in zip(passages, domains):
                h = _md5(p)
                if h not in seen:
                    seen.add(h)
                    out_p.append(p); out_d.append(d)
            return out_p, out_d

        pos_passages, pos_domain_list_raw = _dedup_list(pos_passages, pos_domain_list_raw)
        neg_passages, neg_domain_list_raw = _dedup_list(neg_passages, neg_domain_list_raw)

        pos_hashes = {_md5(p) for p in pos_passages}
        cross_n = sum(1 for n in neg_passages if _md5(n) in pos_hashes)
        if cross_n:
            print(f"  [dedup] {concept}: removing {cross_n} label-ambiguous "
                  f"neg passages that also appear as pos")
            filtered = [(n, d) for n, d in zip(neg_passages, neg_domain_list_raw)
                        if _md5(n) not in pos_hashes]
            neg_passages = [x[0] for x in filtered]
            neg_domain_list_raw = [x[1] for x in filtered]

    # ── Seed-word contamination enforcement ───────────────────────────────────
    # Negative passages must NOT contain positive-class seed words.
    # Enforced here at construction time so contaminated records never reach disk.
    if seed_words:
        _sw_lower = [s.lower() for s in seed_words]
        if matched_pair_ids is not None:
            kept = [
                (p, n, dp, dn, pid)
                for p, n, dp, dn, pid in zip(
                    pos_passages, neg_passages,
                    pos_domain_list_raw, neg_domain_list_raw, matched_pair_ids
                )
                if not has_seed_words(n, _sw_lower)
            ]
            n_removed = len(pos_passages) - len(kept)
            if n_removed:
                print(f"  [seed-clean] {concept}: removed {n_removed} matched pairs "
                      f"where neg contained seed words")
            if kept:
                pos_passages, neg_passages, pos_domain_list_raw, neg_domain_list_raw, matched_pair_ids = (
                    [x[0] for x in kept], [x[1] for x in kept],
                    [x[2] for x in kept], [x[3] for x in kept], [x[4] for x in kept]
                )
            else:
                pos_passages, neg_passages = [], []
                pos_domain_list_raw, neg_domain_list_raw = [], []
                matched_pair_ids = []
        else:
            clean = [(n, d) for n, d in zip(neg_passages, neg_domain_list_raw)
                     if not has_seed_words(n, _sw_lower)]
            n_removed = len(neg_passages) - len(clean)
            if n_removed:
                print(f"  [seed-clean] {concept}: removed {n_removed} neg passages "
                      f"containing seed words")
            neg_passages = [x[0] for x in clean]
            neg_domain_list_raw = [x[1] for x in clean]

    # Rebuild domain inputs as lists for uniform downstream handling
    pos_domain = pos_domain_list_raw
    neg_domain = neg_domain_list_raw

    if len(pos_passages) < total_needed or len(neg_passages) < total_needed:
        print(
            f"  [WARNING] {concept}: only {len(pos_passages)} pos / "
            f"{len(neg_passages)} neg available after dedup (need {total_needed} each). "
            "Corpus will be smaller than target — see README for fallback steps."
        )

    # Normalise domain args to lists (for uniform handling below)
    def _to_domain_list(d, length):
        return d if isinstance(d, list) else [d] * length

    # Zip pairs if matched, else shuffle independently
    if matched_pair_ids is not None:
        pos_domains_list = _to_domain_list(pos_domain, len(pos_passages))
        neg_domains_list = _to_domain_list(neg_domain, len(neg_passages))
        quintuples = list(zip(pos_passages, neg_passages, matched_pair_ids,
                               pos_domains_list, neg_domains_list))
        random.shuffle(quintuples)
        pos_passages     = [q[0] for q in quintuples]
        neg_passages     = [q[1] for q in quintuples]
        matched_pair_ids = [q[2] for q in quintuples]
        pos_domains_list = [q[3] for q in quintuples]
        neg_domains_list = [q[4] for q in quintuples]
    else:
        if isinstance(pos_domain, list):
            pos_pairs = list(zip(pos_passages, pos_domain))
            random.shuffle(pos_pairs)
            pos_passages, pos_domain = zip(*pos_pairs) if pos_pairs else ([], [])
            pos_passages = list(pos_passages)
            pos_domain   = list(pos_domain)
        else:
            random.shuffle(pos_passages)
        if isinstance(neg_domain, list):
            neg_pairs = list(zip(neg_passages, neg_domain))
            random.shuffle(neg_pairs)
            neg_passages, neg_domain = zip(*neg_pairs) if neg_pairs else ([], [])
            neg_passages = list(neg_passages)
            neg_domain   = list(neg_domain)
        else:
            random.shuffle(neg_passages)
        pos_domains_list = _to_domain_list(pos_domain, len(pos_passages))
        neg_domains_list = _to_domain_list(neg_domain, len(neg_passages))

    pos_passages     = pos_passages[:total_needed]
    neg_passages     = neg_passages[:total_needed]
    pos_domains_list = pos_domains_list[:total_needed]
    neg_domains_list = neg_domains_list[:total_needed]
    if matched_pair_ids:
        matched_pair_ids = matched_pair_ids[:total_needed]

    print(f"\n  {concept}: {len(pos_passages)} pos / {len(neg_passages)} neg collected")
    domain_set = set(pos_domains_list + neg_domains_list)
    print(f"  Domains covered: {sorted(domain_set)}")

    if dry_run:
        print(f"  [dry-run] Would save train ({n_train}+{n_train}) and test ({n_test}+{n_test})")
        return

    out_dir = corpora_dir / concept
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name in ("train", "test"):
        sl = slice(0, n_train) if split_name == "train" else slice(n_train, total_needed)
        pos_split = pos_passages[sl]
        neg_split = neg_passages[sl]
        pd_split  = pos_domains_list[sl]
        nd_split  = neg_domains_list[sl]
        pair_slice = matched_pair_ids[sl] if matched_pair_ids else None

        pos_records, neg_records = [], []
        for i, text in enumerate(pos_split):
            pair_id = pair_slice[i] if pair_slice else None
            pos_records.append(make_record(i, concept, split_name, 1, text, pd_split[i], tok, pair_id))
        for i, text in enumerate(neg_split):
            pair_id = pair_slice[i] if pair_slice else None
            neg_records.append(make_record(i, concept, split_name, 0, text, nd_split[i], tok, pair_id))
        save_jsonl(pos_records, out_dir / f"{split_name}_pos.jsonl")
        save_jsonl(neg_records, out_dir / f"{split_name}_neg.jsonl")



# ─────────────────────────────────────────────────────────────────────────────
# Shared streaming helpers
# ─────────────────────────────────────────────────────────────────────────────

def _stream_rewrite_pairs(
    ds_name: str,
    config,
    field: str,
    domain: str,
    filter_positive_fn,
    rewriter_fn,
    n_needed: int,
    tok,
    pair_prefix: str,
    start_counter: int = 0,
    chunk_long_docs: bool = False,
) -> tuple:
    """
    Stream a dataset and build matched rewrite-pairs.
    chunk_long_docs: if True, split each document into 300-500 token windows first.
    Returns (pos_texts, neg_texts, pos_domains, neg_domains, pair_ids).
    """
    from datasets import load_dataset
    ds = load_dataset(ds_name, config, split="train", streaming=True)
    pos, neg, p_dom, n_dom, pair_ids = [], [], [], [], []
    counter = start_counter
    for ex in ds:
        full_text = clean_text(ex.get(field, "") or "")
        if not full_text:
            continue
        candidates = _chunk_tokens(full_text, tok) if chunk_long_docs else [full_text]
        for text in candidates:
            if not is_valid_length(text, tok):
                continue
            if not filter_positive_fn(text):
                continue
            rewritten = rewriter_fn(text)
            if rewritten is None:
                continue
            if not is_valid_length(rewritten, tok):
                continue
            if not is_length_matched(text, rewritten, tok):
                continue
            pos.append(text)
            neg.append(rewritten)
            p_dom.append(domain)
            n_dom.append(domain)
            pair_ids.append(f"{pair_prefix}_{counter:05d}")
            counter += 1
            if counter - start_counter >= n_needed:
                break
        if counter - start_counter >= n_needed:
            break
    return pos, neg, p_dom, n_dom, pair_ids


def _stream_independent(
    ds_name: str,
    config,
    field: str,
    domain: str,
    filter_fn,
    n_needed: int,
    tok,
    chunk_long_docs: bool = False,
) -> tuple:
    """
    Stream a dataset and collect passages passing filter_fn.
    Returns (texts, domains).
    """
    from datasets import load_dataset
    ds = load_dataset(ds_name, config, split="train", streaming=True)
    texts, domains = [], []
    for ex in ds:
        full_text = clean_text(ex.get(field, "") or "")
        if not full_text:
            continue
        candidates = _chunk_tokens(full_text, tok) if chunk_long_docs else [full_text]
        for text in candidates:
            if not is_valid_length(text, tok):
                continue
            if filter_fn(text):
                texts.append(text)
                domains.append(domain)
            if len(texts) >= n_needed:
                break
        if len(texts) >= n_needed:
            break
    return texts, domains


# ─────────────────────────────────────────────────────────────────────────────
# Per-concept builders
# Each builder returns (pos_passages, neg_passages, pos_domains, neg_domains,
#                       matched_pair_ids_or_None)
# ─────────────────────────────────────────────────────────────────────────────

def build_hedging(n_total: int):
    """
    Matched-pair corpus (rule-based rewriting).
    3 domains: scientific (ArXiv), news (CC-News), government (BillSum).
    Each source targets n_total//2 to ensure all 3 domains contribute.
    """
    from poolbench.rewriters import rewrite_hedging
    tok = get_tokenizer()
    n_pri = n_total // 2   # 500 from ArXiv
    n_sec = n_total // 3   # 333 from CC-News and BillSum

    sources = [
        ("gfissore/arxiv-abstracts-2021", None,  "abstract", "scientific",  n_pri, False),
        ("cc_news",                        None,  "text",     "news",        n_sec, True),
        ("FiscalNote/billsum",             None,  "text",     "government",  n_sec, True),
    ]

    pos, neg, p_dom, n_dom, pair_ids = [], [], [], [], []
    for ds_name, config, field, domain, n_target, chunk in sources:
        _p, _n, _pd, _nd, _pids = _stream_rewrite_pairs(
            ds_name, config, field, domain,
            filter_hedging_positive, rewrite_hedging,
            n_target, tok, "hedging", len(pos), chunk,
        )
        pos += _p; neg += _n; p_dom += _pd; n_dom += _nd; pair_ids += _pids

    return pos, neg, p_dom, n_dom, pair_ids


def build_legal_formality(n_total: int):
    """
    Matched-pair corpus (rule-based rewriting from legal sources).
    3 domains: legal_us (SCOTUS), legal_eu (EurLex), government (BillSum).
    All sources capped at n_total//2 so all 3 get a chance to contribute.
    All sources require chunking — opinions/directives are thousands of tokens.
    """
    from poolbench.rewriters import rewrite_legal_formality
    tok = get_tokenizer()
    n_pri = n_total // 2   # 500 from SCOTUS
    n_sec = n_total // 3   # 333 from EurLex and BillSum

    sources = [
        ("lex_glue", "scotus",  "text",    "legal_us",    n_pri, True),
        ("lex_glue", "eurlex",  "text",    "legal_eu",    n_sec, True),
        ("FiscalNote/billsum",  None, "text",  "government", n_sec, True),
    ]

    pos, neg, p_dom, n_dom, pair_ids = [], [], [], [], []
    for ds_name, config, field, domain, n_target, chunk in sources:
        _p, _n, _pd, _nd, _pids = _stream_rewrite_pairs(
            ds_name, config, field, domain,
            filter_legal_positive, rewrite_legal_formality,
            n_target, tok, "legal", len(pos), chunk,
        )
        pos += _p; neg += _n; p_dom += _pd; n_dom += _nd; pair_ids += _pids

    return pos, neg, p_dom, n_dom, pair_ids


def build_math_certainty(n_total: int):
    """
    Independent samples (NOT matched pairs).
    Positives: hendrycks_math solutions — contain "therefore/thus/hence/QED" in proof context.
    Negatives: ArXiv abstracts — academic text without mathematical certainty markers.
    Rationale: problem texts are 15-100 tokens (median 41), far below MIN_TOKENS=300.
    Pairing solutions with problems produces ~0 valid pairs. Independent sampling is correct.
    """
    from datasets import load_dataset, get_dataset_config_names
    tok = get_tokenizer()

    pos, neg, p_dom, n_dom = [], [], [], []

    # Positives: hendrycks_math solutions (proof language with certainty markers)
    print("  Loading EleutherAI/hendrycks_math configs for math_certainty positives ...")
    for cfg in get_dataset_config_names("EleutherAI/hendrycks_math"):
        if len(pos) >= n_total:
            break
        for split_name in ("train", "test"):
            if len(pos) >= n_total:
                break
            ds = load_dataset("EleutherAI/hendrycks_math", cfg, split=split_name, streaming=True)
            for ex in ds:
                if len(pos) >= n_total:
                    break
                solution = clean_text(ex.get("solution", ""))
                if not is_valid_length(solution, tok):
                    continue
                if not filter_math_certainty_positive(solution):
                    continue
                pos.append(solution)
                p_dom.append("math")

    # Negatives: ArXiv abstracts — formal academic writing WITHOUT certainty markers
    # These describe results observationally ("we show", "results suggest") rather
    # than with deductive proof certainty ("therefore", "it follows that", "QED").
    print("  Loading gfissore/arxiv-abstracts-2021 for math_certainty negatives ...")
    ds_arxiv = load_dataset("gfissore/arxiv-abstracts-2021", split="train", streaming=True)
    for ex in ds_arxiv:
        if len(neg) >= n_total:
            break
        text = clean_text(ex.get("abstract", ""))
        if not is_valid_length(text, tok):
            continue
        if filter_math_certainty_negative(text):
            neg.append(text)
            n_dom.append("academic")

    # Third domain: CC-News — general news text without proof markers (adds "news" domain)
    print("  Loading cc_news for math_certainty negatives (news domain) ...")
    n_news_target = n_total // 3
    n_news = 0
    ds_news = load_dataset("cc_news", split="train", streaming=True)
    for ex in ds_news:
        if n_news >= n_news_target:
            break
        full_text = clean_text(ex.get("text", "") or "")
        if not full_text:
            continue
        for chunk in _chunk_tokens(full_text, tok):
            if n_news >= n_news_target:
                break
            if not is_valid_length(chunk, tok):
                continue
            if filter_math_certainty_negative(chunk):
                neg.append(chunk)
                n_dom.append("news")
                n_news += 1

    return pos, neg, p_dom, n_dom, None


def build_frustration(n_total: int):
    """
    Matched-pair corpus: frustration marker removed/softened to create neutral negative.
    3 domains: review (Yelp), social (Reddit), news (CC-News).
    """
    from datasets import load_dataset
    from poolbench.rewriters import rewrite_frustration
    from poolbench.filters import filter_frustration_positive_text
    tok = get_tokenizer()
    n_sec = n_total // 3

    pos, neg, p_dom, n_dom, pair_ids = [], [], [], [], []

    # Source 1: Yelp 1-star reviews (review domain)
    print("  Loading Yelp/yelp_review_full for frustration (review domain) ...")
    ds_yelp = load_dataset("Yelp/yelp_review_full", split="train", streaming=True)
    for ex in ds_yelp:
        text = clean_text(ex.get("text", ""))
        label = int(ex.get("label", -1))
        if label != 0 or not is_valid_length(text, tok):
            continue
        if not filter_frustration_positive_text(text):
            continue
        rewritten = rewrite_frustration(text)
        if rewritten is None or not is_valid_length(rewritten, tok):
            continue
        if not is_length_matched(text, rewritten, tok):
            continue
        pos.append(text); neg.append(rewritten)
        p_dom.append("review"); n_dom.append("review")
        pair_ids.append(f"frustration_pair_{len(pair_ids):05d}")
        if len(pos) >= n_total:
            break

    # Sources 2 & 3: Reddit (social) and CC-News (news)
    for ds_name, config, field, domain, n_target, chunk in [
        ("sentence-transformers/reddit", None, "body",  "social", n_sec, False),
        ("cc_news",                      None, "text",  "news",   n_sec, True),
    ]:
        _p, _n, _pd, _nd, _pids = _stream_rewrite_pairs(
            ds_name, config, field, domain,
            filter_frustration_positive_text, rewrite_frustration,
            n_target, tok, "frustration", len(pos), chunk,
        )
        pos += _p; neg += _n; p_dom += _pd; n_dom += _nd; pair_ids += _pids

    return pos, neg, p_dom, n_dom, pair_ids


def build_pos_sentiment(n_total: int):
    """
    3 domains: review (Yelp), movies (IMDB), product (Amazon).
    Label-based sampling — no seed-word filtering needed.
    Yelp: 5-star = pos, 1-star = neg. IMDB: label=1 = pos, label=0 = neg.
    Amazon: label=1 = pos, label=0 = neg (title+content combined for length).
    """
    from datasets import load_dataset
    tok = get_tokenizer()
    n_yelp = n_total // 2   # 500 from Yelp (reliable, high-yield)
    n_imdb = n_total // 4   # 250 from IMDB (movies domain complement)

    pos, neg, p_dom, n_dom = [], [], [], []

    print("  Loading Yelp/yelp_review_full for pos_sentiment ...")
    ds_yelp = load_dataset("Yelp/yelp_review_full", split="train", streaming=True)
    for ex in ds_yelp:
        text = clean_text(ex.get("text", ""))
        label = int(ex.get("label", -1))  # 0=1-star, 4=5-star
        if not is_valid_length(text, tok):
            continue
        if label == 4 and len(pos) < n_yelp:
            pos.append(text); p_dom.append("restaurant")
        elif label == 0 and len(neg) < n_yelp:
            neg.append(text); n_dom.append("restaurant")
        if len(pos) >= n_yelp and len(neg) >= n_yelp:
            break

    print("  Loading IMDB for pos_sentiment (movies domain) ...")
    ds_imdb = load_dataset("imdb", split="train", streaming=True)
    for ex in ds_imdb:
        text = clean_text(ex.get("text", ""))
        label = int(ex.get("label", -1))  # 1=positive, 0=negative
        if not is_valid_length(text, tok):
            continue
        if label == 1 and len(pos) < n_yelp + n_imdb:
            pos.append(text); p_dom.append("movies")
        elif label == 0 and len(neg) < n_yelp + n_imdb:
            neg.append(text); n_dom.append("movies")
        if len(pos) >= n_yelp + n_imdb and len(neg) >= n_yelp + n_imdb:
            break

    print("  Loading fancyzhx/amazon_polarity for pos_sentiment (product domain) ...")
    ds_amz = load_dataset("fancyzhx/amazon_polarity", split="train", streaming=True)
    for ex in ds_amz:
        text = clean_text((ex.get("title", "") or "") + " " + (ex.get("content", "") or ""))
        label = int(ex.get("label", -1))  # 1=positive, 0=negative
        if not is_valid_length(text, tok):
            continue
        if label == 1 and len(pos) < n_total:
            pos.append(text); p_dom.append("product")
        elif label == 0 and len(neg) < n_total:
            neg.append(text); n_dom.append("product")
        if len(pos) >= n_total and len(neg) >= n_total:
            break

    return pos, neg, p_dom, n_dom, None


def build_toxicity(n_total: int):
    """
    3 domains: restaurant (Yelp), movies (IMDB), news (CC-News).
    Positives: hostile 1-star reviews (Yelp primary, IMDB bonus) with ≥1 hostile word.
    Negatives: civil text from all 3 domains (333 each).
      CC-News replaces Amazon as 3rd domain — news chunks are civil and abundant
      at 300-500 tokens, while Amazon product reviews are mostly <300 tokens.
    """
    import re
    from datasets import load_dataset
    from poolbench.filters import filter_toxicity_negative_text
    tok = get_tokenizer()

    pos, neg, p_dom, n_dom = [], [], [], []

    _hostile_re = re.compile(
        r'\b(horrible|disgusting|pathetic|awful|atrocious|worthless|'
        r'incompetent|rude|nasty|scam|fraud|liar|cheated|ripped.?off|'
        r'disgraceful|unacceptable|offensive|abysmal|deplorable|'
        r'unprofessional|absolutely terrible|stay away|worst|disgusted|'
        r'threatening|abusive|harass|racist|slur|hate)\b',
        re.IGNORECASE,
    )

    n_neg_each = n_total // 3   # ~333 neg per domain

    # Positives: Yelp fills all n_total; IMDB adds bonus records for domain variety
    print("  Loading Yelp/yelp_review_full for toxicity ...")
    ds_yelp = load_dataset("Yelp/yelp_review_full", split="train", streaming=True)
    for ex in ds_yelp:
        if len(pos) >= n_total and len(neg) >= n_neg_each:
            break
        label = int(ex.get("label", -1))
        text = clean_text(ex.get("text", ""))
        if not is_valid_length(text, tok):
            continue
        if label == 0 and len(pos) < n_total:
            if len(_hostile_re.findall(text)) >= 1:
                pos.append(text); p_dom.append("restaurant")
        elif label in (3, 4) and len(neg) < n_neg_each:
            if filter_toxicity_negative_text(text):
                neg.append(text); n_dom.append("restaurant")

    print("  Loading IMDB for toxicity (movies domain) ...")
    ds_imdb = load_dataset("imdb", split="train", streaming=True)
    for ex in ds_imdb:
        if len(pos) >= n_total and len(neg) >= n_neg_each * 2:
            break
        text = clean_text(ex.get("text", ""))
        label = int(ex.get("label", -1))
        if not is_valid_length(text, tok):
            continue
        if label == 0 and len(pos) < n_total:
            if len(_hostile_re.findall(text)) >= 1:
                pos.append(text); p_dom.append("movies")
        elif label == 1 and len(neg) < n_neg_each * 2:
            if filter_toxicity_negative_text(text):
                neg.append(text); n_dom.append("movies")

    print("  Loading cc_news for toxicity negatives (news domain) ...")
    ds_news = load_dataset("cc_news", split="train", streaming=True)
    for ex in ds_news:
        if len(pos) >= n_total and len(neg) >= n_total:
            break
        full_text = clean_text(ex.get("text", "") or "")
        if not full_text:
            continue
        for chunk in _chunk_tokens(full_text, tok):
            if len(neg) >= n_total:
                break
            if not is_valid_length(chunk, tok):
                continue
            if filter_toxicity_negative_text(chunk):
                neg.append(chunk); n_dom.append("news")
                break  # one chunk per article

    return pos, neg, p_dom, n_dom, None


def build_depression(n_total: int):
    """
    3 domains: social_mh (Reddit mental health subreddits), social (Reddit general),
    news (CC-News — news stories about mental health with first-person accounts).
    Independent samples.
    """
    from datasets import load_dataset
    from poolbench.filters import filter_depression_positive_text, filter_depression_negative_text
    tok = get_tokenizer()
    n_sec = n_total // 3

    pos, neg, p_dom, n_dom = [], [], [], []

    # Source 1: solomonk/reddit_mental_health_posts (subreddit-based)
    print("  Loading solomonk/reddit_mental_health_posts ...")
    _DEPRESSION_SUBS = {"depression", "SuicideWatch", "Anxiety", "mentalhealth",
                        "depression_help", "MentalHealthSupport"}
    _CONTROL_SUBS = {"AskReddit", "AskScience", "todayilearned", "worldnews",
                     "technology", "science", "explainlikeimfive"}
    ds_mh = load_dataset("solomonk/reddit_mental_health_posts", split="train", streaming=True)
    for ex in ds_mh:
        text = clean_text(ex.get("body", "") or ex.get("title", ""))
        sub = ex.get("subreddit", "")
        if not is_valid_length(text, tok):
            continue
        if sub in _DEPRESSION_SUBS and len(pos) < n_total:
            pos.append(text); p_dom.append("social_mh")
        elif (sub in _CONTROL_SUBS
              and filter_depression_negative_text(text)
              and len(neg) < n_total):
            neg.append(text); n_dom.append("social_mh")
        if len(pos) >= n_total and len(neg) >= n_total:
            break
    # Fallback: keyword filter if control subs thin
    if len(neg) < n_total:
        _DEP_KW = {"depress", "suicid", "anxiet", "mental health", "therapy", "self-harm"}
        ds_mh2 = load_dataset("solomonk/reddit_mental_health_posts", split="train", streaming=True)
        for ex in ds_mh2:
            text = clean_text(ex.get("body", "") or ex.get("title", ""))
            if not is_valid_length(text, tok): continue
            lower = text.lower()
            if not any(kw in lower for kw in _DEP_KW) and len(neg) < n_total:
                neg.append(text); n_dom.append("social_mh")
            if len(neg) >= n_total: break

    # Source 2: sentence-transformers/reddit (general Reddit text-filtered)
    rp, rpd = _stream_independent("sentence-transformers/reddit", None, "body", "social",
                                   filter_depression_positive_text, n_sec, tok)
    rn, rnd = _stream_independent("sentence-transformers/reddit", None, "body", "social",
                                   filter_depression_negative_text, n_sec, tok)
    pos += rp; p_dom += rpd; neg += rn; n_dom += rnd

    # Source 3: CC-News (news coverage of mental health)
    np_, npd = _stream_independent("cc_news", None, "text", "news",
                                    filter_depression_positive_text, n_sec, tok,
                                    chunk_long_docs=True)
    nn, nnd = _stream_independent("cc_news", None, "text", "news",
                                   filter_depression_negative_text, n_sec, tok,
                                   chunk_long_docs=True)
    pos += np_; p_dom += npd; neg += nn; n_dom += nnd

    return pos, neg, p_dom, n_dom, None


def build_causation(n_total: int):
    """
    Matched-pair corpus (rewrite: causal connectives removed).
    3 domains: academic (ArXiv), news (CC-News), government (BillSum).
    Each source targeted at n_total//2 so all 3 contribute.
    """
    from poolbench.rewriters import rewrite_causation
    from poolbench.filters import filter_causation_positive as _filter_causation_pos
    tok = get_tokenizer()
    n_pri = n_total // 2   # 500 from ArXiv
    n_sec = n_total // 3   # 333 from CC-News and BillSum

    sources = [
        ("gfissore/arxiv-abstracts-2021", None, "abstract", "academic",    n_pri, False),
        ("cc_news",                        None, "text",     "news",        n_sec, True),
        ("FiscalNote/billsum",             None, "text",     "government",  n_sec, True),
    ]

    pos, neg, p_dom, n_dom, pair_ids = [], [], [], [], []
    for ds_name, config, field, domain, n_target, chunk in sources:
        _p, _n, _pd, _nd, _pids = _stream_rewrite_pairs(
            ds_name, config, field, domain,
            _filter_causation_pos, rewrite_causation,
            n_target, tok, "causation", len(pos), chunk,
        )
        pos += _p; neg += _n; p_dom += _pd; n_dom += _nd; pair_ids += _pids

    return pos, neg, p_dom, n_dom, pair_ids


def build_contrast(n_total: int):
    """
    Matched-pair corpus (rewrite: contrastive connectives removed).
    3 domains: academic (ArXiv), news (CC-News), government (BillSum).
    Each source targeted at n_total//2 so all 3 contribute.
    """
    from poolbench.rewriters import rewrite_contrast
    from poolbench.filters import filter_contrast_positive
    tok = get_tokenizer()
    n_pri = n_total // 2   # 500 from ArXiv
    n_sec = n_total // 3   # 333 from CC-News and BillSum

    sources = [
        ("gfissore/arxiv-abstracts-2021", None, "abstract", "academic",    n_pri, False),
        ("cc_news",                        None, "text",     "news",        n_sec, True),
        ("FiscalNote/billsum",             None, "text",     "government",  n_sec, True),
    ]

    pos, neg, p_dom, n_dom, pair_ids = [], [], [], [], []
    for ds_name, config, field, domain, n_target, chunk in sources:
        _p, _n, _pd, _nd, _pids = _stream_rewrite_pairs(
            ds_name, config, field, domain,
            filter_contrast_positive, rewrite_contrast,
            n_target, tok, "contrast", len(pos), chunk,
        )
        pos += _p; neg += _n; p_dom += _pd; n_dom += _nd; pair_ids += _pids

    return pos, neg, p_dom, n_dom, pair_ids


def build_conditionality(n_total: int):
    """
    Matched-pair corpus (rewrite: conditional markers removed).
    3 domains: academic (ArXiv), news (CC-News), government (BillSum).
    Each source targeted at n_total//2 so all 3 contribute.
    """
    from poolbench.rewriters import rewrite_conditionality
    from poolbench.filters import filter_conditionality_positive
    tok = get_tokenizer()
    n_pri = n_total // 2   # 500 from ArXiv
    n_sec = n_total // 3   # 333 from CC-News and BillSum

    sources = [
        ("gfissore/arxiv-abstracts-2021", None, "abstract", "academic",    n_pri, False),
        ("cc_news",                        None, "text",     "news",        n_sec, True),
        ("FiscalNote/billsum",             None, "text",     "government",  n_sec, True),
    ]

    pos, neg, p_dom, n_dom, pair_ids = [], [], [], [], []
    for ds_name, config, field, domain, n_target, chunk in sources:
        _p, _n, _pd, _nd, _pids = _stream_rewrite_pairs(
            ds_name, config, field, domain,
            filter_conditionality_positive, rewrite_conditionality,
            n_target, tok, "conditionality", len(pos), chunk,
        )
        pos += _p; neg += _n; p_dom += _pd; n_dom += _nd; pair_ids += _pids

    return pos, neg, p_dom, n_dom, pair_ids


def build_academic_tone(n_total: int):
    """
    5 domains: academic (ArXiv pos), legal (SCOTUS pos), biomedical (PubMed pos),
    social (Reddit neg), news (CC-News neg), review (Yelp neg). Independent samples.
    """
    tok = get_tokenizer()
    n_sec = n_total // 3

    pos, neg, p_dom, n_dom = [], [], [], []

    # Positives: ArXiv (academic)
    print("  Loading ArXiv for academic_tone positives ...")
    p1, pd1 = _stream_independent("gfissore/arxiv-abstracts-2021", None, "abstract",
                                   "academic", filter_academic_positive, n_total, tok)
    pos += p1; p_dom += pd1

    # Positives: SCOTUS (legal domain — formal/academic register)
    p2, pd2 = _stream_independent("lex_glue", "scotus", "text", "legal",
                                   filter_academic_positive, n_sec, tok, chunk_long_docs=True)
    pos += p2; p_dom += pd2

    # Positives: PubMed abstracts (biomedical — formal academic biomedical text)
    p3, pd3 = _stream_independent("ccdv/pubmed-summarization", None, "abstract",
                                   "biomedical", filter_academic_positive, n_sec, tok)
    pos += p3; p_dom += pd3

    # Negatives: Reddit (social)
    print("  Loading Reddit for academic_tone negatives ...")
    n1, nd1 = _stream_independent("sentence-transformers/reddit", None, "body",
                                   "social", filter_academic_negative, n_total, tok)
    neg += n1; n_dom += nd1

    # Negatives: CC-News (news)
    n2, nd2 = _stream_independent("cc_news", None, "text", "news",
                                   filter_academic_negative, n_sec, tok, chunk_long_docs=True)
    neg += n2; n_dom += nd2

    # Negatives: Yelp reviews (review domain — informal consumer text, clearly non-academic)
    n3, nd3 = _stream_independent("Yelp/yelp_review_full", None, "text",
                                   "review", filter_academic_negative, n_sec, tok)
    neg += n3; n_dom += nd3

    return pos, neg, p_dom, n_dom, None


def build_code_docs(n_total: int):
    """
    5 domains: code (CodeSearchNet pos), academic_cs (ArXiv pos), biomedical (PubMed pos),
    social (Reddit neg), news (CC-News neg), legal (SCOTUS neg). Independent samples.
    """
    tok = get_tokenizer()
    n_sec = n_total // 3

    pos, neg, p_dom, n_dom = [], [], [], []

    # Positives: CodeSearchNet — Python, Java, JavaScript, Ruby docstrings only.
    # ArXiv and PubMed were incorrectly included as positives: their abstracts
    # are academic prose, not API/function documentation.
    print("  Loading CodeSearchNet for code_docs positives (4 language configs) ...")
    n_per_lang = max(n_total // 4, 1)
    for lang in ("python", "java", "javascript", "ruby"):
        lp, lpd = _stream_independent(
            "code_search_net", lang, "func_documentation_string", "code",
            filter_code_docs_positive, n_per_lang, tok,
        )
        pos += lp; p_dom += lpd

    # Negatives: Reddit tech (social) + CC-News tech (news).
    # SCOTUS legal opinions were removed: they are formal/bureaucratic texts,
    # not casual technical explanations — they confused the register signal.
    print("  Loading Reddit for code_docs negatives ...")
    n1, nd1 = _stream_independent("sentence-transformers/reddit", None, "body",
                                   "social", filter_code_docs_negative, n_total, tok)
    neg += n1; n_dom += nd1

    n2, nd2 = _stream_independent("cc_news", None, "text", "news",
                                   filter_code_docs_negative, n_sec, tok, chunk_long_docs=True)
    neg += n2; n_dom += nd2

    return pos, neg, p_dom, n_dom, None


def build_bureaucratic(n_total: int):
    """
    3 domains: government (BillSum), legal_us (SCOTUS), social (Reddit neg), review (Yelp neg).
    Independent samples.
    """
    from datasets import load_dataset
    tok = get_tokenizer()
    n_sec = n_total // 3

    pos, neg, p_dom, n_dom = [], [], [], []

    # Positives: BillSum (government)
    print("  Loading FiscalNote/billsum for bureaucratic positives (government domain) ...")
    ds_bill = load_dataset("FiscalNote/billsum", split="train", streaming=True)
    for ex in ds_bill:
        text = clean_text(ex.get("summary", ""))
        if is_valid_length(text, tok) and filter_bureaucratic_positive(text):
            pos.append(text); p_dom.append("government")
        if len(pos) >= n_total:
            break

    # Test split top-up for BillSum (to fix 249-record shortfall)
    ds_bill_test = load_dataset("FiscalNote/billsum", split="test", streaming=True)
    for ex in ds_bill_test:
        text = clean_text(ex.get("summary", ""))
        if is_valid_length(text, tok) and filter_bureaucratic_positive(text):
            pos.append(text); p_dom.append("government")
        if len(pos) >= n_total + n_total // 4:
            break
    ds_bill_ca = load_dataset("FiscalNote/billsum", split="ca_test", streaming=True)
    for ex in ds_bill_ca:
        text = clean_text(ex.get("summary", ""))
        if is_valid_length(text, tok) and filter_bureaucratic_positive(text):
            pos.append(text); p_dom.append("government")
        if len(pos) >= n_total + n_total // 3:
            break

    # Positives: SCOTUS (legal_us domain — also highly bureaucratic formal)
    p2, pd2 = _stream_independent("lex_glue", "scotus", "text", "legal_us",
                                   filter_bureaucratic_positive, n_sec, tok,
                                   chunk_long_docs=True)
    pos += p2; p_dom += pd2

    # Positives: EurLex (legal_eu — EU legislative/regulatory text, highly bureaucratic)
    p3, pd3 = _stream_independent("lex_glue", "eurlex", "text", "legal_eu",
                                   filter_bureaucratic_positive, n_sec, tok, chunk_long_docs=True)
    pos += p3; p_dom += pd3

    # Negatives: Yelp (review)
    print("  Loading Yelp for bureaucratic negatives ...")
    ds_yelp = load_dataset("Yelp/yelp_review_full", split="train", streaming=True)
    for ex in ds_yelp:
        text = clean_text(ex.get("text", ""))
        if is_valid_length(text, tok) and filter_bureaucratic_negative(text):
            neg.append(text); n_dom.append("review")
        if len(neg) >= n_total:
            break

    # Negatives: Reddit (social)
    n2, nd2 = _stream_independent("sentence-transformers/reddit", None, "body", "social",
                                   filter_bureaucratic_negative, n_sec, tok)
    neg += n2; n_dom += nd2

    # Negatives: CC-News (news — general news, conversational register)
    n3, nd3 = _stream_independent("cc_news", None, "text", "news",
                                   filter_bureaucratic_negative, n_sec, tok, chunk_long_docs=True)
    neg += n3; n_dom += nd3

    return pos, neg, p_dom, n_dom, None


def build_uncertainty(n_total: int):
    """
    3 domains: academic (ArXiv), news (CC-News), social (Reddit).
    Independent samples.
    """
    tok = get_tokenizer()
    n_sec = n_total // 3

    pos, neg, p_dom, n_dom = [], [], [], []

    for ds_name, config, field, domain, n_target, chunk in [
        ("gfissore/arxiv-abstracts-2021", None, "abstract", "academic", n_total, False),
        ("cc_news",                        None, "text",     "news",     n_sec,   True),
        ("sentence-transformers/reddit",   None, "body",     "social",   n_sec,   False),
    ]:
        p, pd_ = _stream_independent(ds_name, config, field, domain,
                                      filter_uncertainty_positive, n_target, tok, chunk)
        n, nd_ = _stream_independent(ds_name, config, field, domain,
                                      filter_uncertainty_negative, n_target, tok, chunk)
        pos += p; p_dom += pd_; neg += n; n_dom += nd_

    return pos, neg, p_dom, n_dom, None


def build_deference(n_total: int):
    """
    3 domains: academic (ArXiv), news (CC-News), legal (SCOTUS).
    Deference markers ("previous work", "as shown by", "it has been established")
    appear in academic citations AND legal opinions citing prior cases.
    Reddit excluded: casual posts rarely contain academic attribution phrases.
    Independent samples.
    """
    from poolbench.filters import filter_deference_positive, filter_deference_negative
    tok = get_tokenizer()
    n_sec = n_total // 3

    pos, neg, p_dom, n_dom = [], [], [], []

    for ds_name, config, field, domain, n_target, chunk in [
        ("gfissore/arxiv-abstracts-2021", None, "abstract", "academic", n_total, False),
        ("cc_news",                        None, "text",     "news",     n_sec,   True),
        ("lex_glue",                        "scotus", "text", "legal",   n_sec,   True),
    ]:
        p, pd_ = _stream_independent(ds_name, config, field, domain,
                                      filter_deference_positive, n_target, tok, chunk)
        n, nd_ = _stream_independent(ds_name, config, field, domain,
                                      filter_deference_negative, n_target, tok, chunk)
        pos += p; p_dom += pd_; neg += n; n_dom += nd_

    return pos, neg, p_dom, n_dom, None


def build_planning(n_total: int):
    """
    3 domains: academic (ArXiv), news (CC-News), social (Reddit).
    Positives: texts with ≥1 forward-planning marker ("future work", "we will",
      "we plan to", "next steps", etc.). With ≥1 threshold, CC-News and Reddit
      yield planning language (business goals, political plans, personal plans).
    Negatives: ArXiv only — anti-planning markers ("we show", "we present") are
      academic phrases; news/Reddit would require multi-million scans for negatives.
    Independent samples.
    """
    from poolbench.filters import filter_planning_positive, filter_planning_negative
    tok = get_tokenizer()
    n_sec = n_total // 3

    pos, neg, p_dom, n_dom = [], [], [], []

    # Positives: 3 domains
    for ds_name, config, field, domain, n_target, chunk in [
        ("gfissore/arxiv-abstracts-2021", None, "abstract", "academic", n_total, False),
        ("cc_news",                        None, "text",     "news",     n_sec,   True),
        ("sentence-transformers/reddit",   None, "body",     "social",   n_sec,   False),
    ]:
        p, pd_ = _stream_independent(ds_name, config, field, domain,
                                      filter_planning_positive, n_target, tok, chunk)
        pos += p; p_dom += pd_

    # Negatives: ArXiv only (anti-planning markers are academic phrases)
    n, nd_ = _stream_independent("gfissore/arxiv-abstracts-2021", None, "abstract",
                                  "academic", filter_planning_negative, n_total, tok)
    neg += n; n_dom += nd_

    return pos, neg, p_dom, n_dom, None


def build_negation_density(n_total: int):
    """
    Matched-pair corpus (rewrite: negation words removed to create zero-negation negative).
    3 domains: academic (ArXiv), news (CC-News), government (BillSum).
    Fast regex filter (≥3 negation tokens) avoids slow spaCy streaming.
    Each source targeted at n_total//2 so all 3 contribute.
    """
    import re as _re
    from poolbench.rewriters import rewrite_negation_density
    tok = get_tokenizer()

    _FAST_NEG_RE = _re.compile(r"\b(not|no|never|neither|nor|without|cannot|n't)\b", _re.IGNORECASE)

    def fast_neg_positive(text: str) -> bool:
        return len(_FAST_NEG_RE.findall(text)) >= 3

    n_pri = n_total // 2   # 500 from ArXiv
    n_sec = n_total // 3   # 333 from CC-News and BillSum

    sources = [
        ("gfissore/arxiv-abstracts-2021", None, "abstract", "academic",    n_pri, False),
        ("cc_news",                        None, "text",     "news",        n_sec, True),
        ("FiscalNote/billsum",             None, "text",     "government",  n_sec, True),
    ]

    pos, neg, p_dom, n_dom, pair_ids = [], [], [], [], []
    for ds_name, config, field, domain, n_target, chunk in sources:
        _p, _n, _pd, _nd, _pids = _stream_rewrite_pairs(
            ds_name, config, field, domain,
            fast_neg_positive, rewrite_negation_density,
            n_target, tok, "negation", len(pos), chunk,
        )
        pos += _p; neg += _n; p_dom += _pd; n_dom += _nd; pair_ids += _pids

    return pos, neg, p_dom, n_dom, pair_ids


def build_numerical_precision(n_total: int):
    """
    5 domains: academic (ArXiv pos), news (CC-News pos/neg), legal (SCOTUS pos),
    social (Reddit neg), review (Yelp neg). Independent samples.
    """
    tok = get_tokenizer()
    n_sec = n_total // 3

    pos, neg, p_dom, n_dom = [], [], [], []

    # Positives: ArXiv (academic — most reliable for ≥4 numeric tokens)
    p1, pd1 = _stream_independent("gfissore/arxiv-abstracts-2021", None, "abstract",
                                   "academic", filter_numerical_positive, n_total, tok)
    pos += p1; p_dom += pd1

    # Positives: CC-News (news — data journalism, sports scores, financial results)
    p2, pd2 = _stream_independent("cc_news", None, "text", "news",
                                   filter_numerical_positive, n_sec, tok, chunk_long_docs=True)
    pos += p2; p_dom += pd2

    # Positives: SCOTUS legal opinions (legal — opinions cite statutes, case numbers, dollar amounts)
    p3, pd3 = _stream_independent("lex_glue", "scotus", "text", "legal",
                                   filter_numerical_positive, n_sec, tok, chunk_long_docs=True)
    pos += p3; p_dom += pd3

    # Negatives: CC-News (news narratives — vague quantifiers)
    n1, nd1 = _stream_independent("cc_news", None, "text", "news",
                                   filter_numerical_negative, n_total, tok, chunk_long_docs=True)
    neg += n1; n_dom += nd1

    # Negatives: Reddit (social — informal, vague quantifiers)
    n2, nd2 = _stream_independent("sentence-transformers/reddit", None, "body", "social",
                                   filter_numerical_negative, n_sec, tok)
    neg += n2; n_dom += nd2

    # Negatives: Yelp reviews (review — casual consumer text with vague quantities)
    n3, nd3 = _stream_independent("Yelp/yelp_review_full", None, "text",
                                   "review", filter_numerical_negative, n_sec, tok)
    neg += n3; n_dom += nd3

    return pos, neg, p_dom, n_dom, None


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

        # Minimum record count check
        min_expected = 100  # hard floor — anything less means the builder failed
        if len(pos_recs) < min_expected:
            print(f"  [FAIL] {concept}/{split}_pos: only {len(pos_recs)} records (minimum {min_expected})")
            ok = False
        if len(neg_recs) < min_expected:
            print(f"  [FAIL] {concept}/{split}_neg: only {len(neg_recs)} records (minimum {min_expected})")
            ok = False
        if not ok:
            continue

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
        if len(domains) < 3:
            print(f"  [WARN] {concept}/{split}: only {len(domains)} domain(s) covered (target ≥ 3)")

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
            seed_words=CONCEPTS[concept].get("contamination_markers", CONCEPTS[concept].get("seed_words", [])),
        )

        if not args.dry_run:
            validate_corpus(concept, args.corpora_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
