"""
dataset_builder.py
==================
Main corpus construction script for PoolBench.

Usage:
    # Build a single concept (recommended — run sequentially):
    python dataset_builder.py --concept hedging

    # Build all 17 concepts one by one:
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
    clean_text, is_valid_length, is_length_matched, token_count,
    make_record, save_jsonl, has_seed_words, infer_domain,
)
from poolbench.filters import (
    filter_hedging_positive, filter_hedging_negative,
    filter_legal_positive, filter_legal_negative,
    filter_frustration_positive_label, filter_frustration_negative_label,
    filter_toxicity_positive, filter_toxicity_negative,
    filter_depression_positive_label,
    filter_causation_positive_label, filter_causation_negative_label,
    filter_contrast_positive_label, filter_contrast_negative_label,
    filter_conditionality_positive_label, filter_conditionality_negative_label,
    filter_academic_positive, filter_academic_negative,
    filter_code_docs_positive, filter_code_docs_negative,
    filter_bureaucratic_positive, filter_bureaucratic_negative,
    filter_narrative_positive, filter_narrative_negative,
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


def _pick_text_field(example: dict) -> str:
    """Return the first non-empty text-like field from a dataset example."""
    for key in ("text", "sentence", "utterance", "content", "prompt", "response", "body", "comment"):
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return clean_text(value)
    return ""


def _label_name(example: dict, label_names: list[str] | None = None) -> str:
    """Normalise a dataset label to lowercase text."""
    raw = example.get("label")
    if isinstance(raw, str):
        return raw.strip().lower()
    if label_names and isinstance(raw, int) and 0 <= raw < len(label_names):
        return label_names[raw].strip().lower()
    return str(raw).strip().lower()


def _collect_top_token_examples(
    dataset_name: str,
    keep_labels: set[str],
    n_needed: int,
    tok,
    domain: str = "social",
    min_tokens: int = 8,
    max_tokens: int = 128,
) -> tuple[list[str], list[str]]:
    """
    Collect the longest examples for a labelled dataset.
    The resulting corpus is token-length filtered first, then sorted by token count
    so we preferentially keep the most information-rich short-form examples.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split="train")
    label_feature = ds.features.get("label") if getattr(ds, "features", None) else None
    label_names = getattr(label_feature, "names", None) if label_feature is not None else None

    buckets: dict[str, list[tuple[int, str]]] = {label: [] for label in keep_labels}
    for ex in ds:
        text = _pick_text_field(ex)
        if not text:
            continue
        label = _label_name(ex, label_names)
        if label not in keep_labels:
            continue
        n_tok = token_count(text, tok)
        if not (min_tokens <= n_tok <= max_tokens):
            continue
        buckets[label].append((n_tok, text))

    texts: list[str] = []
    domains: list[str] = []
    per_label_target = max(1, n_needed // max(1, len(keep_labels)))
    extra = n_needed - per_label_target * len(keep_labels)

    for idx, label in enumerate(sorted(keep_labels)):
        target = per_label_target + (1 if idx < extra else 0)
        chosen = sorted(buckets[label], key=lambda item: item[0], reverse=True)[:target]
        texts.extend(text for _, text in chosen)
        domains.extend([domain] * len(chosen))

    if len(texts) < n_needed:
        raise ValueError(
            f"{dataset_name}: only collected {len(texts)} examples "
            f"for labels {sorted(keep_labels)} within token range [{min_tokens}, {max_tokens}]"
        )

    # Keep the longest available examples overall for the final cap.
    paired = sorted(zip(texts, domains), key=lambda item: token_count(item[0], tok), reverse=True)
    paired = paired[:n_needed]
    texts = [p[0] for p in paired]
    domains = [p[1] for p in paired]
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


def build_frustration(n_total: int):
    """
    Independent natural sampling — NOT matched pairs.
    Positives: Yelp 1-star reviews (review) + Reddit angry posts (social).
    Negatives: Yelp 3-star reviews (review, same platform → controls topic distribution)
               + CC-News (news).
    Matched-pair rewrites were dropped: stripping 2–3 rare frustration words from a
    400-token passage creates near-identical pairs that inflate probe AUROC artificially.
    `_stream_independent` is used throughout — Yelp label filtering happens via the
    text-based filter, which already requires frustration markers for positives and
    bans them from negatives.
    """
    from poolbench.filters import filter_frustration_positive_text, filter_frustration_negative_text
    tok = get_tokenizer()
    n_half = (n_total + 1) // 2

    pos, neg, p_dom, n_dom = [], [], [], []

    # ── Positives: Yelp 1-star reviews (review domain) ───────────────────────
    print("  Loading Yelp/yelp_review_full for frustration positives (review) ...")
    p1, pd1 = _stream_independent(
        "Yelp/yelp_review_full", None, "text", "review",
        filter_frustration_positive_text, n_half, tok,
    )
    pos += p1; p_dom += pd1

    # ── Positives: Reddit (social domain) ────────────────────────────────────
    if len(pos) < n_total:
        print("  Loading sentence-transformers/reddit for frustration positives (social) ...")
        p2, pd2 = _stream_independent(
            "sentence-transformers/reddit", None, "body", "social",
            filter_frustration_positive_text, n_total - len(pos), tok,
        )
        pos += p2; p_dom += pd2

    # ── Negatives: Yelp reviews (neutral tone, same platform) ────────────────
    print("  Loading Yelp/yelp_review_full for frustration negatives (review) ...")
    n1, nd1 = _stream_independent(
        "Yelp/yelp_review_full", None, "text", "review",
        filter_frustration_negative_text, n_half, tok,
    )
    neg += n1; n_dom += nd1

    # ── Negatives: CC-News (news domain) ─────────────────────────────────────
    if len(neg) < n_total:
        print("  Loading cc_news for frustration negatives (news) ...")
        n2, nd2 = _stream_independent(
            "cc_news", None, "text", "news",
            filter_frustration_negative_text, n_total - len(neg), tok,
            chunk_long_docs=True,
        )
        neg += n2; n_dom += nd2

    return pos, neg, p_dom, n_dom, None


def build_imdb_sentiment(n_total: int):
    """
    Single-source IMDb sentiment corpus.

    Positive and negative passages are drawn from the same review dataset so the
    benchmark measures sentiment polarity rather than source/domain shift.
    HTML break artifacts (e.g. <br /><br />) are stripped by clean_text().
    """
    from datasets import load_dataset

    tok = get_tokenizer()
    pos, neg, p_dom, n_dom = [], [], [], []

    def _is_positive(label) -> bool:
        if isinstance(label, str):
            norm = label.strip().lower()
            return norm in {"positive", "pos", "1", "true"}
        try:
            return int(label) == 1
        except Exception:
            return False

    def _is_negative(label) -> bool:
        if isinstance(label, str):
            norm = label.strip().lower()
            return norm in {"negative", "neg", "0", "false"}
        try:
            return int(label) == 0
        except Exception:
            return False

    print("  Loading yin001/imdb_dataset_positive_negative for imdb_sentiment ...")
    ds_imdb = load_dataset("yin001/imdb_dataset_positive_negative", split="train", streaming=True)
    for ex in ds_imdb:
        raw = ex.get("review", ex.get("text", "")) or ""
        lbl = ex.get("sentiment", ex.get("label", ex.get("labels", None)))
        if not raw or not is_valid_length(clean_text(raw), tok):
            continue
        text = clean_text(raw)
        if _is_positive(lbl) and len(pos) < n_total:
            pos.append(text); p_dom.append("review")
        elif _is_negative(lbl) and len(neg) < n_total:
            neg.append(text); n_dom.append("review")
        if len(pos) >= n_total and len(neg) >= n_total:
            break

    return pos, neg, p_dom, n_dom, None


def build_toxicity(n_total: int):
    """
        3 domains: online (surge-ai), comments (civil_comments), twitter (Davidson hate speech).
        Positives: explicitly labeled toxic/hate speech/offensive or strict high-confidence
            Civil Comments samples with hostile-text confirmation.
        Negatives: explicitly labeled non-toxic/neither or strict low-toxicity Civil Comments
            samples with no hostile-text marker.
    Sources:
      1. surge-ai toxicity CSV — broad English toxic content, binary labeled.
      2. tdavidson/hate_speech_offensive — Twitter data; class 0=hate_speech, 1=offensive (POS);
         class 2=neither (NEG). Provides the third domain (twitter) and explicit slurs/hate content.
      3. google/civil_comments — score-based fallback to fill any shortfall.
    """
    import pandas as pd
    import hashlib
    from datasets import load_dataset
    from poolbench.filters import filter_toxicity_positive_text
    tok = get_tokenizer()
    n_sec = n_total // 3   # ~333 per source

    pos, neg, p_dom, n_dom = [], [], [], []
    seen_pos, seen_neg = set(), set()

    def _len_ok(text: str, min_tokens: int) -> bool:
        n_tok = token_count(text, tok)
        return min_tokens <= n_tok <= 500

    def _domain_count(domains: list[str], name: str) -> int:
        return sum(1 for d in domains if d == name)

    def _add_pos(text: str, domain: str) -> bool:
        h = hashlib.md5(text.encode()).hexdigest()
        if h in seen_pos:
            return False
        seen_pos.add(h)
        pos.append(text)
        p_dom.append(domain)
        return True

    def _add_neg(text: str, domain: str) -> bool:
        h = hashlib.md5(text.encode()).hexdigest()
        if h in seen_neg:
            return False
        seen_neg.add(h)
        neg.append(text)
        n_dom.append(domain)
        return True

    # Source 1: surge-ai toxicity CSV (binary labeled — "online" domain)
    _SURGE_URL = (
        "https://raw.githubusercontent.com/surge-ai/toxicity/main/toxicity_en.csv"
    )
    print("  Loading surge-ai toxicity CSV ...")
    try:
        df_surge = pd.read_csv(_SURGE_URL)
        df_surge.columns = [c.strip().lower() for c in df_surge.columns]
        for _, row in df_surge.iterrows():
            text = clean_text(str(row.get("text", "") or ""))
            if not _len_ok(text, 30):
                continue
            label = str(row.get("is_toxic", "")).strip().lower()
            if label in {"toxic", "1", "true"} and len(pos) < n_sec:
                _add_pos(text, "online")
            elif label in {"not toxic", "0", "false"} and len(neg) < n_sec:
                _add_neg(text, "online")
    except Exception as e:
        print(f"  [WARN] surge-ai CSV load failed ({e})")

    # Source 2: Davidson hate speech (Twitter — adds "twitter" domain, solves 3-domain requirement)
    # class 0 = hate speech (slurs, targeted hate), class 1 = offensive language,
    # class 2 = neither (clean tweets). Using class 0+1 as POS, class 2 as NEG.
    print("  Loading tdavidson/hate_speech_offensive for toxicity (twitter domain) ...")
    ds_tw = load_dataset("tdavidson/hate_speech_offensive", split="train", streaming=True)
    for ex in ds_tw:
        if _domain_count(p_dom, "twitter") >= n_sec and _domain_count(n_dom, "twitter") >= n_sec:
            break
        text = clean_text(ex.get("tweet", "") or "")
        # Strip RT prefix noise and HTML entities
        text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        if not _len_ok(text, 15):   # tweets are short; use 15-token floor
            continue
        cls = int(ex.get("class", 2))
        if cls in (0, 1) and _domain_count(p_dom, "twitter") < n_sec:
            _add_pos(text, "twitter")
        elif cls == 2 and _domain_count(n_dom, "twitter") < n_sec:
            _add_neg(text, "twitter")

    # Source 3: civil_comments with stricter intent guardrails (comments domain)
    # High score alone is insufficient: neutral/critical discussion can score high.
    # Require hostile-text confirmation for positives and explicit hostility absence for negatives.
    print("  Loading google/civil_comments for strict toxicity comments domain ...")
    ds = load_dataset("google/civil_comments", split="train", streaming=True)
    for ex in ds:
        if _domain_count(p_dom, "comments") >= n_sec and _domain_count(n_dom, "comments") >= n_sec:
            break
        text = clean_text(ex.get("text", "") or "")
        if not _len_ok(text, 30):
            continue
        score = float(ex.get("toxicity", 0.0))
        is_hostile_text = filter_toxicity_positive_text(text)
        if score >= 0.7 and is_hostile_text and _domain_count(p_dom, "comments") < n_sec:
            _add_pos(text, "comments")
        elif score <= 0.1 and (not is_hostile_text) and _domain_count(n_dom, "comments") < n_sec:
            _add_neg(text, "comments")

    # Shortfall top-up: prioritize explicit-label sources (surge-ai, davidson),
    # then strict civil_comments fallback.
    if len(pos) < n_total or len(neg) < n_total:
        print("  Top-up from explicit-label sources for toxicity shortfall ...")

        # surge-ai top-up
        try:
            for _, row in df_surge.iterrows():
                if len(pos) >= n_total and len(neg) >= n_total:
                    break
                text = clean_text(str(row.get("text", "") or ""))
                if not _len_ok(text, 30):
                    continue
                label = str(row.get("is_toxic", "")).strip().lower()
                if label in {"toxic", "1", "true"} and len(pos) < n_total:
                    _add_pos(text, "online")
                elif label in {"not toxic", "0", "false"} and len(neg) < n_total:
                    _add_neg(text, "online")
        except Exception:
            pass

        # davidson top-up
        if len(pos) < n_total or len(neg) < n_total:
            ds_tw_topup = load_dataset("tdavidson/hate_speech_offensive", split="train", streaming=True)
            for ex in ds_tw_topup:
                if len(pos) >= n_total and len(neg) >= n_total:
                    break
                text = clean_text(ex.get("tweet", "") or "")
                text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
                if not _len_ok(text, 15):
                    continue
                cls = int(ex.get("class", 2))
                if cls in (0, 1) and len(pos) < n_total:
                    _add_pos(text, "twitter")
                elif cls == 2 and len(neg) < n_total:
                    _add_neg(text, "twitter")

        # strict civil_comments final fallback
        if len(pos) < n_total or len(neg) < n_total:
            ds_civil_topup = load_dataset("google/civil_comments", split="train", streaming=True)
            for ex in ds_civil_topup:
                if len(pos) >= n_total and len(neg) >= n_total:
                    break
                text = clean_text(ex.get("text", "") or "")
                if not _len_ok(text, 30):
                    continue
                score = float(ex.get("toxicity", 0.0))
                is_hostile_text = filter_toxicity_positive_text(text)
                if score >= 0.7 and is_hostile_text and len(pos) < n_total:
                    _add_pos(text, "comments")
                elif score <= 0.1 and (not is_hostile_text) and len(neg) < n_total:
                    _add_neg(text, "comments")

    return pos, neg, p_dom, n_dom, None


def build_depression(n_total: int):
    """
    Hybrid Reddit corpus.
    Positive = depression examples from mrjunos/depression-reddit-cleaned.
    Negative = general Reddit comments from dlb/mentalreddit, filtered to remove
    explicit depression markers.
    This concept uses a 200-500 token window and then keeps the longest valid
    examples so the split stays dense-lexical and length-controlled.
    """
    from datasets import load_dataset
    from poolbench.filters import filter_depression_negative_general_text
    tok = get_tokenizer()
    pos, neg, p_dom, n_dom = [], [], [], []

    pos_url = (
        "https://huggingface.co/datasets/mrjunos/"
        "depression-reddit-cleaned/resolve/main/depression_reddit_cleaned_ds.csv"
    )
    print("  Loading mrjunos/depression-reddit-cleaned ...")
    ds_pos = load_dataset(
        "csv",
        data_files=pos_url,
        split="train",
    )

    pos_candidates: list[tuple[int, str]] = []
    for ex in ds_pos:
        text = clean_text(_pick_text_field(ex))
        if not text:
            continue
        n_tok = token_count(text, tok)
        if not (200 <= n_tok <= 500):
            continue
        raw_label = ex.get("labels", ex.get("label"))
        if raw_label is None:
            continue
        label = str(raw_label).strip().lower()
        if filter_depression_positive_label(label):
            pos_candidates.append((n_tok, text))

    # ── Supplementary positives: second pass of mrjunos with wider window ───
    # mrjunos/depression-reddit-cleaned yields ~479 in strict 200-500 window.
    # For the remaining ~221, widen to 180-520 tokens — a ±20 token relaxation
    # that stays well within the original spirit of the length constraint and
    # avoids scanning a multi-billion-record streaming corpus.
    if len(pos_candidates) < n_total:
        import hashlib as _hl
        _needed = n_total - len(pos_candidates)
        print(f"  Second pass for {_needed} more depression positives (window 180-520) ...")
        _seen = {_hl.md5(t.encode()).hexdigest() for _, t in pos_candidates}
        # Re-load mrjunos (already cached locally after first pass)
        ds_pass2 = load_dataset("csv", data_files=pos_url, split="train")
        extras: list[tuple[int, str]] = []
        for ex in ds_pass2:
            text = clean_text(_pick_text_field(ex))
            if not text:
                continue
            n_tok = token_count(text, tok)
            if not (180 <= n_tok <= 520):
                continue
            if n_tok in range(200, 501):
                continue  # already captured in first pass
            raw_label = ex.get("labels", ex.get("label"))
            if raw_label is None:
                continue
            if not filter_depression_positive_label(str(raw_label).strip().lower()):
                continue
            h = _hl.md5(text.encode()).hexdigest()
            if h in _seen:
                continue
            _seen.add(h)
            extras.append((n_tok, text))
        extras.sort(key=lambda item: item[0], reverse=True)
        pos_candidates.extend(extras[:_needed])

    pos_candidates.sort(key=lambda item: item[0], reverse=True)
    pos = [text for _, text in pos_candidates[:n_total]]
    p_dom = ["social"] * len(pos)

    target = n_total
    if len(pos) == 0:
        raise ValueError("depression: no positive examples found within the 200-500 token window")

    print("  Loading dlb/mentalreddit for depression negatives ...")
    ds_neg = load_dataset("dlb/mentalreddit", split="train", streaming=True)
    allowed_general_subs = {
        "askreddit", "askscience", "casualconversation", "daddit", "explainlikeimfive",
        "legaladvice", "movies", "news", "nostupidquestions", "personalfinance",
        "politics", "showerthoughts", "technology", "todayilearned", "worldnews",
    }
    excluded_substrs = {
        "depress", "mental", "anxiet", "suicid", "therapy", "bipolar", "ptsd",
        "selfharm", "self-harm", "ocd", "adhd", "autism", "schiz", "eating",
        "recovery", "addict",
    }

    neg_candidates: list[tuple[int, str]] = []
    neg_cap = max(target * 3, 1200)
    for ex in ds_neg:
        subreddit = str(ex.get("subreddit", "")).strip().lower()
        if subreddit in allowed_general_subs:
            pass
        elif any(mark in subreddit for mark in excluded_substrs):
            continue

        text = clean_text(_pick_text_field(ex))
        if not text or not filter_depression_negative_general_text(text):
            continue
        n_tok = token_count(text, tok)
        if not (200 <= n_tok <= 500):
            continue
        neg_candidates.append((n_tok, text))
        if len(neg_candidates) >= neg_cap:
            break

    if len(neg_candidates) < target:
        raise ValueError(
            f"depression: only {len(neg_candidates)} negative candidates found, "
            f"need at least {target} to match the positive pool"
        )

    neg_candidates.sort(key=lambda item: item[0], reverse=True)
    neg = [text for _, text in neg_candidates[:target]]
    n_dom = ["social"] * len(neg)

    # Deduplicate before handing the corpus to the generic splitter so the
    # depression split ratio is computed on the true post-clean candidate pool.
    import hashlib

    def _dedup_pairs(pairs: list[tuple[int, str]]) -> list[tuple[int, str]]:
        seen: set[str] = set()
        out: list[tuple[int, str]] = []
        for score, text in pairs:
            h = hashlib.md5(text.encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            out.append((score, text))
        return out

    pos_candidates = _dedup_pairs(pos_candidates)
    neg_candidates = _dedup_pairs(neg_candidates)
    pos_hashes = {hashlib.md5(text.encode()).hexdigest() for _, text in pos_candidates}
    neg_candidates = [
        (score, text)
        for score, text in neg_candidates
        if hashlib.md5(text.encode()).hexdigest() not in pos_hashes
    ]
    pos = [text for _, text in pos_candidates]
    neg = [text for _, text in neg_candidates[:target]]
    p_dom = ["social"] * len(pos)
    n_dom = ["social"] * len(neg)

    print(
        f"  Depression candidates after label + length filtering: "
        f"{len(pos)} pos / {len(neg)} neg"
    )

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

    # Positives: PubMedQA long answers (biomedical — formal medical academic writing)
    # Uses pqa_artificial (211k examples) for volume; long_answer field contains
    # the formal biomedical conclusion text, typically 2-4 sentences of dense
    # academic prose with methodology and result language.
    print("  Loading qiaojin/PubMedQA for academic_tone positives (biomedical domain) ...")
    ds_pubmedqa = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train", streaming=True)
    _p3, _pd3 = [], []
    for ex in ds_pubmedqa:
        if len(_p3) >= n_sec:
            break
        text = clean_text(ex.get("long_answer", "") or "")
        if not text:
            # Fall back to first context passage
            ctx = ex.get("context", {})
            ctxs = ctx.get("contexts", []) if isinstance(ctx, dict) else []
            text = clean_text(ctxs[0]) if ctxs else ""
        if not text or not is_valid_length(text, tok):
            continue
        if filter_academic_positive(text):
            _p3.append(text)
            _pd3.append("biomedical")
    pos += _p3; p_dom += _pd3

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


def build_narrative(n_total: int):
    """
    fiction vs factual prose — 3 domains.
    Positives: euclaise/writingprompts story field (fiction domain).
    Negatives: wikimedia/wikipedia (encyclopaedia domain) + CC-News (news) to fill shortfall.
    Wikipedia articles about fictional works are excluded via filter_narrative_negative.
    """
    from datasets import load_dataset
    tok = get_tokenizer()

    pos, p_dom = [], []
    neg, n_dom = [], []
    n_half = (n_total + 1) // 2

    # ── Positives: WritingPrompts stories ────────────────────────────────────
    print("  Loading euclaise/writingprompts for narrative positives (source 1/1) ...")
    ds_wp = load_dataset("euclaise/writingprompts", split="train", streaming=True)
    for ex in ds_wp:
        if len(pos) >= n_total:
            break
        story = clean_text(ex.get("story", "") or "")
        if not story:
            continue
        # Chunk if the story is very long; keep chunks that hit the token window
        candidates = _chunk_tokens(story, tok)
        for chunk in candidates:
            if not is_valid_length(chunk, tok):
                continue
            if filter_narrative_positive(chunk):
                pos.append(chunk)
                p_dom.append("fiction")
            if len(pos) >= n_total:
                break

    # ── Negatives: Wikipedia (encyclopaedia) ─────────────────────────────────
    print("  Loading wikimedia/wikipedia for narrative negatives (source 1/2) ...")
    n1, nd1 = _stream_independent(
        "wikimedia/wikipedia", "20231101.en", "text", "encyclopaedia",
        filter_narrative_negative, n_half, tok, chunk_long_docs=True,
    )
    neg += n1; n_dom += nd1

    # ── Negatives: CC-News (news) — fills any shortfall ──────────────────────
    if len(neg) < n_total:
        print("  Loading cc_news for narrative negatives (source 2/2) ...")
        n2, nd2 = _stream_independent(
            "cc_news", None, "text", "news",
            filter_narrative_negative, n_total - len(neg), tok,
            chunk_long_docs=True,
        )
        neg += n2; n_dom += nd2

    return pos, neg, p_dom, n_dom, None


def build_deference(n_total: int):
    """
    Two-source corpus for domain diversity.
    Source 1: Intel/polite-guard — customer-service sentence-level politeness
              (positive = polite + somewhat polite; negative = neutral + impolite).
    Source 2: allenai/scicite — citation intent labels;
              background = deferring to prior authority (positive),
              result = asserting own findings (negative).
    Uses a relaxed token window (8-128) because both sources are sentence-level.
    """
    from datasets import load_dataset
    tok = get_tokenizer()
    tok_min, tok_max = CONCEPTS["deference"].get("token_range", [8, 128])

    # Source 1: Intel/polite-guard (customer_service domain)
    pos, p_dom = _collect_top_token_examples(
        "Intel/polite-guard",
        {"polite", "somewhat polite"},
        n_total,
        tok,
        domain="customer_service",
        min_tokens=tok_min,
        max_tokens=tok_max,
    )
    neg, n_dom = _collect_top_token_examples(
        "Intel/polite-guard",
        {"neutral", "impolite"},
        n_total,
        tok,
        domain="customer_service",
        min_tokens=tok_min,
        max_tokens=tok_max,
    )

    # Source 2: academic-domain deference signal
    # Try allenai/scicite first (background=deferring, result=asserting).
    # HuggingFace deprecated custom dataset scripts, so fall back to streaming
    # sentence-transformers/reddit filtered by text markers when scicite fails.
    print("  Loading academic source for deference (academic_citation domain) ...")
    from poolbench.data.filters import filter_deference_positive as _fdp, filter_deference_negative as _fdn
    sci_pos_cands: list[tuple[int, str]] = []
    sci_neg_cands: list[tuple[int, str]] = []
    _sci_loaded = False
    try:
        ds_sci = load_dataset("allenai/scicite", split="train", trust_remote_code=True)
        sci_label_names = getattr(ds_sci.features.get("label"), "names", None)
        for ex in ds_sci:
            text = (ex.get("string") or "").strip()
            if not text:
                continue
            n_tok = token_count(text, tok)
            if not (tok_min <= n_tok <= tok_max):
                continue
            lname = _label_name(ex, sci_label_names)
            if lname == "background":
                sci_pos_cands.append((n_tok, text))
            elif lname == "result":
                sci_neg_cands.append((n_tok, text))
        _sci_loaded = bool(sci_pos_cands)
    except Exception as e:
        print(f"  [WARN] allenai/scicite unavailable ({e}). Falling back to Reddit academic text.")

    if not _sci_loaded:
        # Fallback: stream Reddit and keep posts that contain academic citation
        # language ("previous work", "building on", "we show", etc.).
        # r/AskScience, r/MachineLearning, r/science naturally produce these.
        print("  Loading sentence-transformers/reddit for deference academic fallback ...")
        ds_reddit_sci = load_dataset("sentence-transformers/reddit", split="train", streaming=True)
        _cap = n_total * 5
        _scanned = 0
        for ex in ds_reddit_sci:
            text = clean_text(ex.get("body", ex.get("text", "")) or "")
            if not text:
                _scanned += 1
                if _scanned > _cap:
                    break
                continue
            n_tok = token_count(text, tok)
            if not (tok_min <= n_tok <= tok_max):
                _scanned += 1
                if _scanned > _cap:
                    break
                continue
            if _fdp(text):
                sci_pos_cands.append((n_tok, text))
            elif _fdn(text):
                sci_neg_cands.append((n_tok, text))
            _scanned += 1
            if len(sci_pos_cands) >= n_total and len(sci_neg_cands) >= n_total:
                break
            if _scanned > _cap:
                break

    sci_pos_cands.sort(key=lambda x: x[0], reverse=True)
    sci_neg_cands.sort(key=lambda x: x[0], reverse=True)
    _dom2 = "academic_citation" if _sci_loaded else "social_academic"
    if sci_pos_cands:
        pos += [t for _, t in sci_pos_cands[:n_total]]
        p_dom += [_dom2] * len(sci_pos_cands[:n_total])
    if sci_neg_cands:
        neg += [t for _, t in sci_neg_cands[:n_total]]
        n_dom += [_dom2] * len(sci_neg_cands[:n_total])

    return pos, neg, p_dom, n_dom, None


def build_planning(n_total: int):
    """
    Positives: WikiHow `title`+`text` concatenation — naturally planning/goal-directed.
    Negatives: Reddit (social) + CC-News (news) chunks with planning markers absent.
    Yelp is intentionally excluded — its short reviews rarely pass the 300-token floor.
    """
    from poolbench.filters import filter_planning_negative
    from datasets import load_dataset
    tok = get_tokenizer()

    pos, neg, p_dom, n_dom = [], [], [], []
    n_half = (n_total + 1) // 2

    # ── Positives: WikiHow (concatenate title + steps text) ──────────────────
    print("  Loading wikihow/wikihow for planning positives (source 1/1) ...")
    try:
        ds_wikihow = load_dataset("wikihow/wikihow", "all", split="train", streaming=True, trust_remote_code=True)
        title_field, body_field = "title", "text"
    except Exception:
        ds_wikihow = load_dataset("gursi26/wikihow-cleaned", split="train", streaming=True)
        title_field, body_field = "title", "text"

    for ex in ds_wikihow:
        title = clean_text(ex.get(title_field, "") or "")
        body  = clean_text(ex.get(body_field,  ex.get("summary", "")) or "")
        text  = (title + " " + body).strip() if title else body
        if not text or not is_valid_length(text, tok):
            continue
        pos.append(text); p_dom.append("howto")
        if len(pos) >= n_total:
            break

    # ── Negatives: Reddit (social) ───────────────────────────────────────────
    print("  Loading sentence-transformers/reddit for planning negatives (source 1/2) ...")
    n1, nd1 = _stream_independent(
        "sentence-transformers/reddit", None, "body", "social",
        filter_planning_negative, n_half, tok,
    )
    neg += n1; n_dom += nd1

    # ── Negatives: CC-News (news) — fills any shortfall ──────────────────────
    if len(neg) < n_total:
        print("  Loading cc_news for planning negatives (source 2/2) ...")
        n2, nd2 = _stream_independent(
            "cc_news", None, "text", "news",
            filter_planning_negative, n_total - len(neg), tok,
            chunk_long_docs=True,
        )
        neg += n2; n_dom += nd2

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
    5 domains: academic (ArXiv pos), math (MetaMathQA pos), news (CC-News pos/neg),
    legal (SCOTUS pos), social (Reddit neg), review (Yelp neg). Independent samples.
    MetaMathQA 'response' field = step-by-step math solutions with explicit plain-text
    numbers (no LaTeX formatting), providing cleaner numeric signal than ArXiv.
    """
    from datasets import load_dataset
    tok = get_tokenizer()
    n_sec = n_total // 3

    pos, neg, p_dom, n_dom = [], [], [], []

    # Positives: MetaMathQA responses (math — step-by-step word-problem solutions
    # with explicit plain-text numbers like "120 apples", "45%", "3.14").
    # These avoid the LaTeX-formatting issue that affects ArXiv positives.
    print("  Loading meta-math/MetaMathQA for numerical_precision positives (math domain) ...")
    ds_meta = load_dataset("meta-math/MetaMathQA", split="train", streaming=True)
    _p0, _pd0 = [], []
    for ex in ds_meta:
        if len(_p0) >= n_total:
            break
        text = clean_text(ex.get("response", "") or "")
        if not is_valid_length(text, tok):
            continue
        if filter_numerical_positive(text):
            _p0.append(text); _pd0.append("math")
    pos += _p0; p_dom += _pd0

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
    "frustration":         build_frustration,
    "imdb_sentiment":      build_imdb_sentiment,
    "toxicity":            build_toxicity,
    "depression":          build_depression,
    "causation":           build_causation,
    "contrast":            build_contrast,
    "conditionality":      build_conditionality,
    "academic_tone":       build_academic_tone,
    "code_docs":           build_code_docs,
    "bureaucratic":        build_bureaucratic,
    "narrative":           build_narrative,
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
        tok_min, tok_max = meta.get("token_range", [300, 500])
        bad_pos = [r for r in pos_recs if not (tok_min <= r["token_count"] <= tok_max)]
        bad_neg = [r for r in neg_recs if not (tok_min <= r["token_count"] <= tok_max)]
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
        min_domains = meta.get("min_domains", 3)
        if len(domains) < min_domains:
            print(f"  [WARN] {concept}/{split}: only {len(domains)} domain(s) covered (target ≥ {min_domains})")

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
                       help="Build all 17 concept corpora sequentially")
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

        if concept == "depression":
            available = min(len(pos), len(neg))
            n_train = (available * 7) // 10
            n_test = available - n_train
            print(
                f"  [depression split] using {available} examples per class "
                f"({n_train} train / {n_test} test, ~70/30)"
            )
        else:
            n_train = args.n_train
            n_test = args.n_test

        split_and_save(
            concept=concept,
            pos_passages=pos,
            neg_passages=neg,
            pos_domain=pos_domain,
            neg_domain=neg_domain,
            n_train=n_train,
            n_test=n_test,
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
