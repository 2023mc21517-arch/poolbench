"""
src/pooling_strategies.py
All 19 ranked pooling strategies + 2 oracle strategies.
Strategy registry and batch application helper.

19 ranked strategies on the main leaderboard:
  P1, P2 (2) + A1–A6 (6) + W1–W4 (4) + S1/S3/S4 (3) + L1–L4 (4) = 19

Off-leaderboard (oracle / concat tier):
  G1_human_span, G2_IxG, P3_first_last_concat
"""

from __future__ import annotations
import numpy as np
import spacy
from functools import lru_cache

# Load spaCy once at module level (disabled components that aren't needed for pooling)
@lru_cache(maxsize=1)
def _nlp():
    return spacy.load("en_core_web_sm", disable=["lemmatizer"])


# ── POSITION-ANCHORED ────────────────────────────────────────────────────────

def pool_last_token(h: np.ndarray) -> np.ndarray:
    """P1 — Last token position."""
    return h[-1]


def pool_first_token(h: np.ndarray) -> np.ndarray:
    """P2 — First token position (CLS / BOS token)."""
    return h[0]


def pool_first_last_concat(h: np.ndarray) -> np.ndarray:
    """
    P3 — Concatenation of first and last token.
    NOTE: output is 2×d_model — excluded from the main ranked leaderboard.
    Probe/SCP/cosine heatmap all operate in a different embedding space.
    Reported in a separate appendix table only.
    """
    return np.concatenate([h[0], h[-1]])


# ── UNIFORM AGGREGATION ──────────────────────────────────────────────────────

def pool_mean(h: np.ndarray) -> np.ndarray:
    """A1 — Mean pooling (industry default)."""
    return h.mean(axis=0)


def pool_sum(h: np.ndarray) -> np.ndarray:
    """A2 — Sum pooling."""
    return h.sum(axis=0)


def pool_max(h: np.ndarray) -> np.ndarray:
    """A3 — Element-wise max pooling."""
    return h.max(axis=0)


def pool_min(h: np.ndarray) -> np.ndarray:
    """A4 — Element-wise min pooling."""
    return h.min(axis=0)


def pool_median(h: np.ndarray) -> np.ndarray:
    """A5 — Median pooling."""
    return np.median(h, axis=0)


def pool_random(h: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    A6 — Random token sampling (noise-floor baseline).
    Selects a uniformly random 50% subset of token positions and averages.
    Seed fixed at 42 for reproducibility. Any strategy that does not significantly
    outperform A6_random is not performing meaningful token selection.
    """
    rng = np.random.default_rng(seed)
    n_select = max(1, len(h) // 2)
    idx = rng.choice(len(h), size=n_select, replace=False)
    return h[idx].mean(axis=0)


# ── WINDOW ───────────────────────────────────────────────────────────────────

def pool_mean_last_4(h: np.ndarray) -> np.ndarray:
    """W1 — Mean of last 4 tokens."""
    return h[-4:].mean(axis=0)


def pool_mean_last_8(h: np.ndarray) -> np.ndarray:
    """W2 — Mean of last 8 tokens."""
    return h[-8:].mean(axis=0)


def pool_mean_last_16(h: np.ndarray) -> np.ndarray:
    """W3 — Mean of last 16 tokens."""
    return h[-16:].mean(axis=0)


def pool_hierarchical(h: np.ndarray, n_chunks: int = 4) -> np.ndarray:
    """W4 — Hierarchical: split into n_chunks, mean each chunk, then mean chunk means."""
    chunk_size = max(1, len(h) // n_chunks)
    chunks = [h[i : i + chunk_size].mean(axis=0) for i in range(0, len(h), chunk_size)]
    return np.stack(chunks).mean(axis=0)


# ── SALIENCY-WEIGHTED ────────────────────────────────────────────────────────

def pool_attention_weighted(h: np.ndarray, attn_weights: np.ndarray | None) -> np.ndarray:
    """
    S1 — Attention-weighted mean pooling.
    attn_weights: (seq_len,) mean inflow attention per token, pre-computed from the
    per-head attention matrices captured during extraction.
    Falls back to pool_mean if attn_weights is None (Mamba2 / encoder-decoder).
    """
    if attn_weights is None:
        return pool_mean(h)
    w = attn_weights / (attn_weights.sum() + 1e-9)
    return (w[:, None] * h).sum(axis=0)


def pool_SIF_adapted(h: np.ndarray, token_ids: list[int], unigram_probs: dict,
                     a: float = 1e-3) -> np.ndarray:
    """
    S3 — Smoothed Inverse Frequency pooling (Arora et al. 2017).
    Weight each token by a / (a + p(token)) where p(token) is its corpus unigram prob.
    Tokens with low unigram probability (rare words) receive higher weight.
    First-PC subtraction is done at the corpus level in compute_all_pooling_strategies,
    not per-passage.
    unigram_probs: {token_id: float} pre-computed from the PoolBench training corpora.
    Falls back to pool_mean if unigram_probs is empty.
    """
    if not unigram_probs:
        return pool_mean(h)
    weights = np.array([a / (a + unigram_probs.get(tid, a)) for tid in token_ids],
                       dtype=np.float32)
    weights = weights / (weights.sum() + 1e-9)
    return (weights[:, None] * h).sum(axis=0)


def pool_attn_head(h: np.ndarray, attn_weights_per_head: np.ndarray | None,
                   top_k_heads: int = 20) -> np.ndarray:
    """
    S4 — ITI-inspired (variance proxy) per-head attention pooling.
    Inspired by Inference-Time Intervention (Li et al., NeurIPS 2023).

    APPROXIMATION NOTE: The original ITI paper selects heads by probing *accuracy* on
    labeled data (supervised). This implementation uses attention-weight *variance* as an
    unsupervised proxy. Labeled "S4_attn_head (ITI-inspired proxy)" throughout the paper.
    Validation against pool_attn_head_ITI_exact() is REQUIRED (Spearman ρ ≥ 0.80 on
    hedging, pos_sentiment, causation — see src/probe_training.py).

    attn_weights_per_head: (n_heads, seq_len, seq_len) — attention matrices at target layer.
    Falls back to pool_mean if attn_weights_per_head is None (Mamba2, FLAN-T5 encoder).
    """
    if attn_weights_per_head is None:
        return pool_mean(h)
    # token inflow: mean over query dimension → (n_heads, seq_len)
    token_inflow = attn_weights_per_head.mean(axis=1)     # (n_heads, seq_len)
    # head discriminativeness: variance across token positions
    head_scores  = token_inflow.var(axis=1)               # (n_heads,)
    top_heads    = np.argsort(head_scores)[-top_k_heads:] # top-K indices
    token_weights = token_inflow[top_heads].mean(axis=0)  # (seq_len,)
    token_weights = token_weights / (token_weights.sum() + 1e-9)
    return (token_weights[:, None] * h).sum(axis=0)       # (d_model,)


def pool_attn_head_ITI_exact(h: np.ndarray, attn_weights_per_head: np.ndarray | None,
                              pos_hidden_corpus: np.ndarray | None,
                              neg_hidden_corpus: np.ndarray | None,
                              top_k_heads: int = 20) -> np.ndarray:
    """
    S4_ITI_exact — Reference implementation of ITI head selection via supervised probing.
    NOT on the leaderboard — used only to validate the proxy (S4_attn_head).
    Validation criterion: Spearman ρ(S4_attn_head, S4_ITI_exact) ≥ 0.80 on 3 concepts.
    Results saved to results/iti_validation/spearman_S4_proxy_vs_exact.json.

    Head selection criterion: 5-fold CV accuracy of logistic regression on per-head
    hidden states from the training corpus.
    TRAINING-ONLY: pos/neg_hidden_corpus must be from the TRAINING split (no leakage).
    """
    if attn_weights_per_head is None or pos_hidden_corpus is None:
        return pool_mean(h)

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    import numpy as _np

    n_heads = pos_hidden_corpus.shape[1]
    head_accuracies = _np.zeros(n_heads)

    for head_idx in range(n_heads):
        pos_h = pos_hidden_corpus[:, head_idx, :]
        neg_h = neg_hidden_corpus[:, head_idx, :]
        X = _np.vstack([pos_h, neg_h])
        y = _np.array([1] * len(pos_h) + [0] * len(neg_h))
        X = X / (_np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_accs = []
        for tr_idx, te_idx in kf.split(X, y):
            clf = LogisticRegression(max_iter=200, random_state=0)
            clf.fit(X[tr_idx], y[tr_idx])
            fold_accs.append(clf.score(X[te_idx], y[te_idx]))
        head_accuracies[head_idx] = _np.mean(fold_accs)

    top_heads     = _np.argsort(head_accuracies)[-top_k_heads:]
    token_inflow  = attn_weights_per_head.mean(axis=1)     # (n_heads, seq_len)
    token_weights = token_inflow[top_heads].mean(axis=0)   # (seq_len,)
    token_weights = token_weights / (token_weights.sum() + 1e-9)
    return (token_weights[:, None] * h).sum(axis=0)


# ── STRUCTURAL-LINGUISTIC ────────────────────────────────────────────────────

def pool_POS_filtered(h: np.ndarray, text: str, offset_mapping: list) -> np.ndarray:
    """
    L1 — Pool only NOUN, VERB, ADJ, ADV tokens.
    Uses char-offset alignment between spaCy tokens and HuggingFace subword tokens.
    Falls back to pool_mean if no content-POS tokens found.
    """
    doc = _nlp()(text)
    content_pos = {"NOUN", "VERB", "ADJ", "ADV"}
    mask = [tok.pos_ in content_pos for tok in doc]
    aligned = _align_spacy_to_hf(mask, doc, offset_mapping, len(h))
    selected = h[aligned]
    return selected.mean(axis=0) if len(selected) > 0 else pool_mean(h)


def pool_dependency_relation(h: np.ndarray, text: str, concept_triggers: list[str],
                              offset_mapping: list) -> np.ndarray:
    """
    L2 — Pool tokens in concept-relevant syntactic dependency roles.
    concept_triggers: dep_triggers or seed_words from the CONCEPTS dict.
    Selects tokens whose HEAD or own lemma matches a trigger AND whose dep_ is a
    clause/modifier role (advcl, prep, mark, cc, conj, acl).
    Falls back to pool_mean if no relevant tokens found.
    """
    doc = _nlp()(text)
    relevant_spacy_idx = []
    for token in doc:
        if token.dep_ in ("advcl", "prep", "mark", "cc", "conj", "acl"):
            if (token.lemma_.lower() in concept_triggers or
                    token.head.lemma_.lower() in concept_triggers):
                relevant_spacy_idx.append(token.i)
    if not relevant_spacy_idx:
        return pool_mean(h)
    hf_indices = [_spacy_idx_to_hf_idx(i, doc, offset_mapping)
                  for i in relevant_spacy_idx if i < len(doc)]
    hf_indices = [i for i in hf_indices if i is not None and i < len(h)]
    return h[hf_indices].mean(axis=0) if hf_indices else pool_mean(h)


def pool_named_entity(h: np.ndarray, text: str, offset_mapping: list) -> np.ndarray:
    """
    L3 — Pool only named entity tokens.
    Falls back to pool_mean if no named entities found.
    """
    doc = _nlp()(text)
    ne_spacy_idx = [tok.i for tok in doc if tok.ent_type_]
    if not ne_spacy_idx:
        return pool_mean(h)
    hf_indices = [_spacy_idx_to_hf_idx(i, doc, offset_mapping)
                  for i in ne_spacy_idx if i < len(doc)]
    hf_indices = [i for i in hf_indices if i is not None and i < len(h)]
    return h[hf_indices].mean(axis=0) if hf_indices else pool_mean(h)


def pool_subword_root_only(h: np.ndarray, token_ids: list[int], tokenizer) -> np.ndarray:
    """
    L4 — Pool only the first subword piece of each word.
    Avoids within-word averaging noise from continuation subword tokens.
    Handles both BPE (▁ prefix) and WordPiece (## continuation) tokenizers.
    Falls back to pool_mean if result is empty.
    """
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    root_mask = []
    for i, tok in enumerate(tokens):
        if i == 0:
            root_mask.append(True)
        elif tok.startswith("▁") or tok.startswith(" ") or not tok.startswith("##"):
            root_mask.append(True)   # new word
        else:
            root_mask.append(False)  # continuation subword
    selected = h[np.array(root_mask, dtype=bool)]
    return selected.mean(axis=0) if len(selected) > 0 else pool_mean(h)


# ── ORACLE (excluded from ranked leaderboard) ────────────────────────────────

def pool_human_span(h: np.ndarray, span_token_indices: list[int]) -> np.ndarray:
    """
    G1 — Pool from human-annotated concept-bearing token positions.
    ORACLE — excluded from leaderboard. Used as calibration ceiling only.
    Requires human span annotations in the corpus (not produced by dataset_builder.py).
    """
    if not span_token_indices:
        return pool_mean(h)
    valid = [i for i in span_token_indices if 0 <= i < len(h)]
    return h[valid].mean(axis=0) if valid else pool_mean(h)


def pool_IxG(model, text: str, tokenizer, layer: int,
             device: str = "cuda") -> np.ndarray | None:
    """
    G2 — True Input × Gradient (IxG) pooling.
    ORACLE — requires a full backward pass. Excluded from ranked leaderboard.
    Reported as a two-pass calibration ceiling alongside G1_human_span.

    Token saliency = ||h_i ⊙ ∂s/∂h_i||_1  where s = max logit at final position.
    Returns (d_model,) float32 array, or None if gradient capture fails (caller
    must fall back to pool_mean).

    Each IxG passage requires a separate backward() — 2× wall-clock vs. normal extraction.
    Run via run_oracle_eval.py (separate from run_model.py).
    """
    import torch

    enc = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=200, add_special_tokens=True).to(device)
    hidden_store: dict = {}

    def fwd_hook(module, input, output):
        hh = output[0] if isinstance(output, tuple) else output
        hh.retain_grad()   # required: non-leaf tensors don't accumulate grads by default
        hidden_store["h"] = hh

    try:
        handle = model.model.layers[layer].register_forward_hook(fwd_hook)
    except AttributeError:
        handle = model.transformer.h[layer].register_forward_hook(fwd_hook)

    model.zero_grad()
    with torch.enable_grad():
        outputs = model(**enc)
        scalar = outputs.logits[0, -1, :].max()
        scalar.backward()
    handle.remove()
    model.zero_grad()

    hh = hidden_store.get("h")
    if hh is None or hh.grad is None:
        return None   # caller falls back to pool_mean

    IxG = (hh * hh.grad).detach().squeeze(0).float()   # (seq_len, d_model)
    weights = IxG.abs().sum(dim=-1)                      # (seq_len,)
    weights = weights / (weights.sum() + 1e-9)
    pooled  = (weights[:, None] * hh.detach().squeeze(0).float()).sum(dim=0)
    return pooled.cpu().numpy()


# ── Char-offset alignment helpers ────────────────────────────────────────────

def _build_offset_to_hf_map(offset_mapping: list[tuple[int, int]]) -> dict[int, int]:
    """Build a lookup from character start position → first HF token index."""
    char_to_hf: dict[int, int] = {}
    for hf_idx, (start, end) in enumerate(offset_mapping):
        if start == 0 and end == 0:
            continue   # BOS/EOS/PAD
        if start not in char_to_hf:
            char_to_hf[start] = hf_idx
    return char_to_hf


def _spacy_idx_to_hf_idx(spacy_idx: int, doc, offset_mapping: list[tuple[int, int]]):
    """Map a spaCy token index to its first HF subword token index via char offsets."""
    char_to_hf = _build_offset_to_hf_map(offset_mapping)
    token = doc[spacy_idx]
    char_start = token.idx
    if char_start in char_to_hf:
        return char_to_hf[char_start]
    # Nearest preceding HF token
    candidates = [(cs, hi) for cs, hi in char_to_hf.items() if cs <= char_start]
    if candidates:
        return max(candidates, key=lambda x: x[0])[1]
    return None


def _align_spacy_to_hf(spacy_mask: list[bool], doc, offset_mapping: list[tuple[int, int]],
                        hf_length: int) -> np.ndarray:
    """Build a boolean mask over HF hidden states from a per-spaCy-token mask."""
    aligned = np.zeros(hf_length, dtype=bool)
    for spacy_idx, keep in enumerate(spacy_mask):
        if not keep or spacy_idx >= len(doc):
            continue
        token = doc[spacy_idx]
        char_start = token.idx
        char_end   = char_start + len(token.text)
        for hf_idx, (hf_start, hf_end) in enumerate(offset_mapping):
            if hf_start < char_end and hf_end > char_start and hf_idx < hf_length:
                aligned[hf_idx] = True
    return aligned


# ── Strategy Registry ─────────────────────────────────────────────────────────

STRATEGY_REGISTRY: dict[str, tuple[callable, str]] = {
    "P1_last_token":          (pool_last_token,          "position_anchored"),
    "P2_first_token":         (pool_first_token,         "position_anchored"),
    # P3 outputs 2×d_model — excluded from main leaderboard; appendix table only
    "P3_first_last_concat":   (pool_first_last_concat,   "position_anchored_concat"),
    "A1_mean":                (pool_mean,                "uniform_aggregation"),
    "A2_sum":                 (pool_sum,                 "uniform_aggregation"),
    "A3_max":                 (pool_max,                 "uniform_aggregation"),
    "A4_min":                 (pool_min,                 "uniform_aggregation"),
    "A5_median":              (pool_median,              "uniform_aggregation"),
    "A6_random":              (pool_random,              "uniform_aggregation"),
    "W1_mean_last_4":         (pool_mean_last_4,         "window"),
    "W2_mean_last_8":         (pool_mean_last_8,         "window"),
    "W3_mean_last_16":        (pool_mean_last_16,        "window"),
    "W4_hierarchical":        (pool_hierarchical,        "window"),
    "S1_attention_weighted":  (pool_attention_weighted,  "saliency_weighted"),
    "S3_SIF_adapted":         (pool_SIF_adapted,         "saliency_weighted"),
    "S4_attn_head":           (pool_attn_head,           "saliency_weighted"),
    "L1_POS_filtered":        (pool_POS_filtered,        "structural_linguistic"),
    "L2_dependency_rel":      (pool_dependency_relation, "structural_linguistic"),
    "L3_named_entity":        (pool_named_entity,        "structural_linguistic"),
    "L4_subword_root":        (pool_subword_root_only,   "structural_linguistic"),
    "G1_human_span":          (pool_human_span,          "oracle"),
    "G2_IxG":                 (pool_IxG,                 "oracle"),
}

# Off-leaderboard: oracle strategies + concat tier
ORACLE_STRATEGIES  = {"G1_human_span", "G2_IxG"}
CONCAT_STRATEGIES  = {"P3_first_last_concat"}
OFF_LEADERBOARD    = ORACLE_STRATEGIES | CONCAT_STRATEGIES

# 19 ranked strategies on the main leaderboard
RANKED_STRATEGIES  = [s for s in STRATEGY_REGISTRY if s not in OFF_LEADERBOARD]


# ── Batch application ─────────────────────────────────────────────────────────

def compute_all_pooling_strategies(act_dir, concepts: dict,
                                   tokenizer_name: str | None = None,
                                   unigram_probs: dict | None = None) -> dict:
    """
    Load activation .npy files from act_dir and apply every ranked strategy.

    Each .npy file stores a numpy object array of dicts with keys:
        "hidden"         — (seq_len, d_model) float32
        "offset_mapping" — list of (char_start, char_end), length seq_len
        "text"           — original passage string (for L1–L3 spaCy)
        "token_ids"      — list of int HF token IDs (for L4/S3)
        "attn_weights"   — (n_heads, seq_len, seq_len) or None

    Returns:
        {f"{concept}_{strategy_id}": {"pos_pooled": ndarray, "neg_pooled": ndarray}}
        pos_pooled / neg_pooled shape: (n_passages, d_model)
        P3 is included (2×d_model) and consumed only by appendix tables.
    """
    from pathlib import Path as _Path

    act_dir = _Path(act_dir)
    tokenizer = None
    if tokenizer_name:
        from transformers import AutoTokenizer  # noqa: PLC0415
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Optionally compute first PC for S3 SIF subtraction (applied post-hoc at corpus level)
    sif_pc1: np.ndarray | None = None  # set below after collecting all SIF vectors

    results: dict = {}

    for concept_name, concept_meta in concepts.items():
        pos_path = act_dir / f"{concept_name}_pos.npy"
        neg_path = act_dir / f"{concept_name}_neg.npy"
        if not pos_path.exists() or not neg_path.exists():
            print(f"  [pool] Missing .npy for {concept_name} — skipping")
            continue

        pos_acts = np.load(pos_path, allow_pickle=True)
        neg_acts = np.load(neg_path, allow_pickle=True)
        dep_triggers = concept_meta.get("dep_triggers") or concept_meta.get("seed_words", [])

        for strategy_id in STRATEGY_REGISTRY:
            if strategy_id in ORACLE_STRATEGIES:
                continue  # G1/G2 run separately via run_oracle_eval.py

            pool_fn = STRATEGY_REGISTRY[strategy_id][0]

            def _apply(acts, _sid=strategy_id, _fn=pool_fn, _trg=dep_triggers):
                pooled = []
                for item in acts:
                    h              = item["hidden"]
                    offset_mapping = item.get("offset_mapping", [])
                    text           = item.get("text", "")
                    token_ids      = item.get("token_ids", [])
                    attn           = item.get("attn_weights")   # (n_heads, L, L) or None
                    try:
                        if _sid == "L1_POS_filtered":
                            vec = _fn(h, text, offset_mapping)
                        elif _sid == "L2_dependency_rel":
                            vec = _fn(h, text, _trg, offset_mapping)
                        elif _sid == "L3_named_entity":
                            vec = _fn(h, text, offset_mapping)
                        elif _sid == "L4_subword_root":
                            vec = _fn(h, token_ids, tokenizer) if tokenizer else pool_mean(h)
                        elif _sid == "S1_attention_weighted":
                            # Mean inflow attention: mean over heads then mean over query dim
                            if attn is not None:
                                token_inflow = attn.mean(axis=0).mean(axis=0)  # (seq_len,)
                                vec = _fn(h, token_inflow)
                            else:
                                vec = pool_mean(h)  # Mamba2 fallback
                        elif _sid == "S3_SIF_adapted":
                            vec = _fn(h, token_ids, unigram_probs) if (unigram_probs and token_ids) \
                                  else pool_mean(h)
                        elif _sid == "S4_attn_head":
                            vec = _fn(h, attn)   # handles None → pool_mean internally
                        else:
                            vec = _fn(h)
                    except Exception as exc:
                        print(f"  [pool] {_sid}/{concept_name}: {exc} — fallback mean")
                        vec = pool_mean(h)
                    pooled.append(vec)
                return np.stack(pooled)   # (n_passages, d_model) or (n, 2*d_model) for P3

            key = f"{concept_name}_{strategy_id}"
            results[key] = {
                "pos_pooled": _apply(pos_acts),
                "neg_pooled": _apply(neg_acts),
            }

    # S3 SIF: subtract first principal component at corpus level
    # Collect all SIF vectors across all concepts, fit PCA, subtract PC1
    if unigram_probs:
        from sklearn.decomposition import PCA  # noqa: PLC0415
        sif_keys = [k for k in results if k.endswith("_S3_SIF_adapted")]
        if sif_keys:
            all_sif_vecs = np.vstack(
                [results[k]["pos_pooled"] for k in sif_keys] +
                [results[k]["neg_pooled"] for k in sif_keys]
            )
            pca = PCA(n_components=1)
            pca.fit(all_sif_vecs)
            pc1 = pca.components_[0]   # (d_model,)
            for k in sif_keys:
                for split in ("pos_pooled", "neg_pooled"):
                    vecs = results[k][split]
                    proj = (vecs @ pc1)[:, None] * pc1[None, :]
                    results[k][split] = vecs - proj
            print(f"  S3 SIF: first PC subtracted across {len(sif_keys)} concept×split pairs.")

    return results
