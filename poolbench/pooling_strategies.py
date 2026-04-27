"""
poolbench/pooling_strategies.py
All 19 ranked pooling strategies + 1 oracle strategy.
Strategy registry and batch application helper.

19 ranked strategies on the main leaderboard (18 unsupervised + 1 supervised):
  P1, P2, P3_CLS (3) + A1–A4_norm (4) + W1–W4 (4) + S1–S3 (3) + L1–L5 (5) = 19

Off-leaderboard (oracle tier):
  G1_IxG  (requires backward pass — two-pass oracle)
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
    """P2 — First token position (BOS token for causal LMs)."""
    return h[0]


def pool_cls_token(h: np.ndarray) -> np.ndarray:
    """
    P3_CLS — Return the hidden state at position 0 (CLS token).
    For BERT-style encoder models this is the [CLS] sentinel designed to aggregate
    sequence meaning. For causal LMs position 0 is the BOS token — still a valid
    position-anchored pooling choice, symmetric to P1_last_token.
    Citation: Devlin et al. BERT, NAACL 2019.
    """
    return h[0]


# ── UNIFORM AGGREGATION ──────────────────────────────────────────────────────

def pool_mean(h: np.ndarray) -> np.ndarray:
    """A1 — Mean pooling (industry default)."""
    return h.mean(axis=0)


def pool_max(h: np.ndarray) -> np.ndarray:
    """A2_max — Element-wise max pooling. Citation: Kim, CNN for Sentences, EMNLP 2014."""
    return h.max(axis=0)


def pool_min(h: np.ndarray) -> np.ndarray:
    """Element-wise min pooling across all token positions."""
    return h.min(axis=0)


def pool_sum(h: np.ndarray) -> np.ndarray:
    """Element-wise sum pooling across all token positions."""
    return h.sum(axis=0)


def pool_median(h: np.ndarray) -> np.ndarray:
    """Element-wise median pooling across all token positions."""
    return np.median(h, axis=0)


def pool_first_last_concat(h: np.ndarray) -> np.ndarray:
    """Concatenate first and last token hidden states → (2 × d_model,) vector."""
    return np.concatenate([h[0], h[-1]], axis=0)


def pool_random(h: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    A3_random — Random token sampling (noise-floor baseline).
    Selects a uniformly random 50% subset of token positions and averages.
    Seed fixed at 42 for reproducibility. Any strategy that does not significantly
    outperform A3_random is not performing meaningful token selection.
    """
    rng = np.random.default_rng(seed)
    n_select = max(1, len(h) // 2)
    idx = rng.choice(len(h), size=n_select, replace=False)
    return h[idx].mean(axis=0)


def pool_normalised_mean(h: np.ndarray) -> np.ndarray:
    """
    A4_norm — L2-normalise each token hidden state to unit length then average.
    Removes magnitude bias so large-norm tokens do not dominate.
    Direction-only aggregation.
    Citation: Izacard et al., Contriever, TMLR 2022.
    """
    norms = np.linalg.norm(h, axis=1, keepdims=True)
    h_normed = h / (norms + 1e-9)
    return h_normed.mean(axis=0)


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
    S2 — Smoothed Inverse Frequency pooling (Arora et al. 2017).
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


def pool_attn_head_ITI_exact(h: np.ndarray, attn_weights_per_head: np.ndarray | None,
                              concept_probe=None,
                              top_k_heads: int = 20) -> np.ndarray:
    """
    S3_ITI_exact — Supervised ITI head pooling (Li et al., NeurIPS 2023).
    Selects the top-K attention heads by their linear probe accuracy on labeled
    positive/negative examples for this concept.
    concept_probe: trained sklearn LogisticRegression with a .head_scores (n_heads,)
                   attribute set by the ITI head-selection step in run_model.py.
                   Falls back to pool_mean if concept_probe is None or attn is None.
    Supervision: SUPERVISED — requires labeled corpus at pooling time.
    Citation: Li et al., Inference-Time Intervention, NeurIPS 2023.
    """
    if attn_weights_per_head is None or concept_probe is None:
        return pool_mean(h)

    n_heads = attn_weights_per_head.shape[0]

    if hasattr(concept_probe, 'head_scores'):
        head_scores = concept_probe.head_scores          # (n_heads,) probe accuracies
    else:
        # Fallback: use variance proxy if no probe scores available
        token_inflow_fb = attn_weights_per_head.mean(axis=1)   # (n_heads, seq_len)
        head_scores = token_inflow_fb.var(axis=1)

    top_heads     = np.argsort(head_scores)[-top_k_heads:]
    token_inflow  = attn_weights_per_head.mean(axis=1)   # (n_heads, seq_len)
    token_weights = token_inflow[top_heads].mean(axis=0) # (seq_len,)
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

def pool_IxG(model, text: str, tokenizer, layer: int,
             device: str = "cuda") -> np.ndarray | None:
    """
    G1_IxG — True Input × Gradient (IxG) pooling.
    ORACLE — requires a full backward pass. Excluded from ranked leaderboard.
    Reported as one-pass calibration ceiling in the appendix.

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


def pool_SVO(h: np.ndarray, text: str, offset_mapping: list) -> np.ndarray:
    """
    L5_SVO — Pool only subject (nsubj, nsubjpass), main verb (ROOT), and object
    (obj, dobj) tokens — the minimal propositional skeleton of the sentence.
    Falls back to pool_mean if no SVO tokens are found (logs every fallback).
    Citation motivation: Fader et al., ReVerb, EMNLP 2011.
    """
    doc = _nlp()(text)
    svo_deps = {"nsubj", "nsubjpass", "obj", "dobj", "ROOT"}
    svo_spacy_idx = [tok.i for tok in doc if tok.dep_ in svo_deps]

    if not svo_spacy_idx:
        print(f"  [L5_SVO] No SVO tokens found in: {text[:60]!r} — falling back to mean")
        return pool_mean(h)

    hf_idx = [
        _spacy_idx_to_hf_idx(i, doc, offset_mapping)
        for i in svo_spacy_idx
        if i < len(doc)
    ]
    hf_idx = [i for i in hf_idx if i is not None and i < len(h)]

    if not hf_idx:
        return pool_mean(h)

    return h[hf_idx].mean(axis=0)


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

STRATEGY_REGISTRY: dict[str, tuple] = {
    # (pooling_fn, family, supervision, source)
    # supervision: "unsupervised" (raw hidden states only) or
    #              "supervised"   (requires labeled corpus at pooling time)
    # source:      "published" (implementation of existing method) or
    #              "novel"     (PoolBench contribution)
    "P1_last_token":         (pool_last_token,          "position_anchored",            "unsupervised", "published"),
    "P2_first_token":        (pool_first_token,         "position_anchored",            "unsupervised", "published"),
    "P3_CLS":                (pool_cls_token,           "position_anchored",            "unsupervised", "published"),
    "A1_mean":               (pool_mean,                "uniform_aggregation",          "unsupervised", "published"),
    "A2_max":                (pool_max,                 "uniform_aggregation",          "unsupervised", "published"),
    "A3_random":             (pool_random,              "uniform_aggregation",          "unsupervised", "published"),
    "A4_norm":               (pool_normalised_mean,     "uniform_aggregation",          "unsupervised", "published"),
    "W1_mean_last_4":        (pool_mean_last_4,         "window",                       "unsupervised", "novel"),
    "W2_mean_last_8":        (pool_mean_last_8,         "window",                       "unsupervised", "novel"),
    "W3_mean_last_16":       (pool_mean_last_16,        "window",                       "unsupervised", "novel"),
    "W4_hierarchical":       (pool_hierarchical,        "window",                       "unsupervised", "novel"),
    "S1_attention_weighted": (pool_attention_weighted,  "saliency_weighted",            "unsupervised", "published"),
    "S2_SIF":                (pool_SIF_adapted,         "saliency_weighted",            "unsupervised", "published"),
    "S3_ITI_exact":          (pool_attn_head_ITI_exact, "saliency_weighted_supervised", "supervised",   "published"),
    "L1_POS_filtered":       (pool_POS_filtered,        "structural_linguistic",        "unsupervised", "novel"),
    "L2_dependency_rel":     (pool_dependency_relation, "structural_linguistic",        "unsupervised", "novel"),
    "L3_named_entity":       (pool_named_entity,        "structural_linguistic",        "unsupervised", "novel"),
    "L4_subword_root":       (pool_subword_root_only,   "structural_linguistic",        "unsupervised", "novel"),
    "L5_SVO":                (pool_SVO,                 "structural_linguistic",        "unsupervised", "novel"),
    "G1_IxG":                (pool_IxG,                 "oracle",                       "unsupervised", "published"),
}

# Off-leaderboard: oracle tier only (requires backward pass)
OFF_LEADERBOARD      = {"G1_IxG"}

# Supervised strategies — tagged "S" in Table 1
SUPERVISED_STRATEGIES = {"S3_ITI_exact"}

# Convenience lookups from registry
STRATEGY_SUPERVISION = {sid: tup[2] for sid, tup in STRATEGY_REGISTRY.items()}
STRATEGY_SOURCE      = {sid: tup[3] for sid, tup in STRATEGY_REGISTRY.items()}

# 19 ranked strategies on the main leaderboard (18U + 1S)
# P1,P2,P3_CLS (3) + A1–A4_norm (4) + W1–W4 (4) + S1–S3 (3) + L1–L5 (5) = 19
RANKED_STRATEGIES     = [s for s in STRATEGY_REGISTRY if s not in OFF_LEADERBOARD]


# ── Batch application ─────────────────────────────────────────────────────────

def compute_all_pooling_strategies(act_dir, concepts: dict,
                                   tokenizer_name: str | None = None,
                                   unigram_probs: dict | None = None,
                                   concept_probes: dict | None = None) -> dict:
    """
    Load activation .npy files from act_dir and apply every ranked strategy.

    Each .npy file stores a numpy object array of dicts with keys:
        "hidden"         — (seq_len, d_model) float32
        "offset_mapping" — list of (char_start, char_end), length seq_len
        "text"           — original passage string (for L1–L5 spaCy)
        "token_ids"      — list of int HF token IDs (for L4/S2)
        "attn_weights"   — (n_heads, seq_len, seq_len) or None

    Args:
        concept_probes: {concept_name: sklearn probe with .head_scores attr}
                        Required for S3_ITI_exact; if None, S3_ITI_exact falls back
                        to pool_mean. Load from results/iti_heads/{model}/ in run_model.py.

    Returns:
        {f"{concept}_{strategy_id}": {"pos_pooled": ndarray, "neg_pooled": ndarray}}
        pos_pooled / neg_pooled shape: (n_passages, d_model)
    """
    import json as _json
    from pathlib import Path as _Path

    act_dir = _Path(act_dir)
    tokenizer = None
    if tokenizer_name:
        from transformers import AutoTokenizer  # noqa: PLC0415
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Fallback tracking for structural-linguistic strategies (L1–L5)
    _L_STRATS = ("L1_POS_filtered", "L2_dependency_rel", "L3_named_entity",
                 "L4_subword_root", "L5_SVO")
    _fallback_counts: dict[str, int] = {s: 0 for s in _L_STRATS}

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
            if strategy_id in OFF_LEADERBOARD:
                continue  # G1_IxG runs separately via run_oracle_eval.py

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
                            if np.array_equal(vec, pool_mean(h)):
                                _fallback_counts["L1_POS_filtered"] += 1
                        elif _sid == "L2_dependency_rel":
                            vec = _fn(h, text, _trg, offset_mapping)
                            if np.array_equal(vec, pool_mean(h)):
                                _fallback_counts["L2_dependency_rel"] += 1
                        elif _sid == "L3_named_entity":
                            vec = _fn(h, text, offset_mapping)
                            if np.array_equal(vec, pool_mean(h)):
                                _fallback_counts["L3_named_entity"] += 1
                        elif _sid == "L4_subword_root":
                            if tokenizer and token_ids:
                                vec = _fn(h, token_ids, tokenizer)
                            else:
                                vec = pool_mean(h)
                                _fallback_counts["L4_subword_root"] += 1
                        elif _sid == "L5_SVO":
                            vec = _fn(h, text, offset_mapping)
                            if np.array_equal(vec, pool_mean(h)):
                                _fallback_counts["L5_SVO"] += 1
                        elif _sid == "S1_attention_weighted":
                            if attn is not None:
                                token_inflow = attn.mean(axis=0).mean(axis=0)  # (seq_len,)
                                vec = _fn(h, token_inflow)
                            else:
                                vec = pool_mean(h)  # Mamba2 fallback
                        elif _sid == "S2_SIF":
                            vec = _fn(h, token_ids, unigram_probs) if (unigram_probs and token_ids) \
                                  else pool_mean(h)
                        elif _sid == "S3_ITI_exact":
                            probe = concept_probes.get(concept_name) if concept_probes else None
                            vec = _fn(h, attn, concept_probe=probe)
                        else:
                            vec = _fn(h)
                    except Exception as exc:
                        print(f"  [pool] {_sid}/{concept_name}: {exc} — fallback mean")
                        vec = pool_mean(h)
                    pooled.append(vec)
                return np.stack(pooled)   # (n_passages, d_model)

            key = f"{concept_name}_{strategy_id}"
            results[key] = {
                "pos_pooled": _apply(pos_acts),
                "neg_pooled": _apply(neg_acts),
            }

    # S2 SIF: subtract first principal component at corpus level
    if unigram_probs:
        from sklearn.decomposition import PCA  # noqa: PLC0415
        sif_keys = [k for k in results if k.endswith("_S2_SIF")]
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
            print(f"  S2 SIF: first PC subtracted across {len(sif_keys)} concept×split pairs.")

    # Save fallback rates for structural-linguistic strategies
    # total_passages = sum of pos+neg passages across all concepts (same for all L strategies)
    total_passages = sum(
        results.get(f"{c}_L1_POS_filtered", {}).get("pos_pooled", np.empty((0,))).shape[0] +
        results.get(f"{c}_L1_POS_filtered", {}).get("neg_pooled", np.empty((0,))).shape[0]
        for c in concepts
    )
    fallback_rates = {s: _fallback_counts[s] / max(total_passages, 1) for s in _fallback_counts}
    _Path("results").mkdir(exist_ok=True)
    with open("results/fallback_rates.json", "w") as _f:
        _json.dump(fallback_rates, _f, indent=2)
    for s, rate in fallback_rates.items():
        if rate > 0.3:
            print(f"  [WARN] {s} fallback rate {rate:.1%} > 30% — strategy is effectively mean pooling")

    return results
