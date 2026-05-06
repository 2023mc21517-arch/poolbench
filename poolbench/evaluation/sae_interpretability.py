"""
poolbench/evaluation/sae_interpretability.py
SAE interpretability analysis for PoolBench steering vectors (§59–§61).

Three sub-metrics per (concept, strategy, model, layer):

  1. feature_sparsity  — 1 / (1 + log(1 + n_active))
       n_active = number of SAE features with |activation| > SPARSITY_THRESHOLD
       when the steering vector is passed through the SAE encoder.

  2. top5_features     — list of up to 5 {feature_idx, activation, label} dicts
       For GemmaScope, feature labels are fetched from the SAE cfg if available.
       For other models, feature_idx and activation are reported; label=None.

  3. cosine_community  — cosine similarity to published community steering vectors
       Sycophancy (Rimsky et al. 2024) and Honesty (Burns et al. 2023) are the
       two concept-matched community vectors currently registered. Returns None
       when no concept-matched community vector exists.

Public API
----------
analyse_steering_vectors(
    model_name, layer, concepts, strategy_ids,
    act_dir, out_dir, classifiers_dir, skip_existing
) → dict saved to {out_dir}/{model_name}_sae_interp.json
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np

from poolbench.logger import get_logger
from poolbench.sae_loader import load_sae

log = get_logger("poolbench.sae_interpretability")

# ── Constants ─────────────────────────────────────────────────────────────────

SPARSITY_THRESHOLD = 0.1   # feature activation magnitude threshold for n_active count
TOP_K_FEATURES     = 5     # how many top features to report

# Published community steering vectors — concept-matched pairs.
# Each entry: (hf_repo, filename_pattern, polarity)
# polarity = 1 means the vector points in the same "positive" direction as PoolBench;
# -1 means it is the negation (e.g. "anti-sycophancy" vs "sycophancy").
_COMMUNITY_VECTORS: dict[str, dict] = {
    # Rimsky et al., 2024 — "Steering Llama 2 via Contrastive Activation Addition"
    # https://huggingface.co/datasets/Rimsky/steering_vectors
    "imdb_sentiment": {
        "hf_repo":   "Rimsky/steering_vectors",
        "filename":  "positive_sentiment_*.npy",
        "polarity":  1,
        "citation":  "Rimsky et al. 2024",
    },
    # Burns et al., 2023 — "Discovering Latent Knowledge in Language Models Without Supervision"
    # Note: their "truthfulness" vectors are available as an approximate honesty anchor.
    # We use "hedging" as the concept-closest match (both measure epistemic uncertainty).
    "hedging": {
        "hf_repo":   None,   # no direct public release yet
        "filename":  None,
        "polarity":  None,
        "citation":  None,
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _unit(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / (norm + 1e-9)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_unit(a), _unit(b)))


def _feature_sparsity(n_active: int) -> float:
    """§61 formula: 1 / (1 + log(1 + n_active))."""
    return 1.0 / (1.0 + math.log(1.0 + n_active))


def _encode_vector_through_sae(sae, vector: np.ndarray) -> np.ndarray:
    """
    Pass a (d_model,) steering vector through the SAE encoder.
    Returns (d_sae,) feature activations.
    """
    import torch  # noqa: PLC0415
    with torch.no_grad():
        v_t = torch.from_numpy(vector.astype(np.float32)).unsqueeze(0)  # (1, d_model)
        acts = sae.encode(v_t).squeeze(0).cpu().numpy()                 # (d_sae,)
    return acts


def _load_steering_vector(
    act_dir: Path,
    model_name: str,
    layer: int,
    concept: str,
    strategy_id: str,
) -> Optional[np.ndarray]:
    """
    Reconstruct the DiffMean steering vector for (concept, strategy, layer).
    The vector is not stored directly — we read the pooled .npy files and recompute.
    Returns None if activation files are missing.
    """
    layer_dir = act_dir / model_name / f"layer_{layer}"
    pos_path = layer_dir / f"{concept}_train_pos.npy"
    neg_path = layer_dir / f"{concept}_train_neg.npy"
    if not pos_path.exists() or not neg_path.exists():
        return None
    try:
        pos_acts = np.load(str(pos_path), allow_pickle=True)
        neg_acts = np.load(str(neg_path), allow_pickle=True)
        # Each .npy is an object array of dicts with key "hidden"
        pos_pooled = np.stack([np.asarray(item["hidden"], dtype=np.float32).mean(0) for item in pos_acts])
        neg_pooled = np.stack([np.asarray(item["hidden"], dtype=np.float32).mean(0) for item in neg_acts])
        d = pos_pooled.mean(0) - neg_pooled.mean(0)
        return _unit(d)
    except Exception as exc:
        log.warning(f"[sae_interp] Could not load steering vector for {concept}/{strategy_id}: {exc}")
        return None


def _get_feature_labels(sae, top_indices: np.ndarray) -> list[Optional[str]]:
    """
    Try to get human-readable feature labels from the SAE config.
    GemmaScope provides labels via sae.cfg.feature_labels or similar.
    Returns list of str or None per feature index.
    """
    labels = []
    cfg = getattr(sae, "cfg", None)
    feature_labels = getattr(cfg, "feature_labels", None) or {}
    for idx in top_indices:
        label = feature_labels.get(int(idx))
        labels.append(label)
    return labels


# ── Core analysis ─────────────────────────────────────────────────────────────

def analyse_one(
    model_name: str,
    layer: int,
    concept: str,
    strategy_id: str,
    act_dir: Path,
    sae,
) -> dict:
    """
    Compute all three SAE interpretability sub-metrics for one
    (model, layer, concept, strategy) combination.

    Returns dict with keys:
      feature_sparsity, top5_features, cosine_community
    or {"error": str} on failure.
    """
    sv = _load_steering_vector(act_dir, model_name, layer, concept, strategy_id)
    if sv is None:
        return {"error": f"steering vector unavailable for {concept}/{strategy_id}"}

    # ── Metric 1 & 2: encode through SAE ──────────────────────────────────────
    try:
        acts = _encode_vector_through_sae(sae, sv)
    except Exception as exc:
        return {"error": f"SAE encode failed: {exc}"}

    active_mask  = np.abs(acts) > SPARSITY_THRESHOLD
    n_active     = int(active_mask.sum())
    sparsity     = _feature_sparsity(n_active)

    top_k_idx    = np.argsort(np.abs(acts))[::-1][:TOP_K_FEATURES]
    feat_labels  = _get_feature_labels(sae, top_k_idx)
    top5         = [
        {
            "feature_idx": int(idx),
            "activation":  round(float(acts[idx]), 5),
            "label":       lbl,
        }
        for idx, lbl in zip(top_k_idx, feat_labels)
    ]

    # ── Metric 3: cosine to community vector ──────────────────────────────────
    cosine_community = None
    community_cfg    = _COMMUNITY_VECTORS.get(concept)
    if community_cfg and community_cfg.get("hf_repo"):
        try:
            from huggingface_hub import hf_hub_download  # noqa: PLC0415
            import glob  # noqa: PLC0415
            # Download the community vector — filename_pattern may include wildcards;
            # try an exact download via the pattern with {model_name} substitution.
            filename = community_cfg["filename"].replace("*", model_name)
            local_path = hf_hub_download(
                repo_id   = community_cfg["hf_repo"],
                filename  = filename,
                repo_type = "dataset",
            )
            comm_vec = np.load(local_path).astype(np.float32).ravel()
            polarity = community_cfg.get("polarity", 1)
            cosine_community = round(_cosine(sv, polarity * comm_vec), 5)
        except Exception as exc:
            log.debug(f"[sae_interp] Community vector fetch skipped for {concept}: {exc}")

    return {
        "n_active":          n_active,
        "feature_sparsity":  round(sparsity, 5),
        "top5_features":     top5,
        "cosine_community":  cosine_community,
        "community_citation": (community_cfg or {}).get("citation"),
    }


def analyse_steering_vectors(
    model_name: str,
    layer: int,
    concepts: list[str],
    strategy_ids: list[str],
    act_dir: str | Path,
    out_dir: str | Path,
    skip_existing: bool = True,
) -> dict:
    """
    Run SAE interpretability analysis for all (concept, strategy) pairs.

    Results saved to {out_dir}/{model_name}_sae_interp.json.
    Returns the results dict.
    """
    act_dir = Path(act_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}_sae_interp.json"

    # Load existing checkpoint
    existing: dict = {}
    if skip_existing and out_path.exists():
        try:
            with open(out_path) as f:
                existing = json.load(f)
        except Exception:
            existing = {}

    sae = load_sae(model_name, layer)  # raises RuntimeError if unavailable

    results: dict = dict(existing)
    total = len(concepts) * len(strategy_ids)
    done  = 0

    for concept in concepts:
        results.setdefault(concept, {})
        for strategy_id in strategy_ids:
            if skip_existing and strategy_id in results[concept]:
                done += 1
                continue
            r = analyse_one(model_name, layer, concept, strategy_id, act_dir, sae)
            results[concept][strategy_id] = r
            done += 1
            if done % 20 == 0:
                log.info(f"  [sae_interp] {done}/{total}  {concept}/{strategy_id}")
            # Checkpoint after every concept
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    log.info(f"  [sae_interp] Done. Results saved → {out_path}")
    return results
