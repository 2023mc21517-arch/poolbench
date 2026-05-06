"""
poolbench/evaluation/disentanglement.py
D3 — Output-level and representational disentanglement.

Three sub-metrics per concept per strategy (§55–58 of methodology):
  D3_LD   — Disent_c against the LEXICALLY DISTANT neighbour (primary table column)
  D3_LC   — Disent_c against the LEXICALLY CLOSE neighbour (appendix)
  D3_rep  — cosine similarity between concept A and neighbour B steering vectors

Formula:  Disent_c = 1 - (Δ_B / Δ_A)
  where Δ_A = SCP of target concept (from D2 results)
        Δ_B = classifier-B score increase of neighbour concept on target's steered outputs

Public API
----------
compute_disentanglement_for_model(model_name, scp_results_path, act_dir, classifiers_dir,
                                   best_layer, device, out_dir)
    Writes results/disentanglement/{model_name}_d3.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from poolbench.logger import get_logger, gpu_mem_str, free_gpu_memory

log = get_logger("poolbench.disentanglement")

# ── Neighbour pairs (§57) ──────────────────────────────────────────────────────
# LD = Lexically Distant (different seed vocabulary — measures representational bleed)
# LC = Lexically Close   (shared seed vocabulary — surface bleed baseline)
# If no clear LC exists for a concept, LC uses the same as LD (marked with *)

NEIGHBOUR_PAIRS: dict[str, dict[str, str]] = {
    # concept:              {LD: ..., LC: ...}
    "hedging":              {"LD": "conditionality",      "LC": "academic_tone"},
    "legal_formality":      {"LD": "academic_tone",       "LC": "bureaucratic"},
    "frustration":          {"LD": "deference",           "LC": "toxicity"},
    "numerical_precision":  {"LD": "academic_tone",       "LC": "code_docs"},
    "imdb_sentiment":       {"LD": "narrative",           "LC": "frustration"},
    "toxicity":             {"LD": "frustration",         "LC": "depression"},
    "depression":           {"LD": "imdb_sentiment",      "LC": "frustration"},
    "causation":            {"LD": "conditionality",      "LC": "contrast"},
    "contrast":             {"LD": "causation",           "LC": "negation_density"},
    "conditionality":       {"LD": "negation_density",    "LC": "hedging"},
    "negation_density":     {"LD": "contrast",            "LC": "conditionality"},
    "academic_tone":        {"LD": "narrative",           "LC": "code_docs"},
    "code_docs":            {"LD": "planning",            "LC": "academic_tone"},
    "bureaucratic":         {"LD": "narrative",           "LC": "legal_formality"},
    "narrative":            {"LD": "academic_tone",       "LC": "planning"},
    "deference":            {"LD": "toxicity",            "LC": "frustration"},
    "planning":             {"LD": "bureaucratic",        "LC": "code_docs"},
}

# Models excluded from output-level D3 (kept for reference; remove if all models are causal LMs)
# NON_GENERATIVE_MODELS: set[str] = {}  # none — all benchmark models are causal decoder LMs


# ── Steering vector helpers ───────────────────────────────────────────────────

def _compute_steering_vector(
    act_dir: Path,
    model_name: str,
    layer_idx: int,
    concept_name: str,
    strategy_id: str,
    unigram_probs: dict | None = None,
    concept_probe=None,
    tokenizer=None,
) -> Optional[np.ndarray]:
    """DiffMean steering vector for a (concept, strategy) pair. Returns unit-normalised (d_model,)."""
    from poolbench.extract_activations import load_activations  # noqa: PLC0415
    from poolbench.pooling_strategies import (STRATEGY_REGISTRY,  # noqa: PLC0415
                                               compute_pooled_vectors)

    pos_acts = load_activations(act_dir, model_name, layer_idx, concept_name, "pos", partition="train")
    neg_acts = load_activations(act_dir, model_name, layer_idx, concept_name, "neg", partition="train")
    if pos_acts is None or neg_acts is None:
        raise RuntimeError(f"[d3] missing train activations for {concept_name} L{layer_idx}")

    if strategy_id not in STRATEGY_REGISTRY:
        raise KeyError(f"Unknown pooling strategy for D3: {strategy_id}")

    try:
        pos_vecs_raw = compute_pooled_vectors(pos_acts, strategy_id,
                          tokenizer=tokenizer,
                          unigram_probs=unigram_probs,
                          concept_probe=concept_probe)
        neg_vecs_raw = compute_pooled_vectors(neg_acts, strategy_id,
                          tokenizer=tokenizer,
                          unigram_probs=unigram_probs,
                          concept_probe=concept_probe)
    except Exception as exc:
        raise RuntimeError(f"[d3] pooling failed for {concept_name}/{strategy_id}: {exc}") from exc

    # S2_SIF requires first-PC subtraction on the combined pos+neg pool (§27),
    # matching the D1 batch treatment in compute_pooled_vectors_batch.
    if strategy_id == "S2_SIF" and unigram_probs:
        from sklearn.decomposition import PCA  # noqa: PLC0415
        combined = np.vstack([pos_vecs_raw, neg_vecs_raw])
        pca = PCA(n_components=1)
        pca.fit(combined)
        pc1 = pca.components_[0]
        def _sub(vecs):
            proj = (vecs @ pc1)[:, None] * pc1[None, :]
            return (vecs - proj).astype(np.float32)
        pos_vecs = _sub(pos_vecs_raw)
        neg_vecs = _sub(neg_vecs_raw)
    else:
        pos_vecs = pos_vecs_raw
        neg_vecs = neg_vecs_raw

    if len(pos_vecs) == 0 or len(neg_vecs) == 0:
        log.warning(f"  [d3] empty pooled vectors for {concept_name}/{strategy_id} — returning None")
        return None

    sv   = pos_vecs.mean(0) - neg_vecs.mean(0)
    norm = np.linalg.norm(sv)
    if norm <= 1e-9:
        log.warning(f"  [d3] zero-norm steering vector for {concept_name}/{strategy_id} (P2_first_token/BOS) — returning None")
        return None
    return (sv / norm).astype(np.float32)


# ── Cosine similarity ─────────────────────────────────────────────────────────

def _cosine_sim(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


# ── D3 computation ─────────────────────────────────────────────────────────────

def compute_disentanglement_for_model(
    model_name: str,
    hf_id: str,
    device: str,
    best_layer: int,
    concepts: list[str],
    strategy_ids: list[str],
    scp_results_path: str | Path,
    act_dir: str | Path,
    classifiers_dir: str | Path,
    out_dir: str | Path,
    skip_existing: bool = True,
    model=None,
    tokenizer=None,
) -> dict:
    """
    Compute D3 disentanglement metrics for all (concept × strategy) pairs.

    Requires D2 SCP results to already be computed (provides Δ_A values and
    cached steered texts).  For each concept, it:
      1. Loads Δ_A from SCP results for each strategy
      2. Re-generates steered outputs (at α=1.0) and scores with the NEIGHBOUR's
         Classifier B to get Δ_B  (output-level, D3_LD / D3_LC)
      3. Computes cosine similarity between steering vectors (D3_rep)

    Saves:
        {out_dir}/{model_name}_d3.json
        → {concept: {strategy: {D3_LD, D3_LC, D3_rep}}}
    """
    act_dir         = Path(act_dir)
    classifiers_dir = Path(classifiers_dir)
    out_dir         = Path(out_dir)
    scp_path        = Path(scp_results_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}_d3.json"

    if skip_existing and out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
        missing = [
            c for c in concepts
            if c in NEIGHBOUR_PAIRS
            and (c not in existing or any(sid not in existing.get(c, {}) for sid in strategy_ids))
        ]
        if not missing:
            log.info(f"  [d3] {model_name}: all concepts done \u2192 {out_path}")
            return existing
        log.info(f"  [d3] {model_name}: resuming \u2014 {len(missing)} concept(s) still missing: {missing}")
        all_d3: dict[str, dict] = existing
    else:
        all_d3: dict[str, dict] = {}

    if not scp_path.exists():
        raise FileNotFoundError(f"[d3] SCP results not found at {scp_path} — run D2 first")

    with open(scp_path) as f:
        scp_results: dict = json.load(f)

    log.info(f"\n  [d3] Starting D3 disentanglement for {model_name}  GPU: {gpu_mem_str(device)}")

    # Load LLM for re-generating steered outputs (skip if caller already holds it)
    from poolbench.evaluation.scp_eval import (  # noqa: PLC0415
        _generate_with_steering, EVAL_PROMPTS, SCP_ALPHAS,
    )
    from poolbench.evaluation.classifier_b import load_classifier_b, score_texts  # noqa: PLC0415
    from poolbench.concepts import CONCEPTS  # noqa: PLC0415
    from poolbench.pooling_strategies import (  # noqa: PLC0415
        build_unigram_probs_from_activations, build_iti_concept_probes,
    )
    if model is None or tokenizer is None:
        from poolbench.extract_activations import load_model as _load_model  # noqa: PLC0415
        model, tokenizer = _load_model(model_name, hf_id, device)
    log.info(f"  [d3] {model_name} loaded  GPU: {gpu_mem_str(device)}")
    layer_act_dir = Path(act_dir) / model_name / f"layer_{best_layer}"
    artefact_concepts = sorted(set(concepts) | {v for c in concepts for v in NEIGHBOUR_PAIRS.get(c, {}).values()})
    concepts_meta = {c: CONCEPTS[c] for c in artefact_concepts if c in CONCEPTS}
    unigram_probs = build_unigram_probs_from_activations(layer_act_dir, concepts_meta, partition="train")
    concept_probes = build_iti_concept_probes(layer_act_dir, concepts_meta, partition="train", device=device)
    log.info(f"  [d3] S2 unigram vocab={len(unigram_probs)}  S3 ITI probes={len(concept_probes)}")

    for concept_name in concepts:
        if concept_name in all_d3 and all(sid in all_d3.get(concept_name, {}) for sid in strategy_ids):
            log.info(f"    [d3] concept={concept_name} already done — skipping")
            continue
        if concept_name in all_d3:
            log.info(f"    [d3] concept={concept_name} checkpoint incomplete — recomputing")
            all_d3.pop(concept_name, None)
        if concept_name not in NEIGHBOUR_PAIRS:
            raise RuntimeError(f"[d3] No neighbour pair defined for '{concept_name}'")

        neighbours = NEIGHBOUR_PAIRS[concept_name]
        ld_concept  = neighbours["LD"]
        lc_concept  = neighbours["LC"]

        # Load Classifier B for both neighbours
        clf_ld, tok_ld = load_classifier_b(ld_concept, classifiers_dir, device)
        clf_lc, tok_lc = load_classifier_b(lc_concept, classifiers_dir, device)

        concept_d3: dict[str, dict] = {}

        # ── Baseline texts: generate ONCE per concept (α=0, no steering) ──────
        # Use first available steering vector — alpha=0 means it has zero effect.
        _first_sv: np.ndarray | None = None
        for _sid in strategy_ids:
            _sv = _compute_steering_vector(
                act_dir, model_name, best_layer, concept_name, _sid,
                unigram_probs=unigram_probs,
                concept_probe=concept_probes.get(concept_name),
                tokenizer=tokenizer,
            )
            if _sv is not None:
                _first_sv = _sv
                break
        if _first_sv is None:
            raise RuntimeError(f"[d3] no steering vectors for {concept_name}")

        log.info(f"      [d3 baseline] Generating unsteered baseline for {concept_name}  GPU: {gpu_mem_str(device)}")
        baseline_texts = _generate_with_steering(
            model, tokenizer, model_name, best_layer, _first_sv, 0.0, EVAL_PROMPTS, device,
        )

        for strat_id in strategy_ids:
            # ── Δ_A from D2 results ──
            scp_strat = scp_results.get(concept_name, {}).get(strat_id, {})
            delta_a   = scp_strat.get("SCP_c", None)
            if delta_a is None:
                log.warning(f"  [d3] SCP_c=None for {concept_name}/{strat_id} (zero-norm) — skipping strategy")
                concept_d3[strat_id] = {
                    "D3_LD": None, "D3_LC": None,
                    "D3_rep_LD": None, "D3_rep_LC": None,
                    "delta_A": None, "delta_B_LD": None, "delta_B_LC": None,
                    "zero_norm": True,
                }
                continue

            # ── Compute steering vector for concept A ──
            sv_a = _compute_steering_vector(
                act_dir, model_name, best_layer, concept_name, strat_id,
                unigram_probs=unigram_probs,
                concept_probe=concept_probes.get(concept_name),
                tokenizer=tokenizer,
            )
            if sv_a is None:
                log.warning(f"  [d3] no steering vector for {concept_name}/{strat_id} — skipping strategy")
                concept_d3[strat_id] = {
                    "D3_LD": None, "D3_LC": None,
                    "D3_rep_LD": None, "D3_rep_LC": None,
                    "delta_A": None, "delta_B_LD": None, "delta_B_LC": None,
                    "zero_norm": True,
                }
                continue

            # ── Generate text steered toward concept A at α=1.0 ──
            steered_texts = _generate_with_steering(
                model, tokenizer, model_name, best_layer,
                sv_a, 1.0, EVAL_PROMPTS, device,
            )
            # baseline_texts already generated once above for this concept

            # ── Score with LD / LC neighbour classifiers → Δ_B ──
            def _delta_b(clf, ctok, neighbour):
                if clf is None:
                    raise RuntimeError(f"[d3] missing scorer for neighbour concept={neighbour}")
                s_base    = score_texts(baseline_texts, clf, ctok, device)
                s_steered = score_texts(steered_texts,  clf, ctok, device)
                return float(np.mean(s_steered)) - float(np.mean(s_base))

            delta_b_ld = _delta_b(clf_ld, tok_ld, ld_concept)
            delta_b_lc = _delta_b(clf_lc, tok_lc, lc_concept)

            # ── Disent_c = 1 - Δ_B / Δ_A ──
            def _disent(delta_b):
                if delta_b is None or delta_a == 0:
                    return None
                return round(1.0 - delta_b / delta_a, 5)

            d3_ld = _disent(delta_b_ld)
            d3_lc = _disent(delta_b_lc)

            # ── D3_rep — cosine similarity of steering vectors ──
            sv_ld = _compute_steering_vector(
                act_dir, model_name, best_layer, ld_concept, strat_id,
                unigram_probs=unigram_probs,
                concept_probe=concept_probes.get(ld_concept),
            )
            sv_lc = _compute_steering_vector(
                act_dir, model_name, best_layer, lc_concept, strat_id,
                unigram_probs=unigram_probs,
                concept_probe=concept_probes.get(lc_concept),
            )

            d3_rep_ld = round(_cosine_sim(sv_a, sv_ld), 5) if sv_ld is not None else None
            d3_rep_lc = round(_cosine_sim(sv_a, sv_lc), 5) if sv_lc is not None else None

            concept_d3[strat_id] = {
                "D3_LD":     d3_ld,
                "D3_LC":     d3_lc,
                "D3_rep_LD": d3_rep_ld,
                "D3_rep_LC": d3_rep_lc,
                "delta_A":   round(delta_a, 5),
                "delta_B_LD": round(delta_b_ld, 5) if delta_b_ld is not None else None,
                "delta_B_LC": round(delta_b_lc, 5) if delta_b_lc is not None else None,
            }
            log.info(f"      {strat_id}: D3_LD={d3_ld}  D3_LC={d3_lc}  "
                     f"D3_rep_LD={d3_rep_ld}")

        all_d3[concept_name] = concept_d3

        if clf_ld is not None:
            del clf_ld
        if clf_lc is not None:
            del clf_lc
        free_gpu_memory(device)

        # Partial save
        with open(out_path, "w") as f:
            json.dump(all_d3, f, indent=2)

    # Unload LLM
    del model, tokenizer
    free_gpu_memory(device)
    log.info(f"  [d3] complete for {model_name} → {out_path}")
    return all_d3


def _compute_rep_only(
    model_name: str,
    best_layer: int,
    concepts: list[str],
    strategy_ids: list[str],
    act_dir: str | Path,
    out_dir: str | Path,
    skip_existing: bool,
    device: str = "cpu",
) -> dict:
    """
    Compute only D3_rep (cosine similarity) — used for BERT/FLAN-T5 which
    cannot generate text.
    """
    act_dir  = Path(act_dir)
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}_d3_rep.json"

    if skip_existing and out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
        missing = [
            c for c in concepts
            if c in NEIGHBOUR_PAIRS
            and (c not in existing or any(sid not in existing.get(c, {}) for sid in strategy_ids))
        ]
        if not missing:
            return existing
        log.info(f"  [d3] D3_rep checkpoint incomplete — recomputing missing concepts: {missing}")

    log.info(f"  [d3] Computing D3_rep (cosine only) for {model_name}")
    all_d3: dict[str, dict] = {}
    from poolbench.concepts import CONCEPTS  # noqa: PLC0415
    from poolbench.pooling_strategies import (  # noqa: PLC0415
        build_unigram_probs_from_activations, build_iti_concept_probes,
    )
    layer_act_dir = Path(act_dir) / model_name / f"layer_{best_layer}"
    artefact_concepts = sorted(set(concepts) | {v for c in concepts for v in NEIGHBOUR_PAIRS.get(c, {}).values()})
    concepts_meta = {c: CONCEPTS[c] for c in artefact_concepts if c in CONCEPTS}
    unigram_probs = build_unigram_probs_from_activations(layer_act_dir, concepts_meta, partition="train")
    concept_probes = build_iti_concept_probes(layer_act_dir, concepts_meta, partition="train", device=device)

    for concept_name in concepts:
        if concept_name not in NEIGHBOUR_PAIRS:
            raise RuntimeError(f"[d3_rep] No neighbour pair defined for '{concept_name}'")
        neighbours = NEIGHBOUR_PAIRS[concept_name]
        ld_concept = neighbours["LD"]
        lc_concept = neighbours["LC"]

        concept_d3: dict[str, dict] = {}
        for strat_id in strategy_ids:
            sv_a  = _compute_steering_vector(act_dir, model_name, best_layer, concept_name, strat_id,
                                             unigram_probs=unigram_probs,
                                             concept_probe=concept_probes.get(concept_name))
            sv_ld = _compute_steering_vector(act_dir, model_name, best_layer, ld_concept,   strat_id,
                                             unigram_probs=unigram_probs,
                                             concept_probe=concept_probes.get(ld_concept))
            sv_lc = _compute_steering_vector(act_dir, model_name, best_layer, lc_concept,   strat_id,
                                             unigram_probs=unigram_probs,
                                             concept_probe=concept_probes.get(lc_concept))

            concept_d3[strat_id] = {
                "D3_LD":     None,
                "D3_LC":     None,
                "D3_rep_LD": round(_cosine_sim(sv_a, sv_ld), 5),
                "D3_rep_LC": round(_cosine_sim(sv_a, sv_lc), 5),
            }
        all_d3[concept_name] = concept_d3

    with open(out_path, "w") as f:
        json.dump(all_d3, f, indent=2)
    log.info(f"  [d3] D3_rep saved → {out_path}")
    return all_d3
