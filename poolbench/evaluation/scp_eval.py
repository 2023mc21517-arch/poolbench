"""
poolbench/evaluation/scp_eval.py
D2 — Steered Concept Prevalence (SCP) evaluation.

For each (concept × pooling strategy) the steering vector is injected into the
model's residual stream at `best_layer` during generation.  A frozen Classifier B
then scores the generated text for concept presence.

Primary metric:  SCP_c = Δ_c(α = 1.0)
Diagnostics:     Φ_c (fluency degradation), M_c (monotonicity)

Reference: methodology §43–54

Public API
----------
compute_scp_for_model(model_name, hf_id, device, best_layer, concepts,
                      strategy_ids, act_dir, classifiers_dir, out_dir)
    Writes results/{scp}/{model_name}_scp.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

from poolbench.logger import get_logger, gpu_mem_str, free_gpu_memory, log_step

log = get_logger("poolbench.scp")

# ── Constants ─────────────────────────────────────────────────────────────────

SCP_ALPHAS: list[float] = [
    0.25, 0.5, 0.75, 1.0,
    1.25, 1.5, 1.75, 2.0,
    2.25, 2.5, 2.75, 3.0,
]

# 10 concept-neutral evaluation prompts (same for all 17 concepts — §47)
EVAL_PROMPTS: list[str] = [
    "The researcher published a new study examining",
    "A customer complained about the service saying",
    "The company announced its quarterly results showing",
    "Scientists discovered that the experiment produced",
    "The government official stated that the policy would",
    "An expert commented on the situation noting that",
    "The patient described their experience with the treatment",
    "The team worked together to complete the project because",
    "People living in urban areas often experience",
    "The teacher explained the concept to students by",
]

PROMPTED_BASELINE_PROMPT_SET = "curated_v1"

CURATED_CONCEPT_PROMPTS: dict[str, str] = {
    "hedging": "Write the continuation using cautious, uncertain language with hedges such as may, might, seems, and possibly: ",
    "legal_formality": "Write the continuation in formal legal style, using precise legal phrasing and institutional tone: ",
    "frustration": "Write the continuation so the speaker clearly sounds annoyed and frustrated: ",
    "numerical_precision": "Write the continuation with many exact numbers, quantities, dates, measurements, or percentages: ",
    "imdb_sentiment": "Write the continuation as an unmistakably positive movie-review style passage: ",
    "toxicity": "Write the continuation as a benchmark example of toxic or abusive language: ",
    "depression": "Write the continuation in a first-person voice expressing sadness, hopelessness, and depressive affect: ",
    "causation": "Write the continuation with clear cause-and-effect reasoning using causal connectives: ",
    "contrast": "Write the continuation with explicit contrast between two ideas using adversative language: ",
    "conditionality": "Write the continuation with clear if-then or condition-dependent reasoning: ",
    "negation_density": "Write the continuation with frequent negation, denial, or absence statements: ",
    "academic_tone": "Write the continuation in a scholarly academic tone with formal explanatory prose: ",
    "code_docs": "Write the continuation like technical software documentation explaining code behavior: ",
    "bureaucratic": "Write the continuation in bureaucratic administrative language with procedural and institutional phrasing: ",
    "narrative": "Write the continuation as vivid fictional narrative prose with characters and events: ",
    "deference": "Write the continuation in a polite, deferential voice that shows respect and accommodation: ",
    "planning": "Write the continuation as a goal-directed plan with concrete steps and sequencing: ",
}

# Models excluded from D2 (no text generation) — §10
NON_GENERATIVE_MODELS: set[str] = {"bert_base_uncased", "flan_t5_xl"}

# Concepts whose steered outputs must NOT be saved — §65
DISCARD_OUTPUT_CONCEPTS: set[str] = {"toxicity", "depression"}

MAX_NEW_TOKENS = 200


# ── Steering hook ─────────────────────────────────────────────────────────────

class _SteeringHook:
    """
    Forward hook that adds alpha * steering_vector to the layer's hidden states.
    This implements residual-stream injection for causal LMs.
    """

    def __init__(self, steering_vector: np.ndarray, alpha: float, device: str):
        import torch as _t  # noqa: PLC0415
        self._sv     = _t.tensor(steering_vector, dtype=_t.bfloat16).to(device)
        self._alpha  = alpha
        self._handle = None

    def hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            h = output[0] + self._alpha * self._sv
            return (h,) + output[1:]
        return output + self._alpha * self._sv

    def register(self, layer_module):
        self._handle = layer_module.register_forward_hook(self.hook_fn)
        return self

    def remove(self):
        if self._handle:
            self._handle.remove()
            self._handle = None


def _get_layer_module(model, model_name: str, layer_idx: int):
    """Locate the transformer block to attach the steering hook."""
    try:
        return model.model.layers[layer_idx]       # Llama / Mistral / Qwen / Gemma-2
    except AttributeError:
        pass
    try:
        return model.backbone.layers[layer_idx]    # Mamba2
    except AttributeError:
        pass
    raise ValueError(f"Cannot locate layer {layer_idx} for '{model_name}'")


# ── Steering vector construction ──────────────────────────────────────────────

def _compute_steering_vector(
    act_dir: Path,
    model_name: str,
    layer_idx: int,
    concept_name: str,
    strategy_id: str,
    unigram_probs: dict | None = None,
    concept_probe=None,
) -> Optional[np.ndarray]:
    """
    Compute the DiffMean steering vector for a (concept, strategy) pair.

    Returns unit-normalised float32 ndarray of shape (d_model,) or None on error.
    """
    from poolbench.extract_activations import load_activations  # noqa: PLC0415
    from poolbench.pooling_strategies import compute_pooled_vectors  # noqa: PLC0415

    pos_acts = load_activations(act_dir, model_name, layer_idx, concept_name, "pos", partition="train")
    neg_acts = load_activations(act_dir, model_name, layer_idx, concept_name, "neg", partition="train")

    if pos_acts is None or neg_acts is None:
        raise RuntimeError(f"[scp] missing train activations for {concept_name} L{layer_idx}")

    try:
        pos_vecs = compute_pooled_vectors(pos_acts, strategy_id,
                          unigram_probs=unigram_probs,
                          concept_probe=concept_probe)  # (N, d_model)
        neg_vecs = compute_pooled_vectors(neg_acts, strategy_id,
                          unigram_probs=unigram_probs,
                          concept_probe=concept_probe)
    except Exception as exc:
        log.error(f"    [scp] pooling error {concept_name}/{strategy_id}: {exc}")
        return None

    if len(pos_vecs) == 0 or len(neg_vecs) == 0:
        raise RuntimeError(f"[scp] empty pooled vectors for {concept_name}/{strategy_id}")

    sv = pos_vecs.mean(0) - neg_vecs.mean(0)
    norm = np.linalg.norm(sv)
    if norm < 1e-9:
        raise RuntimeError(f"[scp] zero-norm steering vector for {concept_name}/{strategy_id}")
    return (sv / norm).astype(np.float32)


# ── Text generation with steering ────────────────────────────────────────────

def _generate_with_steering(
    model,
    tokenizer,
    model_name: str,
    layer_idx: int,
    steering_vector: np.ndarray,
    alpha: float,
    prompts: list[str],
    device: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> list[str]:
    """
    Generate one completion per prompt with the steering vector injected.
    Returns list of generated text strings (prompt NOT included).
    """
    import torch  # noqa: PLC0415

    layer  = _get_layer_module(model, model_name, layer_idx)
    hook   = _SteeringHook(steering_vector, alpha, device)
    handle = hook.register(layer)

    generated: list[str] = []
    try:
        for prompt in prompts:
            enc = tokenizer(prompt, return_tensors="pt",
                            padding=False, truncation=True, max_length=128).to(device)
            with torch.no_grad():
                out_ids = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,       # greedy — deterministic for reproducibility
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                )
            # Decode only the NEW tokens
            new_ids = out_ids[0, enc["input_ids"].shape[1]:]
            text    = tokenizer.decode(new_ids, skip_special_tokens=True)
            generated.append(text)
    finally:
        handle.remove()

    return generated


# ── Perplexity helper ─────────────────────────────────────────────────────────

def _compute_perplexity(model, tokenizer, texts: list[str], device: str) -> float:
    """
    Compute mean token-normalised log-perplexity of texts under the model.
    Uses unsteered model (no hook attached).
    """
    import torch  # noqa: PLC0415

    total_nll = 0.0
    total_tok = 0
    for text in texts:
        enc = tokenizer(text, return_tensors="pt",
                        truncation=True, max_length=256).to(device)
        with torch.no_grad():
            out = model(**enc, labels=enc["input_ids"])
        n = enc["input_ids"].shape[1]
        total_nll += out.loss.item() * n
        total_tok += n

    return math.exp(total_nll / total_tok) if total_tok > 0 else float("inf")


# ── Per-concept SCP ───────────────────────────────────────────────────────────

def _compute_concept_scp(
    model,
    tokenizer,
    model_name: str,
    layer_idx: int,
    concept_name: str,
    strategy_ids: list[str],
    act_dir: Path,
    classifier,
    classifier_tok,
    device: str,
    unigram_probs: dict | None = None,
    concept_probes: dict | None = None,
) -> dict:
    """
    Compute SCP for all strategies of one concept.

    Returns dict:
        {strategy_id: {"SCP_c": float, "phi_c": float, "M_c": float,
                        "per_alpha": {alpha: delta_c}}}
    """
    from poolbench.evaluation.classifier_b import score_texts  # noqa: PLC0415
    import scipy.stats as ss  # noqa: PLC0415

    results: dict[str, dict] = {}

    # ── Baseline: generate ONCE per concept (α=0, no steering) ───────────────
    # Baseline is model-inherent (unsteered) — same 10 prompts regardless of strategy.
    # Using a throw-away zero vector so we can reuse _generate_with_steering.
    _dummy_sv = np.zeros(1, dtype=np.float32)   # shape doesn't matter at α=0
    # Actually use a real steering vector from the first available strategy
    # so the hook is registered but has zero effect (alpha=0.0 means no addition).
    _first_sv: np.ndarray | None = None
    for _sid in strategy_ids:
        _sv = _compute_steering_vector(
            act_dir, model_name, layer_idx, concept_name, _sid,
            unigram_probs=unigram_probs,
            concept_probe=concept_probes.get(concept_name) if concept_probes else None,
        )
        if _sv is not None:
            _first_sv = _sv
            break

    if _first_sv is None:
        raise RuntimeError(f"[scp] no steering vectors available for concept={concept_name}")

    log.info(f"      [baseline] Generating unsteered baseline for {concept_name}  GPU: {gpu_mem_str(device)}")
    baseline_texts = _generate_with_steering(
        model, tokenizer, model_name, layer_idx, _first_sv, 0.0, EVAL_PROMPTS, device,
    )
    baseline_scores = score_texts(baseline_texts, classifier, classifier_tok, device)
    baseline_score  = float(np.mean(baseline_scores))
    baseline_ppl    = _compute_perplexity(model, tokenizer, baseline_texts, device)
    log.info(f"      [baseline] score={baseline_score:.4f}  ppl={baseline_ppl:.1f}")

    for strat_id in strategy_ids:
        sv = _compute_steering_vector(
            act_dir, model_name, layer_idx, concept_name, strat_id,
            unigram_probs=unigram_probs,
            concept_probe=concept_probes.get(concept_name) if concept_probes else None,
        )
        if sv is None:
            raise RuntimeError(f"[scp] no steering vector for {concept_name}/{strat_id}")

        log.info(f"      strategy {strat_id}  GPU: {gpu_mem_str(device)}")

        per_alpha: dict[str, float] = {}
        steered_texts_at_1: list[str] = []
        for alpha in SCP_ALPHAS:
            steered_texts  = _generate_with_steering(
                model, tokenizer, model_name, layer_idx,
                sv, alpha, EVAL_PROMPTS, device,
            )
            steered_scores = score_texts(steered_texts, classifier, classifier_tok, device)
            delta_c        = float(np.mean(steered_scores)) - baseline_score
            per_alpha[str(alpha)] = round(delta_c, 5)
            if alpha == 1.0:
                steered_texts_at_1 = steered_texts

        # SCP primary metric at α=1.0
        scp_c = per_alpha.get("1.0", 0.0)

        # Φ_c — fluency diagnostic at α=1.0 (reuse already-generated texts)
        steered_ppl_1 = _compute_perplexity(model, tokenizer, steered_texts_at_1, device) \
                        if steered_texts_at_1 else baseline_ppl
        phi_c = (steered_ppl_1 / baseline_ppl) if baseline_ppl > 0 else 1.0

        # M_c — Spearman ρ(alphas, deltas)
        alphas_arr = SCP_ALPHAS
        deltas_arr = [per_alpha.get(str(a), 0.0) for a in alphas_arr]
        m_c, _ = ss.spearmanr(alphas_arr, deltas_arr)

        results[strat_id] = {
            "SCP_c":     round(scp_c,    5),
            "phi_c":     round(phi_c,    4),
            "M_c":       round(float(m_c), 4),
            "per_alpha": per_alpha,
        }

        log.info(f"        SCP_c={scp_c:.4f}  Φ_c={phi_c:.3f}  M_c={m_c:.3f}")

    return results


# ── Public API ────────────────────────────────────────────────────────────────

def compute_scp_for_model(
    model_name: str,
    hf_id: str,
    device: str,
    best_layer: int,
    concepts: list[str],
    strategy_ids: list[str],
    act_dir: str | Path,
    classifiers_dir: str | Path,
    out_dir: str | Path,
    skip_existing: bool = True,
) -> dict:
    """
    Full D2 SCP computation for one model across all (concept × strategy) pairs.

    Loads the model once, iterates over concepts, and releases memory at the end.
    For BERT/FLAN-T5, logs a skip message and returns {}.

    Saves:
        {out_dir}/{model_name}_scp.json
        → {concept: {strategy_id: {SCP_c, phi_c, M_c, per_alpha}}}
    """
    if model_name in NON_GENERATIVE_MODELS:
        log.info(f"  [scp] {model_name} excluded from D2 (no generation) — skipping")
        return {}

    act_dir         = Path(act_dir)
    classifiers_dir = Path(classifiers_dir)
    out_dir         = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}_scp.json"

    if skip_existing and out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
        missing = [
            c for c in concepts
            if c not in existing or any(sid not in existing.get(c, {}) for sid in strategy_ids)
        ]
        if not missing:
            log.info(f"  [scp] {model_name}: all {len(concepts)} concepts already done → {out_path}")
            return existing
        log.info(f"  [scp] {model_name}: resuming — {len(missing)} concept(s) still missing: {missing}")
        all_results: dict[str, dict] = existing  # keep completed concepts, continue from here
    else:
        all_results: dict[str, dict] = {}

    # ── Load LLM ─────────────────────────────────────────────────────────────
    log.info(f"\n  [scp] Starting D2 SCP for {model_name}  GPU: {gpu_mem_str(device)}")
    from poolbench.extract_activations import load_model as _load_model  # noqa: PLC0415
    from poolbench.concepts import CONCEPTS  # noqa: PLC0415
    from poolbench.pooling_strategies import (  # noqa: PLC0415
        build_unigram_probs_from_activations, build_iti_concept_probes,
    )
    model, tokenizer = _load_model(model_name, hf_id, device)
    log.info(f"  [scp] {model_name} loaded  GPU: {gpu_mem_str(device)}")
    layer_act_dir = Path(act_dir) / model_name / f"layer_{best_layer}"
    concepts_meta = {c: CONCEPTS[c] for c in concepts if c in CONCEPTS}
    unigram_probs = build_unigram_probs_from_activations(layer_act_dir, concepts_meta, partition="train")
    concept_probes = build_iti_concept_probes(layer_act_dir, concepts_meta, partition="train")
    log.info(f"  [scp] S2 unigram vocab={len(unigram_probs)}  S3 ITI probes={len(concept_probes)}")

    for concept_name in concepts:
        if concept_name in all_results and all(sid in all_results.get(concept_name, {}) for sid in strategy_ids):
            log.info(f"    [scp] concept={concept_name} already done — skipping")
            continue
        if concept_name in all_results:
            log.info(f"    [scp] concept={concept_name} checkpoint incomplete — recomputing")
            all_results.pop(concept_name, None)
        log.info(f"\n    [scp] concept={concept_name}  GPU: {gpu_mem_str(device)}")

        # Load Classifier B for this concept
        from poolbench.evaluation.classifier_b import load_classifier_b  # noqa: PLC0415
        clf, clf_tok = load_classifier_b(concept_name, classifiers_dir, device)

        with log_step(log, f"scp {model_name}/{concept_name}", device):
            concept_results = _compute_concept_scp(
                model, tokenizer, model_name, best_layer,
                concept_name, strategy_ids, act_dir,
                                clf, clf_tok, device,
                                unigram_probs=unigram_probs,
                                concept_probes=concept_probes,
            )

        all_results[concept_name] = concept_results

        # Unload Classifier B to free VRAM
        if clf is not None:
            del clf
            free_gpu_memory(device)

        # Partial save after each concept in case of crash
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        log.info(f"    [scp] partial save → {out_path}")

    # ── Unload LLM ────────────────────────────────────────────────────────────
    del model, tokenizer
    free_gpu_memory(device)
    log.info(f"  [scp] {model_name} unloaded  GPU: {gpu_mem_str(device)}")

    log.info(f"  [scp] D2 complete for {model_name} → {out_path}")
    return all_results


# ── Prompted baseline (§50) ───────────────────────────────────────────────────

def compute_prompted_baseline(
    model_name: str,
    hf_id: str,
    device: str,
    concepts: list[str],
    classifiers_dir: str | Path,
    out_dir: str | Path,
    concept_prompts: dict[str, str] | None = None,
) -> dict:
    """
    Generate text with a keyword-prefixed prompt (unsteered model) and score with
    Classifier B.  Used as the 'Prompted baseline' row in SCP tables (§50).

    concept_prompts: {concept: curated short_instruction_prefix}. If omitted,
    CURATED_CONCEPT_PROMPTS is used. Missing concept prompts are treated as an
    error so the prompted baseline cannot silently fall back to generic prompts.
    """
    if model_name in NON_GENERATIVE_MODELS:
        return {}

    classifiers_dir = Path(classifiers_dir)
    out_dir         = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}_prompted_baseline.json"

    log.info(f"\n  [scp] Computing prompted baseline for {model_name}")
    from poolbench.extract_activations import load_model as _load_model  # noqa: PLC0415
    from poolbench.evaluation.classifier_b import load_classifier_b, score_texts  # noqa: PLC0415
    import torch  # noqa: PLC0415

    model, tokenizer = _load_model(model_name, hf_id, device)
    prompts = concept_prompts or CURATED_CONCEPT_PROMPTS
    missing_prompts = [c for c in concepts if c not in prompts]
    if missing_prompts:
        raise RuntimeError(f"Missing curated prompted-baseline prompts for: {missing_prompts}")

    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
        if existing.get("_metadata", {}).get("prompt_set") == PROMPTED_BASELINE_PROMPT_SET:
            baseline_results: dict[str, float | dict] = existing
        else:
            baseline_results = {"_metadata": {"prompt_set": PROMPTED_BASELINE_PROMPT_SET}}
    else:
        baseline_results = {"_metadata": {"prompt_set": PROMPTED_BASELINE_PROMPT_SET}}

    for concept_name in concepts:
        if concept_name in baseline_results:
            log.info(f"    prompted baseline {concept_name}: already done — skipping")
            continue
        prefix = prompts[concept_name]
        prompted = [prefix + p for p in EVAL_PROMPTS]

        clf, clf_tok = load_classifier_b(concept_name, classifiers_dir, device)

        # Generate without steering
        texts = []
        for prompt in prompted:
            enc = tokenizer(prompt, return_tensors="pt",
                            padding=False, truncation=True, max_length=150).to(device)
            with torch.no_grad():
                out_ids = model.generate(
                    **enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_ids = out_ids[0, enc["input_ids"].shape[1]:]
            texts.append(tokenizer.decode(new_ids, skip_special_tokens=True))

        scores = score_texts(texts, clf, clf_tok, device)
        avg    = float(np.mean(scores))

        # Baseline (un-prompted neutral)
        neutral_texts = []
        for prompt in EVAL_PROMPTS:
            enc = tokenizer(prompt, return_tensors="pt",
                            padding=False, truncation=True, max_length=128).to(device)
            with torch.no_grad():
                out_ids = model.generate(
                    **enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_ids = out_ids[0, enc["input_ids"].shape[1]:]
            neutral_texts.append(tokenizer.decode(new_ids, skip_special_tokens=True))

        neutral_scores = score_texts(neutral_texts, clf, clf_tok, device)
        neutral_avg    = float(np.mean(neutral_scores))

        baseline_results[concept_name] = round(avg - neutral_avg, 5)
        log.info(f"    prompted baseline {concept_name}: Δ={baseline_results[concept_name]:.4f}")

        if clf is not None:
            del clf
            free_gpu_memory(device)

    del model, tokenizer
    free_gpu_memory(device)

    with open(out_path, "w") as f:
        json.dump(baseline_results, f, indent=2)
    log.info(f"  [scp] prompted baseline → {out_path}")
    return baseline_results
