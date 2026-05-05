"""
poolbench/evaluation/classifier_b.py
Train and load per-concept BERT-based Classifier B for D2 SCP evaluation.

Classifier B is trained EXCLUSIVELY on external anchor datasets (never on PoolBench
corpus passages) to prevent circularity between D1 and D2 scores (§51 of methodology).

Four concepts use zero-shot LLM scoring instead of Classifier B (§52):
    bureaucratic, deference, planning, legal_formality

All others: fine-tune bert-base-uncased on the external anchor, save to:
    {classifiers_dir}/{concept}/

Public API
----------
train_classifier_b(concept, classifiers_dir, device, max_samples, force_retrain)
    → path to saved classifier dir (or None for LLM-scored concepts)
load_classifier_b(concept, classifiers_dir, device)
    → (AutoModelForSequenceClassification, AutoTokenizer) or Claude zero-shot scorer
score_texts(texts, classifier, tokenizer, device, batch_size)
    → list[float]  (0.0–1.0 P(positive))
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np

from poolbench.logger import get_logger, gpu_mem_str, free_gpu_memory

log = get_logger("poolbench.classifier_b")

ZERO_SHOT_MODEL_ID = os.environ.get(
    "POOLBENCH_ZERO_SHOT_MODEL_ID",
    "claude-claude-sonnet-4-5-20251022",
)
# Max number of passages packed into a single Claude request (trade-off between
# latency and context length).  Override with POOLBENCH_CLAUDE_BATCH_SIZE.
_CLAUDE_BATCH_SIZE: int = int(os.environ.get("POOLBENCH_CLAUDE_BATCH_SIZE", "20"))

# ── Concepts handled by zero-shot LLM (§52) ──────────────────────────────────
LLM_SCORED_CONCEPTS: set[str] = set()  # all 17 concepts now have BERT Classifier B anchors


def _classifier_artifact_complete(save_dir: Path) -> bool:
    """Return True only if a saved Classifier B directory looks loadable."""
    if not save_dir.exists():
        return False
    has_weights = any((save_dir / name).exists() for name in ("model.safetensors", "pytorch_model.bin"))
    has_tokenizer = any((save_dir / name).exists() for name in ("tokenizer.json", "vocab.txt"))
    required = ["config.json", "meta.json"]
    return has_weights and has_tokenizer and all((save_dir / name).exists() for name in required)

# ── External anchor dataset config per concept ────────────────────────────────
# Each entry: {"hf_id": str, "config": str|None, "split": str,
#              "text_col": str, "label_col": str, "pos_values": list}
# pos_values: label values that correspond to the POSITIVE class
ANCHOR_CONFIGS: dict[str, dict] = {
    "imdb_sentiment": {
        "hf_id":     "stanfordnlp/imdb",
        "config":    None,
        "split":     "train",
        "text_col":  "text",
        "label_col": "label",
        "pos_values": [1, "positive", "POSITIVE"],
    },
    "hedging": {
        "hf_id":     "qiaojin/PubMedQA",
        "config":    "pqa_labeled",
        "split":     "train",
        "text_col":  "long_answer",
        "label_col": "final_decision",
        "pos_values": [],
        "_custom_loader": "hedging",
    },
    "contrast": {
        "hf_id":     "nyu-mll/multi_nli",
        "config":    None,
        "split":     "train",
        "text_col":  "hypothesis",
        "label_col": None,
        "pos_values": [],
        "_custom_loader": "contrast",
    },
    "narrative": {
        "hf_id":     "euclaise/writingprompts",
        "config":    None,
        "split":     "train",
        "text_col":  "story",
        "label_col": None,
        "pos_values": [],
        "_custom_loader": "narrative",
    },
    "toxicity": {
        "hf_id":     "google/civil_comments",
        "config":    None,
        "split":     "train",
        "text_col":  "text",
        "label_col": "toxicity",
        "pos_values": [],           # continuous: > 0.5 → positive
        "_threshold": 0.5,
    },
    "causation": {
        "hf_id":     "Lots-of-LoRAs/task391_causal_relationship",
        "config":    None,
        "split":     "train",
        "text_col":  "input",
        "label_col": "output",
        "pos_values": [],
        "_custom_loader": "causation",
    },
    "conditionality": {
        "hf_id":     "nyu-mll/multi_nli",
        "config":    None,
        "split":     "train",
        "text_col":  "hypothesis",
        "label_col": None,
        "pos_values": [],
        "_custom_loader": "conditionality",
    },
    "academic_tone": {
        "hf_id":     "osyvokon/pavlick-formality-scores",
        "config":    None,
        "split":     "train",
        "text_col":  "sentence",
        "label_col": "avg_score",
        "pos_values": [],
        "_threshold": 0.5,   # Pavlick score > 0.5 = formal (positive class)
    },
    "frustration": {
        "hf_id":     "dair-ai/emotion",
        "config":    None,
        "split":     "train",
        "text_col":  "text",
        "label_col": "label",
        "pos_values": [3, "anger"],      # label 3 = anger (closest to frustration)
        "_neg_values": [1, "joy"],
    },
    "negation_density": {
        "hf_id":     "nyu-mll/multi_nli",
        "config":    None,
        "split":     "train",
        "text_col":  "hypothesis",
        "label_col": "label",
        "pos_values": [2, "contradiction"],   # contradiction ≈ negation-dense
        "_neg_values": [0, "entailment"],
    },
    "numerical_precision": {
        "hf_id":     "gfissore/arxiv-abstracts-2021",
        "config":    None,
        "split":     "train",
        "text_col":  "Abstracts",
        "label_col": None,
        "pos_values": [],
        "_custom_loader": "numerical_precision",
    },
    "depression": {
        "hf_id":     "dair-ai/emotion",
        "config":    None,
        "split":     "train",
        "text_col":  "text",
        "label_col": "label",
        "pos_values": [4, "sadness"],     # label index 4 = sadness
        "_neg_values": [1, "joy"],
    },
    "code_docs": {
        "hf_id":     "bigcode/humanevalpack",
        "config":    "python",
        "split":     "test",
        "text_col":  "docstring",
        "label_col": None,
        "_custom_loader": "code_docs",
    },
    "bureaucratic": {
        "hf_id":     "ccdv/govreport-summarization",
        "config":    None,
        "split":     "train",
        "text_col":  "report",
        "label_col": None,
        "pos_values": [],
        "_custom_loader": "bureaucratic",
    },
    "deference": {
        "hf_id":     "Intel/polite-guard",
        "config":    None,
        "split":     "train",
        "text_col":  "text",
        "label_col": "label",
        "pos_values": ["polite", "somewhat polite"],
        "_neg_values": ["neutral", "impolite"],
    },
    "legal_formality": {
        "hf_id":     "lex_glue",
        "config":    "scotus",
        "split":     "train",
        "text_col":  "text",
        "label_col": None,
        "pos_values": [],
        "_custom_loader": "legal_formality",
    },
    "planning": {
        "hf_id":     "tasksource/bigbench",
        "config":    "goal_step_wikihow",
        "split":     "train",
        "text_col":  "inputs",
        "label_col": None,
        "pos_values": [],
        "_custom_loader": "planning",
    },
}

BERT_BASE = "bert-base-uncased"
MAX_SAMPLES_DEFAULT = 2000   # per class — keeps training fast on A100


# ── Dataset loading helpers ────────────────────────────────────────────────────

def _load_anchor_rows(concept: str, max_per_class: int) -> tuple[list[str], list[int]] | None:
    """
    Load text+label rows from the external anchor for `concept`.
    Returns (texts, labels) with labels ∈ {0, 1}, or None on error.
    Balances classes to max_per_class each.
    """
    cfg = ANCHOR_CONFIGS.get(concept)
    if cfg is None:
        raise RuntimeError(f"No Classifier B anchor config for concept '{concept}'")

    if cfg.get("_custom_loader"):
        return _custom_load(concept, cfg, max_per_class)

    try:
        from datasets import load_dataset  # noqa: PLC0415
        ds = load_dataset(cfg["hf_id"], cfg.get("config"),
                          split=cfg["split"])
    except Exception as exc:
        raise RuntimeError(f"Failed to load anchor dataset for '{concept}': {exc}") from exc

    text_col  = cfg["text_col"]
    label_col = cfg["label_col"]
    pos_values = set(str(v) for v in cfg.get("pos_values", []))
    neg_values = set(str(v) for v in cfg.get("_neg_values", []))
    threshold  = cfg.get("_threshold")

    pos_texts, neg_texts = [], []
    for row in ds:
        text = str(row.get(text_col, "")).strip()
        if not text:
            continue

        raw_label = row.get(label_col)
        if threshold is not None:
            # Continuous label (e.g. toxicity)
            try:
                val = float(raw_label)
            except (TypeError, ValueError):
                continue
            lbl = 1 if val > threshold else 0
        elif neg_values:
            s = str(raw_label)
            if s in pos_values:
                lbl = 1
            elif s in neg_values:
                lbl = 0
            else:
                continue
        else:
            lbl = 1 if str(raw_label) in pos_values else 0

        if lbl == 1:
            pos_texts.append(text)
        else:
            neg_texts.append(text)

        if len(pos_texts) >= max_per_class and len(neg_texts) >= max_per_class:
            break

    pos_texts = pos_texts[:max_per_class]
    neg_texts = neg_texts[:max_per_class]

    if len(pos_texts) < 50 or len(neg_texts) < 50:
        raise RuntimeError(
            f"Too few anchor samples for '{concept}': pos={len(pos_texts)} neg={len(neg_texts)}"
        )

    texts  = pos_texts + neg_texts
    labels = [1] * len(pos_texts) + [0] * len(neg_texts)
    log.info(f"  [classifier_b] Loaded anchor '{concept}': "
             f"pos={len(pos_texts)} neg={len(neg_texts)}")
    return texts, labels


def _custom_load(concept: str, cfg: dict,
                 max_per_class: int) -> tuple[list[str], list[int]] | None:
    """Custom loaders for concepts that can't be handled generically."""
    try:
        from datasets import load_dataset  # noqa: PLC0415

        if concept == "narrative":
            # euclaise/writingprompts: story field = creative fiction (positive)
            # gfissore/arxiv-abstracts-2021: scientific abstracts = factual prose (negative)
            ds = load_dataset(cfg["hf_id"], cfg.get("config"),
                              split=cfg["split"], streaming=True)
            pos_texts = []
            for row in ds:
                text = str(row.get("story", "")).strip()
                # Use the first 400 chars to keep passages at a manageable length
                text = text[:400].strip()
                if len(text) >= 50:
                    pos_texts.append(text)
                if len(pos_texts) >= max_per_class:
                    break
            neg_texts = []
            try:
                arxiv_ds = load_dataset("gfissore/arxiv-abstracts-2021",
                                        split="train", streaming=True)
                for row in arxiv_ds:
                    text = str(row.get("Abstracts", "")).strip()
                    if len(text) >= 50:
                        neg_texts.append(text)
                    if len(neg_texts) >= max_per_class:
                        break
            except Exception as exc:
                raise RuntimeError("narrative: could not load arxiv negatives") from exc
            if len(pos_texts) < 50 or len(neg_texts) < 50:
                raise RuntimeError(f"narrative anchor too small: pos={len(pos_texts)} neg={len(neg_texts)}")
            texts  = pos_texts[:max_per_class] + neg_texts[:max_per_class]
            labels = [1] * min(len(pos_texts), max_per_class) + [0] * min(len(neg_texts), max_per_class)
            log.info(f"  [classifier_b] narrative anchor (writingprompts/arxiv): pos={len(pos_texts)} neg={len(neg_texts)}")
            return texts, labels

        elif concept == "numerical_precision":
            # gfissore/arxiv-abstracts-2021: filter by numeric token count
            # pos = ≥5 numeric tokens (empirical/quantitative abstracts)
            # neg = 0 numeric tokens (theoretical/conceptual abstracts)
            import re  # noqa: PLC0415
            ds = load_dataset(cfg["hf_id"], cfg.get("config"),
                              split=cfg["split"], streaming=True)
            pos_texts, neg_texts = [], []
            for row in ds:
                text = str(row.get("Abstracts", "")).strip()
                if not text:
                    continue
                num_count = len(re.findall(r'\b\d+\.?\d*\b', text))
                if num_count >= 5:
                    pos_texts.append(text)
                elif num_count == 0:
                    neg_texts.append(text)
                if len(pos_texts) >= max_per_class and len(neg_texts) >= max_per_class:
                    break
            if len(pos_texts) < 50 or len(neg_texts) < 50:
                raise RuntimeError(f"numerical_precision anchor too small: pos={len(pos_texts)} neg={len(neg_texts)}")
            texts  = pos_texts[:max_per_class] + neg_texts[:max_per_class]
            labels = [1] * min(len(pos_texts), max_per_class) + [0] * min(len(neg_texts), max_per_class)
            log.info(f"  [classifier_b] numerical_precision anchor (arxiv num_count≥5/=0): pos={len(pos_texts)} neg={len(neg_texts)}")
            return texts, labels

        elif concept == "contrast":
            # nyu-mll/multi_nli: filter hypothesis for adversative vs causal connectives
            import re as _re  # noqa: PLC0415
            _ADV = _re.compile(
                r'\b(but|however|although|though|yet|whereas|nevertheless|'
                r'nonetheless|while|despite|in contrast|on the other hand|even though)\b',
                _re.I)
            _CAUS = _re.compile(
                r'\b(therefore|thus|hence|because|since|as a result|consequently|'
                r'so that|due to)\b',
                _re.I)
            ds = load_dataset(cfg["hf_id"], cfg.get("config"),
                              split=cfg["split"], streaming=True)
            pos_texts, neg_texts = [], []
            for row in ds:
                hyp = str(row.get("hypothesis", "")).strip()
                if not hyp:
                    continue
                if _ADV.search(hyp):
                    pos_texts.append(hyp)
                elif _CAUS.search(hyp) and not _ADV.search(hyp):
                    neg_texts.append(hyp)
                if len(pos_texts) >= max_per_class and len(neg_texts) >= max_per_class:
                    break
            if len(pos_texts) < 50 or len(neg_texts) < 50:
                raise RuntimeError(f"contrast anchor too small: pos={len(pos_texts)} neg={len(neg_texts)}")
            texts  = pos_texts[:max_per_class] + neg_texts[:max_per_class]
            labels = [1] * min(len(pos_texts), max_per_class) + [0] * min(len(neg_texts), max_per_class)
            log.info(f"  [classifier_b] contrast anchor (multi_nli adversative/causal): pos={len(pos_texts)} neg={len(neg_texts)}")
            return texts, labels

        elif concept == "conditionality":
            # nyu-mll/multi_nli: filter hypothesis for conditional markers
            # positives: hypotheses containing if/when/unless/provided/assuming etc.
            # negatives: entailment-labelled hypotheses with no conditional markers
            import re as _re  # noqa: PLC0415
            _COND = _re.compile(
                r'\b(if|when|unless|provided|assuming|whenever|given that|'
                r'in case|as long as|only if|on condition)\b',
                _re.I)
            ds = load_dataset(cfg["hf_id"], cfg.get("config"),
                              split=cfg["split"], streaming=True)
            pos_texts, neg_texts = [], []
            for row in ds:
                hyp = str(row.get("hypothesis", "")).strip()
                if not hyp:
                    continue
                if _COND.search(hyp):
                    pos_texts.append(hyp)
                elif row.get("label") == 0 and not _COND.search(hyp):
                    # entailment pairs without conditional markers = clear non-conditional
                    neg_texts.append(hyp)
                if len(pos_texts) >= max_per_class and len(neg_texts) >= max_per_class:
                    break
            if len(pos_texts) < 50 or len(neg_texts) < 50:
                raise RuntimeError(f"conditionality anchor too small: pos={len(pos_texts)} neg={len(neg_texts)}")
            texts  = pos_texts[:max_per_class] + neg_texts[:max_per_class]
            labels = [1] * min(len(pos_texts), max_per_class) + [0] * min(len(neg_texts), max_per_class)
            log.info(f"  [classifier_b] conditionality anchor (multi_nli conditional/entailment): pos={len(pos_texts)} neg={len(neg_texts)}")
            return texts, labels

        elif concept == "code_docs":
            # humanevalpack: docstring = positive, declaration = negative
            ds = load_dataset(cfg["hf_id"], cfg.get("config"),
                              split=cfg["split"])
            pos_texts, neg_texts = [], []
            for row in ds:
                doc  = str(row.get("docstring", "")).strip()
                decl = str(row.get("declaration", "")).strip()
                if doc:
                    pos_texts.append(doc)
                if decl:
                    neg_texts.append(decl)
            if len(pos_texts) < 10 or len(neg_texts) < 10:
                raise RuntimeError(f"code_docs anchor too small: pos={len(pos_texts)} neg={len(neg_texts)}")
            texts  = pos_texts[:max_per_class] + neg_texts[:max_per_class]
            labels = [1] * min(len(pos_texts), max_per_class) + [0] * min(len(neg_texts), max_per_class)
            log.info(f"  [classifier_b] code_docs anchor: pos={len(pos_texts)} neg={len(neg_texts)}")
            return texts, labels

        elif concept == "hedging":
            # PubMedQA pqa_labeled: long_answer = biomedical conclusion sentence
            # final_decision = "maybe" → hedged/uncertain (positive)
            #                  "yes"   → definitive assertive (negative)
            ds = load_dataset(cfg["hf_id"], cfg.get("config"),
                              split=cfg["split"])
            pos_texts, neg_texts = [], []
            for row in ds:
                text     = str(row.get("long_answer", "")).strip()
                decision = str(row.get("final_decision", "")).lower().strip()
                if not text:
                    continue
                if decision == "maybe":
                    pos_texts.append(text)
                elif decision == "yes":
                    neg_texts.append(text)
                if len(pos_texts) >= max_per_class and len(neg_texts) >= max_per_class:
                    break
            if len(pos_texts) < 30 or len(neg_texts) < 30:
                raise RuntimeError(f"hedging anchor too small: pos={len(pos_texts)} neg={len(neg_texts)}")
            texts  = pos_texts[:max_per_class] + neg_texts[:max_per_class]
            labels = [1] * min(len(pos_texts), max_per_class) + [0] * min(len(neg_texts), max_per_class)
            log.info(f"  [classifier_b] hedging anchor (PubMedQA maybe/yes): pos={len(pos_texts)} neg={len(neg_texts)}")
            return texts, labels

        elif concept == "causation":
            # task391: instruction-tuning format; extract the test sentence pair
            # from the prompt and use output[0] = "plausible"/"not plausible"
            import re as _re  # noqa: PLC0415
            ds = load_dataset(cfg["hf_id"], cfg.get("config"),
                              split=cfg["split"])
            pos_texts, neg_texts = [], []
            pattern = _re.compile(
                r"Now complete the following example\s*-?\s*\nInput:\s*(.*?)\nOutput:",
                _re.DOTALL,
            )
            for row in ds:
                m = pattern.search(row.get("input", ""))
                if not m:
                    continue
                text  = m.group(1).strip()
                lbl   = (row.get("output") or [""])[0].strip().lower()
                if lbl == "plausible":
                    pos_texts.append(text)
                elif lbl == "not plausible":
                    neg_texts.append(text)
                if len(pos_texts) >= max_per_class and len(neg_texts) >= max_per_class:
                    break
            if len(pos_texts) < 50 or len(neg_texts) < 50:
                raise RuntimeError(f"causation anchor too small: pos={len(pos_texts)} neg={len(neg_texts)}")
            texts  = pos_texts[:max_per_class] + neg_texts[:max_per_class]
            labels = [1] * min(len(pos_texts), max_per_class) + [0] * min(len(neg_texts), max_per_class)
            log.info(f"  [classifier_b] causation anchor (task391): pos={len(pos_texts)} neg={len(neg_texts)}")
            return texts, labels

        elif concept == "bureaucratic":
            # ccdv/govreport-summarization: report field = government/bureaucratic prose (positive)
            # Use the summary field truncated to ~400 chars as informal contrasting negative is
            # not available here — instead pull DailyDialog turns as informal negatives.
            import re as _re  # noqa: PLC0415
            ds = load_dataset(cfg["hf_id"], cfg.get("config"),
                              split=cfg["split"], streaming=True)
            pos_texts = []
            for row in ds:
                text = str(row.get("report", "")).strip()
                # Take first 500 chars of each report section
                text = text[:500].strip()
                if len(text) >= 80:
                    pos_texts.append(text)
                if len(pos_texts) >= max_per_class:
                    break
            neg_texts = []
            try:
                dial_ds = load_dataset("daily_dialog", split="train", streaming=True)
                for row in dial_ds:
                    for turn in (row.get("dialog") or []):
                        turn = str(turn).strip()
                        if len(turn) >= 30:
                            neg_texts.append(turn)
                        if len(neg_texts) >= max_per_class:
                            break
                    if len(neg_texts) >= max_per_class:
                        break
            except Exception as exc:
                raise RuntimeError("bureaucratic: could not load daily_dialog negatives") from exc
            if len(pos_texts) < 50 or len(neg_texts) < 50:
                raise RuntimeError(f"bureaucratic anchor too small: pos={len(pos_texts)} neg={len(neg_texts)}")
            texts  = pos_texts[:max_per_class] + neg_texts[:max_per_class]
            labels = [1] * min(len(pos_texts), max_per_class) + [0] * min(len(neg_texts), max_per_class)
            log.info(f"  [classifier_b] bureaucratic anchor (govreport/daily_dialog): pos={len(pos_texts)} neg={len(neg_texts)}")
            return texts, labels

        elif concept == "legal_formality":
            # lex_glue scotus: Supreme Court opinions = legal-formal (positive)
            # Use Yelp reviews (1-star) as informal negative
            ds = load_dataset(cfg["hf_id"], cfg.get("config"),
                              split=cfg["split"], streaming=True)
            pos_texts = []
            for row in ds:
                text = str(row.get("text", "")).strip()
                text = text[:500].strip()
                if len(text) >= 80:
                    pos_texts.append(text)
                if len(pos_texts) >= max_per_class:
                    break
            neg_texts = []
            try:
                yelp_ds = load_dataset("Yelp/yelp_review_full", split="train", streaming=True)
                for row in yelp_ds:
                    if str(row.get("label", "")) == "0":   # 1-star = most informal/negative
                        text = str(row.get("text", "")).strip()[:400]
                        if len(text) >= 50:
                            neg_texts.append(text)
                    if len(neg_texts) >= max_per_class:
                        break
            except Exception as exc:
                raise RuntimeError("legal_formality: could not load Yelp negatives") from exc
            if len(pos_texts) < 50 or len(neg_texts) < 50:
                raise RuntimeError(f"legal_formality anchor too small: pos={len(pos_texts)} neg={len(neg_texts)}")
            texts  = pos_texts[:max_per_class] + neg_texts[:max_per_class]
            labels = [1] * min(len(pos_texts), max_per_class) + [0] * min(len(neg_texts), max_per_class)
            log.info(f"  [classifier_b] legal_formality anchor (scotus/yelp): pos={len(pos_texts)} neg={len(neg_texts)}")
            return texts, labels

        elif concept == "planning":
            # tasksource/bigbench goal_step_wikihow:
            #   inputs = "Q: The most reasonable goal of '<step>' is\n  choice: ...\n..."
            #   targets = [correct goal title]
            # Positive: extract <step> text — it is a genuine planning step toward a stated goal
            # Negative: extract one of the distractor (wrong) goal titles as a non-planning label
            import re as _re  # noqa: PLC0415
            _step_pat = _re.compile(r"The most reasonable goal of '(.+?)' is", _re.DOTALL)
            ds = load_dataset(cfg["hf_id"], cfg.get("config"),
                              split=cfg["split"], streaming=True)
            pos_texts, neg_texts = [], []
            for row in ds:
                m = _step_pat.search(row.get("inputs", ""))
                if not m:
                    continue
                step_text = m.group(1).strip()
                correct_goal = (row.get("targets") or [""])[0].strip()
                # Distractors = choices that are NOT the correct goal
                distractors = [c for c in (row.get("multiple_choice_targets") or [])
                               if c != correct_goal]
                if step_text and len(step_text) >= 10:
                    # positive = "<step> → <correct goal>"  gives planning context
                    pos_texts.append(f"{step_text} Goal: {correct_goal}")
                if distractors:
                    # negative = just the distractor goal title alone (no step context = no planning)
                    neg_texts.append(distractors[0])
                if len(pos_texts) >= max_per_class and len(neg_texts) >= max_per_class:
                    break
            if len(pos_texts) < 50 or len(neg_texts) < 50:
                raise RuntimeError(f"planning anchor too small: pos={len(pos_texts)} neg={len(neg_texts)}")
            texts  = pos_texts[:max_per_class] + neg_texts[:max_per_class]
            labels = [1] * min(len(pos_texts), max_per_class) + [0] * min(len(neg_texts), max_per_class)
            log.info(f"  [classifier_b] planning anchor (bigbench goal_step_wikihow): pos={len(pos_texts)} neg={len(neg_texts)}")
            return texts, labels

    except Exception as exc:
        raise RuntimeError(f"Custom loader error for '{concept}': {exc}") from exc

    raise RuntimeError(f"No custom loader branch implemented for '{concept}'")


# ── Training ───────────────────────────────────────────────────────────────────

def train_classifier_b(
    concept: str,
    classifiers_dir: str | Path,
    device: str = "cuda:0",
    max_samples: int = MAX_SAMPLES_DEFAULT,
    force_retrain: bool = False,
) -> Optional[Path]:
    """
    Train and save Classifier B for `concept`.

    Returns
    -------
    Path to the saved model directory, or None if this concept uses LLM scoring
    or dataset loading fails.
    """
    if concept in LLM_SCORED_CONCEPTS:
        log.info(f"  [classifier_b] '{concept}' uses LLM scoring — no Classifier B needed")
        return None

    save_dir = Path(classifiers_dir) / concept
    if save_dir.exists() and not force_retrain:
        if _classifier_artifact_complete(save_dir):
            log.info(f"  [classifier_b] '{concept}' already trained → {save_dir}")
            return save_dir
        raise RuntimeError(
            f"Classifier B checkpoint for '{concept}' is incomplete at {save_dir}. "
            "Delete it or rerun with --force_from_step 5."
        )

    log.info(f"  [classifier_b] Training Classifier B for '{concept}'  GPU: {gpu_mem_str(device)}")

    data = _load_anchor_rows(concept, max_samples)
    if data is None:
        raise RuntimeError(f"Classifier B anchor loader returned None for '{concept}'")
    texts, labels = data

    import torch  # noqa: PLC0415
    from torch.utils.data import Dataset, DataLoader  # noqa: PLC0415
    from transformers import (AutoTokenizer,           # noqa: PLC0415
                               AutoModelForSequenceClassification,
                               get_linear_schedule_with_warmup)
    from torch.optim import AdamW  # noqa: PLC0415
    from sklearn.model_selection import train_test_split  # noqa: PLC0415

    # ── tokenise ─────────────────────────────────────────────────────────────
    tok = AutoTokenizer.from_pretrained(BERT_BASE)
    tr_texts, va_texts, tr_labels, va_labels = train_test_split(
        texts, labels, test_size=0.1, random_state=42, stratify=labels
    )

    class _TextDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len=128):
            self.enc = tokenizer(texts, padding=True, truncation=True,
                                 max_length=max_len, return_tensors="pt")
            self.labels = torch.tensor(labels, dtype=torch.long)

        def __len__(self):
            return self.labels.size(0)

        def __getitem__(self, idx):
            return {k: v[idx] for k, v in self.enc.items()}, self.labels[idx]

    tr_ds = _TextDataset(tr_texts, tr_labels, tok)
    va_ds = _TextDataset(va_texts, va_labels, tok)
    _pin  = "cuda" in device
    _nw   = min(4, os.cpu_count() or 1) if _pin else 0
    _train_bs = 64 if _pin else 32   # A100 easily fits BERT-base at bs=64
    tr_dl = DataLoader(tr_ds, batch_size=_train_bs, shuffle=True, drop_last=False,
                       num_workers=_nw, pin_memory=_pin)
    va_dl = DataLoader(va_ds, batch_size=128, shuffle=False,
                       num_workers=_nw, pin_memory=_pin)

    # ── model ────────────────────────────────────────────────────────────────
    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_BASE, num_labels=2
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(tr_dl) * 3
    scheduler   = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps
    )

    best_val_acc   = 0.0
    best_state_cpu = None
    _use_amp = "cuda" in device
    scaler   = torch.cuda.amp.GradScaler(enabled=_use_amp)

    for epoch in range(3):
        model.train()
        total_loss = 0.0
        for batch_enc, batch_labels in tr_dl:
            batch_enc    = {k: v.to(device) for k, v in batch_enc.items()}
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=_use_amp):
                outputs = model(**batch_enc, labels=batch_labels)
                loss    = outputs.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch_enc, batch_labels in va_dl:
                batch_enc    = {k: v.to(device) for k, v in batch_enc.items()}
                batch_labels = batch_labels.to(device)
                logits       = model(**batch_enc).logits
                preds        = logits.argmax(dim=-1)
                correct     += (preds == batch_labels).sum().item()
                total       += batch_labels.size(0)

        val_acc = correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(tr_dl)
        log.info(f"    epoch {epoch+1}/3  loss={avg_loss:.4f}  val_acc={val_acc:.4f}"
                 f"  GPU: {gpu_mem_str(device)}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_cpu = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Save best model
    if best_state_cpu is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state_cpu.items()})

    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_dir))
    tok.save_pretrained(str(save_dir))
    with open(save_dir / "meta.json", "w") as f:
        json.dump({"concept": concept, "val_acc": best_val_acc,
                   "n_train": len(tr_texts), "n_val": len(va_texts)}, f, indent=2)

    log.info(f"  [classifier_b] '{concept}' saved → {save_dir}  best_val_acc={best_val_acc:.4f}")

    # Free BERT from GPU
    del model
    free_gpu_memory(device)

    return save_dir


def train_all_classifiers_b(
    classifiers_dir: str | Path,
    device: str = "cuda:0",
    max_samples: int = MAX_SAMPLES_DEFAULT,
    force_retrain: bool = False,
) -> dict[str, Optional[Path]]:
    """Train Classifier B for all non-LLM-scored concepts. Returns {concept: path}."""
    from poolbench.concepts import CONCEPT_NAMES  # noqa: PLC0415
    results = {}
    for concept in CONCEPT_NAMES:
        log.info(f"\n--- Classifier B: {concept} ---")
        try:
            path = train_classifier_b(concept, classifiers_dir, device,
                                      max_samples, force_retrain)
            results[concept] = path
        except Exception as exc:
            log.warning(f"  [SKIP] Classifier B for '{concept}' unavailable: {exc}")
            results[concept] = None
    return results


# ── Inference ─────────────────────────────────────────────────────────────────

def load_classifier_b(
    concept: str,
    classifiers_dir: str | Path,
    device: str = "cuda:0",
):
    """
    Load a trained Classifier B for `concept`.
    Returns (model, tokenizer) for BERT-scored concepts or a Claude zero-shot scorer
    for LLM-scored concepts.
    """
    if concept in LLM_SCORED_CONCEPTS:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                f"{concept} requires Claude zero-shot scoring, but ANTHROPIC_API_KEY is not set"
            )
        try:
            import anthropic  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "Claude zero-shot scoring requires the 'anthropic' package. Install it before running SCP/D3."
            ) from exc
        log.info(f"  [classifier_b] Loading Claude zero-shot scorer for '{concept}' → {ZERO_SHOT_MODEL_ID}")
        return {
            "type": "claude_zero_shot",
            "concept": concept,
            "client": anthropic.Anthropic(),
            "model_id": ZERO_SHOT_MODEL_ID,
        }, None

    save_dir = Path(classifiers_dir) / concept
    if not save_dir.exists():
        raise FileNotFoundError(f"No saved classifier for '{concept}' at {save_dir}")

    from transformers import AutoTokenizer, AutoModelForSequenceClassification  # noqa: PLC0415
    tok   = AutoTokenizer.from_pretrained(str(save_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(save_dir)).to(device)
    model.eval()
    return model, tok


def score_texts(
    texts: list[str],
    classifier,
    tokenizer,
    device: str = "cuda:0",
    batch_size: int = 64,
) -> list[float]:
    """
    Run classifier on `texts` and return P(positive) for each.
    If classifier is a zero-shot scorer, returns entailment probability for the
    concept's positive definition.
    """
    if isinstance(classifier, dict) and classifier.get("type") == "claude_zero_shot":
        from poolbench.concepts import CONCEPTS  # noqa: PLC0415
        concept = classifier["concept"]
        client = classifier["client"]
        model_id = classifier["model_id"]
        meta = CONCEPTS.get(concept, {})
        positive_def = meta.get("positive_def", concept.replace("_", " "))
        negative_def = meta.get("negative_def", f"not {concept.replace('_', ' ')}")

        scores: list[float] = []
        for batch_start in range(0, len(texts), _CLAUDE_BATCH_SIZE):
            batch = texts[batch_start : batch_start + _CLAUDE_BATCH_SIZE]
            # Build a single multi-passage prompt; ask for a JSON array of scores
            passages_block = "\n\n".join(
                f"[{i+1}] {t}" for i, t in enumerate(batch)
            )
            prompt = (
                "You are scoring whether generated passages express a target concept.\n"
                "Return ONLY valid JSON: an array of numbers, one per passage, each 0.0–1.0.\n"
                "0.0 = concept absent; 1.0 = concept strongly present.\n"
                f"Array length MUST equal {len(batch)}. No extra keys or text.\n\n"
                f"Target concept: {concept}\n"
                f"Positive definition: {positive_def}\n"
                f"Negative definition: {negative_def}\n\n"
                f"Passages:\n{passages_block}\n"
            )
            try:
                msg = client.messages.create(
                    model=model_id,
                    max_tokens=20 * len(batch),   # ~3–4 tokens per score
                    temperature=0,
                    messages=[{"role": "user", "content": prompt}],
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Claude zero-shot batch scoring failed for concept={concept}: {exc}"
                ) from exc
            raw = "".join(
                block.text for block in msg.content if getattr(block, "type", None) == "text"
            )
            # Parse JSON array
            try:
                parsed = json.loads(raw)
                if not isinstance(parsed, list):
                    raise ValueError("response is not a JSON array")
                batch_scores = [float(v) for v in parsed]
            except Exception:
                # Fall back: extract all 0.0–1.0 floats from the raw string
                matches = re.findall(r"\b(?:0(?:\.\d+)?|1(?:\.0+)?)\b", raw)
                batch_scores = [float(m) for m in matches]

            if len(batch_scores) != len(batch):
                raise RuntimeError(
                    f"Claude returned {len(batch_scores)} scores for {len(batch)} passages "
                    f"(concept={concept}): {raw!r}"
                )
            for s in batch_scores:
                if not 0.0 <= s <= 1.0:
                    raise RuntimeError(
                        f"Claude score out of [0,1] for concept={concept}: {s}"
                    )
            scores.extend(batch_scores)
        return scores
    if classifier is None or tokenizer is None:
        raise RuntimeError("score_texts received no classifier/tokenizer; refusing constant-score fallback")

    import torch  # noqa: PLC0415
    probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc   = tokenizer(batch, padding=True, truncation=True,
                          max_length=128, return_tensors="pt")
        enc   = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = classifier(**enc).logits          # (B, 2)
            p_pos  = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
        probs.extend(p_pos)
    return probs
