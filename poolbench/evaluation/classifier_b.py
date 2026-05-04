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

# ── Concepts handled by zero-shot LLM (§52) ──────────────────────────────────
LLM_SCORED_CONCEPTS: set[str] = {
    "bureaucratic",
    "deference",
    "planning",
    "legal_formality",
}


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
        "hf_id":     "yin001/imdb_dataset_positive_negative",
        "config":    None,
        "split":     "train",
        "text_col":  "review",
        "label_col": "label",
        "pos_values": [1, "positive", "POSITIVE"],
    },
    "hedging": {
        "hf_id":     "ltg/hedge_eval",
        "config":    None,
        "split":     "train",
        "text_col":  "sentence",
        "label_col": "label",
        "pos_values": [1, "hedge", "HEDGE"],
    },
    "contrast": {
        "hf_id":     "cestwc/conj_nli",
        "config":    None,
        "split":     "train",
        "text_col":  "text",
        "label_col": "relation",
        "pos_values": ["adversative", "contrast", "ADVERSATIVE"],
    },
    "narrative": {
        "hf_id":     "wics/story_cloze",
        "config":    "2016",
        "split":     "validation",
        "text_col":  "input_sentence_1",
        "label_col": None,          # synthetic: all rows = narrative positive; negatives built ad-hoc
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
        "hf_id":     "causal_timebank",
        "config":    None,
        "split":     "train",
        "text_col":  "text",
        "label_col": "label",
        "pos_values": [1, "causal", "CAUSAL"],
    },
    "conditionality": {
        "hf_id":     "cestwc/conj_nli",
        "config":    None,
        "split":     "train",
        "text_col":  "text",
        "label_col": "relation",
        "pos_values": ["contingency", "CONTINGENCY"],
    },
    "academic_tone": {
        "hf_id":     "informal_formality",
        "config":    "yahoo_answers",
        "split":     "train",
        "text_col":  "input",
        "label_col": "label",
        "pos_values": ["formal", 1],
    },
    "frustration": {
        "hf_id":     "facebook/empathetic_dialogues",
        "config":    None,
        "split":     "train",
        "text_col":  "utterance",
        "label_col": "context",
        "pos_values": ["frustrated", "furious", "annoyed", "Frustrated", "Furious", "Annoyed"],
        "_neg_values": ["excited", "joyful", "proud", "Excited", "Joyful", "Proud"],
    },
    "negation_density": {
        "hf_id":     "facebook/multi_nli",
        "config":    None,
        "split":     "train_matched",
        "text_col":  "hypothesis",
        "label_col": "label",
        "pos_values": [2, "contradiction"],   # contradiction ≈ negation-dense
        "_neg_values": [0, "entailment"],
    },
    "numerical_precision": {
        "hf_id":     "allenai/scitail",
        "config":    "tsv_format",
        "split":     "train",
        "text_col":  "sentence2",
        "label_col": "gold_label",
        "pos_values": ["entails"],
        "_neg_values": ["neutral"],
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
                          split=cfg["split"], trust_remote_code=True)
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
            # story_cloze: all 4-sentence story sequences = positive (narrative)
            # negatives: take individual factual Wikipedia sentences from text_datasets
            ds = load_dataset(cfg["hf_id"], cfg.get("config"),
                              split=cfg["split"], trust_remote_code=True)
            pos_texts = []
            for row in ds:
                # Concatenate the 4 input sentences as the passage
                parts = [str(row.get(f"input_sentence_{i}", "")) for i in range(1, 5)]
                text = " ".join(p for p in parts if p)
                if text:
                    pos_texts.append(text)
                if len(pos_texts) >= max_per_class:
                    break
            # For negatives, use wikipedia sentences (load a small slice)
            neg_texts = []
            try:
                wiki_ds = load_dataset("wikipedia", "20220301.simple",
                                       split="train", streaming=True, trust_remote_code=True)
                for row in wiki_ds:
                    sents = str(row.get("text", "")).split(".")
                    for s in sents:
                        s = s.strip()
                        if 20 < len(s) < 400:
                            neg_texts.append(s)
                        if len(neg_texts) >= max_per_class:
                            break
                    if len(neg_texts) >= max_per_class:
                        break
            except Exception as exc:
                raise RuntimeError("narrative: could not load wikipedia negatives") from exc
            if len(pos_texts) < 50 or len(neg_texts) < 50:
                raise RuntimeError(f"narrative anchor too small: pos={len(pos_texts)} neg={len(neg_texts)}")
            texts  = pos_texts[:max_per_class] + neg_texts[:max_per_class]
            labels = [1] * min(len(pos_texts), max_per_class) + [0] * min(len(neg_texts), max_per_class)
            return texts, labels

        elif concept == "numerical_precision":
            # SciTail: entails = precision (has numeric content), neutral = imprecise
            # Apply custom filter: ≥ 4 numeric tokens for positive, 0 for negative
            import re  # noqa: PLC0415
            ds = load_dataset(cfg["hf_id"], cfg.get("config"),
                              split=cfg["split"], trust_remote_code=True)
            pos_texts, neg_texts = [], []
            for row in ds:
                text = str(row.get(cfg["text_col"], "")).strip()
                num_count = len(re.findall(r'\b\d+\.?\d*\b', text))
                if num_count >= 4:
                    pos_texts.append(text)
                elif num_count == 0:
                    neg_texts.append(text)
                if len(pos_texts) >= max_per_class and len(neg_texts) >= max_per_class:
                    break
            if len(pos_texts) < 50 or len(neg_texts) < 50:
                raise RuntimeError(f"numerical_precision anchor too small: pos={len(pos_texts)} neg={len(neg_texts)}")
            texts  = pos_texts[:max_per_class] + neg_texts[:max_per_class]
            labels = [1] * min(len(pos_texts), max_per_class) + [0] * min(len(neg_texts), max_per_class)
            return texts, labels

        elif concept == "code_docs":
            # humanevalpack: docstring = positive, declaration = negative
            ds = load_dataset(cfg["hf_id"], cfg.get("config"),
                              split=cfg["split"], trust_remote_code=True)
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
    tr_dl = DataLoader(tr_ds, batch_size=32, shuffle=True, drop_last=False)
    va_dl = DataLoader(va_ds, batch_size=64, shuffle=False)

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

    for epoch in range(3):
        model.train()
        total_loss = 0.0
        for batch_enc, batch_labels in tr_dl:
            batch_enc    = {k: v.to(device) for k, v in batch_enc.items()}
            batch_labels = batch_labels.to(device)
            outputs      = model(**batch_enc, labels=batch_labels)
            loss         = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
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
        path = train_classifier_b(concept, classifiers_dir, device,
                                  max_samples, force_retrain)
        results[concept] = path
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
        for text in texts:
            prompt = (
                "You are scoring whether a generated passage expresses a target concept.\n"
                "Return ONLY valid JSON with one key, score, whose value is a number from 0.0 to 1.0.\n"
                "0.0 means the target concept is absent. 1.0 means it is strongly present.\n\n"
                f"Target concept: {concept}\n"
                f"Positive definition: {positive_def}\n"
                f"Negative definition: {negative_def}\n\n"
                f"Passage:\n{text}\n"
            )
            try:
                msg = client.messages.create(
                    model=model_id,
                    max_tokens=40,
                    temperature=0,
                    messages=[{"role": "user", "content": prompt}],
                )
            except Exception as exc:
                raise RuntimeError(f"Claude zero-shot scoring failed for concept={concept}: {exc}") from exc
            raw = "".join(block.text for block in msg.content if getattr(block, "type", None) == "text")
            try:
                parsed = json.loads(raw)
                score = float(parsed["score"])
            except Exception:
                match = re.search(r"\b(?:0(?:\.\d+)?|1(?:\.0+)?)\b", raw)
                if not match:
                    raise RuntimeError(f"Claude returned an unparsable score for concept={concept}: {raw!r}")
                score = float(match.group(0))
            if not 0.0 <= score <= 1.0:
                raise RuntimeError(f"Claude score out of [0,1] for concept={concept}: {score}")
            scores.append(score)
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
