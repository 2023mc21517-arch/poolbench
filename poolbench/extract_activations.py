"""
src/extract_activations.py
GPU-side activation extraction for PoolBench.

Supports:
  - Causal LM (Llama 3.1 8B, Gemma-2 9B, Mistral 7B)
  - Encoder-Decoder (FLAN-T5 XL) — encoder hidden states only
  - Encoder-only (BERT base uncased)

Each call saves per-concept .npy files (object arrays of dicts) to:
  {out_dir}/{model_name}/{concept}_pos.npy
  {out_dir}/{model_name}/{concept}_neg.npy

Each element in the saved object array is a dict:
    "hidden"          — (seq_len, d_model) float16 by default (set
                         POOLBENCH_ACTIVATION_SAVE_DTYPE=float32 to override);
                         downstream pooling casts back to float32 for computation
  "offset_mapping"  — list of (char_start, char_end) int tuples
  "text"            — original passage string (for L1–L3 spaCy)
  "token_ids"       — list of int HF token IDs (for L4 / S3_SIF)
    "attn_weights"    — compact per-head token inflow (n_heads, seq_len) float16,
                                             or None for Mamba2. Older files may contain full
                                             (n_heads, seq_len, seq_len) matrices.

Usage (called from run_model.py):
  from poolbench.extract_activations import extract_activations_for_model
  extract_activations_for_model(
      model_name="llama3_8b",
      concept_corpus_dir="data/corpora",
      out_dir="results/activations",
      candidate_layers=[16, 24, 31],
      batch_size=8,
      device="cuda:0",
  )
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from poolbench.logger import get_logger, gpu_mem_str, free_gpu_memory

log = get_logger("poolbench.extract")


# ── architecture detection ────────────────────────────────────────────────────

# All benchmark models are causal decoder LMs—no encoder-only, encoder-decoder, or SSM architectures
# in scope. These sets are kept empty; the helpers below always return False.
_ENCODER_DECODER_MODELS: set[str] = set()
_ENCODER_ONLY_MODELS: set[str]    = set()
_SSM_MODELS: set[str]             = set()


def _is_encoder_decoder(model_name: str) -> bool:
    return model_name in _ENCODER_DECODER_MODELS


def _is_encoder_only(model_name: str) -> bool:
    return model_name in _ENCODER_ONLY_MODELS


def _is_ssm(model_name: str) -> bool:
    return model_name in _SSM_MODELS


def _activation_save_dtype(name: str | None = None) -> np.dtype:
    """Resolve on-disk activation dtype. Defaults to float16 (50% disk savings; downstream pooling upcasts to float32 before computation)."""
    dtype_name = (name or os.environ.get("POOLBENCH_ACTIVATION_SAVE_DTYPE", "float16")).lower()
    if dtype_name in {"float32", "fp32"}:
        return np.dtype(np.float32)
    if dtype_name in {"float16", "fp16"}:
        return np.dtype(np.float16)
    raise ValueError(
        "POOLBENCH_ACTIVATION_SAVE_DTYPE must be 'float32' or 'float16'; "
        f"got {dtype_name!r}"
    )


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(model_name: str, hf_id: str, device: str = "cuda"):
    """
    Load model + tokenizer for the given model_name.
    Returns (model, tokenizer).

    Memory:
      - All models loaded in bfloat16 (4-bit quantisation NOT used — activations
        must be full-precision for reliable AUROC measurements).
      - For 8B models: at least 16 GB VRAM required. For Gemma-2 9B: 20 GB.
      - Mamba2: loaded via MambaForCausalLM from transformers (>=4.40.0 required).
    """
    import torch  # noqa: PLC0415
    from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: PLC0415
    from transformers import AutoModelForSeq2SeqLM, AutoModelForMaskedLM

    log.info(f"  [extract] Loading {model_name} ({hf_id}) on {device}  GPU before: {gpu_mem_str(device)}")
    tokenizer_kwargs = {
        "trust_remote_code": True,
        "padding_side": "left",  # for causal LMs
    }
    tokenizer = AutoTokenizer.from_pretrained(hf_id, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": device,
        "trust_remote_code": True,
    }

    if _is_encoder_decoder(model_name):
        model = AutoModelForSeq2SeqLM.from_pretrained(hf_id, **model_kwargs)
    elif _is_encoder_only(model_name):
        model_kwargs["padding_side"] = "right"  # BERT convention
        tokenizer.padding_side = "right"
        model = AutoModelForMaskedLM.from_pretrained(hf_id, **model_kwargs)
    else:
        # Causal LM or SSM (Mamba2) — both handled by AutoModelForCausalLM
        if not _is_ssm(model_name):
            # SDPA/Flash attention backends do not return attention maps with
            # output_attentions=True. S1 and S3 need token attention inflow, so
            # force eager attention for Transformer causal LMs.
            model_kwargs["attn_implementation"] = "eager"
        model = AutoModelForCausalLM.from_pretrained(hf_id, **model_kwargs)

    model.eval()
    log.info(f"  [extract] Model loaded  GPU after: {gpu_mem_str(device)}")
    return model, tokenizer


# ── Hook utilities ─────────────────────────────────────────────────────────────

class _LayerCaptureHook:
    """Forward hook that stores the hidden state and (optionally) attention weights."""

    def __init__(self):
        self.hidden: Optional[np.ndarray]    = None
        self.attn:   Optional[np.ndarray]    = None
        self._handle = None

    def hook_fn(self, module, input, output):
        import torch  # noqa: PLC0415
        if isinstance(output, tuple):
            h        = output[0]
            attn_map = output[1] if len(output) > 1 else None
        else:
            h, attn_map = output, None

        self.hidden = h.detach().float().cpu().numpy()  # (batch, seq_len, d_model)
        if attn_map is not None and torch.is_tensor(attn_map):
            self.attn = attn_map.detach().float().cpu().numpy()  # (batch, n_heads, L, L)

    def register(self, layer_module):
        self._handle = layer_module.register_forward_hook(self.hook_fn)
        return self

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def _get_layer_module(model, model_name: str, layer_idx: int):
    """Return the transformer block at layer_idx for arbitrary architectures."""
    try:
        # Llama, Mistral, Qwen, Gemma-2 — model.model.layers[i]
        return model.model.layers[layer_idx]
    except AttributeError:
        pass
    try:
        # BERT — model.bert.encoder.layer[i]
        return model.bert.encoder.layer[layer_idx]
    except AttributeError:
        pass
    try:
        # FLAN-T5 encoder — model.encoder.block[i]
        return model.encoder.block[layer_idx]
    except AttributeError:
        pass
    try:
        # Mamba2 — model.backbone.layers[i]
        return model.backbone.layers[layer_idx]
    except AttributeError:
        pass
    raise ValueError(f"Cannot locate layer {layer_idx} for model {model_name}. "
                     "Please add the architecture in _get_layer_module().")


# ── Tokenisation ──────────────────────────────────────────────────────────────

def _tokenise_batch(texts: list[str], tokenizer, max_length: int = 512,
                    is_encoder_decoder: bool = False):
    """Tokenise a list of texts, returning inputs dict and offset_mapping per text."""
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        add_special_tokens=True,
    )
    # offset_mapping is returned as (batch, seq_len, 2); keep on CPU
    offset_mapping = enc.pop("offset_mapping")  # remove before passing to model
    return enc, offset_mapping


# ── Core extraction ───────────────────────────────────────────────────────────

def _extract_batch(
    model,
    tokenizer,
    model_name: str,
    texts: list[str],
    layer_idx: int,
    device: str,
    activation_dtype: np.dtype = np.dtype(np.float32),
) -> list[dict]:
    """
    Extract activations for one batch at a specific layer.
    Returns a list of per-passage dicts.
    """
    import torch  # noqa: PLC0415
    is_ed = _is_encoder_decoder(model_name)
    is_eo = _is_encoder_only(model_name)
    max_length = 512 if is_ed or is_eo else 600   # slightly over 500 to be safe

    enc, offset_mapping = _tokenise_batch(texts, tokenizer, max_length=max_length,
                                          is_encoder_decoder=is_ed)
    enc = {k: v.to(device) for k, v in enc.items()}

    hook = _LayerCaptureHook()
    layer = _get_layer_module(model, model_name, layer_idx)
    hook.register(layer)

    try:
        import torch as _t  # noqa: PLC0415
        with _t.inference_mode():
            if is_ed:
                # Only run encoder; decoder output not needed
                _ = model.encoder(**enc)
            else:
                _ = model(**enc, output_attentions=not _is_ssm(model_name))
    except Exception as exc:
        hook.remove()
        raise RuntimeError(f"forward pass error during extraction (layer {layer_idx}): {exc}") from exc

    hook.remove()

    hidden_batch = hook.hidden   # (batch, seq_len, d_model) or None
    attn_batch   = hook.attn     # (batch, n_heads, seq_len, seq_len) or None

    if hidden_batch is None:
        raise RuntimeError(f"Layer hook captured no hidden states for {model_name} layer {layer_idx}")

    results = []
    attention_mask = enc["attention_mask"].cpu().numpy()  # (batch, seq_len)

    for i, text in enumerate(texts):
        seq_len = int(attention_mask[i].sum())
        h_full  = hidden_batch[i]    # (padded_seq_len, d_model)

        if is_eo or is_ed:
            # BERT / T5: no strong position direction; take from left
            h = h_full[:seq_len].astype(activation_dtype)
            offsets = [(int(s), int(e)) for s, e in offset_mapping[i, :seq_len].numpy()]
        else:
            # Causal LM with left-padding: last seq_len tokens are the real tokens
            padded_len = h_full.shape[0]
            start_pos  = padded_len - seq_len
            h          = h_full[start_pos:].astype(activation_dtype)
            offsets    = [(int(s), int(e)) for s, e in
                          offset_mapping[i, start_pos:padded_len].numpy()]

        attn = None
        if attn_batch is not None and not _is_ssm(model_name):
            if is_eo or is_ed:
                attn_full = attn_batch[i, :, :seq_len, :seq_len]
            else:
                attn_full = attn_batch[i, :, start_pos:, start_pos:]
            # Store only per-head mean inflow per token. S1 and S3_ITI_exact use
            # token inflow, not the full query×key attention matrix. This reduces
            # Llama activation files by roughly 20 MB per 400-token passage.
            attn = attn_full.mean(axis=1).astype(np.float16)  # (n_heads, seq_len)

        token_ids = enc["input_ids"][i].cpu().numpy()
        if is_eo or is_ed:
            token_ids_clean = token_ids[:seq_len].tolist()
        else:
            token_ids_clean = token_ids[start_pos:].tolist()

        results.append({
            "hidden":         h,              # (seq_len, d_model) activation_dtype on disk
            "offset_mapping": offsets,        # list of (start, end)
            "text":           text,
            "token_ids":      token_ids_clean,
            "attn_weights":   attn,           # (n_heads, seq_len) compact inflow or None
        })

    return results


# ── Public API ────────────────────────────────────────────────────────────────

def extract_activations_for_model(
    model_name: str,
    hf_id: str,
    concept_corpus_dir: str | Path,
    out_dir: str | Path,
    candidate_layers: list[int],
    batch_size: int   = 8,
    device: str       = "cuda:0",
    skip_existing: bool = True,
    activation_save_dtype: str | None = None,
) -> None:
    """
    Extract and save activations for all concepts at each candidate layer.

    For each concept × layer × split (pos/neg), saves:
        {out_dir}/{model_name}/layer_{layer_idx}/{concept}_{split}.npy

    The .npy files are numpy object arrays where each element is a dict
    (see module docstring for schema).

    candidate_layers: typically 3–5 layer indices to probe, e.g. [16, 24, 31] for 8B.
    Layers are 0-indexed.
    """
    from poolbench.utils import load_jsonl  # noqa: PLC0415
    import torch  # noqa: PLC0415
    import gc     # noqa: PLC0415

    concept_corpus_dir = Path(concept_corpus_dir)
    out_dir            = Path(out_dir)
    activation_dtype   = _activation_save_dtype(activation_save_dtype)
    model, tokenizer   = load_model(model_name, hf_id, device=device)

    if (concept_corpus_dir / "train_pos.jsonl").exists():
        concept_dirs = [concept_corpus_dir]
    else:
        concept_dirs = [d for d in sorted(concept_corpus_dir.iterdir()) if d.is_dir()]

    n_layers   = len(candidate_layers)
    n_concepts = len(concept_dirs)
    log.info(f"  [extract] {model_name}: {n_layers} layers × {n_concepts} concepts  "
             f"batch={batch_size}  activation_save_dtype={activation_dtype.name}  "
             f"GPU: {gpu_mem_str(device)}")

    for layer_idx in candidate_layers:
        layer_dir = out_dir / model_name / f"layer_{layer_idx}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"  [extract] Layer {layer_idx}  GPU: {gpu_mem_str(device)}")

        for concept_dir in concept_dirs:
            concept_name = concept_dir.name

            for split in ("pos", "neg"):
                for partition in ("train", "test"):
                    jsonl_path = concept_dir / f"{partition}_{split}.jsonl"
                    if not jsonl_path.exists():
                        log.warning(f"  [extract] {concept_name}/{split}/{partition}: not found")
                        continue

                    out_path = layer_dir / f"{concept_name}_{partition}_{split}.npy"
                    if skip_existing and out_path.exists():
                        log.info(f"  [extract] {out_path.name} exists — skipping")
                        continue

                    records = load_jsonl(str(jsonl_path))
                    texts   = [r["text"] for r in records]
                    log.info(f"  [extract] {concept_name}/{partition}/{split} L{layer_idx}: "
                             f"{len(texts)} passages  GPU: {gpu_mem_str(device)}")

                    all_items: list[dict] = []
                    for batch_start in tqdm(range(0, len(texts), batch_size),
                                            desc=f"{concept_name}_{split} L{layer_idx}",
                                            leave=False):
                        batch_texts = texts[batch_start: batch_start + batch_size]
                        items = _extract_batch(model, tokenizer, model_name,
                                               batch_texts, layer_idx, device,
                                               activation_dtype=activation_dtype)
                        if len(items) != len(batch_texts):
                            raise RuntimeError(
                                f"Extraction returned {len(items)} items for batch of {len(batch_texts)} "
                                f"({concept_name}/{partition}/{split} L{layer_idx})"
                            )
                        all_items.extend(items)

                    arr = np.empty(len(all_items), dtype=object)
                    for k, item in enumerate(all_items):
                        arr[k] = item
                    np.save(out_path, arr)
                    log.info(f"  [extract] Saved {len(all_items)} passages → {out_path}  "
                             f"GPU: {gpu_mem_str(device)}")

    # Free GPU memory after all extractions complete
    del model
    torch.cuda.empty_cache()
    gc.collect()
    log.info(f"  [extract] {model_name}: all layers done. GPU freed: {gpu_mem_str(device)}")


def load_activations(act_dir: str | Path, model_name: str,
                     layer_idx: int, concept_name: str,
                     split: str = "pos",
                     partition: str = "train") -> np.ndarray | None:
    """
    Load a saved activation object array.
    Returns np.ndarray of dicts, or None if file not found.
    """
    path = Path(act_dir) / model_name / f"layer_{layer_idx}" / f"{concept_name}_{partition}_{split}.npy"
    if not path.exists():
        return None
    return np.load(path, allow_pickle=True)
