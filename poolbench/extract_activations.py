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

def load_model(
    model_name: str,
    hf_id: str,
    device: str = "cuda",
    attn_implementation: str | None = None,
):
    """
    Load model + tokenizer for the given model_name.
    Returns (model, tokenizer).

    Memory:
      - All models loaded in bfloat16 (4-bit quantisation NOT used — activations
        must be full-precision for reliable AUROC measurements).
      - For 8B models: at least 16 GB VRAM required. For Gemma-2 9B: 20 GB.
      - Mamba2: loaded via MambaForCausalLM from transformers (>=4.40.0 required).

    attn_implementation:
      - None / omitted → "eager" for transformer causal LMs (required by activation
        extraction because output_attentions=True is incompatible with flash/SDPA).
      - "sdpa"          → PyTorch scaled-dot-product attention (torch>=2.0), uses
        flash attention automatically on Ampere/Hopper GPUs.  Use this when loading
        purely for generation (D2/D3 steps) to get throughput gains on H100.
      - "flash_attention_2" → requires the flash-attn package to be installed.
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
            # default to eager attention for Transformer causal LMs.
            # Pass attn_implementation="sdpa" or "flash_attention_2" when the
            # caller only needs generation throughput (no attention weights).
            model_kwargs["attn_implementation"] = attn_implementation if attn_implementation is not None else "eager"
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

def _extract_batch_multilayer(
    model,
    tokenizer,
    model_name: str,
    texts: list[str],
    layer_indices: list[int],
    device: str,
    activation_dtype: np.dtype = np.dtype(np.float32),
) -> dict[int, list[dict]]:
    """
    Extract activations for one batch at multiple layers in a single forward pass.
    Registers hooks on all requested layers simultaneously so the model is only
    called once regardless of how many candidate layers are requested.
    Returns {layer_idx: list_of_per_passage_dicts}.
    """
    import torch  # noqa: PLC0415
    is_ed = _is_encoder_decoder(model_name)
    is_eo = _is_encoder_only(model_name)
    max_length = 512 if is_ed or is_eo else 600

    enc, offset_mapping = _tokenise_batch(texts, tokenizer, max_length=max_length,
                                          is_encoder_decoder=is_ed)
    enc = {k: v.to(device) for k, v in enc.items()}

    # Register one hook per target layer — all captured in a single forward pass
    hooks: dict[int, _LayerCaptureHook] = {}
    for layer_idx in layer_indices:
        hook = _LayerCaptureHook()
        hook.register(_get_layer_module(model, model_name, layer_idx))
        hooks[layer_idx] = hook

    try:
        import torch as _t  # noqa: PLC0415
        with _t.inference_mode():
            if is_ed:
                _ = model.encoder(**enc)
            else:
                _ = model(**enc, output_attentions=not _is_ssm(model_name))
    except Exception as exc:
        for h in hooks.values():
            h.remove()
        raise RuntimeError(f"forward pass error during extraction: {exc}") from exc

    for h in hooks.values():
        h.remove()

    attention_mask = enc["attention_mask"].cpu().numpy()   # (batch, seq_len)
    input_ids_cpu  = enc["input_ids"].cpu().numpy()        # (batch, seq_len)

    results_per_layer: dict[int, list[dict]] = {}
    for layer_idx, hook in hooks.items():
        hidden_batch = hook.hidden   # (batch, seq_len, d_model)
        attn_batch   = hook.attn     # (batch, n_heads, seq_len, seq_len) or None

        if hidden_batch is None:
            raise RuntimeError(
                f"Layer hook captured no hidden states for {model_name} layer {layer_idx}"
            )

        layer_results: list[dict] = []
        for i, text in enumerate(texts):
            seq_len = int(attention_mask[i].sum())
            h_full  = hidden_batch[i]

            if is_eo or is_ed:
                h       = h_full[:seq_len].astype(activation_dtype)
                offsets = [(int(s), int(e)) for s, e in offset_mapping[i, :seq_len].numpy()]
                token_ids_clean = input_ids_cpu[i, :seq_len].tolist()
            else:
                # Causal LM with left-padding: real tokens are at the end
                padded_len = h_full.shape[0]
                start_pos  = padded_len - seq_len
                h          = h_full[start_pos:].astype(activation_dtype)
                offsets    = [(int(s), int(e)) for s, e in
                              offset_mapping[i, start_pos:padded_len].numpy()]
                token_ids_clean = input_ids_cpu[i, start_pos:padded_len].tolist()

            attn = None
            if attn_batch is not None and not _is_ssm(model_name):
                if is_eo or is_ed:
                    attn_full = attn_batch[i, :, :seq_len, :seq_len]
                else:
                    attn_full = attn_batch[i, :, start_pos:, start_pos:]
                # Compact to per-head mean inflow (n_heads, seq_len) — ~20 MB saving per passage
                attn = attn_full.mean(axis=1).astype(np.float16)

            layer_results.append({
                "hidden":         h,
                "offset_mapping": offsets,
                "text":           text,
                "token_ids":      token_ids_clean,
                "attn_weights":   attn,
            })

        results_per_layer[layer_idx] = layer_results

    return results_per_layer


def _extract_batch(
    model,
    tokenizer,
    model_name: str,
    texts: list[str],
    layer_idx: int,
    device: str,
    activation_dtype: np.dtype = np.dtype(np.float32),
) -> list[dict]:
    """Single-layer wrapper around _extract_batch_multilayer (kept for compatibility)."""
    return _extract_batch_multilayer(
        model, tokenizer, model_name, texts, [layer_idx], device, activation_dtype
    )[layer_idx]


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

    if (concept_corpus_dir / "train_pos.jsonl").exists():
        concept_dirs = [concept_corpus_dir]
    else:
        concept_dirs = [d for d in sorted(concept_corpus_dir.iterdir()) if d.is_dir()]

    # Early-exit before loading the model if all activations already exist
    if skip_existing:
        all_exist = all(
            (out_dir / model_name / f"layer_{li}" / f"{cd.name}_{p}_{s}.npy").exists()
            for cd in concept_dirs
            for li in candidate_layers
            for p in ("train", "test")
            for s in ("pos", "neg")
        )
        if all_exist:
            log.info(f"  [extract] {model_name}: all activations exist — skipping model load")
            return

    model, tokenizer   = load_model(model_name, hf_id, device=device)

    n_layers   = len(candidate_layers)
    n_concepts = len(concept_dirs)
    log.info(f"  [extract] {model_name}: {n_layers} layers × {n_concepts} concepts  "
             f"batch={batch_size}  activation_save_dtype={activation_dtype.name}  "
             f"GPU: {gpu_mem_str(device)}")

    # Create all layer output dirs up front
    layer_dirs: dict[int, Path] = {}
    for layer_idx in candidate_layers:
        layer_dir = out_dir / model_name / f"layer_{layer_idx}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        layer_dirs[layer_idx] = layer_dir

    # Reordered loop: concept → partition → split → batch.
    # All candidate layers are captured in a single forward pass per batch,
    # reducing GPU forward passes by a factor of len(candidate_layers).
    for concept_dir in concept_dirs:
        concept_name = concept_dir.name
        for partition in ("train", "test"):
            for split in ("pos", "neg"):
                jsonl_path = concept_dir / f"{partition}_{split}.jsonl"
                if not jsonl_path.exists():
                    log.warning(f"  [extract] {concept_name}/{partition}/{split}: not found")
                    continue

                # Which layers still need this (concept, partition, split)?
                layers_needed = [
                    li for li in candidate_layers
                    if not (skip_existing and
                            (layer_dirs[li] / f"{concept_name}_{partition}_{split}.npy").exists())
                ]
                layers_skipped = [li for li in candidate_layers if li not in layers_needed]
                for li in layers_skipped:
                    log.info(f"  [extract] {concept_name}_{partition}_{split} L{li} exists — skipping")
                if not layers_needed:
                    continue

                records = load_jsonl(str(jsonl_path))
                texts   = [r["text"] for r in records]
                log.info(f"  [extract] {concept_name}/{partition}/{split} "
                         f"layers={layers_needed}: {len(texts)} passages  "
                         f"GPU: {gpu_mem_str(device)}")

                # Accumulate per-layer results across all batches
                all_items_per_layer: dict[int, list[dict]] = {li: [] for li in layers_needed}
                for batch_start in tqdm(range(0, len(texts), batch_size),
                                        desc=f"{concept_name}_{partition}_{split}",
                                        leave=False):
                    batch_texts = texts[batch_start: batch_start + batch_size]
                    batch_results = _extract_batch_multilayer(
                        model, tokenizer, model_name, batch_texts,
                        layers_needed, device, activation_dtype=activation_dtype,
                    )
                    for li, items in batch_results.items():
                        if len(items) != len(batch_texts):
                            raise RuntimeError(
                                f"Extraction returned {len(items)} items for batch of "
                                f"{len(batch_texts)} ({concept_name}/{partition}/{split} L{li})"
                            )
                        all_items_per_layer[li].extend(items)

                # Save each layer's output
                for li in layers_needed:
                    all_items = all_items_per_layer[li]
                    arr = np.empty(len(all_items), dtype=object)
                    for k, item in enumerate(all_items):
                        arr[k] = item
                    out_path = layer_dirs[li] / f"{concept_name}_{partition}_{split}.npy"
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
