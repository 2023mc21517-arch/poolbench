"""
poolbench/sae_loader.py
Load a pre-trained public SAE for a given model + layer.

Requires the optional dependency:
    pip install "poolbench[sae]"    # i.e. sae-lens >= 3.0.0

Supported models and their sae-lens release identifiers:

  mistral_7b  → jbloom/Mistral_7B_v0.1_SAEs  (mistral-7b-res-wg)
                Layers 8, 16, 24 available (hook_resid_post)

  llama3_8b   → llama_scope_lxr_8x
                NousResearch/Meta-Llama-3.1-8B  (LLaMA-Scope)
                Layers 16, 24, 31 (hook_resid_post)

  gemma2_9b   → gemma-scope-9b-pt-res
                google/gemma-2-9b (GemmaScope)
                Layers 14, 28, 41 (hook_resid_post)

The layer→sae_id mapping uses the TransformerLens convention:
    blocks.{layer}.hook_resid_post

Returns a SAE object exposing:
  .encode(h: Tensor) → (batch, d_sae) feature activations
  .W_dec             → (d_sae, d_model) decoder weight tensor
"""
from __future__ import annotations

from poolbench.logger import get_logger

log = get_logger("poolbench.sae_loader")

# sae-lens release strings and layer id templates per model
# fmt: off
_SAE_RELEASES: dict[str, dict] = {
    "mistral_7b": {
        "release":   "mistral-7b-res-wg",
        "sae_id_tpl": "blocks.{layer}.hook_resid_pre",
        "layers":    [8, 16, 24],
    },
    "llama3_8b": {
        "release":   "llama_scope_lxr_8x",
        "sae_id_tpl": "l{layer}r_8x",
        "layers":    [16, 24, 31],
    },
    "gemma2_9b": {
        "release":   "gemma-scope-9b-pt-res-canonical",
        "sae_id_tpl": "layer_{layer}/width_16k/canonical",
        "layers":    [14, 20, 28],
    },
}
# fmt: on

# Module-level cache so we only download/load once per (model, layer) pair
_sae_cache: dict[tuple[str, int], object] = {}


def load_sae(model_name: str, layer: int):
    """
    Return a pre-trained SAE for `model_name` at residual-stream `layer`.

    Caches across calls. Returns None with a warning if:
      - sae-lens is not installed
      - the model is not in the supported list
      - the layer is not covered by any public release

    Callers should treat None as "SAE unavailable; fall back to C1 DifMean".
    """
    cache_key = (model_name, layer)
    if cache_key in _sae_cache:
        return _sae_cache[cache_key]

    cfg = _SAE_RELEASES.get(model_name)
    if cfg is None:
        raise RuntimeError(
            f"[sae_loader] No public SAE release registered for '{model_name}'. "
            "Run with --skip_sae_interp if you want to skip Step 8."
        )

    if layer not in cfg["layers"]:
        raise RuntimeError(
            f"[sae_loader] Layer {layer} not in supported SAE layers {cfg['layers']} "
            f"for '{model_name}'. Run with --skip_sae_interp if you want to skip Step 8."
        )

    try:
        from sae_lens import SAE  # noqa: PLC0415
    except ImportError:
        raise RuntimeError(
            "[sae_loader] sae-lens is not installed. "
            "Install with: pip install 'poolbench[sae]' or pip install sae-lens>=3.0.0 . "
            "Run with --skip_sae_interp if you want to skip Step 8."
        )

    release = cfg["release"]
    sae_id  = cfg["sae_id_tpl"].format(layer=layer)
    log.info(f"[sae_loader] Loading SAE  release={release}  sae_id={sae_id} ...")
    try:
        sae, _cfg_dict, _log_sparsity = SAE.from_pretrained(
            release=release,
            sae_id=sae_id,
        )
        sae = sae.eval()
        log.info(
            f"[sae_loader] Loaded SAE for {model_name} L{layer}: "
            f"d_in={sae.cfg.d_in}  d_sae={sae.cfg.d_sae}"
        )
    except Exception as exc:
        raise RuntimeError(
            f"[sae_loader] Failed to load SAE for {model_name} L{layer} "
            f"(release={release}, sae_id={sae_id}): {exc}. "
            "Run with --skip_sae_interp if you want to skip Step 8."
        ) from exc

    _sae_cache[cache_key] = sae
    return sae


def clear_sae_cache() -> None:
    """Release all cached SAE objects (frees GPU/CPU memory)."""
    _sae_cache.clear()
