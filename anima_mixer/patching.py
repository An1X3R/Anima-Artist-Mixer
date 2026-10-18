"""Model validation, patch bookkeeping, conditioning, and CFG helpers."""

import logging

import torch

logger = logging.getLogger(__name__)


def should_reraise(error):
    """Return True for errors that must abort sampling immediately."""
    for name in ("OutOfMemoryError",):
        cuda_error = getattr(getattr(torch, "cuda", None), name, None)
        if cuda_error is not None and isinstance(error, cuda_error):
            return True
        torch_error = getattr(torch, name, None)
        if torch_error is not None and isinstance(error, torch_error):
            return True
    try:
        from comfy.model_management import InterruptProcessingException
        if isinstance(error, InterruptProcessingException):
            return True
    except ImportError:
        pass
    return False


def in_stabilizer_window(state):
    threshold = state.get("stabilizer_min_sigma")
    if threshold is None:
        return True
    cur = state.get("current_sigma")
    if cur is None:
        return True
    return float(cur) >= float(threshold)


def context_fingerprint(context):
    if context is None or not torch.is_tensor(context):
        return None
    try:
        sample = context.detach()
        flat = sample.reshape(-1)
        step = max(1, flat.numel() // 1024)
        digest = flat[::step].to(torch.float32).sum().item()
        return (tuple(context.shape), str(context.dtype), round(digest, 3))
    except Exception:
        return (tuple(context.shape), str(context.dtype), None)


def tensor_cache_signature(tensor):
    """Describe one retained tensor without reading or synchronizing its values."""
    if tensor is None or not torch.is_tensor(tensor):
        return None
    try:
        version = int(tensor._version)
    except Exception:
        version = None
    try:
        stride = tuple(tensor.stride())
        storage_offset = int(tensor.storage_offset())
    except Exception:
        stride = None
        storage_offset = None
    return (
        id(tensor),
        tuple(tensor.shape),
        stride,
        storage_offset,
        str(tensor.dtype),
        tensor.device.type,
        tensor.device.index,
        version,
    )


def forward_fingerprint(state, context):
    if context is None:
        return None
    memo = state.setdefault("_ctx_fp_memo", {})
    key = id(context)
    cached = memo.get(key)
    if cached is not None:
        return cached
    fp = context_fingerprint(context)
    memo[key] = fp
    return fp


def reset_run_state(state):
    state["_disabled_layers"] = set()
    state["_disable_batched"] = False
    state["_warned_batched"] = False
    state["_warned"] = False
    state["_warned_svd"] = False
    state["_ema_cache"] = {}
    state["_static_cache"] = {}
    state["_ctx_fp_memo"] = {}
    state["_artist_chunk_cache"] = {}
    state["_anchor_failed"] = False
    state["_adapter_anchor_failed"] = False


def extract_conditioning(conditioning):
    if conditioning is None:
        return None, None, None
    if not isinstance(conditioning, (list, tuple)) or len(conditioning) == 0:
        return None, None, None
    first = conditioning[0]
    if not isinstance(first, (list, tuple)) or len(first) == 0:
        return None, None, None
    raw = first[0] if torch.is_tensor(first[0]) else None
    extra = first[1] if len(first) > 1 and isinstance(first[1], dict) else {}
    return raw, extra.get("t5xxl_ids"), extra.get("t5xxl_weights")


def unwrap_cross_attn(ca):
    from .wrapper import CrossAttnWrapper
    while isinstance(ca, CrossAttnWrapper):
        ca = ca.original_module
    return ca


class CrossAttnForwardPatch:
    """Callable object patch for cross_attn.forward.

    Patching only forward keeps the original module in the model tree, avoiding
    ComfyUI restore/state-dict path issues caused by wrapping the whole module.
    """

    _anima_artist_mixer_forward_patch = True

    def __init__(self, wrapper):
        self.wrapper = wrapper
        self.original_forward = wrapper.original

    def __call__(self, *args, **kwargs):
        return self.wrapper.forward(*args, **kwargs)


def unwrap_cross_attn_forward(ca):
    forward = getattr(ca, "forward", None)
    while isinstance(forward, CrossAttnForwardPatch):
        forward = forward.original_forward
    return forward


def make_cross_attn_forward_patch(wrapper):
    return CrossAttnForwardPatch(wrapper)


def validate_model(diffusion_model):
    if not hasattr(diffusion_model, "blocks"):
        return False, 0, 0, f"{type(diffusion_model).__name__} has no .blocks"
    blocks = diffusion_model.blocks
    if len(blocks) == 0:
        return False, 0, 0, ".blocks is empty"
    b0 = blocks[0]
    if not hasattr(b0, "cross_attn"):
        return False, 0, 0, "blocks[0] has no cross_attn"
    ca = unwrap_cross_attn(b0.cross_attn)
    if not hasattr(ca, "context_dim"):
        return False, 0, 0, "cross_attn has no context_dim"
    return True, len(blocks), int(ca.context_dim), "ok"


def preprocess_one(dm, raw, ids, weights, target_device, target_dtype):
    if ids is None:
        artist = raw.to(device=target_device, dtype=target_dtype)
        if artist.dim() == 2:
            artist = artist.unsqueeze(0)
        return artist
    raw_b = raw if raw.dim() == 3 else raw.unsqueeze(0)
    ids_b = ids if ids.dim() >= 2 else ids.unsqueeze(0)
    weights_b = None
    if weights is not None:
        if weights.dim() == 1:
            weights_b = weights.unsqueeze(0).unsqueeze(-1)
        elif weights.dim() == 2:
            weights_b = weights.unsqueeze(-1)
        else:
            weights_b = weights
    raw_b = raw_b.to(device=target_device, dtype=target_dtype)
    ids_b = ids_b.to(device=target_device)
    if weights_b is not None:
        weights_b = weights_b.to(device=target_device, dtype=target_dtype)
    with torch.inference_mode():
        return dm.preprocess_text_embeds(raw_b, ids_b, t5xxl_weights=weights_b)


def build_artists(state, ref_context):
    if state.get("individuals") is not None:
        return state["individuals"], state["real_lens"]
    dm = state["dm_ref"]
    individuals, real_lens = [], []
    for raw, ids, w_t in zip(state["raws"], state["ids_list"], state["w_list"]):
        artist = preprocess_one(dm, raw, ids, w_t, ref_context.device, ref_context.dtype)
        individuals.append(artist)
        real_lens.append(int(ids.shape[-1]) if ids is not None else artist.shape[1])
    state["individuals"] = individuals
    state["real_lens"] = real_lens
    return individuals, real_lens


def broadcast_batch(t, batch_size):
    if t.shape[0] == batch_size:
        return t
    if t.shape[0] == 1:
        return t.expand(batch_size, -1, -1)
    if batch_size % t.shape[0] == 0:
        return t.repeat(batch_size // t.shape[0], 1, 1)
    return t[:1].expand(batch_size, -1, -1)


def _expand_cond_flags(cou, batch_size):
    if cou is not None and len(cou) > 0:
        if len(cou) == batch_size:
            return [c == 0 for c in cou]
        if batch_size % len(cou) == 0:
            chunk = batch_size // len(cou)
            flags = []
            for c in cou:
                flags.extend([c == 0] * chunk)
            return flags
    return None


def resolve_mask(cou, batch_size, apply_to_uncond, state):
    """Build a per-row mask from ComfyUI's cond_or_uncond marker.

    ComfyUI can batch several latents per cond row. In that case
    len(cond_or_uncond) is smaller than batch_size and markers must be expanded
    over contiguous row chunks; otherwise the old fallback injected into
    uncond rows and weakened CFG.
    """
    if apply_to_uncond:
        return [True] * batch_size
    mask = _expand_cond_flags(cou, batch_size)
    if mask is not None:
        return mask
    if not state.get("_warned", False):
        logger.warning(
            "[AnimaCrossAttn] cond_or_uncond markers unusable (got=%s, batch=%d); "
            "falling back to injecting into every row. CFG may weaken.",
            cou, batch_size,
        )
        state["_warned"] = True
    return [True] * batch_size


def resolve_strengths(cou, batch_size, apply_to_uncond, strength, uncond_strength):
    strength = float(strength)
    if not apply_to_uncond:
        return [strength] * batch_size
    cond_flags = _expand_cond_flags(cou, batch_size)
    if cond_flags is None:
        return [strength] * batch_size
    uncond_strength = max(0.0, min(1.0, float(uncond_strength)))
    return [
        strength if is_cond else strength * uncond_strength
        for is_cond in cond_flags
    ]


def in_sigma_range(state):
    rng = state.get("sigma_range")
    if rng is None:
        return True
    cur = state.get("current_sigma")
    if cur is None:
        return True
    lo, hi = rng
    return lo <= cur <= hi
