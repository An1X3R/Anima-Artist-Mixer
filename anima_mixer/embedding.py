"""Post-adapter embedding mixing without patching individual attention layers."""

import logging

import torch
import torch.nn.functional as F

from .alignment import align_artist_embeddings, align_base_context
from .constants import ALIGN_BASE_ANCHORED, ALIGN_SHARED_BASE_IDS
from .math_utils import project_perpendicular
from .parsing import normalize_weights
from .patching import (
    adapter_mixer_state_is_active,
    begin_mixer_execution,
    broadcast_batch,
    call_with_mixer_owner,
    clear_mixer_run_state,
    execution_tensor_signature,
    MixerFatalError,
    preprocess_one,
    resolve_clone_local_mixer_wrapper,
    resolve_mask,
    resolve_strengths,
    resolve_multigpu_worker_wrapper,
    runtime_input_signature,
    should_reraise,
    tensor_cache_signature,
)

logger = logging.getLogger(__name__)


def right_pad_embedding(embedding, target_length):
    """Right-pad a [B, T, D] embedding with zero token rows."""
    if not torch.is_tensor(embedding) or embedding.dim() != 3:
        raise ValueError(
            f"expected a [batch, tokens, channels] tensor, got "
            f"{type(embedding).__name__} {getattr(embedding, 'shape', None)}"
        )
    target_length = int(target_length)
    if target_length < embedding.shape[1]:
        raise ValueError(
            f"target length {target_length} is shorter than {embedding.shape[1]}"
        )
    if target_length == embedding.shape[1]:
        return embedding
    return F.pad(embedding, (0, 0, 0, target_length - embedding.shape[1]))


def pad_embeddings_to_longest(embeddings):
    if not embeddings:
        return []
    feature_dims = {int(embedding.shape[-1]) for embedding in embeddings}
    if len(feature_dims) != 1:
        raise ValueError(f"adapter embedding widths differ: {sorted(feature_dims)}")
    longest = max(int(embedding.shape[1]) for embedding in embeddings)
    return [right_pad_embedding(embedding, longest) for embedding in embeddings]


def weighted_embedding_sum(embeddings, weights, normalize=True):
    """Align adapter outputs and form their weighted sum in float32."""
    if not embeddings:
        raise ValueError("at least one artist embedding is required")
    if len(embeddings) != len(weights):
        raise ValueError(
            f"artist embedding/weight count differs: {len(embeddings)} != {len(weights)}"
        )

    resolved_weights = (
        normalize_weights(weights) if normalize else [float(weight) for weight in weights]
    )
    target_batch = max(int(embedding.shape[0]) for embedding in embeddings)
    aligned = pad_embeddings_to_longest([
        broadcast_batch(embedding, target_batch) for embedding in embeddings
    ])

    output_dtype = aligned[0].dtype
    total = torch.zeros_like(aligned[0], dtype=torch.float32)
    for embedding, weight in zip(aligned, resolved_weights):
        total.add_(embedding.to(torch.float32), alpha=float(weight))
    return total.to(output_dtype)


def mix_projected_context(base, artist_sum, strengths, mask, fallback_base=None):
    """Apply base + strength * perpendicular(artist_sum - base, base) per token."""
    if not torch.is_tensor(base) or base.dim() != 3:
        raise ValueError(f"base context must be [B, T, D], got {getattr(base, 'shape', None)}")
    if not torch.is_tensor(artist_sum) or artist_sum.dim() != 3:
        raise ValueError(
            f"artist context must be [B, T, D], got {getattr(artist_sum, 'shape', None)}"
        )
    if base.shape[-1] != artist_sum.shape[-1]:
        raise ValueError(
            f"base/artist embedding widths differ: {base.shape[-1]} != {artist_sum.shape[-1]}"
        )

    fallback_base = base if fallback_base is None else fallback_base
    if not torch.is_tensor(fallback_base) or fallback_base.dim() != 3:
        raise ValueError(
            f"fallback base must be [B, T, D], got "
            f"{getattr(fallback_base, 'shape', None)}"
        )
    if fallback_base.shape[0] != base.shape[0]:
        raise ValueError(
            f"base/fallback batches differ: {base.shape[0]} != {fallback_base.shape[0]}"
        )
    if fallback_base.shape[-1] != base.shape[-1]:
        raise ValueError(
            f"base/fallback widths differ: {base.shape[-1]} != "
            f"{fallback_base.shape[-1]}"
        )

    artist_sum = broadcast_batch(artist_sum, base.shape[0]).to(
        device=base.device, dtype=base.dtype,
    )
    fallback_base = fallback_base.to(device=base.device, dtype=base.dtype)
    target_length = max(
        int(base.shape[1]),
        int(artist_sum.shape[1]),
        int(fallback_base.shape[1]),
    )
    base = right_pad_embedding(base, target_length)
    artist_sum = right_pad_embedding(artist_sum, target_length)
    fallback_base = right_pad_embedding(fallback_base, target_length)

    if len(mask) != base.shape[0] or len(strengths) != base.shape[0]:
        raise ValueError(
            f"CFG row metadata differs from context batch: "
            f"mask={len(mask)}, strengths={len(strengths)}, batch={base.shape[0]}"
        )

    delta_perp = project_perpendicular(artist_sum - base, base)
    row_strength = torch.tensor(
        strengths, device=base.device, dtype=base.dtype,
    ).view(base.shape[0], 1, 1)
    mixed = base + row_strength * delta_perp
    row_mask = torch.tensor(
        mask, device=base.device, dtype=torch.bool,
    ).view(base.shape[0], 1, 1)
    return torch.where(row_mask, mixed, fallback_base)


def build_artist_embedding_sum(state, ref_context, dm=None):
    """Run artist prompts through the model adapter once and cache their sum."""
    alignment_mode = state["alignment_mode"]
    dm = state["dm_ref"] if dm is None else dm
    cache_key = (
        state.get("_cache_namespace"),
        state.get("_model_weight_patch_identity"),
        runtime_input_signature(state),
        id(dm),
        ref_context.device.type,
        ref_context.device.index,
        str(ref_context.dtype),
        alignment_mode,
    )
    cache = state.setdefault("_artist_embedding_cache", {})
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    embeddings = []
    labels = state.get("labels") or []
    for index, (raw, artist_ids, artist_weights) in enumerate(zip(
        state["raws"], state["ids_list"], state["t5_weights_list"],
    )):
        if alignment_mode == ALIGN_SHARED_BASE_IDS:
            target_ids = state.get("base_ids")
            target_weights = state.get("base_t5_weights")
            if target_ids is None:
                raise ValueError(
                    "shared_base_ids requires t5xxl_ids in the base conditioning"
                )
        elif alignment_mode == ALIGN_BASE_ANCHORED:
            target_ids = artist_ids
            target_weights = artist_weights
        else:
            raise ValueError(f"unsupported alignment mode {alignment_mode!r}")

        try:
            embedding = preprocess_one(
                dm,
                raw,
                target_ids,
                target_weights,
                ref_context.device,
                ref_context.dtype,
            )
        except BaseException as error:
            if should_reraise(error):
                clear_mixer_run_state(state, interrupted=True)
                raise
            label = labels[index] if index < len(labels) else f"#{index}"
            raise ValueError(
                f"failed to build post-adapter embedding for artist {label!r}: {error}"
            ) from error
        embeddings.append(embedding)

    if alignment_mode == ALIGN_BASE_ANCHORED:
        embeddings = align_artist_embeddings(
            embeddings,
            state["alignment_plan"],
        )

    artist_sum = weighted_embedding_sum(
        embeddings,
        state["user_weights"],
        normalize=state.get("normalize_weights", True),
    ).detach()
    # Dynamic-VRAM/quantized Adapter weights may run on auxiliary CUDA streams.
    # Finish that one cached Adapter build before the main denoiser reuses or
    # unloads the same weights.  This is once per cache miss, never per sigma.
    if torch.cuda.is_available():
        sync_device = ref_context.device if ref_context.device.type == "cuda" else None
        torch.cuda.synchronize(sync_device)
    cache[cache_key] = artist_sum
    return artist_sum


def _call_underlying(prev_wrapper, apply_model, options, state=None):
    def _call():
        if prev_wrapper is not None:
            return prev_wrapper(apply_model, options)
        return apply_model(
            options["input"],
            options["timestep"],
            **options["c"],
        )

    if state is None:
        return _call()
    return call_with_mixer_owner(state, apply_model, _call)


def _mixed_context_cache_key(state, context, mask, strengths):
    return (
        state.get("_cache_namespace"),
        state.get("_model_weight_patch_identity"),
        runtime_input_signature(state),
        execution_tensor_signature(state, context),
        tuple(bool(value) for value in mask),
        tuple(float(value) for value in strengths),
    )


def _cached_mixed_context(state, context, cache_key):
    entry = state.get("_mixed_context_cache")
    if not isinstance(entry, dict):
        return None
    if entry.get("source") is not context or entry.get("key") != cache_key:
        return None
    mixed = entry.get("mixed")
    if not torch.is_tensor(mixed):
        return None
    if entry.get("mixed_signature") != tensor_cache_signature(mixed):
        return None
    return mixed


def make_adapter_embedding_wrapper(state, prev_wrapper):
    """Replace post-adapter context at the model boundary, preserving wrapper chains."""
    def _wrapper_body(apply_model, options):
        clone_wrapper = resolve_clone_local_mixer_wrapper(
            apply_model,
            wrapper,
            state,
        )
        if clone_wrapper is not None:
            return clone_wrapper(apply_model, options)
        if not adapter_mixer_state_is_active(state, apply_model=apply_model):
            return _call_underlying(prev_wrapper, apply_model, options, state)
        raw_c = options.get("c") or {}
        transformer_options = raw_c.get("transformer_options") or {}
        is_multigpu = (
            isinstance(transformer_options, dict)
            and transformer_options.get("multigpu_thread_device") is not None
        )
        if is_multigpu:
            worker_wrapper = resolve_multigpu_worker_wrapper(
                apply_model,
                options,
                wrapper,
            )
            if worker_wrapper is not None:
                # The multigpu sampler calls the main model-options wrapper for
                # every clone.  Re-enter through the clone-local rebound
                # wrapper so its dm_ref, caches, and failure flags belong to
                # the worker model rather than the first GPU.
                worker_options = dict(options)
                worker_options["_anima_mixer_worker_dispatch"] = True
                return worker_wrapper(apply_model, worker_options)
        is_run_start, _owner_changed = begin_mixer_execution(
            state,
            apply_model,
            options.get("timestep"),
            owner_token_override=(
                ("multigpu_wrapper", id(state)) if is_multigpu else None
            ),
        )
        if is_multigpu:
            # ComfyUI's multigpu sampler invokes the main model-options wrapper
            # concurrently for every device.  Keep this wrapper's owner token
            # stable and avoid sharing a live mixed-context entry across workers.
            state["_multigpu_call"] = True
        # Always hand the result to the optional sigma wrapper.  ``False`` is
        # meaningful: it tells the Adapter path that begin() already ran for
        # this call, preventing the sigma wrapper from running it a second time.
        state["_adapter_mixer_run_start"] = bool(is_run_start)
        if state.get("_embedding_mixer_failed", False):
            return _call_underlying(prev_wrapper, apply_model, options, state)

        c = raw_c
        context_key = None
        for key in ("c_crossattn", "context"):
            if torch.is_tensor(c.get(key)):
                context_key = key
                break
        if context_key is None:
            if not state.get("_warned_no_context", False):
                logger.warning(
                    "[AnimaAdapterMixer] no tensor context was available; "
                    "the original model context is used."
                )
                state["_warned_no_context"] = True
            return _call_underlying(prev_wrapper, apply_model, options, state)

        try:
            context = c[context_key]
            cou = options.get("cond_or_uncond")
            if cou is None:
                transformer_options = c.get("transformer_options") or {}
                cou = transformer_options.get("cond_or_uncond")

            batch_size = int(context.shape[0])
            mask = resolve_mask(
                cou,
                batch_size,
                state.get("apply_to_uncond", False),
                state,
            )
            strengths = resolve_strengths(
                cou,
                batch_size,
                state.get("apply_to_uncond", False),
                state["strength"],
                state.get("uncond_strength", 1.0),
            )
            cache_key = _mixed_context_cache_key(state, context, mask, strengths)
            mixed_context = (
                None
                if is_multigpu
                else _cached_mixed_context(state, context, cache_key)
            )
            if mixed_context is None:
                # ``begin_mixer_execution`` pins ``dm_ref`` to the selected
                # clone. Resolving through BaseModel.current_patcher here would
                # reintroduce the sibling-clone drift the lifecycle repair
                # explicitly filters out.
                active_dm = state.get("dm_ref")
                artist_sum = call_with_mixer_owner(
                    state,
                    apply_model,
                    build_artist_embedding_sum,
                    state,
                    context,
                    dm=active_dm,
                )
                projection_base = context
                fallback_base = context
                if state["alignment_mode"] == ALIGN_BASE_ANCHORED:
                    plan = state["alignment_plan"]
                    if plan["length"] > context.shape[1] and not all(mask):
                        raise ValueError(
                            "base-anchored context exceeds the batched base length; "
                            "unmodified CFG rows cannot be preserved without an "
                            "attention mask"
                        )
                    projection_base = align_base_context(context, plan)
                mixed_context = mix_projected_context(
                    projection_base,
                    artist_sum,
                    strengths,
                    mask,
                    fallback_base=fallback_base,
                )
                if not bool(torch.isfinite(mixed_context).all().item()):
                    raise MixerFatalError(
                        "[AnimaAdapterMixer] post-adapter mixed context contains "
                        "non-finite values; aborting this sampling pass."
                    )
                if not is_multigpu:
                    state["_mixed_context_cache"] = {
                        "source": context,
                        "key": cache_key,
                        "mixed": mixed_context,
                        "mixed_signature": tensor_cache_signature(mixed_context),
                    }

            mixed_c = dict(c)
            mixed_c[context_key] = mixed_context
            mixed_options = dict(options)
            mixed_options["c"] = mixed_c
            return _call_underlying(prev_wrapper, apply_model, mixed_options, state)
        except BaseException as error:
            if should_reraise(error):
                clear_mixer_run_state(state, interrupted=True)
                raise
            if not state.get("_warned_embedding_failure", False):
                logger.exception(
                    "[AnimaAdapterMixer] post-adapter mixing failed; "
                    "the original context will be used: %s",
                    error,
                )
                state["_warned_embedding_failure"] = True
            state["_embedding_mixer_failed"] = True
            return _call_underlying(prev_wrapper, apply_model, options, state)

    def wrapper(apply_model, options):
        # The sampler can raise InterruptProcessingException before the
        # context branch, during clone-local/multi-GPU dispatch, or on the
        # no-context fast path. Keep one abort-safe boundary around the whole
        # wrapper so those paths release Mixer-owned state as well.
        try:
            return _wrapper_body(apply_model, options)
        except BaseException as error:
            if should_reraise(error):
                clear_mixer_run_state(state, interrupted=True)
            raise

    wrapper._anima_adapter_mixer_wrapper = True
    wrapper._anima_adapter_mixer_previous = prev_wrapper
    wrapper._anima_adapter_mixer_state = state
    wrapper._anima_mixer_state = state
    wrapper._anima_mixer_previous = prev_wrapper
    wrapper._anima_mixer_factory = make_adapter_embedding_wrapper
    return wrapper


def unwrap_adapter_embedding_wrapper(wrapper):
    """Remove Adapter Mixer wrappers while preserving external wrappers."""
    seen = set()
    while (
        getattr(wrapper, "_anima_adapter_mixer_wrapper", False)
        or getattr(wrapper, "_anima_adapter_anchor_sigma_wrapper", False)
    ):
        if wrapper is None:
            break
        marker = id(wrapper)
        if marker in seen:
            break
        seen.add(marker)
        if getattr(wrapper, "_anima_adapter_mixer_wrapper", False):
            wrapper = getattr(wrapper, "_anima_adapter_mixer_previous", None)
        else:
            wrapper = getattr(wrapper, "_anima_adapter_anchor_sigma_previous", None)
    return wrapper
