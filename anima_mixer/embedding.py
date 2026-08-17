"""Post-adapter embedding mixing without patching individual attention layers."""

import logging

import torch
import torch.nn.functional as F

from .alignment import align_artist_embeddings, align_base_context
from .constants import ALIGN_BASE_ANCHORED, ALIGN_SHARED_BASE_IDS
from .math_utils import project_perpendicular
from .parsing import normalize_weights
# TEMP_SEMANTIC_DIAG_HOOK: remove with semantic_diagnostics.py after diagnosis.
from .semantic_diagnostics import (
    record_cache_lookup as record_semantic_cache_lookup,
    record_context as record_semantic_context,
    record_stage as record_semantic_stage,
    should_capture_context as should_capture_semantic_context,
    should_capture_stage as should_capture_semantic_stage,
)
from .patching import (
    adapter_mixer_state_is_active,
    begin_mixer_execution,
    broadcast_batch,
    call_with_mixer_owner,
    classify_tensor_diagnostic_snapshot,
    clear_mixer_run_state,
    execution_tensor_signature,
    format_tensor_diagnostic_snapshot,
    MixerFatalError,
    complete_adapter_refresh,
    pending_adapter_refresh_epoch,
    preprocess_one,
    resolve_clone_local_mixer_wrapper,
    resolve_mask,
    resolve_strengths,
    resolve_multigpu_worker_wrapper,
    runtime_input_signature,
    should_reraise,
    tensor_cache_signature,
    tensor_diagnostic_snapshot,
)

logger = logging.getLogger(__name__)


def _tensor_finite_summary(tensor):
    """Return compact diagnostics for a tensor that failed the finite check."""
    if not torch.is_tensor(tensor):
        return f"type={type(tensor).__name__}"
    try:
        finite = torch.isfinite(tensor)
        bad = int((~finite).sum().item())
        nan_count = int(torch.isnan(tensor).sum().item())
        inf_count = int(torch.isinf(tensor).sum().item())
        safe = torch.nan_to_num(
            tensor.detach().to(torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        max_abs = float(safe.abs().max().item()) if safe.numel() else 0.0
        return (
            f"shape={tuple(tensor.shape)} dtype={tensor.dtype} "
            f"device={tensor.device} bad={bad} nan={nan_count} inf={inf_count} "
            f"finite_max_abs={max_abs:.6g}"
        )
    except Exception as error:
        return (
            f"shape={tuple(tensor.shape)} dtype={tensor.dtype} "
            f"device={tensor.device} summary_error={error}"
        )


def _compact_index_ranges(indices, max_ranges=16):
    values = sorted({int(value) for value in indices})
    if not values:
        return "none"
    ranges = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append((start, previous))
        start = previous = value
    ranges.append((start, previous))
    rendered = [
        str(start) if start == end else f"{start}-{end}"
        for start, end in ranges[:max_ranges]
    ]
    if len(ranges) > max_ranges:
        rendered.append(f"...(+{len(ranges) - max_ranges} ranges)")
    return ",".join(rendered)


def _row_marker(row_markers, index):
    if row_markers is None:
        return "unknown"
    try:
        marker = row_markers[index]
        if torch.is_tensor(marker):
            marker = marker.detach().item()
        return str(marker)
    except Exception:
        return "unavailable"


def _nonfinite_tensor_layout(tensor, row_markers=None):
    """Locate bad batch/token rows without dumping conditioning values."""
    if not torch.is_tensor(tensor):
        return f"type={type(tensor).__name__}"
    try:
        bad_mask = ~torch.isfinite(tensor)
        if tensor.dim() == 3:
            batch, tokens = int(tensor.shape[0]), int(tensor.shape[1])
            channel_width = int(tensor[0, 0].numel()) if batch and tokens else 0
            bad_per_token = bad_mask.reshape(batch, tokens, -1).sum(dim=-1).cpu()
            parts = []
            for batch_index in range(batch):
                counts = bad_per_token[batch_index]
                affected = torch.nonzero(counts > 0, as_tuple=False).flatten().tolist()
                full = torch.nonzero(
                    counts == channel_width,
                    as_tuple=False,
                ).flatten().tolist() if channel_width else []
                partial = torch.nonzero(
                    (counts > 0) & (counts < channel_width),
                    as_tuple=False,
                ).flatten().tolist() if channel_width else affected
                parts.append(
                    f"batch={batch_index} marker={_row_marker(row_markers, batch_index)} "
                    f"bad={int(counts.sum().item())} "
                    f"affected_tokens={_compact_index_ranges(affected)} "
                    f"full_tokens={_compact_index_ranges(full)} "
                    f"partial_tokens={_compact_index_ranges(partial)}"
                )
            return "; ".join(parts)
        bad_indices = torch.nonzero(bad_mask, as_tuple=False).cpu().tolist()
        preview = bad_indices[:8]
        suffix = f" ...(+{len(bad_indices) - len(preview)})" if len(bad_indices) > len(preview) else ""
        return f"bad_indices={preview}{suffix}"
    except Exception as error:
        return f"layout_error={error}"


def _tensor_health(tensor):
    if tensor is None:
        return "none"
    if not torch.is_tensor(tensor):
        return f"type={type(tensor).__name__}"
    try:
        bad = int((~torch.isfinite(tensor)).sum().item())
    except Exception:
        bad = "unavailable"
    return (
        f"shape={tuple(tensor.shape)},dtype={tensor.dtype},"
        f"device={tensor.device},bad={bad}"
    )


def _raw_snapshot_at_encode(state, index=None):
    snapshots = state.get("raw_encode_snapshots")
    if not isinstance(snapshots, dict):
        return None
    if index is None:
        return snapshots.get("base")
    artists = snapshots.get("artists")
    if not isinstance(artists, (list, tuple)) or index >= len(artists):
        return None
    return artists[index]


def _raw_snapshot_at_state(state, index=None):
    if index is None:
        return state.get("base_raw_state_snapshot")
    artists = state.get("artist_raw_state_snapshots")
    if not isinstance(artists, (list, tuple)) or index >= len(artists):
        return None
    return artists[index]


def _raw_failure_entry(state, role, raw, index=None):
    encode_snapshot = _raw_snapshot_at_encode(state, index=index)
    state_snapshot = _raw_snapshot_at_state(state, index=index)
    current_snapshot = tensor_diagnostic_snapshot(raw)
    provenance = classify_tensor_diagnostic_snapshot(
        encode_snapshot,
        state_snapshot,
        current_snapshot,
        boundary="after_state",
    )
    return (
        f"role={role} provenance={provenance} "
        f"encode={{{format_tensor_diagnostic_snapshot(encode_snapshot)}}} "
        f"state={{{format_tensor_diagnostic_snapshot(state_snapshot)}}} "
        f"current={{{format_tensor_diagnostic_snapshot(current_snapshot)}}}"
    )


def _raw_failure_summary(state):
    if not isinstance(state, dict):
        return "state=unavailable"
    entries = [
        _raw_failure_entry(state, "base", state.get("base_raw")),
    ]
    labels = state.get("labels") or []
    for index, raw in enumerate(state.get("raws") or []):
        label = labels[index] if index < len(labels) else f"#{index}"
        entries.append(_raw_failure_entry(
            state,
            f"artist[{index}] label={label!r}",
            raw,
            index=index,
        ))
    return "; ".join(entries)


def _state_failure_summary(state):
    if not isinstance(state, dict):
        return "state=unavailable"
    namespace = state.get("_cache_namespace")
    prompt_length = None
    labels = state.get("labels") or []
    weights = state.get("user_weights") or []
    if isinstance(namespace, tuple):
        if len(namespace) > 2 and isinstance(namespace[2], str):
            prompt_length = len(namespace[2])
        if len(namespace) > 3 and isinstance(namespace[3], tuple):
            labels = namespace[3]
        if len(namespace) > 4 and isinstance(namespace[4], tuple):
            weights = namespace[4]
    plan = state.get("alignment_plan")
    plan_length = plan.get("length") if isinstance(plan, dict) else None
    return (
        f"run={int(state.get('_execution_index', 0) or 0)} "
        f"sigma={state.get('current_sigma')} "
        f"alignment={state.get('alignment_mode')} plan_length={plan_length} "
        f"base_prompt_chars={prompt_length} artists={len(labels)} "
        f"artist_weights={tuple(weights)} "
        f"current_patch={state.get('_model_weight_patch_identity')} "
        f"previous_distinct_patch={state.get('_shared_previous_weight_patch_identity')} "
        f"patch_history={state.get('_shared_weight_patch_history')} "
        f"base_raw=({_tensor_health(state.get('base_raw'))}) "
        f"base_ids=({_tensor_health(state.get('base_ids'))}) "
        f"base_t5_weights=({_tensor_health(state.get('base_t5_weights'))})"
    )


def _raise_nonfinite(
    stage,
    tensor,
    label=None,
    *,
    state=None,
    row_markers=None,
    context_key=None,
):
    detail = _tensor_finite_summary(tensor)
    suffix = f" label={label!r}" if label is not None else ""
    message = (
        f"[AnimaAdapterMixer] non-finite tensor at stage={stage}{suffix}; "
        f"{detail}"
    )
    logger.error(message)
    logger.error(
        "[AnimaAdapterMixerDiag] stage=%s context_key=%s layout={%s} state={%s}",
        stage,
        context_key,
        _nonfinite_tensor_layout(tensor, row_markers=row_markers),
        _state_failure_summary(state),
    )
    logger.error(
        "[AnimaAdapterMixerRawDiag] stage=%s raw_timeline={%s}",
        stage,
        _raw_failure_summary(state),
    )
    raise MixerFatalError(message)


def _ensure_finite(
    stage,
    tensor,
    label=None,
    *,
    state=None,
    row_markers=None,
    context_key=None,
):
    """Check a cache-miss tensor once and raise only when it is invalid."""
    if not torch.is_tensor(tensor):
        _raise_nonfinite(
            stage,
            tensor,
            label=label,
            state=state,
            row_markers=row_markers,
            context_key=context_key,
        )
    try:
        finite = bool(torch.isfinite(tensor).all().item())
    except Exception as error:
        message = (
            f"[AnimaAdapterMixer] finite check failed at stage={stage} "
            f"label={label!r}: {error}"
        )
        logger.error(message)
        raise MixerFatalError(message) from error
    if not finite:
        _raise_nonfinite(
            stage,
            tensor,
            label=label,
            state=state,
            row_markers=row_markers,
            context_key=context_key,
        )


def _sync_adapter_boundary(ref_context):
    if not torch.cuda.is_available():
        return
    sync_device = ref_context.device if ref_context.device.type == "cuda" else None
    torch.cuda.synchronize(sync_device)


def _tensor_probe(tensor, sample_count=64):
    """Copy a small deterministic sample after synchronization for diagnostics."""
    try:
        flat = tensor.detach().reshape(-1)
        if flat.numel() == 0:
            return torch.empty((0,), dtype=torch.float32)
        count = min(int(sample_count), int(flat.numel()))
        if count == 1:
            sample = flat[:1]
        else:
            positions = torch.linspace(
                0,
                int(flat.numel()) - 1,
                count,
                device=flat.device,
                dtype=torch.float64,
            ).round().to(dtype=torch.long)
            sample = flat.index_select(0, positions)
        return sample.to(dtype=torch.float32, device="cpu")
    except Exception:
        return None


def _log_safe_refresh_drift(warmup, retained, epoch):
    """Report finite first-pass drift while always retaining the second pass."""
    warm_probe = _tensor_probe(warmup)
    retained_probe = _tensor_probe(retained)
    if warm_probe is None or retained_probe is None:
        logger.info(
            "[AnimaAdapterMixer] completed one-shot Adapter refresh for shared "
            "abort epoch=%d (probe unavailable).",
            epoch,
        )
        return
    if tuple(warm_probe.shape) != tuple(retained_probe.shape):
        logger.warning(
            "[AnimaAdapterMixer] finite Adapter drift detected during shared "
            "abort epoch=%d refresh: probe shapes changed %s -> %s; the warm-up "
            "output was discarded.",
            epoch,
            tuple(warm_probe.shape),
            tuple(retained_probe.shape),
        )
        return
    if warm_probe.numel() == 0:
        relative_rms = 0.0
        max_delta = 0.0
    else:
        delta = retained_probe - warm_probe
        diff_rms = float(delta.square().mean().sqrt().item())
        reference_rms = float(retained_probe.square().mean().sqrt().item())
        relative_rms = diff_rms / max(reference_rms, 1e-12)
        max_delta = float(delta.abs().max().item())
    if relative_rms > 1e-3:
        logger.warning(
            "[AnimaAdapterMixer] finite Adapter drift detected during shared "
            "abort epoch=%d refresh (sample_rel_rms=%.6g, sample_max_delta=%.6g); "
            "the warm-up output was discarded and only the synchronized second "
            "pass will be cached.",
            epoch,
            relative_rms,
            max_delta,
        )
    else:
        logger.info(
            "[AnimaAdapterMixer] completed one-shot Adapter refresh for shared "
            "abort epoch=%d (sample_rel_rms=%.6g).",
            epoch,
            relative_rms,
        )


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
    refresh_epoch = pending_adapter_refresh_epoch(state)
    cached = cache.get(cache_key)
    cache_hit = cached is not None and refresh_epoch is None
    # TEMP_SEMANTIC_DIAG_HOOK: count the once-per-run Adapter cache decision.
    record_semantic_cache_lookup(state, "artist_embedding", cache_hit)
    if cache_hit:
        if should_capture_semantic_stage(state, "artist_sum"):
            record_semantic_stage(
                state,
                "artist_sum",
                tensor_diagnostic_snapshot(cached),
            )
        return cached

    def _compute_artist_sum():
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
            label = labels[index] if index < len(labels) else f"#{index}"
            _ensure_finite(
                "artist_preprocess",
                embedding,
                label=label,
                state=state,
            )
            embeddings.append(embedding)

        if alignment_mode == ALIGN_BASE_ANCHORED:
            embeddings = align_artist_embeddings(
                embeddings,
                state["alignment_plan"],
            )

        result = weighted_embedding_sum(
            embeddings,
            state["user_weights"],
            normalize=state.get("normalize_weights", True),
        ).detach()
        _ensure_finite("artist_weighted_sum", result, state=state)
        return result

    if refresh_epoch is not None:
        # A cancelled sampler can leave finite but stale dynamic-VRAM/quantized
        # runtime data on the BaseModel shared by sibling patchers.  Synchronize,
        # consume one throwaway Adapter pass, then build the value that is
        # actually cached.  This cost is paid once per shared abort generation.
        _sync_adapter_boundary(ref_context)
        warmup = _compute_artist_sum()
        _sync_adapter_boundary(ref_context)
        artist_sum = _compute_artist_sum()
        _sync_adapter_boundary(ref_context)
        _log_safe_refresh_drift(warmup, artist_sum, refresh_epoch)
        complete_adapter_refresh(state, refresh_epoch)
    else:
        artist_sum = _compute_artist_sum()
        # Dynamic-VRAM/quantized Adapter weights may run on auxiliary CUDA
        # streams. Finish this cached build before the denoiser reuses or
        # unloads the same weights. This is once per cache miss, never per sigma.
        _sync_adapter_boundary(ref_context)
    # TEMP_SEMANTIC_DIAG_HOOK: capture the first finite/non-finite artist sum.
    if should_capture_semantic_stage(state, "artist_sum"):
        record_semantic_stage(
            state,
            "artist_sum",
            tensor_diagnostic_snapshot(artist_sum),
        )
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
            # TEMP_SEMANTIC_DIAG_HOOK: record the pre-mix context once per run.
            if should_capture_semantic_context(state):
                context_transformer_options = c.get("transformer_options") or {}
                conditioning_uuids = (
                    context_transformer_options.get("uuids")
                    if isinstance(context_transformer_options, dict)
                    else None
                )
                record_semantic_context(
                    state,
                    snapshot=tensor_diagnostic_snapshot(context),
                    context_key=context_key,
                    cond_or_uncond=cou,
                    conditioning_uuids=conditioning_uuids,
                )
            cache_key = _mixed_context_cache_key(state, context, mask, strengths)
            mixed_context = (
                None
                if is_multigpu
                else _cached_mixed_context(state, context, cache_key)
            )
            # TEMP_SEMANTIC_DIAG_HOOK: distinguish mixed-context hits from misses.
            record_semantic_cache_lookup(
                state,
                "mixed_context",
                mixed_context is not None,
            )
            if mixed_context is None:
                # ``begin_mixer_execution`` pins ``dm_ref`` to the selected
                # clone. Resolving through BaseModel.current_patcher here would
                # reintroduce the sibling-clone drift the lifecycle repair
                # explicitly filters out.
                active_dm = state.get("dm_ref")
                _ensure_finite(
                    "base_context",
                    context,
                    state=state,
                    row_markers=cou,
                    context_key=context_key,
                )
                artist_sum = call_with_mixer_owner(
                    state,
                    apply_model,
                    build_artist_embedding_sum,
                    state,
                    context,
                    dm=active_dm,
                )
                _ensure_finite("artist_sum", artist_sum, state=state)
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
                    _ensure_finite(
                        "aligned_base_context",
                        projection_base,
                        state=state,
                        row_markers=cou,
                        context_key=context_key,
                    )
                mixed_context = mix_projected_context(
                    projection_base,
                    artist_sum,
                    strengths,
                    mask,
                    fallback_base=fallback_base,
                )
                _ensure_finite(
                    "projected_mixed_context",
                    mixed_context,
                    state=state,
                    row_markers=cou,
                    context_key=context_key,
                )
                if not is_multigpu:
                    state["_mixed_context_cache"] = {
                        "source": context,
                        "key": cache_key,
                        "mixed": mixed_context,
                        "mixed_signature": tensor_cache_signature(mixed_context),
                    }

            # TEMP_SEMANTIC_DIAG_HOOK: capture the final post-Adapter context once.
            if should_capture_semantic_stage(state, "mixed_context"):
                record_semantic_stage(
                    state,
                    "mixed_context",
                    tensor_diagnostic_snapshot(mixed_context),
                )

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
