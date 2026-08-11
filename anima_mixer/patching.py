"""Model validation, patch bookkeeping, conditioning, and CFG helpers."""

import functools
import logging

import torch

logger = logging.getLogger(__name__)

_MIXER_CALLBACK_KEY_PREFIX = "anima_artist_mixer_state_"
_ACTIVE_ADAPTER_MIXER_TOKEN_KEY = "_anima_adapter_mixer_active_token"


class MixerFatalError(RuntimeError):
    """A Mixer failure after which the current sampling pass must stop."""


def should_reraise(error):
    """Return True for errors that must abort sampling immediately."""
    # ComfyUI's InterruptProcessingException (and Python's own
    # KeyboardInterrupt/SystemExit) deliberately inherit directly from
    # BaseException.  Never turn those control-flow signals into a Mixer
    # fallback when a caller uses an abort-safe BaseException boundary.
    if isinstance(error, BaseException) and not isinstance(error, Exception):
        return True
    if isinstance(error, MixerFatalError):
        return True
    for name in ("OutOfMemoryError", "AcceleratorError"):
        cuda_error = getattr(getattr(torch, "cuda", None), name, None)
        if cuda_error is not None and isinstance(error, cuda_error):
            return True
        torch_error = getattr(torch, name, None)
        if torch_error is not None and isinstance(error, torch_error):
            return True
    for name in ("CudaError", "DeferredCudaCallError"):
        cuda_error = getattr(getattr(torch, "cuda", None), name, None)
        if cuda_error is not None and isinstance(error, cuda_error):
            return True
    if isinstance(error, RuntimeError):
        message = str(error).lower()
        accelerator_markers = (
            "cuda error",
            "cuda runtime",
            "device-side assert",
            "illegal memory access",
            "misaligned address",
            "launch failure",
            "cublas",
            "cudnn",
            "cusparse",
            "nvrtc",
            "triton",
        )
        if any(marker in message for marker in accelerator_markers):
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


def tensor_value_signature(tensor):
    """Return a small value digest for one tensor.

    ``Tensor._version`` is normally enough to invalidate a cache, but writes via
    ``tensor.data`` can leave that counter unchanged.  This digest is deliberately
    used at an execution boundary (not on every attention call): it makes silent
    prompt/conditioning writes invalidate the next run while keeping the hot path
    free of a per-step GPU synchronization.
    """
    if tensor is None or not torch.is_tensor(tensor):
        return None
    try:
        flat = tensor.detach().reshape(-1)
        if flat.numel() == 0:
            return (0, 0.0, 0.0, 0.0, 0.0)
        # A full sum/squared-sum catches arbitrary in-place writes with one device
        # reduction.  The sampled terms reduce collision risk for signed values
        # and make the fallback useful for integer token-ID tensors as well.
        work = flat.to(dtype=torch.float64)
        total = float(work.sum().item())
        squared = float(work.square().sum().item())
        sample_count = min(64, int(work.numel()))
        if sample_count == 1:
            sample = work[:1]
        else:
            positions = torch.linspace(
                0,
                int(work.numel()) - 1,
                sample_count,
                device=work.device,
                dtype=torch.float64,
            ).round().to(dtype=torch.long)
            sample = work.index_select(0, positions)
        return (
            int(work.numel()),
            round(total, 7),
            round(squared, 7),
            round(float(sample.sum().item()), 7),
            round(float(sample.square().sum().item()), 7),
        )
    except Exception:
        return None


def execution_tensor_signature(state, tensor):
    """Combine structural and execution-scoped value signatures.

    The memo is cleared by ``reset_run_state``.  Consequently a conditioning
    tensor is value-checked once per sampling run, instead of synchronizing the
    GPU for every sigma/area call.  Mutating the tensor while a run is already in
    progress remains unsupported, as it would make the sampler input itself
    ill-defined.
    """
    structural = tensor_cache_signature(tensor)
    if tensor is None or not torch.is_tensor(tensor):
        return structural
    memo = state.setdefault("_execution_value_fp_memo", {})
    key = id(tensor)
    cached = memo.get(key)
    if cached is not None and cached[0] == structural:
        return (structural, cached[1])
    value = tensor_value_signature(tensor)
    memo[key] = (structural, value)
    return (structural, value)


def refresh_runtime_input_signature(state):
    """Re-read conditioning values at a known execution boundary.

    A tensor can be changed through ``.data`` without changing its id or
    ``_version``.  The normal execution path memoizes the value digest for
    speed, so an abort boundary must explicitly invalidate that memo before
    comparing the next run's inputs.
    """
    state["_execution_value_fp_memo"] = {}
    state["_runtime_input_signature"] = None
    return runtime_input_signature(state)


def runtime_input_signature(state):
    """Value-signature all Adapter inputs once for the current execution."""
    cached = state.get("_runtime_input_signature")
    if cached is not None:
        return cached
    values = []
    for key in (
        "raws",
        "ids_list",
        "t5_weights_list",
    ):
        values.append(tuple(
            execution_tensor_signature(state, value)
            for value in (state.get(key) or [])
        ))
    values.append(execution_tensor_signature(state, state.get("base_ids")))
    values.append(execution_tensor_signature(state, state.get("base_t5_weights")))
    signature = tuple(values)
    state["_runtime_input_signature"] = signature
    return signature


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
    state["_execution_value_fp_memo"] = {}
    state["_runtime_input_signature"] = None
    state["_artist_chunk_cache"] = {}
    state["_anchor_failed"] = False
    state["_adapter_anchor_failed"] = False
    state["_embedding_mixer_failed"] = False
    state["_warned_embedding_failure"] = False
    state["_anchor_last_sigma"] = None
    state["_in_anchor_run"] = False
    state["_run_call_count"] = 0
    state["_multigpu_call"] = False
    state["_multigpu_dm_by_worker"] = {}
    state["_interrupt_cleanup_complete"] = False
    state["_warned_owner_drift"] = False


def _sigma_value(timestep):
    """Return the current sigma without depending on a sampler-specific shape."""
    if timestep is None or not torch.is_tensor(timestep) or timestep.numel() == 0:
        return None
    try:
        return float(timestep.detach().max().item())
    except Exception:
        return None


def _model_owner_token(apply_model, owner=None):
    """Identify the active ModelPatcher behind a model-function wrapper.

    ComfyUI copies ``model_options`` when a ModelPatcher is cloned, but functions
    stored in it are copied by reference. A closure-based wrapper can therefore
    survive LoRA and optimization-node clones even when the effective model has
    changed. ``pre_run`` records the active patcher on the shared model, giving
    this node a reliable ownership boundary for its mutable state.
    """
    if owner is not None:
        return ("patcher", id(owner))
    model = getattr(apply_model, "__self__", None)
    patcher = getattr(model, "current_patcher", None)
    if patcher is not None:
        return ("patcher", id(patcher))
    if model is not None:
        return ("model", id(model))
    return None


def _clear_model_bound_mixer_state(state):
    """Discard values produced with another effective model clone."""
    state["_artist_embedding_cache"] = {}
    state["_mixed_context_cache"] = None
    state["_anchor_cache"] = {}
    state["_anchor_cache_key"] = None
    state["_anchor_trajectory"] = None
    state["_anchor_last_sigma"] = None
    state["individuals"] = None
    state["real_lens"] = None
    state["_runtime_input_signature"] = None
    state["_warned_trajectory_invalidated"] = False
    state["_warned_trajectory_reuse"] = False


def clear_mixer_run_state(state, *, interrupted=False):
    """Release Mixer tensors and close the active execution boundary.

    ComfyUI normally invokes ``on_cleanup`` after a sampling pass, but an
    ``InterruptProcessingException`` can leave that callback pending while a
    wrapper is unwinding.  Keep this helper idempotent so both paths can use
    the same cleanup contract without touching the mixer math or model
    configuration.
    """
    if interrupted and state.get("_interrupt_cleanup_complete", False):
        return

    previous_input_signature = state.get("_runtime_input_signature")

    # Reuse the canonical run-state reset so newly added per-run fields do not
    # accidentally survive an interrupt boundary.
    reset_run_state(state)

    state["_artist_embedding_cache"] = {}
    state["_mixed_context_cache"] = None
    state["_anchor_cache"] = {}
    state["_anchor_cache_key"] = None
    trajectory = state.get("_anchor_trajectory")
    preserve_ready_trajectory = (
        not interrupted
        and str(state.get("anchor_refresh_mode", "once")) == "warm_cache"
        and isinstance(trajectory, dict)
        and trajectory.get("ready", False)
    )
    if preserve_ready_trajectory:
        # Warm-cache frames are CPU-owned after finalization.  Drop any
        # in-flight GPU snapshot while retaining the reusable trajectory.
        trajectory["last_cache"] = None
        trajectory["active_sigma"] = None
        trajectory["active_device"] = None
        trajectory["active_dtype"] = None
        state["_anchor_trajectory"] = trajectory
    else:
        state["_anchor_trajectory"] = None
    state["_anchor_last_sigma"] = None
    state["_in_anchor_run"] = False
    state["individuals"] = None
    state["real_lens"] = None

    # Drop execution-scoped tensors/signatures and all flags that could make a
    # following run observe the interrupted pass as still active.
    state["_ctx_fp_memo"] = {}
    state["_execution_value_fp_memo"] = {}
    state["_runtime_input_signature"] = None
    if not interrupted and previous_input_signature is not None:
        # Keep one completed-run fingerprint so a reused wrapper can detect
        # prompt tensors changed in place even when ComfyUI did not interrupt.
        state["_boundary_input_signature"] = previous_input_signature
        state["_force_boundary_check"] = True
    state["_run_last_sigma"] = None
    state["_run_call_count"] = 0
    state["_run_active"] = False
    state["_last_run_had_calls"] = False
    state["_mixer_run_start_pending"] = False
    state["_adapter_mixer_run_start"] = None
    state["_adapter_mixer_finalize_warm_cache"] = False
    state["current_sigma"] = None
    state["_model_owner_ref"] = None

    if interrupted:
        state["_abort_input_signature"] = previous_input_signature
        state["_force_boundary_check"] = True
        state["_interrupt_cleanup_complete"] = True
        if state.get("_adapter_mixer_instance_token") is not None:
            # The normal ModelPatcher cleanup callback also closes lifecycle
            # selection. An outer interrupt can unwind before that callback,
            # so make the stale wrapper a pass-through immediately.
            state["_adapter_mixer_selected_for_run"] = False
        # Aimdo/VBAR can have outstanding work on an offload stream when a
        # prompt is interrupted.  Synchronize only on this exceptional path;
        # normal per-step sampling remains asynchronous.
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except BaseException:
            logger.debug(
                "[AnimaAdapterMixer] CUDA synchronization during abort cleanup "
                "failed; preserving the original sampling exception.",
                exc_info=True,
            )


def _diffusion_model_for_patcher(patcher, fallback=None):
    if patcher is not None:
        getter = getattr(patcher, "get_model_object", None)
        if callable(getter):
            try:
                diffusion_model = getter("diffusion_model")
                if diffusion_model is not None:
                    return diffusion_model
            except Exception:
                pass
        model = getattr(patcher, "model", None)
        diffusion_model = getattr(model, "diffusion_model", None)
        if diffusion_model is not None:
            return diffusion_model
    return fallback


def active_patcher_for_apply_model(apply_model):
    """Return the ModelPatcher owning one bound ``apply_model`` method."""
    model = getattr(apply_model, "__self__", None)
    return getattr(model, "current_patcher", None)


def adapter_mixer_state_is_active(state, apply_model=None, patcher=None):
    """Return whether ``state`` is the Adapter Mixer selected by this patcher.

    A later sampling stage can receive a model whose older Mixer is hidden
    inside an opaque Sage/FBCache/control wrapper.  That wrapper cannot be
    reconstructed safely, so the old Mixer closure remains reachable.  The
    clone-local token makes it a pass-through without mutating sibling model
    branches that still legitimately use the older Mixer.

    Cross-Attn Mixer states do not carry an Adapter token and remain unaffected.
    """
    token = state.get("_adapter_mixer_instance_token")
    if token is None:
        return True
    # ``on_pre_run`` is authoritative for the wrapper/options pair selected by
    # ComfyUI. ``BaseModel.current_patcher`` is shared by sibling clones and can
    # be overwritten between pre_run and the first model call when another
    # sampler/detailer prepares its clone. Once lifecycle selection happened,
    # do not let that shared pointer flip this state mid-run.
    if patcher is None:
        selected = state.get("_adapter_mixer_selected_for_run")
        if selected is not None:
            return bool(selected)
    if patcher is None and apply_model is not None:
        patcher = active_patcher_for_apply_model(apply_model)
    options = getattr(patcher, "model_options", None)
    if not isinstance(options, dict):
        # A third-party callable may hide the bound ``apply_model`` method.
        # Once ComfyUI has established the run boundary, use that local
        # selection marker instead of fail-open executing a superseded state.
        selected = state.get("_adapter_mixer_selected_for_run")
        return True if selected is None else bool(selected)
    active_token = options.get(_ACTIVE_ADAPTER_MIXER_TOKEN_KEY)
    return active_token is None or active_token == token


def select_active_adapter_mixer(patcher, state):
    """Select one Adapter Mixer for a clone-local ModelPatcher branch."""
    options = getattr(patcher, "model_options", None)
    if not isinstance(options, dict):
        return False
    token = state.get("_adapter_mixer_instance_token")
    if token is None:
        return False
    previous = options.get(_ACTIVE_ADAPTER_MIXER_TOKEN_KEY)
    options[_ACTIVE_ADAPTER_MIXER_TOKEN_KEY] = token
    if previous not in (None, token):
        logger.info(
            "[AnimaAdapterMixer] selected a newer Adapter Mixer for this "
            "patcher; older hidden Mixer wrappers will pass through."
        )
    return True


def model_weight_patch_identity(patcher):
    """Describe the effective LoRA/weight-patch set for one ModelPatcher.

    A regular ``ModelPatcher.clone()`` intentionally shares the underlying
    ``BaseModel``.  Consequently ``id(diffusion_model)`` is not sufficient to
    distinguish two LoRA variants: the ModelPatcher has a new ``patches_uuid``
    while the diffusion-model object stays the same.  Keep both UUIDs here:

    * ``patches_uuid`` is the requested patch set on this clone;
    * ``current_weight_patches_uuid`` is the patch set currently materialized
      on the shared model.

    The latter is normally identical by ``pre_run``.  Retaining it separately
    turns a mismatch into useful diagnostics instead of silently reusing an
    artist embedding from another LoRA state.
    """
    if patcher is None:
        return None
    requested = getattr(patcher, "patches_uuid", None)
    model = getattr(patcher, "model", None)
    loaded = getattr(model, "current_weight_patches_uuid", None)
    patches = getattr(patcher, "patches", None)
    try:
        patch_count = len(patches) if patches is not None else None
    except Exception:
        patch_count = None
    if requested is None and loaded is None and patch_count is None:
        return None
    return (requested, loaded, patch_count)


def _format_weight_patch_identity(identity):
    """Format a patch identity without assuming UUID objects are present."""
    if identity is None:
        return "unavailable"
    requested, loaded, patch_count = identity
    return (
        f"requested={requested}, loaded={loaded}, "
        f"patches={patch_count}"
    )


def diffusion_model_for_apply_model(apply_model, fallback=None):
    """Resolve the diffusion model for the currently executing clone."""
    return _diffusion_model_for_patcher(
        active_patcher_for_apply_model(apply_model),
        fallback,
    )


def call_with_mixer_owner(state, apply_model, callback, *args, **kwargs):
    """Run one model/Adapter call with the clone-local patcher selected at pre_run.

    ``BaseModel.current_patcher`` is a single mutable pointer shared by all
    ``ModelPatcher`` clones that share one ``BaseModel``.  A later detailer or
    sampler can overwrite it before an older wrapper reaches the model.  The
    Mixer already pins its cache owner; this companion guard makes the actual
    dynamic-VRAM/LoRA forward observe the same patcher for the duration of the
    call, then restores the pointer so sibling branches keep their own state.
    """
    owner = state.get("_model_owner_ref")
    model = getattr(apply_model, "__self__", None)
    if owner is None or model is None or getattr(owner, "model", None) is not model:
        return callback(*args, **kwargs)

    previous = getattr(model, "current_patcher", None)
    changed = previous is not owner
    if changed:
        model.current_patcher = owner
    try:
        return callback(*args, **kwargs)
    finally:
        # If another explicitly nested branch changed the pointer while this
        # call was running, do not overwrite that branch's newer selection.
        if changed and getattr(model, "current_patcher", None) is owner:
            model.current_patcher = previous


def resolve_multigpu_worker_wrapper(apply_model, options, current_wrapper):
    """Select a clone-local wrapper when ComfyUI shares the main wrapper.

    ``_calc_cond_batch_multigpu`` deliberately invokes the main
    ``model_options['model_function_wrapper']`` with each worker clone's bound
    ``apply_model``.  Clone callbacks give each ModelPatcher its own rebound
    wrapper/state, so dispatching to that wrapper restores the intended
    per-device ownership without changing ComfyUI's scheduler.
    """
    if (options or {}).get("_anima_mixer_worker_dispatch", False):
        return None
    c_dict = (options or {}).get("c") or {}
    transformer_options = c_dict.get("transformer_options") or {}
    if not (
        isinstance(transformer_options, dict)
        and transformer_options.get("multigpu_thread_device") is not None
    ):
        return None
    patcher = active_patcher_for_apply_model(apply_model)
    model_options = getattr(patcher, "model_options", None)
    if not isinstance(model_options, dict):
        return None
    worker_wrapper = model_options.get("model_function_wrapper")
    if callable(worker_wrapper) and worker_wrapper is not current_wrapper:
        return worker_wrapper
    return None


def _raw_object_for_patch_path(patcher, path):
    """Read a clone's unpatched object for an object-patch path."""
    module_path = str(path)
    if module_path.endswith(".forward"):
        module_path = module_path[:-len(".forward")]
    model = getattr(patcher, "model", None)
    if model is None:
        return None
    current = model
    try:
        for component in module_path.split("."):
            if component.isdigit():
                try:
                    current = current[int(component)]
                    continue
                except (IndexError, KeyError, TypeError, AttributeError):
                    pass
            if isinstance(current, dict):
                current = current[component]
            else:
                current = getattr(current, component)
    except Exception:
        return None
    return current


def _raw_forward_for_patch_path(patcher, path):
    """Resolve a cross-attention module's current clone-local forward."""
    module = _raw_object_for_patch_path(patcher, path)
    if module is None:
        return None, None
    try:
        module = unwrap_cross_attn(module)
    except Exception:
        pass
    forward = getattr(module, "forward", None)
    if forward is None:
        return module, None
    try:
        forward = unwrap_cross_attn_forward(module)
    except Exception:
        pass
    return module, forward


def begin_mixer_execution(
    state,
    apply_model,
    timestep,
    *,
    owner=None,
    explicit_run_start=False,
    owner_token_override=None,
):
    """Advance Adapter Mixer state before a sampler call reads cached context.

    Returns ``(is_run_start, model_owner_changed)``. The Adapter wrapper calls
    this first; the optional Anchor sigma wrapper consumes its flags to finalize
    a previous warm-cache trajectory at the correct point in the call chain.
    """
    sigma = _sigma_value(timestep)
    # Once on_pre_run has bound this state to a ModelPatcher, retain that
    # owner/weight identity for the whole pass. The underlying BaseModel is
    # shared by clones, so its live ``current_patcher`` field can point at a
    # sibling second-stage/detailer clone even while this wrapper is executing.
    use_pre_run_owner = bool(
        not explicit_run_start
        and owner is None
        and owner_token_override is None
        and state.get("_adapter_mixer_selected_for_run") is True
        and state.get("_run_active", False)
        and state.get("_model_owner_token") is not None
    )
    if use_pre_run_owner:
        active_patcher = None
        owner_token = state.get("_model_owner_token")
        shared_owner_token = True
        weight_identity = state.get("_model_weight_patch_identity")
        live_model = getattr(apply_model, "__self__", None)
        live_patcher = getattr(live_model, "current_patcher", None)
        live_owner_token = _model_owner_token(apply_model, owner=live_patcher)
        if (
            live_owner_token not in (None, owner_token)
            and not state.get("_warned_owner_drift", False)
        ):
            logger.info(
                "[AnimaAdapterMixer] ignored shared current_patcher drift "
                "during the active run (bound=%s, live=%s, state=%x).",
                owner_token,
                live_owner_token,
                id(state),
            )
            state["_warned_owner_drift"] = True
    else:
        active_patcher = owner
        if active_patcher is None:
            model = getattr(apply_model, "__self__", None)
            active_patcher = getattr(model, "current_patcher", None)
        owner_token = (
            owner_token_override
            if owner_token_override is not None
            else _model_owner_token(apply_model, owner=active_patcher)
        )
        shared_owner_token = owner_token_override is not None
        weight_identity = model_weight_patch_identity(active_patcher)
    previous_owner = state.get("_model_owner_token")
    owner_changed = owner_token is not None and previous_owner not in (None, owner_token)
    previous_weight_identity = state.get("_model_weight_patch_identity")
    weight_changed = (
        weight_identity is not None
        and previous_weight_identity not in (None, weight_identity)
    )
    previous_input_signature = state.pop(
        "_abort_input_signature",
        state.pop(
            "_boundary_input_signature",
            state.get("_runtime_input_signature"),
        ),
    )
    fresh_input_signature = None
    conditioning_changed = False
    # ``on_pre_run`` is the authoritative ComfyUI execution boundary.  A
    # previous interrupted pass is normally followed by ``on_cleanup``, but
    # outer sampler/wrapper paths can unwind before that callback is reached.
    # In that case the old execution-value memo would otherwise be reused and
    # an in-place prompt/conditioning edit could look unchanged.  Re-read the
    # value fingerprints once for every explicit pre-run; ordinary sigma calls
    # still use the memo and remain synchronization-free.
    boundary_check_requested = bool(state.pop("_force_boundary_check", False))
    if explicit_run_start:
        boundary_check_requested = True
    if boundary_check_requested:
        fresh_input_signature = refresh_runtime_input_signature(state)
        conditioning_changed = (
            previous_input_signature is not None
            and fresh_input_signature != previous_input_signature
        )
    if previous_owner is None and owner_token is not None:
        state["_model_owner_token"] = owner_token
        if active_patcher is not None:
            state["_model_owner_ref"] = active_patcher
        if not shared_owner_token:
            state["dm_ref"] = _diffusion_model_for_patcher(
                active_patcher,
                state.get("dm_ref"),
            )
    elif owner_changed:
        _clear_model_bound_mixer_state(state)
        state["_model_owner_token"] = owner_token
        if active_patcher is not None:
            state["_model_owner_ref"] = active_patcher
        if not shared_owner_token:
            state["dm_ref"] = _diffusion_model_for_patcher(
                active_patcher,
                state.get("dm_ref"),
            )
        logger.info(
            "[AnimaAdapterMixer] active model clone changed; discarded "
            "model-bound embedding and Anchor caches."
        )
    if weight_identity is not None:
        state["_model_weight_patch_identity"] = weight_identity
    if weight_changed:
        # ``ModelPatcher.clone`` shares BaseModel but ``add_patches`` changes
        # the clone's UUID afterwards.  Do not let artist/Anchor tensors made
        # under the old LoRA survive that otherwise invisible transition.
        if not owner_changed:
            _clear_model_bound_mixer_state(state)
        logger.info(
            "[AnimaAdapterMixer] active LoRA/weight patch set changed; "
            "discarded model-bound embedding and Anchor caches (%s -> %s).",
            _format_weight_patch_identity(previous_weight_identity),
            _format_weight_patch_identity(weight_identity),
        )
    elif active_patcher is not None and state.get("_model_owner_ref") is None:
        # A direct sigma-wrapper fallback may enter without the lifecycle
        # callback having initialized the owner reference yet.
        state["_model_owner_ref"] = active_patcher
    if conditioning_changed:
        _clear_model_bound_mixer_state(state)
        if str(state.get("alignment_mode", "")) == "base_anchored":
            # ComfyUI can reuse the same model wrapper while replacing prompt
            # tensors after an interrupted queue item.  The Adapter caches are
            # already invalidated above, but their token-row map must follow the
            # new IDs as well or a different-length prompt reuses stale indices.
            from .alignment import build_base_anchored_plan
            state["alignment_plan"] = build_base_anchored_plan(
                state.get("base_ids"),
                state.get("ids_list") or [],
            )
        logger.info(
            "[AnimaAdapterMixer] conditioning value changed after an abort; "
            "discarded artist, mixed-context, and Anchor caches and refreshed "
            "token alignment."
        )

    previous_sigma = state.get("_run_last_sigma")
    pending_boundary = bool(state.pop("_mixer_run_start_pending", False))
    already_reset = pending_boundary and not explicit_run_start
    last_run_marker = state.get("_last_run_had_calls")
    previous_run_had_calls = bool(
        state.get("_run_active", False)
        if last_run_marker is None else last_run_marker
    )
    if explicit_run_start:
        is_run_start = True
    elif pending_boundary:
        is_run_start = True
    else:
        is_run_start = (
            owner_changed
            or weight_changed
            or conditioning_changed
            or not state.get("_run_active", False)
            or previous_sigma is None
            or (sigma is not None and sigma > float(previous_sigma) + 1e-3)
        )
    if is_run_start and not already_reset:
        # A live mixed context is only valid inside one denoising pass. A ready
        # warm trajectory remains available and validates its full conditioning
        # signature before the next pass is allowed to reuse it.
        state["_mixed_context_cache"] = None
        state["_anchor_cache"] = {}
        state["_anchor_cache_key"] = None
        reset_run_state(state)
        state["_execution_index"] = int(state.get("_execution_index", 0)) + 1
        state["_adapter_mixer_finalize_warm_cache"] = bool(
            not owner_changed
            and previous_run_had_calls
            and str(state.get("anchor_refresh_mode", "once")) == "warm_cache"
        )
        state["_run_active"] = True
        # ``None`` means no cleanup callback has closed the previous run yet;
        # the sigma fallback can then use the still-active run as evidence.
        state["_last_run_had_calls"] = None
        if explicit_run_start:
            # The pre_run callback has already performed the reset.  The first
            # model-function wrapper call consumes this marker instead of
            # treating its (possibly identical) sigma as another new run.
            state["_mixer_run_start_pending"] = True

    state["_run_last_sigma"] = sigma
    if fresh_input_signature is not None:
        # Preserve the boundary fingerprint across reset_run_state(); the next
        # cache lookup can then use it without another full value reduction.
        state["_runtime_input_signature"] = fresh_input_signature
    if state.get("_run_active", False) and not explicit_run_start:
        state["_run_call_count"] = int(state.get("_run_call_count", 0)) + 1
    state["_adapter_mixer_run_start"] = is_run_start
    return is_run_start, owner_changed


def _clone_mixer_state_for_patcher(state, patcher):
    cloned = dict(state)
    for key in ("labels", "raws", "ids_list", "t5_weights_list", "user_weights"):
        value = cloned.get(key)
        if isinstance(value, list):
            cloned[key] = list(value)
    cloned["dm_ref"] = _diffusion_model_for_patcher(
        patcher,
        cloned.get("dm_ref"),
    )
    cloned["individuals"] = None
    cloned["real_lens"] = None
    _clear_model_bound_mixer_state(cloned)
    reset_run_state(cloned)
    cloned["_model_owner_token"] = None
    cloned["_model_owner_ref"] = None
    cloned["_model_weight_patch_identity"] = None
    cloned["_run_last_sigma"] = None
    cloned["_run_active"] = False
    cloned["_last_run_had_calls"] = False
    cloned["_mixer_run_start_pending"] = False
    cloned["_adapter_mixer_run_start"] = None
    cloned["_adapter_mixer_finalize_warm_cache"] = False
    cloned["_adapter_mixer_selected_for_run"] = None
    cloned["_execution_index"] = 0
    cloned["_force_boundary_check"] = True
    cloned["_boundary_input_signature"] = None
    cloned.pop("_abort_input_signature", None)
    return cloned


def _iter_nested_objects(root, max_depth=32):
    """Walk callable closures/attributes without executing external code."""
    stack = [(root, 0)]
    seen = set()
    while stack:
        value, depth = stack.pop()
        if value is None or id(value) in seen or depth > max_depth:
            continue
        seen.add(id(value))
        yield value
        if isinstance(value, dict):
            stack.extend((item, depth + 1) for item in value.values())
            continue
        if isinstance(value, (list, tuple, set)):
            stack.extend((item, depth + 1) for item in value)
            continue
        if isinstance(value, functools.partial):
            stack.append((value.func, depth + 1))
            stack.extend((item, depth + 1) for item in value.args)
            if value.keywords:
                stack.extend((item, depth + 1) for item in value.keywords.values())
        bound_self = getattr(value, "__self__", None)
        bound_func = getattr(value, "__func__", None)
        if bound_self is not None and bound_func is not None:
            stack.append((bound_func, depth + 1))
            stack.append((bound_self, depth + 1))
        closure = getattr(value, "__closure__", None)
        if closure:
            for cell in closure:
                try:
                    stack.append((cell.cell_contents, depth + 1))
                except ValueError:
                    pass
        defaults = getattr(value, "__defaults__", None)
        if defaults:
            stack.extend((item, depth + 1) for item in defaults)
        kwdefaults = getattr(value, "__kwdefaults__", None)
        if kwdefaults:
            stack.extend((item, depth + 1) for item in kwdefaults.values())
        attrs = getattr(value, "__dict__", None)
        if isinstance(attrs, dict):
            stack.extend((item, depth + 1) for item in attrs.values())
        slots = getattr(type(value), "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for slot in slots:
            if slot in ("__dict__", "__weakref__"):
                continue
            try:
                stack.append((getattr(value, slot), depth + 1))
            except (AttributeError, TypeError):
                pass


def _find_nested_mixer_wrapper(root, state):
    sigma_fallback = None
    for value in _iter_nested_objects(root):
        if getattr(value, "_anima_mixer_state", None) is not state:
            continue
        if getattr(value, "_anima_adapter_mixer_wrapper", False):
            return value
        elif getattr(value, "_anima_adapter_anchor_sigma_wrapper", False):
            sigma_fallback = value
    return sigma_fallback


def resolve_clone_local_mixer_wrapper(apply_model, current_wrapper, state):
    """Dispatch a Mixer hidden by an external closure to clone-local state."""
    # A state selected by its own pre_run callback already is the clone-local
    # target. Looking at shared BaseModel.current_patcher here could redirect it
    # into a sibling sampler's state.
    if state.get("_adapter_mixer_selected_for_run") is True:
        return None
    patcher = active_patcher_for_apply_model(apply_model)
    options = getattr(patcher, "model_options", None)
    if not isinstance(options, dict):
        return None
    token = state.get("_adapter_mixer_instance_token")
    if token is not None:
        active_token = options.get(_ACTIVE_ADAPTER_MIXER_TOKEN_KEY)
        if active_token not in (None, token):
            return None
    registry = options.get("_anima_mixer_clone_wrappers")
    if not isinstance(registry, dict):
        return None
    rebound = registry.get(id(state))
    if callable(rebound) and rebound is not current_wrapper:
        return rebound
    return None


def _rebind_mixer_wrapper_chain(wrapper, old_state, new_state):
    if wrapper is None:
        return None
    if getattr(wrapper, "_anima_mixer_state", None) is not old_state:
        return wrapper
    factory = getattr(wrapper, "_anima_mixer_factory", None)
    previous = getattr(wrapper, "_anima_mixer_previous", None)
    if not callable(factory):
        return wrapper
    rebound_previous = _rebind_mixer_wrapper_chain(
        previous,
        old_state,
        new_state,
    )
    return factory(new_state, rebound_previous)


def _rebind_mixer_object_patches(patcher, old_state, new_state):
    registry = getattr(patcher, "object_patches", None)
    if not hasattr(registry, "items"):
        return
    for path, patch in tuple(registry.items()):
        if getattr(patch, "_anima_mixer_state", None) is not old_state:
            continue
        original_module, original_forward = _raw_forward_for_patch_path(
            patcher,
            path,
        )
        rebind = getattr(patch, "rebind_state", None)
        if callable(rebind):
            try:
                registry[path] = rebind(
                    new_state,
                    original_forward=original_forward,
                    original_module=original_module,
                )
            except TypeError:
                # Keep compatibility with older local patch objects while all
                # current Mixer patches use the clone-local forward above.
                registry[path] = rebind(new_state)


def _mixer_callback_key(state):
    return f"{_MIXER_CALLBACK_KEY_PREFIX}{id(state):x}"


def _replace_mixer_lifecycle_callbacks(patcher, state, callback_key):
    remove = getattr(patcher, "remove_callbacks_with_key", None)
    add = getattr(patcher, "add_callback_with_key", None)
    if not callable(remove) or not callable(add):
        return
    remove("on_clone", callback_key)
    remove("on_pre_run", callback_key)
    remove("on_cleanup", callback_key)
    add("on_clone", callback_key, _make_mixer_clone_callback(state, callback_key))
    add("on_pre_run", callback_key, _make_mixer_pre_run_callback(state))
    add("on_cleanup", callback_key, _make_mixer_cleanup_callback(state))


def _remove_mixer_lifecycle_callbacks(patcher, callback_key):
    remove = getattr(patcher, "remove_callbacks_with_key", None)
    if not callable(remove):
        return
    remove("on_clone", callback_key)
    remove("on_pre_run", callback_key)
    remove("on_cleanup", callback_key)


def _make_mixer_clone_callback(state, callback_key):
    def _on_clone(_source, cloned):
        if not adapter_mixer_state_is_active(state, patcher=cloned):
            # A newer Adapter Mixer owns this branch.  Drop the dormant
            # lifecycle callbacks while leaving sibling/source branches alone.
            _remove_mixer_lifecycle_callbacks(cloned, callback_key)
            return
        new_state = _clone_mixer_state_for_patcher(state, cloned)
        options = getattr(cloned, "model_options", None)
        if isinstance(options, dict):
            clone_registry = options.get("_anima_mixer_clone_wrappers")
            if isinstance(clone_registry, dict):
                for origin_id, registered in tuple(clone_registry.items()):
                    rebound_registered = _rebind_mixer_wrapper_chain(
                        registered,
                        state,
                        new_state,
                    )
                    if rebound_registered is not registered:
                        clone_registry[origin_id] = rebound_registered
            wrapper = options.get("model_function_wrapper")
            rebound = _rebind_mixer_wrapper_chain(wrapper, state, new_state)
            if rebound is not wrapper:
                options["model_function_wrapper"] = rebound
            else:
                # External optimization nodes may wrap the Mixer in an opaque
                # closure. The outer function itself cannot be reconstructed,
                # so keep a clone-local rebound for the shared inner Mixer to
                # dispatch to when this clone becomes active.
                nested = _find_nested_mixer_wrapper(wrapper, state)
                if nested is not None:
                    nested_rebound = _rebind_mixer_wrapper_chain(
                        nested,
                        state,
                        new_state,
                    )
                    options.setdefault(
                        "_anima_mixer_clone_wrappers",
                        {},
                    )[id(state)] = nested_rebound
        _rebind_mixer_object_patches(cloned, state, new_state)
        _replace_mixer_lifecycle_callbacks(cloned, new_state, callback_key)
    return _on_clone


def _make_mixer_pre_run_callback(state):
    def _on_pre_run(patcher):
        active = adapter_mixer_state_is_active(state, patcher=patcher)
        state["_adapter_mixer_selected_for_run"] = active
        if not active:
            clear_mixer_run_state(state)
            return
        begin_mixer_execution(
            state,
            getattr(getattr(patcher, "model", None), "apply_model", None),
            None,
            owner=patcher,
            explicit_run_start=True,
        )
        identity = state.get("_model_weight_patch_identity")
        logger.info(
            "[AnimaAdapterMixer] run=%d patcher=%x %s",
            int(state.get("_execution_index", 0)),
            id(patcher),
            _format_weight_patch_identity(identity),
        )
        if identity is not None and identity[0] is not None and identity[1] is not None:
            if identity[0] != identity[1]:
                logger.warning(
                    "[AnimaAdapterMixer] requested and loaded patch UUIDs differ at "
                    "pre_run; Mixer caches were cleared, but this run may be using an "
                    "incomplete model reload (%s).",
                    _format_weight_patch_identity(identity),
                )
    return _on_pre_run


def _make_mixer_cleanup_callback(state):
    def _on_cleanup(_patcher):
        had_calls = int(
            state.get("_run_call_count", 0)
        ) > 0
        clear_mixer_run_state(state)
        if state.get("_adapter_mixer_instance_token") is not None:
            state["_adapter_mixer_selected_for_run"] = False
        # This marker is historical boundary information, not live run state;
        # retain it so the next sigma fallback still recognizes a prior pass.
        state["_last_run_had_calls"] = had_calls
    return _on_cleanup


def register_mixer_lifecycle(patcher, state):
    """Bind Mixer state to ModelPatcher clone and sampling boundaries.

    ComfyUI's clone path copies callback registries and calls ``on_clone`` after
    the clone's model options/object-patch registries have been copied.  We use
    that hook to give each clone independent mutable Mixer state, then use
    ``on_pre_run`` as the authoritative run boundary.  Older test doubles or
    ComfyUI forks without these callbacks keep the sigma-based fallback.
    """
    get_callbacks = getattr(patcher, "get_callbacks", None)
    add = getattr(patcher, "add_callback_with_key", None)
    if not callable(get_callbacks) or not callable(add):
        return False
    callback_key = _mixer_callback_key(state)
    if not get_callbacks("on_clone", callback_key):
        add("on_clone", callback_key, _make_mixer_clone_callback(state, callback_key))
    if not get_callbacks("on_pre_run", callback_key):
        add("on_pre_run", callback_key, _make_mixer_pre_run_callback(state))
    if not get_callbacks("on_cleanup", callback_key):
        add("on_cleanup", callback_key, _make_mixer_cleanup_callback(state))
    return True


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
        self._anima_mixer_state = getattr(wrapper, "_st", None)

    def rebind_state(
        self,
        state,
        original_forward=None,
        original_module=None,
    ):
        rebind = getattr(self.wrapper, "rebind_state", None)
        if callable(rebind):
            wrapper = rebind(
                state,
                original_forward=original_forward,
                original_module=original_module,
            )
            return CrossAttnForwardPatch(wrapper)
        return self

    def __call__(self, *args, **kwargs):
        return self.wrapper.forward(*args, **kwargs)


def unwrap_cross_attn_forward(ca):
    forward = getattr(ca, "forward", None)
    seen = set()
    while (
        isinstance(forward, CrossAttnForwardPatch)
        or getattr(forward, "_anima_adapter_anchor_q_forward_patch", False)
    ):
        marker = id(forward)
        if marker in seen:
            break
        seen.add(marker)
        forward = getattr(forward, "original_forward", None)
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


def build_artists(state, ref_context, dm=None):
    dm = state["dm_ref"] if dm is None else dm
    cacheable = not state.get("_multigpu_call", False)
    if cacheable and state.get("individuals") is not None:
        return state["individuals"], state["real_lens"]
    individuals, real_lens = [], []
    for raw, ids, w_t in zip(state["raws"], state["ids_list"], state["w_list"]):
        artist = preprocess_one(dm, raw, ids, w_t, ref_context.device, ref_context.dtype)
        individuals.append(artist)
        real_lens.append(int(ids.shape[-1]) if ids is not None else artist.shape[1])
    if cacheable:
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
