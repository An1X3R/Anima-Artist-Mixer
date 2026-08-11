"""Anchor-Q pre-run and sigma capture helpers."""

import logging
import math

import torch

from .constants import (
    ANCHOR_CACHE_POINTS_DEFAULT,
    ANCHOR_CACHE_POINTS_MAX,
    ANCHOR_CACHE_POINTS_MIN,
    ANCHOR_KEYFRAME_ADAPTIVE_Q,
    ANCHOR_KEYFRAME_MODES,
    ANCHOR_KEYFRAME_UNIFORM_SIGMA,
    ANCHOR_REFRESH_WARM_CACHE,
    ANCHOR_SEEDS_MAX,
    ANCHOR_SEEDS_POOL,
)
from .patching import (
    adapter_mixer_state_is_active,
    begin_mixer_execution,
    call_with_mixer_owner,
    clear_mixer_run_state,
    context_fingerprint,
    diffusion_model_for_apply_model,
    execution_tensor_signature,
    in_stabilizer_window,
    resolve_clone_local_mixer_wrapper,
    resolve_multigpu_worker_wrapper,
    should_reraise,
)

logger = logging.getLogger(__name__)


def _get_crossattn_context(c_dict):
    context = c_dict.get("context")
    if context is None:
        context = c_dict.get("c_crossattn")
    return context


def _condition_row_index(row_count, cond_or_uncond, condition_index):
    """Map a ComfyUI condition-group index to its first tensor row."""
    if cond_or_uncond and row_count % len(cond_or_uncond) == 0:
        return condition_index * (row_count // len(cond_or_uncond))
    return condition_index if condition_index < row_count else 0


def _resolve_anchor_seeds(state):
    manual_anchor_seeds = list(state.get("anchor_seed_list") or [])
    if manual_anchor_seeds:
        return manual_anchor_seeds[:ANCHOR_SEEDS_MAX]
    seeds_count = max(
        1,
        min(int(state.get("anchor_seeds_count", 1)), ANCHOR_SEEDS_MAX),
    )
    return ANCHOR_SEEDS_POOL[:seeds_count]


def _resolve_keyframe_mode(state):
    mode = str(state.get(
        "anchor_keyframe_mode",
        ANCHOR_KEYFRAME_UNIFORM_SIGMA,
    ))
    return mode if mode in ANCHOR_KEYFRAME_MODES else ANCHOR_KEYFRAME_UNIFORM_SIGMA


def _context_signature(state, context):
    if state.get("_identity_context_signature", False):
        entry = state.get("_mixed_context_cache")
        if isinstance(entry, dict) and entry.get("mixed") is context:
            return (
                "adapter_mixed",
                entry.get("key"),
            )
        return ("adapter_context", execution_tensor_signature(state, context))
    return context_fingerprint(context)


def _trajectory_signature(state, user_x, c_dict):
    context = _get_crossattn_context(c_dict)
    if context is None:
        return None
    min_sigma = state.get("stabilizer_min_sigma")
    return (
        tuple(user_x.shape),
        str(user_x.dtype),
        state.get("_cache_namespace"),
        _context_signature(state, context),
        tuple(_resolve_anchor_seeds(state)),
        int(state.get("anchor_cache_points", ANCHOR_CACHE_POINTS_DEFAULT)),
        _resolve_keyframe_mode(state),
        int(state.get("anchor_deep_layer_threshold", -1)),
        None if min_sigma is None else round(float(min_sigma), 6),
    )


def _new_anchor_trajectory(state, signature, start_sigma):
    points = max(
        ANCHOR_CACHE_POINTS_MIN,
        min(
            int(state.get("anchor_cache_points", ANCHOR_CACHE_POINTS_DEFAULT)),
            ANCHOR_CACHE_POINTS_MAX,
        ),
    )
    end_sigma = state.get("stabilizer_min_sigma")
    end_sigma = 0.0 if end_sigma is None else float(end_sigma)
    end_sigma = min(float(start_sigma), max(0.0, end_sigma))
    targets = [
        float(start_sigma) + (end_sigma - float(start_sigma)) * index / (points - 1)
        for index in range(points)
    ]
    return {
        "signature": signature,
        "keyframe_mode": _resolve_keyframe_mode(state),
        "points": points,
        "ready": False,
        "frames": [],
        "targets": targets,
        "next_target_index": 1,
        "last_sigma": None,
        "last_cache": None,
        "bytes": 0,
        "observed_frames": 0,
        "pruned_frames": 0,
        "active_sigma": None,
        "active_device": None,
        "active_dtype": None,
        "observed_sigmas": [],
        "sampling_complete": False,
        "terminal_sigma": None,
        "minimum_observed_frames": (
            max(3, min(points, 8))
            if _resolve_keyframe_mode(state) == ANCHOR_KEYFRAME_ADAPTIVE_Q
            else max(2, min(points, 3))
        ),
    }


def _configure_trajectory_schedule(trajectory, state, c_dict):
    """Record the terminal denoising sigma when ComfyUI exposes the schedule."""
    if trajectory.get("terminal_sigma") is not None:
        return
    transformer_options = c_dict.get("transformer_options", {}) or {}
    sample_sigmas = transformer_options.get("sample_sigmas")
    values = None
    if torch.is_tensor(sample_sigmas) and sample_sigmas.numel() > 0:
        try:
            values = [
                float(value)
                for value in sample_sigmas.detach().reshape(-1).to("cpu").tolist()
            ]
        except Exception:
            values = None
    if values:
        # The final schedule entry is normally the post-denoise zero.  The last
        # model call uses the preceding sigma, so use that as the completion mark.
        terminal = values[-2] if len(values) >= 2 else values[-1]
        trajectory["terminal_sigma"] = float(terminal)
    else:
        threshold = state.get("stabilizer_min_sigma")
        if threshold is not None:
            trajectory["terminal_sigma"] = float(threshold)


def _record_trajectory_progress(trajectory, sigma):
    sigma = float(sigma)
    observed = trajectory.setdefault("observed_sigmas", [])
    if not any(abs(float(value) - sigma) <= 1e-6 for value in observed):
        observed.append(sigma)
    terminal = trajectory.get("terminal_sigma")
    if terminal is not None and sigma <= float(terminal) + 1e-3:
        trajectory["sampling_complete"] = True


def _copy_anchor_cache_to_cpu(state, cache):
    threshold = int(state.get("anchor_deep_layer_threshold", -1))
    output = {}
    for layer_index, hidden in cache.items():
        if threshold >= 0 and int(layer_index) >= threshold:
            continue
        if not torch.is_tensor(hidden):
            continue
        output[int(layer_index)] = hidden.detach().to(
            device="cpu",
            copy=True,
        ).contiguous()
    return output


def _frame_bytes(layers):
    return sum(
        int(hidden.numel()) * int(hidden.element_size())
        for hidden in layers.values()
    )


def _evenly_spaced_positions(length, count, device):
    length = int(length)
    count = int(count)
    if length <= 0 or count <= 0:
        return torch.empty(0, device=device, dtype=torch.long)
    if count == 1:
        return torch.zeros(1, device=device, dtype=torch.long)

    last_index = length - 1
    denominator = count - 1
    numerators = torch.arange(
        count,
        device=device,
        dtype=torch.long,
    ) * last_index
    return torch.div(
        numerators + denominator // 2,
        denominator,
        rounding_mode="floor",
    ).clamp_(0, last_index)


def _cache_sketch(cache, values_per_layer=256):
    pieces = []
    for layer_index in sorted(cache):
        flat = cache[layer_index].reshape(-1)
        count = min(int(values_per_layer), int(flat.numel()))
        if count <= 0:
            continue
        if count == 1:
            sample = flat[:1]
        else:
            positions = _evenly_spaced_positions(
                flat.numel(),
                count,
                flat.device,
            )
            sample = flat.index_select(0, positions)
        pieces.append(sample.to(dtype=torch.float32))
    return torch.cat(pieces) if pieces else torch.empty(0, dtype=torch.float32)


def _adaptive_frame_error(left, middle, right):
    low_sigma = float(left["sigma"])
    mid_sigma = float(middle["sigma"])
    high_sigma = float(right["sigma"])
    span = high_sigma - low_sigma
    if span <= 1e-12:
        return 0.0

    low = left.get("sketch")
    mid = middle.get("sketch")
    high = right.get("sketch")
    if not all(torch.is_tensor(value) for value in (low, mid, high)):
        return 0.0
    count = min(int(low.numel()), int(mid.numel()), int(high.numel()))
    if count <= 0:
        return 0.0

    blend = max(0.0, min(1.0, (mid_sigma - low_sigma) / span))
    predicted = low[:count] * (1.0 - blend) + high[:count] * blend
    residual = mid[:count] - predicted
    scale = mid[:count].square().mean() + predicted.square().mean()
    error = float((residual.square().mean() / scale.clamp_min(1e-12)).item())
    return error if math.isfinite(error) else float("inf")


def _prune_adaptive_frames(trajectory):
    frames = trajectory["frames"]
    points = int(trajectory.get("points", ANCHOR_CACHE_POINTS_DEFAULT))
    while len(frames) > points and len(frames) > 2:
        total_span = max(
            1e-12,
            float(frames[-1]["sigma"]) - float(frames[0]["sigma"]),
        )
        candidates = []
        for index in range(1, len(frames) - 1):
            left, middle, right = frames[index - 1:index + 2]
            error = _adaptive_frame_error(left, middle, right)
            left_gap = float(middle["sigma"]) - float(left["sigma"])
            right_gap = float(right["sigma"]) - float(middle["sigma"])
            coverage = max(0.0, left_gap * right_gap) / (total_span * total_span)
            candidates.append((round(error, 8), coverage, -index, index))
        remove_index = min(candidates)[-1]
        removed = frames.pop(remove_index)
        trajectory["bytes"] -= int(removed.get("bytes", 0))
        trajectory["pruned_frames"] += 1


def _store_trajectory_frame(state, trajectory, sigma, cache):
    sigma = float(sigma)
    if any(abs(float(frame["sigma"]) - sigma) <= 1e-6 for frame in trajectory["frames"]):
        return False
    cpu_cache = _copy_anchor_cache_to_cpu(state, cache)
    if not cpu_cache:
        return False
    byte_count = _frame_bytes(cpu_cache)
    frame = {
        "sigma": sigma,
        "layers": cpu_cache,
        "bytes": byte_count,
    }
    if trajectory.get("keyframe_mode") == ANCHOR_KEYFRAME_ADAPTIVE_Q:
        frame["sketch"] = _cache_sketch(cpu_cache)
    trajectory["frames"].append(frame)
    trajectory["frames"].sort(key=lambda frame: float(frame["sigma"]))
    trajectory["bytes"] += byte_count
    trajectory["observed_frames"] += 1
    if trajectory.get("keyframe_mode") == ANCHOR_KEYFRAME_ADAPTIVE_Q:
        _prune_adaptive_frames(trajectory)
    return True


def _record_trajectory_step(state, trajectory, sigma):
    cache = state.get("_anchor_cache") or {}
    if not cache:
        return

    sigma = float(sigma)
    _record_trajectory_progress(trajectory, sigma)
    trajectory["last_sigma"] = sigma
    trajectory["last_cache"] = cache
    if trajectory.get("keyframe_mode") == ANCHOR_KEYFRAME_ADAPTIVE_Q:
        _store_trajectory_frame(state, trajectory, sigma, cache)
        return
    if not trajectory["frames"]:
        _store_trajectory_frame(state, trajectory, sigma, cache)
        return

    targets = trajectory["targets"]
    next_index = int(trajectory["next_target_index"])
    crossed_target = False
    while next_index < len(targets) - 1 and sigma <= float(targets[next_index]) + 1e-6:
        crossed_target = True
        next_index += 1
    trajectory["next_target_index"] = next_index
    if crossed_target:
        _store_trajectory_frame(state, trajectory, sigma, cache)


def _finalize_anchor_trajectory(state):
    trajectory = state.get("_anchor_trajectory")
    if not trajectory or trajectory.get("ready", False):
        return

    last_cache = trajectory.get("last_cache") or {}
    last_sigma = trajectory.get("last_sigma")
    if last_cache and last_sigma is not None:
        _store_trajectory_frame(state, trajectory, last_sigma, last_cache)
    trajectory["last_cache"] = None

    if trajectory.get("keyframe_mode") == ANCHOR_KEYFRAME_ADAPTIVE_Q:
        targets_complete = True
    else:
        targets_complete = int(trajectory.get("next_target_index", 0)) >= max(
            1,
            len(trajectory.get("targets") or []) - 1,
        )
    enough_observations = (
        trajectory.get("sampling_complete", False)
        or int(trajectory.get("observed_frames", 0)) >= int(
            trajectory.get("minimum_observed_frames", 2)
        )
    )
    if len(trajectory["frames"]) < 2 or not targets_complete or not enough_observations:
        logger.warning(
            "[%s] warm-cache run ended before enough sigma keyframes/progress "
            "were captured; the partial cache is discarded.",
            state.get("anchor_log_name", "AnimaCrossAttn"),
        )
        state["_anchor_trajectory"] = None
        state["_anchor_cache"] = {}
        state["_anchor_cache_key"] = None
        return

    trajectory["ready"] = True
    trajectory["active_sigma"] = None
    trajectory["active_device"] = None
    trajectory["active_dtype"] = None
    for frame in trajectory["frames"]:
        frame.pop("sketch", None)
        frame.pop("bytes", None)
    state["_anchor_cache"] = {}
    state["_anchor_cache_key"] = None
    logger.info(
        "[%s] warm-cache ready: %d sigma keyframes (%s, %d observed), "
        "%.2f GiB CPU RAM. "
        "Later sampler seeds reuse it without anchor model passes.",
        state.get("anchor_log_name", "AnimaCrossAttn"),
        len(trajectory["frames"]),
        trajectory.get("keyframe_mode", ANCHOR_KEYFRAME_UNIFORM_SIGMA),
        int(trajectory.get("observed_frames", 0)),
        float(trajectory["bytes"]) / (1024.0 ** 3),
    )


def _ensure_anchor_trajectory(state, signature, sigma):
    trajectory = state.get("_anchor_trajectory")
    if trajectory is None or trajectory.get("signature") != signature:
        if trajectory is not None and not state.get("_warned_trajectory_invalidated", False):
            logger.info(
                "[%s] warm-cache inputs changed; rebuilding the anchor trajectory.",
                state.get("anchor_log_name", "AnimaCrossAttn"),
            )
            state["_warned_trajectory_invalidated"] = True
        trajectory = _new_anchor_trajectory(state, signature, sigma)
        state["_anchor_trajectory"] = trajectory
        state["_anchor_cache"] = {}
        state["_anchor_cache_key"] = None
    return trajectory


def _load_anchor_trajectory(state, trajectory, sigma, user_x):
    frames = trajectory.get("frames") or []
    if not frames:
        return False

    sigma = float(sigma)
    device_key = (user_x.device.type, user_x.device.index)
    dtype_key = str(user_x.dtype)
    if (
        trajectory.get("active_sigma") is not None
        and abs(float(trajectory["active_sigma"]) - sigma) <= 1e-6
        and trajectory.get("active_device") == device_key
        and trajectory.get("active_dtype") == dtype_key
        and state.get("_anchor_cache")
    ):
        return True

    lower = frames[0]
    upper = frames[-1]
    if sigma <= float(frames[0]["sigma"]):
        lower = upper = frames[0]
    elif sigma >= float(frames[-1]["sigma"]):
        lower = upper = frames[-1]
    else:
        for left, right in zip(frames, frames[1:]):
            if float(left["sigma"]) <= sigma <= float(right["sigma"]):
                lower, upper = left, right
                break

    low_sigma = float(lower["sigma"])
    high_sigma = float(upper["sigma"])
    if lower is upper or abs(high_sigma - low_sigma) <= 1e-12:
        blend = 0.0
    else:
        blend = max(0.0, min(1.0, (sigma - low_sigma) / (high_sigma - low_sigma)))

    layer_indices = sorted(
        set(lower["layers"]).intersection(upper["layers"])
    )
    loaded = {}
    for layer_index in layer_indices:
        low = lower["layers"][layer_index]
        if blend <= 1e-6:
            hidden = low.to(device=user_x.device, dtype=user_x.dtype)
        elif blend >= 1.0 - 1e-6:
            hidden = upper["layers"][layer_index].to(
                device=user_x.device,
                dtype=user_x.dtype,
            )
        else:
            hidden = low.to(
                device=user_x.device,
                dtype=user_x.dtype,
                copy=True,
            )
            high = upper["layers"][layer_index].to(
                device=user_x.device,
                dtype=user_x.dtype,
            )
            hidden.mul_(1.0 - blend).add_(high, alpha=blend)
        loaded[layer_index] = hidden

    if not loaded:
        return False
    state["_anchor_cache"] = loaded
    state["_anchor_cache_key"] = (
        "warm_cache",
        trajectory["signature"],
        round(sigma, 6),
    )
    trajectory["active_sigma"] = sigma
    trajectory["active_device"] = device_key
    trajectory["active_dtype"] = dtype_key
    if not state.get("_warned_trajectory_reuse", False):
        logger.info(
            "[%s] reusing the warmed Anchor-Q trajectory; no anchor model "
            "passes are needed for this run.",
            state.get("anchor_log_name", "AnimaCrossAttn"),
        )
        state["_warned_trajectory_reuse"] = True
    return True


def _run_or_load_warm_anchor(state, user_x, user_timestep, c_dict, apply_model, sigma):
    signature = _trajectory_signature(state, user_x, c_dict)
    if signature is None:
        return
    trajectory = _ensure_anchor_trajectory(state, signature, sigma)
    _configure_trajectory_schedule(trajectory, state, c_dict)
    if trajectory.get("ready", False):
        _load_anchor_trajectory(state, trajectory, sigma, user_x)
        return

    # The previous GPU snapshot is only needed if the run ends here. Release it
    # before capturing the next sigma so warmup does not retain two full Q sets.
    trajectory["last_cache"] = None
    maybe_run_anchor(
        state,
        user_x,
        user_timestep,
        c_dict,
        apply_model=apply_model,
    )
    if not state.get("_anchor_failed", False):
        _record_trajectory_step(state, trajectory, sigma)


def make_sigma_capture(state, prev_wrapper):
    def _wrapper_body(apply_model, options):
        clone_wrapper = resolve_clone_local_mixer_wrapper(
            apply_model,
            wrapper,
            state,
        )
        if clone_wrapper is not None:
            return clone_wrapper(apply_model, options)
        if not adapter_mixer_state_is_active(state, apply_model=apply_model):
            if prev_wrapper is not None:
                return prev_wrapper(apply_model, options)
            return apply_model(
                options["input"],
                options["timestep"],
                **options["c"],
            )
        ts = options.get("timestep")
        user_ts = ts
        c_dict = options.get("c", {}) or {}
        transformer_options = c_dict.get("transformer_options", {}) or {}
        is_multigpu = (
            isinstance(transformer_options, dict)
            and transformer_options.get("multigpu_thread_device") is not None
        )
        if is_multigpu:
            active_dm = diffusion_model_for_apply_model(
                apply_model,
                state.get("dm_ref"),
            )
            if active_dm is not None:
                worker_device = transformer_options.get("multigpu_thread_device")
                device_key = (
                    getattr(worker_device, "type", None),
                    getattr(worker_device, "index", None),
                )
                state.setdefault("_multigpu_dm_by_worker", {})[device_key] = active_dm
            worker_wrapper = resolve_multigpu_worker_wrapper(
                apply_model,
                options,
                wrapper,
            )
            if worker_wrapper is not None:
                # See the Adapter wrapper: model_options is shared by the
                # scheduler, but each clone has a rebound sigma wrapper/state.
                worker_options = dict(options)
                worker_options["_anima_mixer_worker_dispatch"] = True
                try:
                    return worker_wrapper(apply_model, worker_options)
                except BaseException as error:
                    if should_reraise(error):
                        clear_mixer_run_state(state, interrupted=True)
                    raise
        cur_sigma = None
        if ts is not None:
            try:
                cur_sigma = float(ts.flatten()[0].item())
                state["current_sigma"] = cur_sigma
            except Exception:
                pass

        # Adapter Mixer reaches this wrapper through its embedding wrapper,
        # which has already isolated closure state before any cached context is
        # read. Cross-Attn Mixer invokes this wrapper directly, so preserve a
        # compatible fallback for that path.
        is_run_start = state.pop("_adapter_mixer_run_start", None)
        finalize_warm_cache = state.pop(
            "_adapter_mixer_finalize_warm_cache", False,
        )
        if is_run_start is True and state.get("_mixer_run_start_pending", False):
            # Cross-Attn Mixer enters this sigma wrapper directly.  Its
            # on_pre_run callback already reset the run and left a pending
            # boundary marker, so consume that marker here to record the first
            # real sigma/call as well.  The Adapter wrapper has already called
            # begin_mixer_execution and therefore has no pending marker.
            begin_mixer_execution(
                state,
                apply_model,
                ts,
                owner_token_override=(
                    ("multigpu_wrapper", id(state)) if is_multigpu else None
                ),
            )
            state.pop("_adapter_mixer_run_start", None)
        if is_run_start is None:
            last_run_marker = state.get("_last_run_had_calls")
            prev_run_had_calls = bool(
                state.get("_run_active", False)
                if last_run_marker is None else last_run_marker
            )
            is_run_start, _owner_changed = begin_mixer_execution(
                state,
                apply_model,
                ts,
                owner_token_override=(
                    ("multigpu_wrapper", id(state)) if is_multigpu else None
                ),
            )
            # ``begin_mixer_execution`` writes this marker for the Adapter
            # wrapper to hand to us.  Cross-Attn calls begin directly, so clear
            # the hand-off marker here or every other call would skip begin().
            state.pop("_adapter_mixer_run_start", None)
            finalize_warm_cache = bool(
                prev_run_had_calls
                and str(state.get("anchor_refresh_mode", "once"))
                == ANCHOR_REFRESH_WARM_CACHE
            )
        refresh_mode = str(state.get("anchor_refresh_mode", "once"))
        if is_multigpu:
            state["_multigpu_call"] = True
        if is_run_start and finalize_warm_cache:
            _finalize_anchor_trajectory(state)

        if (
            not is_multigpu
            and state.get("artist_anchor_q", False)
            and not state.get("_anchor_failed", False)
            and not state.get("_embedding_mixer_failed", False)
            and in_stabilizer_window(state)
        ):
            prev_anchor_sigma = state.get("_anchor_last_sigma")
            is_anchor_start = (
                prev_anchor_sigma is None
                or (cur_sigma is not None and cur_sigma > prev_anchor_sigma + 1e-3)
            )
            state["_anchor_last_sigma"] = cur_sigma
            user_x = options.get("input")
            if user_x is not None and user_ts is not None and c_dict:
                if refresh_mode == ANCHOR_REFRESH_WARM_CACHE and cur_sigma is not None:
                    call_with_mixer_owner(
                        state,
                        apply_model,
                        _run_or_load_warm_anchor,
                        state,
                        user_x,
                        user_ts,
                        c_dict,
                        apply_model,
                        cur_sigma,
                    )
                elif is_anchor_start or not state.get("_anchor_cache"):
                    call_with_mixer_owner(
                        state,
                        apply_model,
                        maybe_run_anchor,
                        state,
                        user_x,
                        user_ts,
                        c_dict,
                        apply_model,
                    )

        try:
            def _call_previous():
                if prev_wrapper is not None:
                    return prev_wrapper(apply_model, options)
                return apply_model(options["input"], options["timestep"], **options["c"])

            return call_with_mixer_owner(state, apply_model, _call_previous)
        except BaseException as error:
            if should_reraise(error):
                clear_mixer_run_state(state, interrupted=True)
            raise

    def wrapper(apply_model, options):
        # Sigma capture has early clone/multi-GPU and run-boundary work before
        # its underlying-forward try block. Cover those paths with the same
        # abort cleanup contract as the Adapter embedding wrapper.
        try:
            return _wrapper_body(apply_model, options)
        except BaseException as error:
            if should_reraise(error):
                clear_mixer_run_state(state, interrupted=True)
            raise

    # Mark only the sigma wrapper created by Adapter Mixer.  ModelPatcher.clone()
    # carries function wrappers forward, so a later Mixer invocation must be
    # able to remove this layer without touching wrappers owned by other nodes.
    wrapper._anima_adapter_anchor_sigma_wrapper = True
    wrapper._anima_adapter_anchor_sigma_previous = prev_wrapper
    wrapper._anima_adapter_anchor_sigma_state = state
    wrapper._anima_mixer_state = state
    wrapper._anima_mixer_previous = prev_wrapper
    wrapper._anima_mixer_factory = make_sigma_capture
    return wrapper


def maybe_run_anchor(state, user_x, user_timestep, c_dict, apply_model=None):
    log_name = str(state.get("anchor_log_name", "AnimaCrossAttn"))
    base_context = _get_crossattn_context(c_dict)
    if base_context is None:
        return
    original_context = base_context

    transformer_options = c_dict.get("transformer_options", {}) or {}
    cou = transformer_options.get("cond_or_uncond")
    cond_idx = 0
    if cou is not None:
        if 0 not in cou:
            return
        cond_idx = cou.index(0)

    if base_context.dim() >= 2 and base_context.shape[0] > 1:
        row = _condition_row_index(base_context.shape[0], cou, cond_idx)
        base_context = base_context[row:row + 1]

    try:
        sigma_key = round(float(user_timestep.flatten()[0].item()), 4)
    except Exception:
        sigma_key = None
    anchor_seeds = _resolve_anchor_seeds(state)
    new_key = (
        state.get("_cache_namespace"),
        tuple(user_x.shape),
        _context_signature(state, original_context),
        sigma_key,
        tuple(anchor_seeds),
    )
    if state.get("_anchor_cache_key") == new_key and state.get("_anchor_cache"):
        return

    dm = state["dm_ref"]
    state["_anchor_cache"] = {}
    state["_in_anchor_run"] = True

    bsz = user_x.shape[0]
    if base_context.shape[0] != bsz:
        if base_context.shape[0] == 1:
            ctx_for_anchor = base_context.expand(bsz, -1, -1)
        else:
            ctx_for_anchor = base_context[:1].expand(bsz, -1, -1)
    else:
        ctx_for_anchor = base_context
    ctx_for_anchor = ctx_for_anchor.contiguous().to(device=user_x.device, dtype=user_x.dtype)

    anchor_kwargs = {}
    for key in ("t5xxl_ids", "t5xxl_weights"):
        v = c_dict.get(key)
        if v is None or not torch.is_tensor(v):
            continue
        if v.shape[0] != bsz:
            if v.shape[0] == 1:
                v = v.expand(bsz, *v.shape[1:])
            else:
                row = _condition_row_index(v.shape[0], cou, cond_idx)
                v = v[row:row + 1].expand(bsz, *v.shape[1:])
        anchor_kwargs[key] = v.contiguous()

    safe_opts = dict(transformer_options) if isinstance(transformer_options, dict) else {}
    safe_opts.pop("cond_or_uncond", None)
    safe_opts.pop("patches", None)

    try:
        with torch.no_grad():
            t5xxl_ids = anchor_kwargs.get("t5xxl_ids")
            t5xxl_weights = anchor_kwargs.get("t5xxl_weights")
            if t5xxl_ids is not None and hasattr(dm, "preprocess_text_embeds"):
                processed_ctx = dm.preprocess_text_embeds(
                    ctx_for_anchor, t5xxl_ids, t5xxl_weights=t5xxl_weights,
                )
            else:
                processed_ctx = ctx_for_anchor

            accumulator = {}
            for seed in anchor_seeds:
                gen = torch.Generator(device=user_x.device)
                gen.manual_seed(seed)
                anchor_x = torch.randn(
                    user_x.shape, generator=gen,
                    device=user_x.device, dtype=user_x.dtype,
                )
                state["_anchor_cache"] = {}
                if apply_model is not None:
                    kwargs = dict(c_dict)
                    if "context" in kwargs:
                        kwargs["context"] = processed_ctx
                    else:
                        kwargs["c_crossattn"] = processed_ctx
                    kwargs["transformer_options"] = safe_opts
                    apply_model(anchor_x, user_timestep, **kwargs)
                elif hasattr(dm, "_forward"):
                    dm._forward(
                        anchor_x,
                        user_timestep,
                        processed_ctx,
                        transformer_options=safe_opts,
                    )
                else:
                    dm(
                        anchor_x,
                        user_timestep,
                        processed_ctx,
                        transformer_options=safe_opts,
                    )

                for layer_idx, hidden in state["_anchor_cache"].items():
                    if layer_idx not in accumulator:
                        accumulator[layer_idx] = hidden.to(torch.float32)
                    else:
                        accumulator[layer_idx] = accumulator[layer_idx] + hidden.to(torch.float32)

            inv = 1.0 / max(1, len(anchor_seeds))
            state["_anchor_cache"] = {
                idx: (acc * inv).to(user_x.dtype) for idx, acc in accumulator.items()
            }
    except BaseException as e:
        if should_reraise(e):
            clear_mixer_run_state(state, interrupted=True)
            raise
        logger.warning(
            "[%s] anchor pre-run failed; anchor_q is disabled: %s",
            log_name,
            e,
        )
        state["_anchor_cache"] = {}
        state["_anchor_failed"] = True
    finally:
        state["_in_anchor_run"] = False

    if state["_anchor_cache"]:
        state["_anchor_cache_key"] = new_key
        if not state.get("_warned_anchor_ok", False):
            logger.info(
                "[%s] anchor pre-run captured %d layers of hidden state",
                log_name,
                len(state["_anchor_cache"]),
            )
            state["_warned_anchor_ok"] = True
    elif not state.get("_anchor_failed", False):
        state["_anchor_failed"] = True
        if not state.get("_warned_anchor_empty", False):
            logger.warning(
                "[%s] anchor pre-run captured no hidden states; anchor_q is "
                "disabled for this run.",
                log_name,
            )
            state["_warned_anchor_empty"] = True
