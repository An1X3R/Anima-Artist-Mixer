"""Anchor-Q pre-run and sigma capture helpers."""

import logging

import torch

from .constants import ANCHOR_SEEDS_MAX, ANCHOR_SEEDS_POOL
from .patching import context_fingerprint, in_stabilizer_window, reset_run_state

logger = logging.getLogger(__name__)


def _get_crossattn_context(c_dict):
    context = c_dict.get("context")
    if context is None:
        context = c_dict.get("c_crossattn")
    return context


def make_sigma_capture(state, prev_wrapper):
    def wrapper(apply_model, options):
        ts = options.get("timestep")
        cur_sigma = None
        if ts is not None:
            try:
                cur_sigma = float(ts.flatten()[0].item())
                state["current_sigma"] = cur_sigma
            except Exception:
                pass

        prev_run_sigma = state.get("_run_last_sigma")
        if prev_run_sigma is None or (
            cur_sigma is not None and cur_sigma > prev_run_sigma + 1e-3
        ):
            reset_run_state(state)
        state["_run_last_sigma"] = cur_sigma

        if (
            state.get("artist_anchor_q", False)
            and not state.get("_anchor_failed", False)
            and in_stabilizer_window(state)
        ):
            prev_anchor_sigma = state.get("_anchor_last_sigma")
            is_run_start = (
                prev_anchor_sigma is None
                or (cur_sigma is not None and cur_sigma > prev_anchor_sigma + 1e-3)
            )
            state["_anchor_last_sigma"] = cur_sigma
            if is_run_start or not state.get("_anchor_cache"):
                user_x = options.get("input")
                user_ts = options.get("timestep")
                c_dict = options.get("c", {}) or {}
                if user_x is not None and user_ts is not None and c_dict:
                    maybe_run_anchor(state, user_x, user_ts, c_dict, apply_model=apply_model)

        if prev_wrapper is not None:
            return prev_wrapper(apply_model, options)
        return apply_model(options["input"], options["timestep"], **options["c"])
    return wrapper


def maybe_run_anchor(state, user_x, user_timestep, c_dict, apply_model=None):
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
        row = cond_idx if cond_idx < base_context.shape[0] else 0
        base_context = base_context[row:row + 1]

    try:
        sigma_key = round(float(user_timestep.flatten()[0].item()), 4)
    except Exception:
        sigma_key = None
    manual_anchor_seeds = list(state.get("anchor_seed_list") or [])
    if manual_anchor_seeds:
        anchor_seeds = manual_anchor_seeds[:ANCHOR_SEEDS_MAX]
    else:
        seeds_count = max(1, min(int(state.get("anchor_seeds_count", 1)), ANCHOR_SEEDS_MAX))
        anchor_seeds = ANCHOR_SEEDS_POOL[:seeds_count]
    new_key = (
        tuple(user_x.shape),
        context_fingerprint(original_context),
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
                row = cond_idx if cond_idx < v.shape[0] else 0
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
                    dm._forward(anchor_x, user_timestep, processed_ctx, transformer_options=safe_opts)
                else:
                    dm(anchor_x, user_timestep, processed_ctx, transformer_options=safe_opts)

                for layer_idx, hidden in state["_anchor_cache"].items():
                    if layer_idx not in accumulator:
                        accumulator[layer_idx] = hidden.to(torch.float32)
                    else:
                        accumulator[layer_idx] = accumulator[layer_idx] + hidden.to(torch.float32)

            inv = 1.0 / max(1, len(anchor_seeds))
            state["_anchor_cache"] = {
                idx: (acc * inv).to(user_x.dtype) for idx, acc in accumulator.items()
            }
    except Exception as e:
        logger.warning(
            "[AnimaCrossAttn] anchor pre-run failed; anchor_q is disabled: %s", e,
        )
        state["_anchor_cache"] = {}
        state["_anchor_failed"] = True
    finally:
        state["_in_anchor_run"] = False

    if state["_anchor_cache"]:
        state["_anchor_cache_key"] = new_key
        if not state.get("_warned_anchor_ok", False):
            logger.info(
                "[AnimaCrossAttn] anchor pre-run captured %d layers of hidden state",
                len(state["_anchor_cache"]),
            )
            state["_warned_anchor_ok"] = True
    elif not state.get("_anchor_failed", False):
        state["_anchor_failed"] = True
        if not state.get("_warned_anchor_empty", False):
            logger.warning(
                "[AnimaCrossAttn] anchor pre-run captured no hidden states; "
                "anchor_q is disabled for this run."
            )
            state["_warned_anchor_empty"] = True
