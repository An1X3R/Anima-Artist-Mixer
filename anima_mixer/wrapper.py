"""Cross-attention wrapper: the runtime artist-injection engine."""

import logging

import torch
import torch.nn as nn

from .constants import (
    ANCHOR_LAYER_THRESHOLD_DISABLED,
    COMBINE_LOWRANK_AVG,
    COMBINE_OUTPUT_AVG,
    FUSION_BASE_PRESERVE,
    FUSION_CONCAT_WITH_BASE,
    FUSION_INTERPOLATE,
    STATIC_CAPTURE_K_DEFAULT,
)
from .math_utils import limit_delta_norm, lowrank_rows_deterministic, project_perpendicular
from .parsing import normalize_weights
from .patching import (
    broadcast_batch,
    build_artists,
    clear_mixer_run_state,
    forward_fingerprint,
    in_sigma_range,
    in_stabilizer_window,
    resolve_mask,
    resolve_strengths,
    should_reraise,
)

logger = logging.getLogger(__name__)


def _combine_concat(individuals, weights):
    parts = [a * float(w) for a, w in zip(individuals, weights)]
    return torch.cat(parts, dim=1)


def _row_mask_like(mask, ref):
    return torch.tensor(mask, device=ref.device, dtype=torch.bool).view(
        len(mask), *([1] * (ref.dim() - 1))
    )


class CrossAttnWrapper(nn.Module):
    def __init__(self, original_forward, shared_state, layer_idx, original_module=None):
        super().__init__()
        self.original = original_forward
        self.original_module = original_module
        self._st = shared_state
        self._idx = layer_idx

    def rebind_state(
        self,
        shared_state,
        original_forward=None,
        original_module=None,
    ):
        """Create a clone-local wrapper without sharing mutable run state."""
        return type(self)(
            self.original if original_forward is None else original_forward,
            shared_state,
            self._idx,
            self.original_module if original_module is None else original_module,
        )

    def _warn_no_sigma(self):
        if not self._st.get("_warned_no_sigma", False):
            logger.warning(
                "[AnimaCrossAttn] cannot see the sampling sigma; EMA/static "
                "capture is disabled for this run."
            )
            self._st["_warned_no_sigma"] = True

    def _maybe_reset_ema(self):
        cur = self._st.get("current_sigma")
        if cur is None:
            return
        prev = self._st.get("_ema_last_sigma")
        if prev is None or cur > prev + 1e-3:
            self._st["_ema_cache"] = {}
        self._st["_ema_last_sigma"] = cur

    def _apply_ema(self, artist_total, fusion_mode, fp=None):
        if self._st.get("_multigpu_call", False):
            return artist_total
        if self._st.get("artist_static_capture", False):
            return artist_total
        ema_alpha = float(self._st.get("artist_ema_alpha", 0.0))
        if ema_alpha <= 0.0 or fusion_mode not in (FUSION_INTERPOLATE, FUSION_BASE_PRESERVE):
            return artist_total
        if self._st.get("current_sigma") is None:
            self._warn_no_sigma()
            return artist_total
        if not in_stabilizer_window(self._st):
            return artist_total
        self._maybe_reset_ema()
        cache = self._st.setdefault("_ema_cache", {})
        key = (self._idx, fp)
        prev = cache.get(key)
        if prev is not None and prev.shape == artist_total.shape:
            artist_total = ema_alpha * prev.to(artist_total.device, artist_total.dtype) + (
                1.0 - ema_alpha
            ) * artist_total
        cache[key] = artist_total.detach()
        return artist_total

    def _maybe_reset_static(self):
        cur = self._st.get("current_sigma")
        if cur is None:
            return
        prev = self._st.get("_static_last_sigma")
        if prev is None or cur > prev + 1e-3:
            self._st["_static_cache"] = {}
        self._st["_static_last_sigma"] = cur

    def _get_artist_outputs_with_cache(self, x, context, rope_emb, t_opts,
                                       individuals, fusion_mode, fp=None,
                                       extra_fp=None):
        if self._st.get("_multigpu_call", False):
            return self._collect_artist_outputs(
                x, context, rope_emb, t_opts, individuals, fusion_mode
            )
        if not self._st.get("artist_static_capture", False):
            return self._collect_artist_outputs(
                x, context, rope_emb, t_opts, individuals, fusion_mode
            )
        if self._st.get("current_sigma") is None:
            self._warn_no_sigma()
            return self._collect_artist_outputs(
                x, context, rope_emb, t_opts, individuals, fusion_mode
            )
        if not in_stabilizer_window(self._st):
            return self._collect_artist_outputs(
                x, context, rope_emb, t_opts, individuals, fusion_mode
            )
        if fusion_mode == FUSION_CONCAT_WITH_BASE:
            return self._collect_artist_outputs(
                x, context, rope_emb, t_opts, individuals, fusion_mode
            )

        self._maybe_reset_static()
        cache = self._st.setdefault("_static_cache", {})
        entry_fp = (tuple(x.shape), len(individuals), extra_fp)
        cache_key = (self._idx, fp)
        cur_sigma = self._st.get("current_sigma")
        sigma_key = round(float(cur_sigma), 4) if cur_sigma is not None else None

        entry = cache.get(cache_key)
        if entry is None or entry.get("_fp") != entry_fp:
            entry = {
                "_fp": entry_fp,
                "seen_sigmas": set(),
                "accumulator": None,
                "count": 0,
                "frozen": False,
                "frozen_outputs": None,
            }
            cache[cache_key] = entry

        if entry["frozen"]:
            return [o.to(context.device, context.dtype) for o in entry["frozen_outputs"]]

        if sigma_key is not None and sigma_key in entry["seen_sigmas"]:
            if entry["accumulator"] is not None and entry["count"] > 0:
                inv = 1.0 / entry["count"]
                return [(a * inv).to(context.device, context.dtype) for a in entry["accumulator"]]
            return self._collect_artist_outputs(
                x, context, rope_emb, t_opts, individuals, fusion_mode
            )

        outs = self._collect_artist_outputs(
            x, context, rope_emb, t_opts, individuals, fusion_mode
        )
        if entry["accumulator"] is None:
            entry["accumulator"] = [o.detach().to(torch.float32) for o in outs]
        else:
            for i, o in enumerate(outs):
                entry["accumulator"][i] = entry["accumulator"][i] + o.detach().to(torch.float32)
        entry["count"] += 1
        if sigma_key is not None:
            entry["seen_sigmas"].add(sigma_key)

        capture_k = int(self._st.get("static_capture_k", STATIC_CAPTURE_K_DEFAULT))
        if entry["count"] >= capture_k:
            inv = 1.0 / entry["count"]
            entry["frozen_outputs"] = [
                (a * inv).to(context.dtype).detach() for a in entry["accumulator"]
            ]
            entry["frozen"] = True
            entry["accumulator"] = None
            entry["seen_sigmas"] = None
            return [o.to(context.device, context.dtype) for o in entry["frozen_outputs"]]

        inv = 1.0 / entry["count"]
        return [(a * inv).to(context.device, context.dtype) for a in entry["accumulator"]]

    def _apply_fusion(self, base_out, artist_total, mask, fusion_mode, strength):
        row_mask = _row_mask_like(mask, base_out)
        row_strength = self._row_strength_like(strength, base_out)
        preserve = float(self._st.get("structure_preserve", 0.0))
        cap = float(self._st.get("delta_norm_cap", 0.0))
        if preserve <= 0.0 and cap <= 0.0:
            if fusion_mode == FUSION_BASE_PRESERVE:
                delta = artist_total - base_out
                delta_perp = project_perpendicular(delta, base_out)
                blended = base_out + row_strength * delta_perp
                return torch.where(row_mask, blended, base_out)
            if torch.is_tensor(row_strength):
                blended = base_out + row_strength * (artist_total - base_out)
            else:
                blended = base_out * (1.0 - row_strength) + artist_total * row_strength
            return torch.where(row_mask, blended, base_out)

        delta = artist_total - base_out
        if fusion_mode == FUSION_BASE_PRESERVE:
            delta = project_perpendicular(delta, base_out)
            delta = self._limit_structure_delta(delta, base_out)
            blended = base_out + row_strength * delta
            return torch.where(row_mask, blended, base_out)
        delta = self._structure_preserved_delta(delta, base_out)
        blended = base_out + row_strength * delta
        return torch.where(row_mask, blended, base_out)

    def _row_strength_like(self, strength, ref):
        if isinstance(strength, (list, tuple)):
            return torch.tensor(
                strength, device=ref.device, dtype=ref.dtype,
            ).view(len(strength), *([1] * (ref.dim() - 1)))
        return float(strength)

    def _structure_preserved_delta(self, delta, base_out):
        preserve = max(0.0, min(1.0, float(self._st.get("structure_preserve", 0.0))))
        if preserve > 0.0:
            delta_perp = project_perpendicular(delta, base_out)
            delta = delta * (1.0 - preserve) + delta_perp * preserve
        return self._limit_structure_delta(delta, base_out)

    def _limit_structure_delta(self, delta, base_out):
        cap = max(0.0, float(self._st.get("delta_norm_cap", 0.0)))
        if cap <= 0.0:
            return delta
        return limit_delta_norm(delta, base_out, cap)

    def _can_return_artist_directly(self, fusion_mode, strength, mask):
        if isinstance(strength, (list, tuple)):
            strength_is_one = all(
                (not m) or abs(float(s) - 1.0) < 1e-6
                for s, m in zip(strength, mask)
            )
        else:
            strength_is_one = abs(float(strength) - 1.0) < 1e-6
        return (
            fusion_mode == FUSION_INTERPOLATE
            and strength_is_one
            and all(mask)
            and float(self._st.get("structure_preserve", 0.0)) <= 0.0
            and float(self._st.get("delta_norm_cap", 0.0)) <= 0.0
            and float(self._st.get("style_balance", 0.0)) <= 0.0
        )

    def forward(self, x, context=None, rope_emb=None, transformer_options=None):
        st = self._st
        transformer_options = transformer_options or {}
        if (
            isinstance(transformer_options, dict)
            and transformer_options.get("multigpu_thread_device") is not None
        ):
            # MultiGPU workers share the main model-options wrapper.  Disable
            # stateful EMA/static/Anchor-Q reuse for that run rather than racing
            # one device's cache against another device's call.
            st["_multigpu_call"] = True

        if st.get("_in_anchor_run", False):
            cache = st.setdefault("_anchor_cache", {})
            cache[self._idx] = x.detach().clone()
            return self.original(
                x, context, rope_emb=rope_emb,
                transformer_options=transformer_options,
            )

        if not st.get("enabled", False) or context is None:
            return self.original(
                x, context, rope_emb=rope_emb,
                transformer_options=transformer_options,
            )

        if self._idx in st.get("_disabled_layers", set()):
            return self.original(
                x, context, rope_emb=rope_emb,
                transformer_options=transformer_options,
            )

        if not in_sigma_range(st):
            return self.original(
                x, context, rope_emb=rope_emb,
                transformer_options=transformer_options,
            )

        try:
            return self._dispatch(x, context, rope_emb, transformer_options)
        except BaseException as e:
            if should_reraise(e):
                clear_mixer_run_state(st, interrupted=True)
                raise
            logger.exception(
                "[AnimaCrossAttn] L%d injection failed; this layer falls back "
                "to original cross_attn for the rest of this run: %s",
                self._idx, e,
            )
            st.setdefault("_disabled_layers", set()).add(self._idx)
            return self.original(
                x, context, rope_emb=rope_emb,
                transformer_options=transformer_options,
            )

    def _dispatch(self, x, context, rope_emb, transformer_options):
        st = self._st
        dm = None
        if st.get("_multigpu_call", False):
            device_key = (
                getattr(context.device, "type", None),
                getattr(context.device, "index", None),
            )
            dm = (st.get("_multigpu_dm_by_worker") or {}).get(device_key)
        individuals, _ = build_artists(st, context, dm=dm)
        combine_mode = st["combine_mode"]
        fusion_mode = st["fusion_mode"]
        strength = float(st["strength"])
        weights = st["user_weights"]

        cou = transformer_options.get("cond_or_uncond") if isinstance(transformer_options, dict) else None
        bsz = context.shape[0]
        mask = resolve_mask(cou, bsz, st["apply_to_uncond"], st)
        strengths = resolve_strengths(
            cou, bsz, st["apply_to_uncond"], strength, st.get("uncond_strength", 1.0)
        )

        if not any(mask):
            return self.original(
                x, context, rope_emb=rope_emb,
                transformer_options=transformer_options,
            )

        fp = forward_fingerprint(st, context)

        if combine_mode == COMBINE_LOWRANK_AVG and len(individuals) >= 2:
            return self._fwd_lowrank_avg(
                x, context, rope_emb, transformer_options,
                individuals, weights, mask, fusion_mode, strengths, fp=fp,
            )

        if combine_mode in (COMBINE_OUTPUT_AVG, COMBINE_LOWRANK_AVG):
            return self._fwd_output_avg(
                x, context, rope_emb, transformer_options,
                individuals, weights, mask, fusion_mode, strengths, fp=fp,
            )

        combined = _combine_concat(individuals, weights)
        combined_fp = tuple(round(float(w), 6) for w in weights)
        return self._fwd_with_combined(
            x, context, rope_emb, transformer_options,
            combined, mask, fusion_mode, strengths, fp=fp, extra_fp=combined_fp,
        )

    def _resolved_weights(self, weights):
        if self._st.get("normalize_weights", True):
            return normalize_weights(weights)
        return list(weights)

    def _fwd_output_avg(self, x, context, rope_emb, t_opts,
                        individuals, weights, mask, fusion_mode, strength, fp=None):
        bsz = context.shape[0]
        ws = self._resolved_weights(weights)
        n = len(individuals)
        force_collect = (
            self._st.get("artist_static_capture", False)
            and fusion_mode != FUSION_CONCAT_WITH_BASE
        )

        artist_total = None
        style_balance = float(self._st.get("style_balance", 0.0))
        if style_balance > 0.0:
            base_out = self.original(x, context, rope_emb=rope_emb, transformer_options=t_opts)
            outs = self._get_artist_outputs_with_cache(
                x, context, rope_emb, t_opts, individuals, fusion_mode, fp=fp,
            )
            outs = self._balance_artist_outputs(outs, base_out)
            for out_i, w in zip(outs, ws):
                artist_total = out_i * w if artist_total is None else artist_total + out_i * w
        elif force_collect:
            outs = self._get_artist_outputs_with_cache(
                x, context, rope_emb, t_opts, individuals, fusion_mode, fp=fp,
            )
            for out_i, w in zip(outs, ws):
                artist_total = out_i * w if artist_total is None else artist_total + out_i * w
        elif n >= 2 and not self._st.get("_disable_batched", False):
            try:
                q_x = self._get_anchor_q_x(x)
                artist_total = self._batched_artists_forward(
                    q_x, context, rope_emb, t_opts, individuals, ws, fusion_mode,
                )
            except BaseException as e:
                if should_reraise(e):
                    clear_mixer_run_state(self._st, interrupted=True)
                    raise
                if not self._st.get("_warned_batched", False):
                    logger.warning(
                        "[AnimaCrossAttn] batched output_avg failed; falling "
                        "back to sequential mode: %s", e,
                    )
                    self._st["_warned_batched"] = True
                    self._st["_disable_batched"] = True
                artist_total = None

        if artist_total is None:
            q_x = self._get_anchor_q_x(x)
            for artist_i, w in zip(individuals, ws):
                artist_b = broadcast_batch(artist_i, bsz).to(
                    device=context.device, dtype=context.dtype,
                )
                kv = torch.cat([context, artist_b], dim=1) \
                    if fusion_mode == FUSION_CONCAT_WITH_BASE else artist_b
                out_i = self.original(q_x, kv, rope_emb=rope_emb, transformer_options=t_opts)
                artist_total = out_i * w if artist_total is None else artist_total + out_i * w

        artist_total = self._apply_ema(artist_total, fusion_mode, fp=fp)

        if self._can_return_artist_directly(fusion_mode, strength, mask):
            return artist_total
        if "base_out" not in locals():
            base_out = self.original(x, context, rope_emb=rope_emb, transformer_options=t_opts)
        return self._apply_fusion(base_out, artist_total, mask, fusion_mode, strength)

    def _balance_artist_outputs(self, artist_outs, base_out):
        balance = max(0.0, min(1.0, float(self._st.get("style_balance", 0.0))))
        if balance <= 0.0 or len(artist_outs) < 2:
            return artist_outs

        base_f32 = base_out.to(torch.float32)
        deltas = torch.stack(
            [(out - base_out).to(torch.float32) for out in artist_outs],
            dim=0,
        )
        flat = deltas.reshape(deltas.shape[0], deltas.shape[1], -1)
        norms = torch.linalg.vector_norm(flat, dim=-1).clamp(min=1e-8)
        target = norms.mean(dim=0, keepdim=True)
        scale = (target / norms).clamp(min=0.5, max=2.0)
        scale = scale.view(deltas.shape[0], deltas.shape[1], *([1] * (deltas.dim() - 2)))
        balanced = deltas * (1.0 - balance) + deltas * scale * balance
        return [
            (base_f32 + balanced[i]).to(device=artist_outs[i].device, dtype=artist_outs[i].dtype)
            for i in range(len(artist_outs))
        ]

    def _get_anchor_q_x(self, x):
        st = self._st
        if st.get("_multigpu_call", False):
            return x
        if not st.get("artist_anchor_q", False):
            return x
        if st.get("_anchor_failed", False):
            return x
        if not in_stabilizer_window(st):
            return x

        threshold = int(st.get("anchor_deep_layer_threshold", ANCHOR_LAYER_THRESHOLD_DISABLED))
        if threshold >= 0 and self._idx >= threshold:
            return x

        anchor_x = st.get("_anchor_cache", {}).get(self._idx)
        if anchor_x is None:
            return x
        if anchor_x.shape != x.shape:
            if anchor_x.shape[1:] == x.shape[1:]:
                ax_bsz = anchor_x.shape[0]
                bsz = x.shape[0]
                if bsz % ax_bsz == 0:
                    anchor_x = anchor_x.repeat(bsz // ax_bsz, *([1] * (anchor_x.dim() - 1)))
                elif ax_bsz % bsz == 0:
                    anchor_x = anchor_x[:bsz]
                else:
                    return x
            else:
                return x
        anchor_x = anchor_x.to(device=x.device, dtype=x.dtype)

        blend = max(0.0, min(1.0, float(st.get("anchor_user_blend", 0.0))))
        if blend > 0.0:
            return blend * x + (1.0 - blend) * anchor_x
        return anchor_x

    def _collect_artist_outputs(self, x, context, rope_emb, t_opts,
                                individuals, fusion_mode):
        bsz = context.shape[0]
        n = len(individuals)
        q_x = self._get_anchor_q_x(x)
        if n >= 2 and not self._st.get("_disable_batched", False):
            try:
                return self._batched_artists_outputs_only(
                    q_x, context, rope_emb, t_opts, individuals, fusion_mode,
                )
            except BaseException as e:
                if should_reraise(e):
                    clear_mixer_run_state(self._st, interrupted=True)
                    raise
                if not self._st.get("_warned_batched", False):
                    logger.warning(
                        "[AnimaCrossAttn] batched outputs failed; falling back "
                        "to sequential mode: %s", e,
                    )
                    self._st["_warned_batched"] = True
                    self._st["_disable_batched"] = True
        outs = []
        for artist_i in individuals:
            artist_b = broadcast_batch(artist_i, bsz).to(
                device=context.device, dtype=context.dtype,
            )
            kv = torch.cat([context, artist_b], dim=1) \
                if fusion_mode == FUSION_CONCAT_WITH_BASE else artist_b
            outs.append(self.original(q_x, kv, rope_emb=rope_emb, transformer_options=t_opts))
        return outs

    def _supports_q_reuse(self):
        module = self.original_module
        return bool(
            module is not None
            and not getattr(module, "is_selfattn", True)
            and all(hasattr(module, name) for name in (
                "q_proj", "q_norm", "k_proj", "k_norm", "v_proj",
                "compute_attention", "n_heads", "head_dim",
            ))
        )

    def _auto_artist_chunk_size(self, x, kv_length, artist_count):
        if artist_count <= 1 or x.device.type not in ("cuda", "xpu", "npu", "mlu"):
            return artist_count

        cache = self._st.setdefault("_artist_chunk_cache", {})
        key = (
            x.device.type, x.device.index, str(x.dtype), tuple(x.shape),
            int(kv_length), int(artist_count),
        )
        cached = cache.get(key)
        if cached is not None:
            return cached

        chunk_size = artist_count
        try:
            from comfy import model_management

            free_memory = int(model_management.get_free_memory(x.device))
            module = self.original_module
            inner_dim = int(
                getattr(module, "n_heads", 1) * getattr(module, "head_dim", x.shape[-1])
            )
            batch_size = int(x.shape[0])
            element_size = int(x.element_size())

            # Per artist: repeated input/Q, attention output/final output, K/V,
            # plus conservative SDPA workspace. The reserve leaves room for
            # the rest of the DiT block and ComfyUI's allocator.
            q_side = 4 * int(x.numel())
            kv_side = 3 * batch_size * int(kv_length) * inner_dim
            per_artist = max(1, int((q_side + kv_side) * element_size * 2.0))
            reserve = max(768 * 1024 ** 2, int(free_memory * 0.20))
            usable = max(0, free_memory - reserve)
            chunk_size = max(1, min(artist_count, usable // per_artist))
        except Exception as e:
            logger.debug("[AnimaCrossAttn] automatic artist chunk sizing unavailable: %s", e)

        cache[key] = int(chunk_size)
        if chunk_size < artist_count and not self._st.get("_warned_auto_chunk", False):
            logger.info(
                "[AnimaCrossAttn] VRAM-aware artist batching: %d artists per chunk "
                "(%d total).",
                chunk_size, artist_count,
            )
            self._st["_warned_auto_chunk"] = True
        return int(chunk_size)

    @staticmethod
    def _repeat_transformer_options(t_opts, repeat_count):
        new_opts = dict(t_opts) if isinstance(t_opts, dict) else {}
        cou = new_opts.get("cond_or_uncond")
        if cou is not None and repeat_count > 1:
            new_opts["cond_or_uncond"] = list(cou) * repeat_count
        return new_opts

    def _project_reusable_q(self, x):
        module = self.original_module
        q_shape = (*x.shape[:-1], module.n_heads, module.head_dim)
        return module.q_norm(module.q_proj(x).view(q_shape))

    def _q_reuse_artists_chunk(self, x, kv_stacked, rope_emb, t_opts, chunk_count,
                               reusable_q=None):
        module = self.original_module
        bsz = x.shape[0]
        kv_shape = (*kv_stacked.shape[:-1], module.n_heads, module.head_dim)

        q = reusable_q if reusable_q is not None else self._project_reusable_q(x)
        k = module.k_norm(module.k_proj(kv_stacked).view(kv_shape))
        v = module.v_proj(kv_stacked).view(kv_shape)
        v_norm = getattr(module, "v_norm", None)
        if v_norm is not None:
            v = v_norm(v)

        q = q.repeat(chunk_count, *([1] * (q.dim() - 1)))
        new_opts = self._repeat_transformer_options(t_opts, chunk_count)
        out = module.compute_attention(q, k, v, transformer_options=new_opts)
        return out.view(chunk_count, bsz, *out.shape[1:])

    def _original_artists_chunk(self, x, kv_stacked, rope_emb, t_opts, chunk_count):
        bsz = x.shape[0]
        x_rep = x.repeat(chunk_count, *([1] * (x.dim() - 1)))
        rope_rep = rope_emb
        if rope_emb is not None and torch.is_tensor(rope_emb):
            if rope_emb.dim() > 0 and rope_emb.shape[0] == bsz:
                rope_rep = rope_emb.repeat(chunk_count, *([1] * (rope_emb.dim() - 1)))
        new_opts = self._repeat_transformer_options(t_opts, chunk_count)
        out = self.original(x_rep, kv_stacked, rope_emb=rope_rep, transformer_options=new_opts)
        return out.view(chunk_count, bsz, *out.shape[1:])

    def _artists_chunk_outputs(self, x, kv_list, rope_emb, t_opts, reusable_q=None):
        chunk_count = len(kv_list)
        kv_stacked = torch.cat(kv_list, dim=0)

        if self._supports_q_reuse():
            validation = self._st.setdefault("_q_reuse_validation", {})
            module_type = type(self.original_module)
            validated = validation.get(module_type)
            if validated is True:
                return self._q_reuse_artists_chunk(
                    x, kv_stacked, rope_emb, t_opts, chunk_count,
                    reusable_q=reusable_q,
                )
            if validated is None:
                try:
                    validation_kv = kv_list[0]
                    optimized = self._q_reuse_artists_chunk(
                        x, validation_kv, rope_emb, t_opts, 1,
                        reusable_q=reusable_q,
                    )
                    reference = self._original_artists_chunk(
                        x, kv_stacked, rope_emb, t_opts, chunk_count,
                    )
                    tolerance = 2e-3 if x.dtype in (torch.float16, torch.bfloat16) else 1e-5
                    validated = bool(torch.allclose(
                        optimized[0], reference[0], rtol=tolerance, atol=tolerance,
                    ))
                    validation[module_type] = validated
                    if validated:
                        logger.info(
                            "[AnimaCrossAttn] Q projection reuse validated and enabled."
                        )
                        self._st["_warned_q_reuse"] = True
                    else:
                        logger.warning(
                            "[AnimaCrossAttn] Q reuse validation differed from the original; "
                            "the standard attention path remains active."
                        )
                    return reference
                except BaseException as e:
                    if should_reraise(e):
                        clear_mixer_run_state(self._st, interrupted=True)
                        raise
                    validation[module_type] = False
                    logger.warning(
                        "[AnimaCrossAttn] Q reuse validation failed; using the standard path: %s",
                        e,
                    )

        return self._original_artists_chunk(
            x, kv_stacked, rope_emb, t_opts, chunk_count,
        )

    def _batched_artists_outputs_only(self, x, context, rope_emb, t_opts,
                                      individuals, fusion_mode):
        bsz = context.shape[0]
        n = len(individuals)
        kv_list = []
        for artist_i in individuals:
            artist_b = broadcast_batch(artist_i, bsz).to(
                device=context.device, dtype=context.dtype,
            )
            if fusion_mode == FUSION_CONCAT_WITH_BASE:
                kv_list.append(torch.cat([context, artist_b], dim=1))
            else:
                kv_list.append(artist_b)
        kv_lens = {kv.shape[1] for kv in kv_list}
        if len(kv_lens) > 1:
            raise ValueError(f"K/V lengths differ {kv_lens}; cannot batch")
        kv_length = next(iter(kv_lens))
        chunk_size = self._auto_artist_chunk_size(x, kv_length, n)
        reusable_q = None
        if self._supports_q_reuse():
            validation = self._st.setdefault("_q_reuse_validation", {})
            if validation.get(type(self.original_module)) is not False:
                reusable_q = self._project_reusable_q(x)
        outputs = []
        for start in range(0, n, chunk_size):
            chunk = kv_list[start:start + chunk_size]
            chunk_out = self._artists_chunk_outputs(
                x, chunk, rope_emb, t_opts, reusable_q=reusable_q,
            )
            outputs.extend(chunk_out[i] for i in range(chunk_out.shape[0]))
        return outputs

    def _batched_artists_forward(self, x, context, rope_emb, t_opts,
                                 individuals, weights, fusion_mode):
        outs = self._batched_artists_outputs_only(
            x, context, rope_emb, t_opts, individuals, fusion_mode,
        )
        total = None
        for out_i, w in zip(outs, weights):
            total = out_i * w if total is None else total + out_i * w
        return total

    def _fwd_lowrank_avg(self, x, context, rope_emb, t_opts,
                         individuals, weights, mask, fusion_mode, strength, fp=None):
        ws = self._resolved_weights(weights)
        n = len(individuals)
        k = max(1, min(int(self._st.get("lowrank_k", 1)), n))

        artist_outs = self._get_artist_outputs_with_cache(
            x, context, rope_emb, t_opts, individuals, fusion_mode, fp=fp,
        )
        base_out = self.original(x, context, rope_emb=rope_emb, transformer_options=t_opts)
        artist_outs = self._balance_artist_outputs(artist_outs, base_out)
        out_dtype = base_out.dtype

        a = torch.stack(artist_outs, dim=0).to(torch.float32)
        base_f32 = base_out.to(torch.float32).unsqueeze(0)
        delta = a - base_f32
        orig_shape = delta.shape
        rows = delta.reshape(n, -1)

        if k < n:
            try:
                rows = lowrank_rows_deterministic(rows, k)
            except BaseException as e:
                if should_reraise(e):
                    clear_mixer_run_state(self._st, interrupted=True)
                    raise
                if not self._st.get("_warned_svd", False):
                    logger.warning(
                        "[AnimaCrossAttn] L%d lowrank_avg failed; this step "
                        "degrades to output_avg: %s", self._idx, e,
                    )
                    self._st["_warned_svd"] = True

        w_t = torch.tensor(ws, device=rows.device, dtype=rows.dtype).view(n, 1)
        delta_avg = (rows * w_t).sum(dim=0).reshape(orig_shape[1:]).to(out_dtype)
        artist_total = base_out + delta_avg

        artist_total = self._apply_ema(artist_total, fusion_mode, fp=fp)
        if self._can_return_artist_directly(fusion_mode, strength, mask):
            return artist_total
        return self._apply_fusion(base_out, artist_total, mask, fusion_mode, strength)

    def _fwd_with_combined(self, x, context, rope_emb, t_opts,
                           combined, mask, fusion_mode, strength, fp=None, extra_fp=None):
        bsz = context.shape[0]
        artist_b = broadcast_batch(combined, bsz).to(
            device=context.device, dtype=context.dtype,
        )

        if fusion_mode in (FUSION_INTERPOLATE, FUSION_BASE_PRESERVE):
            base_out = self.original(x, context, rope_emb=rope_emb, transformer_options=t_opts)
            outs = self._get_artist_outputs_with_cache(
                x, context, rope_emb, t_opts, [artist_b], fusion_mode,
                fp=fp, extra_fp=extra_fp,
            )
            artist_out = self._apply_ema(outs[0], fusion_mode, fp=fp)

            if self._can_return_artist_directly(fusion_mode, strength, mask):
                return artist_out
            return self._apply_fusion(base_out, artist_out, mask, fusion_mode, strength)

        merged = torch.cat([context, artist_b], dim=1)
        if all(mask):
            return self.original(x, merged, rope_emb=rope_emb, transformer_options=t_opts)
        merged_out = self.original(x, merged, rope_emb=rope_emb, transformer_options=t_opts)
        base_out = self.original(x, context, rope_emb=rope_emb, transformer_options=t_opts)
        row_mask = _row_mask_like(mask, merged_out)
        return torch.where(row_mask, merged_out, base_out)
