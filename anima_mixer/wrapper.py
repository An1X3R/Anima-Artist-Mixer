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
    forward_fingerprint,
    in_sigma_range,
    in_stabilizer_window,
    resolve_mask,
)

logger = logging.getLogger(__name__)


def _should_reraise(error):
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
        preserve = float(self._st.get("structure_preserve", 0.0))
        cap = float(self._st.get("delta_norm_cap", 0.0))
        if preserve <= 0.0 and cap <= 0.0:
            if fusion_mode == FUSION_BASE_PRESERVE:
                delta = artist_total - base_out
                delta_perp = project_perpendicular(delta, base_out)
                blended = base_out + strength * delta_perp
                return torch.where(row_mask, blended, base_out)
            blended = base_out * (1.0 - strength) + artist_total * strength
            return torch.where(row_mask, blended, base_out)

        delta = artist_total - base_out
        if fusion_mode == FUSION_BASE_PRESERVE:
            delta = project_perpendicular(delta, base_out)
            delta = self._limit_structure_delta(delta, base_out)
            blended = base_out + strength * delta
            return torch.where(row_mask, blended, base_out)
        delta = self._structure_preserved_delta(delta, base_out)
        blended = base_out + strength * delta
        return torch.where(row_mask, blended, base_out)

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
        return (
            fusion_mode == FUSION_INTERPOLATE
            and strength == 1.0
            and all(mask)
            and float(self._st.get("structure_preserve", 0.0)) <= 0.0
            and float(self._st.get("delta_norm_cap", 0.0)) <= 0.0
        )

    def forward(self, x, context=None, rope_emb=None, transformer_options=None):
        st = self._st
        transformer_options = transformer_options or {}

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
        except Exception as e:
            if _should_reraise(e):
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
        individuals, _ = build_artists(st, context)
        combine_mode = st["combine_mode"]
        fusion_mode = st["fusion_mode"]
        strength = float(st["strength"])
        weights = st["user_weights"]

        cou = transformer_options.get("cond_or_uncond") if isinstance(transformer_options, dict) else None
        bsz = context.shape[0]
        mask = resolve_mask(cou, bsz, st["apply_to_uncond"], st)

        if not any(mask):
            return self.original(
                x, context, rope_emb=rope_emb,
                transformer_options=transformer_options,
            )

        fp = forward_fingerprint(st, context)

        if combine_mode == COMBINE_LOWRANK_AVG and len(individuals) >= 2:
            return self._fwd_lowrank_avg(
                x, context, rope_emb, transformer_options,
                individuals, weights, mask, fusion_mode, strength, fp=fp,
            )

        if combine_mode in (COMBINE_OUTPUT_AVG, COMBINE_LOWRANK_AVG):
            return self._fwd_output_avg(
                x, context, rope_emb, transformer_options,
                individuals, weights, mask, fusion_mode, strength, fp=fp,
            )

        combined = _combine_concat(individuals, weights)
        combined_fp = tuple(round(float(w), 6) for w in weights)
        return self._fwd_with_combined(
            x, context, rope_emb, transformer_options,
            combined, mask, fusion_mode, strength, fp=fp, extra_fp=combined_fp,
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
        if force_collect:
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
            except Exception as e:
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
        base_out = self.original(x, context, rope_emb=rope_emb, transformer_options=t_opts)
        return self._apply_fusion(base_out, artist_total, mask, fusion_mode, strength)

    def _get_anchor_q_x(self, x):
        st = self._st
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
            except Exception as e:
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
        x_rep = x.repeat(n, *([1] * (x.dim() - 1)))
        kv_stacked = torch.cat(kv_list, dim=0)
        rope_rep = rope_emb
        if rope_emb is not None and torch.is_tensor(rope_emb):
            if rope_emb.dim() > 0 and rope_emb.shape[0] == bsz:
                rope_rep = rope_emb.repeat(n, *([1] * (rope_emb.dim() - 1)))
        new_opts = dict(t_opts) if isinstance(t_opts, dict) else {}
        cou = new_opts.get("cond_or_uncond")
        if cou is not None:
            new_opts["cond_or_uncond"] = list(cou) * n
        out = self.original(x_rep, kv_stacked, rope_emb=rope_rep, transformer_options=new_opts)
        out = out.view(n, bsz, *out.shape[1:])
        return [out[i] for i in range(n)]

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
        out_dtype = base_out.dtype

        a = torch.stack(artist_outs, dim=0).to(torch.float32)
        base_f32 = base_out.to(torch.float32).unsqueeze(0)
        delta = a - base_f32
        orig_shape = delta.shape
        rows = delta.reshape(n, -1)

        if k < n:
            try:
                rows = lowrank_rows_deterministic(rows, k)
            except Exception as e:
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
