"""Core ComfyUI nodes: Pack and CrossAttn patcher."""

import logging

from .anchor import make_sigma_capture
from .constants import (
    ANCHOR_LAYER_THRESHOLD_DISABLED,
    ANCHOR_SEEDS_MAX,
    COMBINE_CONCAT,
    COMBINE_LOWRANK_AVG,
    COMBINE_OUTPUT_AVG,
    FUSION_BASE_PRESERVE,
    FUSION_CONCAT_WITH_BASE,
    FUSION_INTERPOLATE,
    MAX_ARTISTS,
    STATIC_CAPTURE_K_DEFAULT,
    STATIC_CAPTURE_K_MAX,
)
from .parsing import (
    expand_prompt_weights,
    parse_anchor_seed_list,
    parse_artist_entries,
    parse_layer_filter,
    split_artist_chain,
)
from .patching import (
    extract_conditioning,
    make_cross_attn_forward_patch,
    register_mixer_lifecycle,
    unwrap_cross_attn,
    unwrap_cross_attn_forward,
    validate_model,
)
from .wrapper import CrossAttnWrapper

logger = logging.getLogger(__name__)


class AnimaArtistPack:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "artist_chain": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "Artist chain. Use comma or newline separators.\n"
                        "Supports CLIP-side weights like (wlop:1.5) and "
                        "linear injection weights like 1.5::wlop / wlop::1.5.\n"
                        "Explicit ::weight remains relative while normalize_weights is enabled."
                    ),
                }),
            },
            "optional": {
                "base_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "Main prompt. Each artist is encoded as "
                        "'<artist>\\n<base_prompt>'."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("ANIMA_PACK",)
    RETURN_NAMES = ("artist_pack",)
    FUNCTION = "pack"
    CATEGORY = "Anima/CrossAttn"

    def pack(self, clip, artist_chain, base_prompt=""):
        parts = split_artist_chain(artist_chain)
        entries = parse_artist_entries(parts)
        names = [entry[0] for entry in entries]
        parsed_weights = [entry[1] for entry in entries]
        explicit_flags = [entry[2] for entry in entries]
        has_explicit = any(explicit_flags)

        base_raw = (base_prompt or "").strip()
        base = expand_prompt_weights(base_raw)
        if base != base_raw:
            logger.info(
                "[AnimaArtistPack] expanded base_prompt ::weight syntax to "
                "ComfyUI bracket syntax before CLIP encoding"
            )

        try:
            base_tokens = clip.tokenize(base)
            base_conditioning = clip.encode_from_tokens_scheduled(base_tokens)
        except Exception as e:
            raise ValueError(
                f"[AnimaArtistPack] failed to encode base_prompt (text={base!r}): {e}"
            )

        if not names:
            return ({
                "conditionings": [],
                "labels": [],
                "weights": [],
                "has_explicit_weights": False,
                "base_prompt": base,
                "base_conditioning": base_conditioning,
            },)

        if len(names) > MAX_ARTISTS:
            logger.warning(
                "[AnimaArtistPack] artist count %d exceeds the limit %d; truncating",
                len(names), MAX_ARTISTS,
            )
            names = names[:MAX_ARTISTS]
            parsed_weights = parsed_weights[:MAX_ARTISTS]
            explicit_flags = explicit_flags[:MAX_ARTISTS]
            has_explicit = any(explicit_flags)

        conditionings = []
        for name in names:
            text = f"{name}\n{base}" if base else name
            try:
                tokens = clip.tokenize(text)
                cond = clip.encode_from_tokens_scheduled(tokens)
            except Exception as e:
                raise ValueError(
                    f"[AnimaArtistPack] encoding failed (text={text!r}): {e}"
                )
            conditionings.append(cond)

        if has_explicit:
            logger.info(
                "[AnimaArtistPack] detected explicit ::weight syntax; "
                "normalize_weights will keep them as relative ratios when enabled"
            )

        return ({
            "conditionings": conditionings,
            "labels": names,
            "weights": parsed_weights,
            "has_explicit_weights": has_explicit,
            "base_prompt": base,
            "base_conditioning": base_conditioning,
        },)


class AnimaArtistCrossAttn:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "artist_pack": ("ANIMA_PACK",),
                "combine_mode": (
                    [COMBINE_CONCAT, COMBINE_OUTPUT_AVG, COMBINE_LOWRANK_AVG],
                    {"default": COMBINE_OUTPUT_AVG},
                ),
                "fusion_mode": (
                    [FUSION_INTERPOLATE, FUSION_CONCAT_WITH_BASE, FUSION_BASE_PRESERVE],
                    {"default": FUSION_INTERPOLATE},
                ),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05,
                }),
                "enabled": ("BOOLEAN", {"default": True}),
                "apply_to_uncond": ("BOOLEAN", {"default": False}),
                "uncond_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "When apply_to_uncond is enabled, scale artist injection "
                        "on uncond rows. 1.0 keeps the old full-uncond behavior."
                    ),
                }),
            },
            "optional": {
                "advanced_options": ("ANIMA_OPTS",),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING")
    RETURN_NAMES = ("model", "base_prompt")
    FUNCTION = "patch"
    CATEGORY = "Anima/CrossAttn"

    def patch(self, model, artist_pack, combine_mode, fusion_mode,
              strength, enabled, apply_to_uncond, uncond_strength=1.0,
              advanced_options=None):
        adv = advanced_options or {}
        sb = int(adv.get("start_block", 0))
        eb = int(adv.get("end_block", -1))
        start_percent = float(adv.get("start_percent", 0.0))
        end_percent = float(adv.get("end_percent", 1.0))
        normalize_w = bool(adv.get("normalize_weights", True))
        structure_preserve = max(0.0, min(1.0, float(adv.get("structure_preserve", 0.0))))
        delta_norm_cap = max(0.0, min(4.0, float(adv.get("delta_norm_cap", 0.0))))
        style_balance = max(0.0, min(1.0, float(adv.get("style_balance", 0.0))))
        artist_ema_alpha = float(adv.get("artist_ema_alpha", 0.0))
        lowrank_k = int(adv.get("lowrank_k", 1))
        artist_static_capture = bool(adv.get("artist_static_capture", False))
        static_capture_k = int(adv.get("static_capture_k", STATIC_CAPTURE_K_DEFAULT))
        static_capture_k = max(1, min(static_capture_k, STATIC_CAPTURE_K_MAX))
        artist_anchor_q = bool(adv.get("artist_anchor_q", False))
        anchor_seed_list = parse_anchor_seed_list(
            adv.get("anchor_seed_list", ""), ANCHOR_SEEDS_MAX
        )
        anchor_seed_list_is_manual = bool(
            adv.get("anchor_seed_list_is_manual", bool(anchor_seed_list))
        )
        anchor_seeds_count = int(adv.get("anchor_seeds_count", 1))
        anchor_seeds_count = max(1, min(anchor_seeds_count, ANCHOR_SEEDS_MAX))
        anchor_user_blend = max(0.0, min(1.0, float(adv.get("anchor_user_blend", 0.0))))
        anchor_deep_layer_threshold = int(
            adv.get("anchor_deep_layer_threshold", ANCHOR_LAYER_THRESHOLD_DISABLED)
        )
        stabilizer_end_percent = max(
            0.0, min(1.0, float(adv.get("stabilizer_end_percent", 1.0)))
        )
        layer_filter_text = str(adv.get("layer_filter", "") or "")
        uncond_strength = max(0.0, min(1.0, float(uncond_strength)))

        if not isinstance(artist_pack, dict):
            raise ValueError(
                "[AnimaCrossAttn] artist_pack has the wrong type; connect the "
                "output of an AnimaArtistPack node"
            )

        base_cond_out = artist_pack.get("base_conditioning")
        if base_cond_out is None:
            raise ValueError(
                "[AnimaCrossAttn] artist_pack is missing base_conditioning. "
                "Restart ComfyUI so AnimaArtistPack reloads at the current version"
            )

        conditionings = artist_pack.get("conditionings") or []
        if not enabled or not conditionings:
            return (model, base_cond_out)

        labels = artist_pack.get("labels") or []
        raws, ids_list, w_list = [], [], []
        for idx, c in enumerate(conditionings):
            raw, ids, w = extract_conditioning(c)
            if raw is None:
                label = labels[idx] if idx < len(labels) else f"#{idx}"
                raise ValueError(
                    f"[AnimaCrossAttn] artist[{label}] conditioning is empty. "
                    "Do the CLIP and model match?"
                )
            raws.append(raw)
            ids_list.append(ids)
            w_list.append(w)

        n = len(raws)
        parsed_weights = artist_pack.get("weights")
        has_explicit_weights = bool(artist_pack.get("has_explicit_weights", False))
        if isinstance(parsed_weights, (list, tuple)) and len(parsed_weights) == n:
            user_weights = [float(w) for w in parsed_weights]
        else:
            user_weights = [1.0] * n
            has_explicit_weights = False

        if artist_static_capture and artist_ema_alpha > 0.0:
            logger.info(
                "[AnimaCrossAttn] artist_static_capture=True ignores "
                "artist_ema_alpha=%.2f", artist_ema_alpha,
            )
        if artist_static_capture and fusion_mode == FUSION_CONCAT_WITH_BASE:
            logger.warning(
                "[AnimaCrossAttn] artist_static_capture=True is incompatible "
                "with fusion=concat_with_base; static capture is ignored."
            )
        if artist_anchor_q and artist_static_capture:
            logger.warning(
                "[AnimaCrossAttn] artist_anchor_q=True conflicts with "
                "artist_static_capture=True; static_capture is disabled."
            )
            artist_static_capture = False
        if artist_anchor_q and fusion_mode == FUSION_CONCAT_WITH_BASE:
            logger.warning(
                "[AnimaCrossAttn] artist_anchor_q=True is incompatible with "
                "fusion=concat_with_base; anchor_q is disabled."
            )
            artist_anchor_q = False

        if fusion_mode == FUSION_BASE_PRESERVE and float(strength) < 0.3:
            logger.info(
                "[AnimaCrossAttn] fusion=base_preserve at strength=%.2f (<0.3) "
                "is very subtle; consider strength >= 0.7.", float(strength),
            )
        if fusion_mode == FUSION_CONCAT_WITH_BASE and (
            structure_preserve > 0.0 or delta_norm_cap > 0.0
        ):
            logger.info(
                "[AnimaCrossAttn] structure_preserve/delta_norm_cap are ignored "
                "by fusion=concat_with_base."
            )
        if structure_preserve > 0.0:
            logger.info(
                "[AnimaCrossAttn] structure_preserve=%.2f is active for "
                "interpolate/base_preserve fusion.", structure_preserve,
            )
        if delta_norm_cap > 0.0:
            logger.info(
                "[AnimaCrossAttn] delta_norm_cap=%.2f limits artist delta "
                "relative to base attention output.", delta_norm_cap,
            )
        if style_balance > 0.0:
            logger.info(
                "[AnimaCrossAttn] style_balance=%.2f is active; artist outputs "
                "are volume-matched before user weights are applied.",
                style_balance,
            )

        if float(strength) > 1.0:
            logger.info(
                "[AnimaCrossAttn] strength=%.2f > 1.0 enters extrapolation: "
                "out = base + %.2f * (artist - base).",
                float(strength), float(strength),
            )
        if apply_to_uncond and uncond_strength < 1.0:
            logger.info(
                "[AnimaCrossAttn] apply_to_uncond=True with "
                "uncond_strength=%.2f; uncond rows receive partial artist injection.",
                uncond_strength,
            )
        if artist_anchor_q and anchor_seed_list_is_manual and anchor_seed_list:
            logger.info(
                "[AnimaCrossAttn] using manual anchor_seed_list=%s; "
                "anchor_seeds_count is ignored.",
                ",".join(str(seed) for seed in anchor_seed_list),
            )

        effective_weight_sum = sum(abs(w) for w in user_weights)
        if (
            not normalize_w
            and n > 1
            and combine_mode in (COMBINE_OUTPUT_AVG, COMBINE_LOWRANK_AVG)
        ):
            if effective_weight_sum >= 4.0:
                raise ValueError(
                    f"[AnimaCrossAttn] normalize_weights=False with {n} artists "
                    f"(effective weight sum {effective_weight_sum:.2f}) will "
                    f"almost always break the image. Enable normalize_weights, "
                    f"lower individual weights, or use combine=concat."
                )
            if effective_weight_sum > 1.5:
                logger.warning(
                    "[AnimaCrossAttn] normalize_weights=False and effective "
                    "weight sum %.2f may be too strong.", effective_weight_sum,
                )

        try:
            dm = model.get_model_object("diffusion_model")
        except Exception:
            dm = model.model.diffusion_model

        ok, num_blocks, _ctx_dim, msg = validate_model(dm)
        if not ok:
            raise ValueError(f"[AnimaCrossAttn] unsupported model: {msg}")
        if not hasattr(dm, "preprocess_text_embeds"):
            raise ValueError(
                "[AnimaCrossAttn] this is not an Anima model "
                "(missing preprocess_text_embeds)"
            )

        explicit_blocks = parse_layer_filter(layer_filter_text, num_blocks)
        if explicit_blocks is not None:
            if not explicit_blocks:
                raise ValueError(
                    f"[AnimaCrossAttn] layer_filter {layer_filter_text!r} "
                    f"matches no blocks (model has {num_blocks} blocks)"
                )
            target_blocks = explicit_blocks
        else:
            sb_real = max(0, sb)
            eb_real = num_blocks - 1 if eb < 0 else min(num_blocks - 1, eb)
            if sb_real > eb_real:
                raise ValueError(
                    f"[AnimaCrossAttn] start_block={sb_real} > end_block={eb_real} "
                    f"(model has {num_blocks} blocks)"
                )
            target_blocks = list(range(sb_real, eb_real + 1))

        use_sigma_range = (start_percent > 0.0) or (end_percent < 1.0)
        use_stabilizer_window = stabilizer_end_percent < 1.0
        sigma_range = None
        if use_sigma_range:
            try:
                ms = model.get_model_object("model_sampling")
                s_at_start = float(ms.percent_to_sigma(start_percent))
                s_at_end = float(ms.percent_to_sigma(end_percent))
                sigma_range = tuple(sorted([s_at_end, s_at_start]))
            except Exception as e:
                logger.warning(
                    "[AnimaCrossAttn] failed to resolve sigma range: %s. "
                    "Step-range control is disabled.", e,
                )

        stabilizer_min_sigma = None
        if use_stabilizer_window:
            try:
                ms = model.get_model_object("model_sampling")
                stabilizer_min_sigma = float(ms.percent_to_sigma(stabilizer_end_percent))
            except Exception as e:
                logger.warning(
                    "[AnimaCrossAttn] failed to resolve stabilizer_end_percent: %s. "
                    "Stabilizers stay active for the whole sampling pass.", e,
                )

        m = model.clone()
        # Resolve all model-bound references from the returned patcher.  A
        # normal clone usually shares the model object, while a later
        # deepclone_multigpu() supplies a fresh diffusion model; keeping the
        # source model's bound methods here would make worker patches call the
        # wrong attention module.
        try:
            dm = m.get_model_object("diffusion_model")
        except Exception:
            dm = m.model.diffusion_model
        state = {
            "enabled": bool(enabled),
            "fusion_mode": fusion_mode,
            "combine_mode": combine_mode,
            "strength": float(strength),
            "apply_to_uncond": bool(apply_to_uncond),
            "uncond_strength": uncond_strength,
            "raws": raws,
            "ids_list": ids_list,
            "w_list": w_list,
            "user_weights": user_weights,
            "normalize_weights": normalize_w,
            "has_explicit_weights": has_explicit_weights,
            "structure_preserve": structure_preserve,
            "delta_norm_cap": delta_norm_cap,
            "style_balance": style_balance,
            "artist_ema_alpha": artist_ema_alpha,
            "lowrank_k": lowrank_k,
            "artist_static_capture": artist_static_capture,
            "static_capture_k": static_capture_k,
            "artist_anchor_q": artist_anchor_q,
            "anchor_seed_list": anchor_seed_list,
            "anchor_seeds_count": anchor_seeds_count,
            "anchor_user_blend": anchor_user_blend,
            "anchor_deep_layer_threshold": anchor_deep_layer_threshold,
            "individuals": None,
            "real_lens": None,
            "dm_ref": dm,
            "sigma_range": sigma_range,
            "stabilizer_min_sigma": stabilizer_min_sigma,
            "current_sigma": None,
            "_disabled_layers": set(),
            "_run_last_sigma": None,
            "_disable_batched": False,
            "_warned_batched": False,
            "_warned": False,
            "_warned_svd": False,
            "_ema_cache": {},
            "_ema_last_sigma": None,
            "_static_cache": {},
            "_static_last_sigma": None,
            "_ctx_fp_memo": {},
            "_artist_chunk_cache": {},
            "_q_reuse_validation": {},
            "_warned_auto_chunk": False,
            "_warned_q_reuse": False,
            "_anchor_cache": {},
            "_anchor_cache_key": None,
            "_anchor_last_sigma": None,
            "_in_anchor_run": False,
            "_anchor_failed": False,
            "_model_owner_token": None,
            "_model_owner_ref": None,
        }

        prev = m.model_options.get("model_function_wrapper")
        m.set_model_unet_function_wrapper(make_sigma_capture(state, prev))

        existing_patches = getattr(m, "object_patches", None) or {}
        adapter_anchor_overlap = [
            i for i in target_blocks
            if getattr(
                existing_patches.get(
                    f"diffusion_model.blocks.{i}.cross_attn.forward"
                ),
                "_anima_adapter_anchor_q_forward_patch",
                False,
            )
        ]
        if adapter_anchor_overlap:
            raise ValueError(
                "[AnimaCrossAttn] the input model already uses Adapter Q-only "
                "Anchor. Do not chain the full Cross-Attn Mixer after it."
            )
        overlapped = [
            i for i in target_blocks
            if f"diffusion_model.blocks.{i}.cross_attn.forward" in existing_patches
        ]
        if overlapped:
            logger.warning(
                "[AnimaCrossAttn] another Anima Artist mixer already patches "
                "blocks %d-%d; the later node wins on overlap.",
                min(overlapped), max(overlapped),
            )

        for i in target_blocks:
            ca = dm.blocks[i].cross_attn
            inner = unwrap_cross_attn_forward(unwrap_cross_attn(ca))
            wrapper = CrossAttnWrapper(inner, state, i, original_module=ca)
            m.add_object_patch(
                f"diffusion_model.blocks.{i}.cross_attn.forward",
                make_cross_attn_forward_patch(wrapper),
            )

        register_mixer_lifecycle(m, state)

        return (m, base_cond_out)
