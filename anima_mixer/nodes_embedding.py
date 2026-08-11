"""ComfyUI node for model-boundary post-adapter artist mixing."""

import logging
import secrets

from .adapter_anchor import make_adapter_anchor_q_forward_patch
from .alignment import build_base_anchored_plan
from .anchor import make_sigma_capture
from .constants import (
    ALIGN_BASE_ANCHORED,
    ALIGN_SHARED_BASE_IDS,
    ANCHOR_CACHE_POINTS_DEFAULT,
    ANCHOR_CACHE_POINTS_MAX,
    ANCHOR_CACHE_POINTS_MIN,
    ANCHOR_SEED_MAX,
    ANCHOR_KEYFRAME_MODES,
    ANCHOR_KEYFRAME_UNIFORM_SIGMA,
    ANCHOR_LAYER_THRESHOLD_DISABLED,
    ANCHOR_REFRESH_MODES,
    ANCHOR_REFRESH_ONCE,
    ANCHOR_SEEDS_MAX,
    WEIGHT_MAX,
    WEIGHT_MIN,
)
from .embedding import (
    make_adapter_embedding_wrapper,
    unwrap_adapter_embedding_wrapper,
)
from .parsing import parse_anchor_seed_list
from .patching import (
    extract_conditioning,
    register_mixer_lifecycle,
    select_active_adapter_mixer,
    tensor_cache_signature,
    unwrap_cross_attn,
    unwrap_cross_attn_forward,
    validate_model,
)

logger = logging.getLogger(__name__)


def _build_cache_namespace(
    artist_pack,
    labels,
    raws,
    ids_list,
    t5_weights_list,
    base_ids,
    base_t5_weights,
    user_weights,
    normalize_weights,
    alignment_mode,
):
    """Identify the prompt/artist inputs represented by one Mixer state."""
    return (
        "anima_adapter_mixer_v2",
        str(alignment_mode),
        str(artist_pack.get("base_prompt", "")),
        tuple(str(label) for label in labels),
        tuple(round(float(weight), 7) for weight in user_weights),
        bool(normalize_weights),
        tuple(tensor_cache_signature(value) for value in raws),
        tuple(tensor_cache_signature(value) for value in ids_list),
        tuple(tensor_cache_signature(value) for value in t5_weights_list),
        tensor_cache_signature(base_ids),
        tensor_cache_signature(base_t5_weights),
    )


class AnimaArtistAdapterMixer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "artist_pack": ("ANIMA_PACK",),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05,
                    "tooltip": (
                        "Strength of the perpendicular artist delta in "
                        "post-adapter embedding space."
                    ),
                }),
                "normalize_weights": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Normalize ::weights to relative artist ratios.",
                }),
                "alignment_mode": (
                    [ALIGN_BASE_ANCHORED, ALIGN_SHARED_BASE_IDS],
                    {
                        "default": ALIGN_BASE_ANCHORED,
                        "tooltip": (
                            "base_anchored keeps every artist's own Adapter input and "
                            "aligns post-Adapter rows by T5 token IDs. "
                            "shared_base_ids is the older common-target-grid mode."
                        ),
                    },
                ),
                "enabled": ("BOOLEAN", {"default": True}),
                "apply_to_uncond": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Experimental uncond mixing for shared_base_ids. "
                        "base_anchored always preserves uncond because negative "
                        "T5 IDs are unavailable for token alignment."
                    ),
                }),
                "uncond_strength": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                }),
            },
            "optional": {
                "advanced_options": ("ANIMA_OPTS",),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING")
    RETURN_NAMES = ("model", "base_prompt")
    FUNCTION = "patch"
    CATEGORY = "Anima/Adapter"

    def patch(self, model, artist_pack, strength, normalize_weights,
              alignment_mode, enabled, apply_to_uncond, uncond_strength=0.0,
              advanced_options=None):
        if not isinstance(artist_pack, dict):
            raise ValueError(
                "[AnimaAdapterMixer] artist_pack has the wrong type; connect "
                "Anima Artist Pack (Split + Encode)."
            )

        base_conditioning = artist_pack.get("base_conditioning")
        if base_conditioning is None:
            raise ValueError(
                "[AnimaAdapterMixer] artist_pack is missing base_conditioning."
            )

        conditionings = artist_pack.get("conditionings") or []
        strength = max(0.0, min(4.0, float(strength)))
        if not enabled or not conditionings or strength <= 0.0:
            return (model, base_conditioning)

        if alignment_mode not in (ALIGN_BASE_ANCHORED, ALIGN_SHARED_BASE_IDS):
            raise ValueError(
                f"[AnimaAdapterMixer] unsupported alignment_mode={alignment_mode!r}"
            )

        if advanced_options is not None and not isinstance(advanced_options, dict):
            raise ValueError(
                "[AnimaAdapterMixer] advanced_options must come from "
                "Anima Artist Options (Advanced)."
            )
        advanced = advanced_options or {}
        artist_anchor_q = bool(advanced.get("artist_anchor_q", False))
        anchor_seed_list = parse_anchor_seed_list(
            advanced.get("anchor_seed_list", ""),
            ANCHOR_SEEDS_MAX,
        )
        anchor_seeds_count = max(
            1,
            min(int(advanced.get("anchor_seeds_count", 1)), ANCHOR_SEEDS_MAX),
        )
        if artist_anchor_q and not anchor_seed_list:
            generated = []
            while len(generated) < anchor_seeds_count:
                seed = secrets.randbelow(ANCHOR_SEED_MAX + 1)
                if seed not in generated:
                    generated.append(seed)
            anchor_seed_list = generated
        anchor_user_blend = max(
            0.0,
            min(1.0, float(advanced.get("anchor_user_blend", 0.0))),
        )
        anchor_deep_layer_threshold = int(advanced.get(
            "anchor_deep_layer_threshold",
            ANCHOR_LAYER_THRESHOLD_DISABLED,
        ))
        stabilizer_end_percent = max(
            0.0,
            min(1.0, float(advanced.get("stabilizer_end_percent", 1.0))),
        )
        anchor_refresh_mode = str(advanced.get(
            "anchor_refresh_mode",
            ANCHOR_REFRESH_ONCE,
        ))
        if anchor_refresh_mode not in ANCHOR_REFRESH_MODES:
            raise ValueError(
                "[AnimaAdapterMixer] unsupported "
                f"anchor_refresh_mode={anchor_refresh_mode!r}"
            )
        anchor_cache_points = max(
            ANCHOR_CACHE_POINTS_MIN,
            min(
                int(advanced.get(
                    "anchor_cache_points",
                    ANCHOR_CACHE_POINTS_DEFAULT,
                )),
                ANCHOR_CACHE_POINTS_MAX,
            ),
        )
        anchor_keyframe_mode = str(advanced.get(
            "anchor_keyframe_mode",
            ANCHOR_KEYFRAME_UNIFORM_SIGMA,
        ))
        if anchor_keyframe_mode not in ANCHOR_KEYFRAME_MODES:
            raise ValueError(
                "[AnimaAdapterMixer] unsupported "
                f"anchor_keyframe_mode={anchor_keyframe_mode!r}"
            )

        try:
            dm = model.get_model_object("diffusion_model")
        except Exception:
            dm = model.model.diffusion_model
        if not callable(getattr(dm, "preprocess_text_embeds", None)):
            raise ValueError(
                "[AnimaAdapterMixer] the diffusion model has no "
                "preprocess_text_embeds Adapter interface."
            )

        anchor_block_count = 0
        stabilizer_min_sigma = None
        if artist_anchor_q:
            valid, anchor_block_count, _context_dim, message = validate_model(dm)
            if not valid:
                raise ValueError(
                    f"[AnimaAdapterMixer] Q-only Anchor is unsupported: {message}"
                )
            if stabilizer_end_percent < 1.0:
                try:
                    model_sampling = model.get_model_object("model_sampling")
                    stabilizer_min_sigma = float(
                        model_sampling.percent_to_sigma(stabilizer_end_percent)
                    )
                except Exception as error:
                    logger.warning(
                        "[AnimaAdapterAnchorQ] failed to resolve "
                        "stabilizer_end_percent: %s. Anchor-Q stays active for "
                        "the whole sampling pass.",
                        error,
                    )

        _base_raw, base_ids, base_t5_weights = extract_conditioning(base_conditioning)
        if base_ids is None:
            raise ValueError(
                "[AnimaAdapterMixer] token alignment requires Anima "
                "t5xxl_ids in the base conditioning."
            )

        apply_to_uncond = bool(apply_to_uncond)
        if alignment_mode == ALIGN_BASE_ANCHORED and apply_to_uncond:
            logger.warning(
                "[AnimaAdapterMixer] base_anchored cannot token-align the "
                "negative context because its T5 IDs are unavailable here; "
                "apply_to_uncond is ignored and uncond rows stay unchanged."
            )
            apply_to_uncond = False

        labels = artist_pack.get("labels") or []
        raws, ids_list, t5_weights_list = [], [], []
        for index, conditioning in enumerate(conditionings):
            raw, ids, t5_weights = extract_conditioning(conditioning)
            if raw is None:
                label = labels[index] if index < len(labels) else f"#{index}"
                raise ValueError(
                    f"[AnimaAdapterMixer] artist {label!r} conditioning is empty."
                )
            raws.append(raw)
            ids_list.append(ids)
            t5_weights_list.append(t5_weights)

        alignment_plan = None
        if alignment_mode == ALIGN_BASE_ANCHORED:
            missing_ids = [
                labels[index] if index < len(labels) else f"#{index}"
                for index, ids in enumerate(ids_list)
                if ids is None
            ]
            if missing_ids:
                raise ValueError(
                    "[AnimaAdapterMixer] base_anchored requires t5xxl_ids for "
                    f"every artist; missing={missing_ids}"
                )
            alignment_plan = build_base_anchored_plan(base_ids, ids_list)

        parsed_weights = artist_pack.get("weights")
        if isinstance(parsed_weights, (list, tuple)) and len(parsed_weights) == len(raws):
            user_weights = [
                max(WEIGHT_MIN, min(WEIGHT_MAX, float(weight)))
                for weight in parsed_weights
            ]
        else:
            user_weights = [1.0] * len(raws)

        m = model.clone()
        # Use the clone-local Adapter model for state and Anchor patch
        # construction.  This matters for ComfyUI's fresh-model
        # deepclone_multigpu path, where the source ``dm`` is on another GPU.
        try:
            dm = m.get_model_object("diffusion_model")
        except Exception:
            dm = m.model.diffusion_model
        existing_patches = getattr(m, "object_patches", None) or {}
        for path, patch in list(existing_patches.items()):
            if (
                str(path).endswith(".cross_attn.forward")
                and getattr(patch, "_anima_adapter_anchor_q_forward_patch", False)
            ):
                existing_patches.pop(path, None)

        cache_namespace = _build_cache_namespace(
            artist_pack,
            labels,
            raws,
            ids_list,
            t5_weights_list,
            base_ids,
            base_t5_weights,
            user_weights,
            normalize_weights,
            alignment_mode,
        )
        state = {
            "enabled": True,
            "dm_ref": dm,
            "labels": labels,
            "raws": raws,
            "ids_list": ids_list,
            "t5_weights_list": t5_weights_list,
            "base_ids": base_ids,
            "base_t5_weights": base_t5_weights,
            "user_weights": user_weights,
            "normalize_weights": bool(normalize_weights),
            "alignment_mode": alignment_mode,
            "alignment_plan": alignment_plan,
            "_cache_namespace": cache_namespace,
            "strength": strength,
            "apply_to_uncond": apply_to_uncond,
            "uncond_strength": max(0.0, min(1.0, float(uncond_strength))),
            "artist_anchor_q": artist_anchor_q,
            "anchor_seed_list": anchor_seed_list,
            "anchor_seeds_count": anchor_seeds_count,
            "anchor_user_blend": anchor_user_blend,
            "anchor_deep_layer_threshold": anchor_deep_layer_threshold,
            "anchor_refresh_mode": anchor_refresh_mode,
            "anchor_cache_points": anchor_cache_points,
            "anchor_keyframe_mode": anchor_keyframe_mode,
            "stabilizer_min_sigma": stabilizer_min_sigma,
            "current_sigma": None,
            "anchor_log_name": "AnimaAdapterAnchorQ",
            "_artist_embedding_cache": {},
            "_mixed_context_cache": None,
            "_identity_context_signature": True,
            "_embedding_mixer_failed": False,
            "_warned_embedding_failure": False,
            "_warned_no_context": False,
            "_warned": False,
            "_run_last_sigma": None,
            "_anchor_cache": {},
            "_anchor_cache_key": None,
            "_anchor_trajectory": None,
            "_anchor_last_sigma": None,
            "_in_anchor_run": False,
            "_anchor_failed": False,
            "_adapter_anchor_failed": False,
            "_warned_adapter_anchor_failure": False,
            "_warned_trajectory_invalidated": False,
            "_warned_trajectory_reuse": False,
            "_model_owner_token": None,
            "_model_owner_ref": None,
            "_execution_index": 0,
            "_adapter_mixer_instance_token": secrets.token_hex(16),
            "_adapter_mixer_selected_for_run": None,
        }

        existing_cross_attn = [
            str(path)
            for path in existing_patches
            if str(path).endswith(".cross_attn.forward")
        ]
        if artist_anchor_q and existing_cross_attn:
            raise ValueError(
                "[AnimaAdapterMixer] Q-only Anchor requires an unpatched model. "
                "Do not chain the full Cross-Attn Mixer with Adapter Mixer."
            )
        if existing_cross_attn:
            logger.warning(
                "[AnimaAdapterMixer] the input model already has cross-attention "
                "patches. Do not chain Adapter Mixer and Cross-Attn Mixer when "
                "performing an A/B comparison."
            )

        prev_wrapper = unwrap_adapter_embedding_wrapper(
            m.model_options.get("model_function_wrapper")
        )
        if artist_anchor_q:
            # Adapter mixing runs first so the anchor pre-run sees exactly the
            # same post-Adapter mixed context as the user denoising pass.
            sigma_wrapper = make_sigma_capture(state, prev_wrapper)
            final_wrapper = make_adapter_embedding_wrapper(state, sigma_wrapper)
        else:
            final_wrapper = make_adapter_embedding_wrapper(state, prev_wrapper)
        m.set_model_unet_function_wrapper(final_wrapper)
        select_active_adapter_mixer(m, state)

        if artist_anchor_q:
            for layer_index in range(anchor_block_count):
                cross_attn = dm.blocks[layer_index].cross_attn
                original_forward = unwrap_cross_attn_forward(
                    unwrap_cross_attn(cross_attn)
                )
                m.add_object_patch(
                    f"diffusion_model.blocks.{layer_index}.cross_attn.forward",
                    make_adapter_anchor_q_forward_patch(
                        original_forward,
                        state,
                        layer_index,
                    ),
                )
            logger.info(
                "[AnimaAdapterAnchorQ] Q-only Anchor is active on %d layers; "
                "seeds=%s, user blend=%.2f, deep threshold=%d, "
                "refresh=%s, cache points=%d, keyframes=%s.",
                anchor_block_count,
                ",".join(str(seed) for seed in anchor_seed_list),
                anchor_user_blend,
                anchor_deep_layer_threshold,
                anchor_refresh_mode,
                anchor_cache_points,
                anchor_keyframe_mode,
            )

        register_mixer_lifecycle(m, state)

        if alignment_mode == ALIGN_BASE_ANCHORED:
            imperfect = [
                (labels[index] if index < len(labels) else f"#{index}", matched)
                for index, (method, matched) in enumerate(zip(
                    alignment_plan["methods"], alignment_plan["matched_counts"],
                ))
                if method != "exact" or matched != alignment_plan["base_token_count"]
            ]
            logger.info(
                "[AnimaAdapterMixer] base_anchored alignment: %d base tokens, "
                "%d common rows.",
                alignment_plan["base_token_count"],
                alignment_plan["length"],
            )
            if imperfect:
                logger.warning(
                    "[AnimaAdapterMixer] tokenizer boundary required LCS fallback "
                    "for %s; all unmatched token rows are preserved in gaps.",
                    imperfect,
                )
        else:
            logger.info(
                "[AnimaAdapterMixer] shared_base_ids is active: artist source "
                "embeddings use the base T5 target grid."
            )
        return (m, base_conditioning)
