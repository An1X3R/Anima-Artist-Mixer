"""UI/helper node definitions kept separate from the core patcher."""

import secrets

from .constants import (
    ANCHOR_CACHE_POINTS_DEFAULT,
    ANCHOR_CACHE_POINTS_MAX,
    ANCHOR_CACHE_POINTS_MIN,
    ANCHOR_KEYFRAME_MODES,
    ANCHOR_KEYFRAME_UNIFORM_SIGMA,
    ANCHOR_LAYER_THRESHOLD_DISABLED,
    ANCHOR_REFRESH_MODES,
    ANCHOR_REFRESH_ONCE,
    ANCHOR_SEED_MAX,
    ANCHOR_SEEDS_MAX,
    MAX_ARTISTS,
    STATIC_CAPTURE_K_DEFAULT,
    STATIC_CAPTURE_K_MAX,
)
from .parsing import parse_anchor_seed_list


class AnimaArtistOptions:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start_block": ("INT", {
                    "default": 0, "min": 0, "max": 63, "step": 1,
                    "tooltip": "Start block (inclusive).",
                }),
                "end_block": ("INT", {
                    "default": -1, "min": -1, "max": 63, "step": 1,
                    "tooltip": "End block (inclusive). -1 means last block.",
                }),
                "start_percent": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Sampling progress start. 0.0 = start.",
                }),
                "end_percent": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Sampling progress end. 1.0 = end.",
                }),
                "normalize_weights": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "True: normalize artist weights to relative ratios. "
                        "False: weights act as direct multipliers. Explicit "
                        "::weight syntax follows this option."
                    ),
                }),
                "artist_ema_alpha": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 0.95, "step": 0.05,
                    "tooltip": "Cross-step EMA smoothing for artist outputs.",
                }),
                "lowrank_k": ("INT", {
                    "default": 1, "min": 1, "max": MAX_ARTISTS, "step": 1,
                    "tooltip": "Rank for lowrank_avg.",
                }),
                "artist_static_capture": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Average the first K steps and freeze artist outputs.",
                }),
                "static_capture_k": ("INT", {
                    "default": STATIC_CAPTURE_K_DEFAULT,
                    "min": 1, "max": STATIC_CAPTURE_K_MAX, "step": 1,
                    "tooltip": "Number of warmup steps for static capture.",
                }),
                "artist_anchor_q": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use selected anchor-seed hidden states as artist-attn Q.",
                }),
                "anchor_seed_list": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": (
                        "Optional fixed anchor seeds, e.g. 12345,67890. "
                        "When filled, anchor_seeds_count is ignored."
                    ),
                }),
                "anchor_seeds_count": ("INT", {
                    "default": 1, "min": 1, "max": ANCHOR_SEEDS_MAX, "step": 1,
                    "tooltip": "Number of fresh random anchor seeds generated per execution.",
                }),
                "anchor_user_blend": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blend user x into anchor Q. 0=pure anchor, 1=pure user.",
                }),
                "anchor_deep_layer_threshold": ("INT", {
                    "default": ANCHOR_LAYER_THRESHOLD_DISABLED,
                    "min": ANCHOR_LAYER_THRESHOLD_DISABLED, "max": 64, "step": 1,
                    "tooltip": "-1 = all layers use anchor. N = layers >= N use user x.",
                }),
                "stabilizer_end_percent": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "When EMA/static/anchor stabilizers stop during sampling.",
                }),
                "anchor_refresh_mode": (list(ANCHOR_REFRESH_MODES), {
                    "default": ANCHOR_REFRESH_ONCE,
                    "tooltip": (
                        "[Adapter Mixer only] once reuses one start-of-run Q "
                        "snapshot. warm_cache "
                        "warms a sigma-keyframed Q trajectory once and reuses "
                        "it for later KSampler seeds in the same session."
                    ),
                }),
                "anchor_cache_points": ("INT", {
                    "default": ANCHOR_CACHE_POINTS_DEFAULT,
                    "min": ANCHOR_CACHE_POINTS_MIN,
                    "max": ANCHOR_CACHE_POINTS_MAX,
                    "step": 1,
                    "tooltip": (
                        "[Adapter Mixer only] CPU keyframes retained by "
                        "warm_cache. More points improve sigma matching but "
                        "use more system RAM."
                    ),
                }),
            },
            "optional": {
                "layer_filter": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Optional comma-separated block list/ranges, e.g. 0,3,5-10,-1.",
                }),
                "anchor_keyframe_mode": (list(ANCHOR_KEYFRAME_MODES), {
                    "default": ANCHOR_KEYFRAME_UNIFORM_SIGMA,
                    "tooltip": (
                        "[Adapter Mixer only] warm_cache keyframe selection. "
                        "uniform_sigma keeps evenly spaced sigma frames. "
                        "adaptive_q observes every warmup sigma and retains "
                        "frames with the largest sampled Q-trajectory interpolation "
                        "error; first warmup uses more CPU transfer."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("ANIMA_OPTS", "STRING")
    RETURN_NAMES = ("advanced_options", "anchor_seeds_used")
    FUNCTION = "build"
    CATEGORY = "Anima/CrossAttn"

    @classmethod
    def IS_CHANGED(cls, anchor_seed_list="", artist_anchor_q=False,
                   anchor_refresh_mode=ANCHOR_REFRESH_ONCE,
                   anchor_cache_points=ANCHOR_CACHE_POINTS_DEFAULT,
                   anchor_keyframe_mode=ANCHOR_KEYFRAME_UNIFORM_SIGMA, **kwargs):
        if artist_anchor_q and not parse_anchor_seed_list(
            anchor_seed_list, ANCHOR_SEEDS_MAX
        ):
            return float("NaN")
        return "|".join((
            str(anchor_seed_list or ""),
            str(anchor_refresh_mode),
            str(int(anchor_cache_points)),
            str(anchor_keyframe_mode),
            str(bool(artist_anchor_q)),
        ))

    def build(self, start_block, end_block, start_percent, end_percent, normalize_weights,
              artist_ema_alpha=0.0, lowrank_k=1, artist_static_capture=False,
              static_capture_k=STATIC_CAPTURE_K_DEFAULT, artist_anchor_q=False,
              anchor_seed_list="", anchor_seeds_count=1, anchor_user_blend=0.0,
              anchor_deep_layer_threshold=ANCHOR_LAYER_THRESHOLD_DISABLED,
              stabilizer_end_percent=1.0,
              anchor_refresh_mode=ANCHOR_REFRESH_ONCE,
              anchor_cache_points=ANCHOR_CACHE_POINTS_DEFAULT,
              layer_filter="",
              anchor_keyframe_mode=ANCHOR_KEYFRAME_UNIFORM_SIGMA):
        manual_seeds = parse_anchor_seed_list(anchor_seed_list, ANCHOR_SEEDS_MAX)
        if manual_seeds:
            seeds_used = manual_seeds
        else:
            seeds_count = max(1, min(int(anchor_seeds_count), ANCHOR_SEEDS_MAX))
            seeds_used = []
            seen = set()
            while len(seeds_used) < seeds_count:
                seed = secrets.randbelow(ANCHOR_SEED_MAX + 1)
                if seed not in seen:
                    seen.add(seed)
                    seeds_used.append(seed)

        resolved_seed_list = ",".join(str(seed) for seed in seeds_used)

        return ({
            "start_block": int(start_block),
            "end_block": int(end_block),
            "start_percent": float(start_percent),
            "end_percent": float(end_percent),
            "normalize_weights": bool(normalize_weights),
            "artist_ema_alpha": float(artist_ema_alpha),
            "lowrank_k": int(lowrank_k),
            "artist_static_capture": bool(artist_static_capture),
            "static_capture_k": int(static_capture_k),
            "artist_anchor_q": bool(artist_anchor_q),
            "anchor_seed_list": resolved_seed_list,
            "anchor_seed_list_is_manual": bool(manual_seeds),
            "anchor_seeds_count": int(anchor_seeds_count),
            "anchor_user_blend": float(anchor_user_blend),
            "anchor_deep_layer_threshold": int(anchor_deep_layer_threshold),
            "stabilizer_end_percent": float(stabilizer_end_percent),
            "anchor_refresh_mode": str(anchor_refresh_mode),
            "anchor_cache_points": max(
                ANCHOR_CACHE_POINTS_MIN,
                min(int(anchor_cache_points), ANCHOR_CACHE_POINTS_MAX),
            ),
            "anchor_keyframe_mode": (
                str(anchor_keyframe_mode)
                if str(anchor_keyframe_mode) in ANCHOR_KEYFRAME_MODES
                else ANCHOR_KEYFRAME_UNIFORM_SIGMA
            ),
            "layer_filter": str(layer_filter or ""),
        }, resolved_seed_list)


class AnimaArtistStructureOptions:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "structure_preserve": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "Keeps artist changes closer to the base prompt structure. "
                        "0.0 = old behavior, 1.0 = strongest directional lock."
                    ),
                }),
                "delta_norm_cap": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 4.0, "step": 0.05,
                    "tooltip": (
                        "Caps artist change magnitude relative to base attention output. "
                        "0.0 disables the cap; try 1.0-1.5 for object stability."
                    ),
                }),
            },
            "optional": {
                "advanced_options": ("ANIMA_OPTS",),
            },
        }

    RETURN_TYPES = ("ANIMA_OPTS",)
    RETURN_NAMES = ("advanced_options",)
    FUNCTION = "build"
    CATEGORY = "Anima/CrossAttn"

    def build(self, structure_preserve=0.0, delta_norm_cap=0.0, advanced_options=None):
        opts = dict(advanced_options or {})
        opts["structure_preserve"] = max(0.0, min(1.0, float(structure_preserve)))
        opts["delta_norm_cap"] = max(0.0, min(4.0, float(delta_norm_cap)))
        return (opts,)


class AnimaArtistStyleBalance:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "style_balance": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "Reduces seed-to-seed artist dominance drift by matching "
                        "artist output volume before user weights are applied."
                    ),
                }),
            },
            "optional": {
                "advanced_options": ("ANIMA_OPTS",),
            },
        }

    RETURN_TYPES = ("ANIMA_OPTS",)
    RETURN_NAMES = ("advanced_options",)
    FUNCTION = "build"
    CATEGORY = "Anima/CrossAttn"

    def build(self, style_balance=0.0, advanced_options=None):
        opts = dict(advanced_options or {})
        opts["style_balance"] = max(0.0, min(1.0, float(style_balance)))
        return (opts,)
