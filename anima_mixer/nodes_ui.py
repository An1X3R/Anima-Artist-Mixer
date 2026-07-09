"""UI/helper node definitions kept separate from the core patcher."""

from .constants import (
    ANCHOR_LAYER_THRESHOLD_DISABLED,
    ANCHOR_SEEDS_MAX,
    MAX_ARTISTS,
    STATIC_CAPTURE_K_DEFAULT,
    STATIC_CAPTURE_K_MAX,
)


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
                    "tooltip": "Use fixed-seed anchor hidden states as artist-attn Q.",
                }),
                "anchor_seeds_count": ("INT", {
                    "default": 1, "min": 1, "max": ANCHOR_SEEDS_MAX, "step": 1,
                    "tooltip": "Number of fixed anchor seeds to average.",
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
            },
            "optional": {
                "layer_filter": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Optional comma-separated block list/ranges, e.g. 0,3,5-10,-1.",
                }),
            },
        }

    RETURN_TYPES = ("ANIMA_OPTS",)
    RETURN_NAMES = ("advanced_options",)
    FUNCTION = "build"
    CATEGORY = "Anima/CrossAttn"

    def build(self, start_block, end_block, start_percent, end_percent, normalize_weights,
              artist_ema_alpha=0.0, lowrank_k=1, artist_static_capture=False,
              static_capture_k=STATIC_CAPTURE_K_DEFAULT, artist_anchor_q=False,
              anchor_seeds_count=1, anchor_user_blend=0.0,
              anchor_deep_layer_threshold=ANCHOR_LAYER_THRESHOLD_DISABLED,
              stabilizer_end_percent=1.0,
              layer_filter=""):
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
            "anchor_seeds_count": int(anchor_seeds_count),
            "anchor_user_blend": float(anchor_user_blend),
            "anchor_deep_layer_threshold": int(anchor_deep_layer_threshold),
            "stabilizer_end_percent": float(stabilizer_end_percent),
            "layer_filter": str(layer_filter or ""),
        },)


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
