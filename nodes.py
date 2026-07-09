"""Backward-compatibility shim for the split Anima Artist Mixer package."""

try:
    from .anima_mixer import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    from .anima_mixer.nodes_core import AnimaArtistCrossAttn, AnimaArtistPack
    from .anima_mixer.nodes_ui import (
        AnimaArtistOptions,
        AnimaArtistStructureOptions,
        AnimaArtistStyleBalance,
    )
except ImportError:
    from anima_mixer import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    from anima_mixer.nodes_core import AnimaArtistCrossAttn, AnimaArtistPack
    from anima_mixer.nodes_ui import (
        AnimaArtistOptions,
        AnimaArtistStructureOptions,
        AnimaArtistStyleBalance,
    )

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "AnimaArtistPack",
    "AnimaArtistCrossAttn",
    "AnimaArtistOptions",
    "AnimaArtistStructureOptions",
    "AnimaArtistStyleBalance",
]
