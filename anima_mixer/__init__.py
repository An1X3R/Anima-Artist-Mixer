"""Anima Artist Mixer package."""

from .nodes_embedding import AnimaArtistAdapterMixer
from .nodes_core import AnimaArtistCrossAttn, AnimaArtistPack
from .nodes_ui import (
    AnimaArtistOptions,
    AnimaArtistStructureOptions,
    AnimaArtistStyleBalance,
)

NODE_CLASS_MAPPINGS = {
    "AnimaArtistPack": AnimaArtistPack,
    "AnimaArtistCrossAttn": AnimaArtistCrossAttn,
    "AnimaArtistAdapterMixer": AnimaArtistAdapterMixer,
    "AnimaArtistOptions": AnimaArtistOptions,
    "AnimaArtistStructureOptions": AnimaArtistStructureOptions,
    "AnimaArtistStyleBalance": AnimaArtistStyleBalance,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaArtistPack": "Anima Artist Pack (Split + Encode)",
    "AnimaArtistCrossAttn": "Anima Artist Cross-Attn (v26 fixed)",
    "AnimaArtistAdapterMixer": "Anima Artist Adapter Mixer (Experimental)",
    "AnimaArtistOptions": "Anima Artist Options (Advanced)",
    "AnimaArtistStructureOptions": "Anima Artist Structure Guard",
    "AnimaArtistStyleBalance": "Anima Artist Style Balance",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
