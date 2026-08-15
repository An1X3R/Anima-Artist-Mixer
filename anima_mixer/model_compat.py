"""ComfyUI loader compatibility for expanded Anima checkpoints.

The regular Anima detector in ComfyUI historically reports a 28-block DiT.
Anima-2.9B keeps the same block schema but expands the checkpoint to 40 blocks.
The block count has to be corrected while ComfyUI is detecting the checkpoint,
before the diffusion model is instantiated; the runtime mixer cannot repair a
model that failed to load.
"""

from dataclasses import dataclass
from functools import wraps
import logging


logger = logging.getLogger(__name__)

_PATCH_MARKER = "_anima_artist_mixer_dynamic_block_patch"

ANIMA_2B_BLOCK_COUNT = 28
ANIMA_29B_BLOCK_COUNT = 40
ANIMA_29B_BLOCK_MODE_AUTO = "auto"
ANIMA_29B_BLOCK_MODE_LEGACY_28 = "legacy_28"
ANIMA_29B_BLOCK_MODE_NATIVE_40 = "native_40"
ANIMA_29B_BLOCK_MODES = (
    ANIMA_29B_BLOCK_MODE_AUTO,
    ANIMA_29B_BLOCK_MODE_LEGACY_28,
    ANIMA_29B_BLOCK_MODE_NATIVE_40,
)

# Anima-2.9B retains the original 28 Anima-2B blocks and inserts 12 new
# blocks between them. Auto mode uses the original 2B-aligned positions so the
# runtime mixer keeps the same injection density on the expanded checkpoint.
# Native mode remains available as an explicit opt-in for all 40 blocks.
ANIMA_2B_TO_29B_BLOCKS = (
    0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19,
    20, 22, 23, 25, 26, 28, 29, 31, 32, 34, 35, 37, 38, 39,
)
ANIMA_29B_INSERTED_BLOCKS = (
    2, 5, 8, 11, 14, 17, 21, 24, 27, 30, 33, 36,
)


@dataclass(frozen=True)
class AnimaBlockLayout:
    """Logical selector space and physical block indices for one model."""

    model_block_count: int
    requested_mode: str
    resolved_mode: str
    physical_blocks: tuple

    @property
    def selector_block_count(self):
        return len(self.physical_blocks)

    @property
    def logical_index_by_physical(self):
        return {
            physical_index: logical_index
            for logical_index, physical_index in enumerate(self.physical_blocks)
        }

    @property
    def skipped_physical_blocks(self):
        selected = set(self.physical_blocks)
        return tuple(
            index for index in range(self.model_block_count)
            if index not in selected
        )

    def map_selector_blocks(self, selector_blocks):
        mapped = []
        for selector_index in selector_blocks:
            index = int(selector_index)
            if 0 <= index < self.selector_block_count:
                mapped.append(self.physical_blocks[index])
        return tuple(mapped)


def resolve_anima_block_layout(num_blocks, mode=ANIMA_29B_BLOCK_MODE_AUTO):
    """Resolve the user-facing block selector to physical model blocks.

    On a 40-block Anima-2.9B model, ``auto`` resolves to ``legacy_28`` and maps
    selectors to the original 2B block positions. ``native_40`` is an opt-in
    mode that exposes every physical block. Other Anima block counts remain
    identity-mapped.
    """

    block_count = max(0, int(num_blocks))
    requested_mode = str(mode or ANIMA_29B_BLOCK_MODE_AUTO)
    if requested_mode not in ANIMA_29B_BLOCK_MODES:
        requested_mode = ANIMA_29B_BLOCK_MODE_AUTO

    if block_count != ANIMA_29B_BLOCK_COUNT:
        return AnimaBlockLayout(
            model_block_count=block_count,
            requested_mode=requested_mode,
            resolved_mode="native",
            physical_blocks=tuple(range(block_count)),
        )

    resolved_mode = requested_mode
    if requested_mode == ANIMA_29B_BLOCK_MODE_AUTO:
        resolved_mode = ANIMA_29B_BLOCK_MODE_LEGACY_28

    physical_blocks = (
        tuple(range(block_count))
        if resolved_mode == ANIMA_29B_BLOCK_MODE_NATIVE_40
        else ANIMA_2B_TO_29B_BLOCKS
    )
    return AnimaBlockLayout(
        model_block_count=block_count,
        requested_mode=requested_mode,
        resolved_mode=resolved_mode,
        physical_blocks=physical_blocks,
    )


def _scan_block_count(state_dict, key_prefix=""):
    """Return the number of serialized Anima blocks, or ``None``.

    ComfyUI passes a prefix for the diffusion model.  The checkpoint format
    stores blocks as ``<prefix>blocks.<index>.*``; looking at the key names
    keeps this check cheap and avoids loading any tensor data.
    """

    if state_dict is None or not hasattr(state_dict, "keys"):
        return None

    prefix = f"{key_prefix or ''}blocks."
    max_block = -1
    try:
        keys = state_dict.keys()
        for key in keys:
            if not isinstance(key, str) or not key.startswith(prefix):
                continue
            remainder = key[len(prefix):]
            block_index, separator, _ = remainder.partition(".")
            if not separator:
                continue
            try:
                max_block = max(max_block, int(block_index))
            except (TypeError, ValueError):
                continue
    except Exception as exc:
        # Detection should remain usable with unusual mapping-like state dicts.
        logger.warning(
            "[AnimaArtistMixer] could not inspect Anima block keys: %s", exc
        )
        return None

    return max_block + 1 if max_block >= 0 else None


def _patch_detected_config(config, state_dict, key_prefix):
    if not isinstance(config, dict) or config.get("image_model") != "anima":
        return config

    actual_blocks = _scan_block_count(state_dict, key_prefix)
    if actual_blocks is None:
        return config

    configured_blocks = config.get("num_blocks")
    if configured_blocks == actual_blocks:
        return config

    # Do not mutate a detector-owned dictionary in place.  A few ComfyUI
    # versions reuse detector results while probing loaders.
    patched = dict(config)
    patched["num_blocks"] = actual_blocks
    logger.info(
        "[AnimaArtistMixer] detected Anima checkpoint with %d blocks; "
        "using that count instead of %s.",
        actual_blocks,
        configured_blocks if configured_blocks is not None else "<unset>",
    )
    return patched


def install_anima_loader_patch(model_detection=None):
    """Install the dynamic Anima block-count detector patch.

    Returns ``True`` when this call wraps ``detect_unet_config`` and ``False``
    when ComfyUI is unavailable or the patch is already installed.  The
    optional module argument makes the behavior testable without importing
    ComfyUI and also lets downstream integrations pass their module object.
    """

    if model_detection is None:
        try:
            import comfy.model_detection as model_detection
        except (ImportError, ModuleNotFoundError):
            return False

    original = getattr(model_detection, "detect_unet_config", None)
    if not callable(original) or getattr(original, _PATCH_MARKER, False):
        return False

    @wraps(original)
    def detect_unet_config_with_dynamic_anima_blocks(*args, **kwargs):
        result = original(*args, **kwargs)

        state_dict = args[0] if args else kwargs.get("state_dict")
        key_prefix = (
            args[1]
            if len(args) > 1
            else kwargs.get("key_prefix", "")
        )
        try:
            return _patch_detected_config(result, state_dict, key_prefix)
        except Exception as exc:
            # Never make an otherwise valid model unloadable because of this
            # optional compatibility layer.
            logger.warning(
                "[AnimaArtistMixer] Anima loader compatibility check failed: %s",
                exc,
            )
            return result

    setattr(detect_unet_config_with_dynamic_anima_blocks, _PATCH_MARKER, True)
    setattr(
        detect_unet_config_with_dynamic_anima_blocks,
        "_anima_artist_mixer_original",
        original,
    )
    model_detection.detect_unet_config = detect_unet_config_with_dynamic_anima_blocks
    logger.info("[AnimaArtistMixer] dynamic Anima block detection enabled")
    return True
