"""ComfyUI loader compatibility for expanded Anima checkpoints.

The regular Anima detector in ComfyUI historically reports a 28-block DiT.
Anima-2.9B keeps the same block schema but expands the checkpoint to 40 blocks.
The block count has to be corrected while ComfyUI is detecting the checkpoint,
before the diffusion model is instantiated; the runtime mixer cannot repair a
model that failed to load.
"""

from functools import wraps
import logging


logger = logging.getLogger(__name__)

_PATCH_MARKER = "_anima_artist_mixer_dynamic_block_patch"


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
