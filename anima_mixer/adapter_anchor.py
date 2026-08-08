"""Q-only Anchor patch for the post-Adapter mixer."""

import logging

import torch

from .constants import ANCHOR_LAYER_THRESHOLD_DISABLED
from .patching import (
    clear_mixer_run_state,
    in_stabilizer_window,
    resolve_mask,
    should_reraise,
)

logger = logging.getLogger(__name__)


def _match_anchor_batch(anchor_x, x):
    if anchor_x.shape == x.shape:
        return anchor_x
    if anchor_x.shape[1:] != x.shape[1:]:
        return None

    anchor_batch = int(anchor_x.shape[0])
    user_batch = int(x.shape[0])
    if user_batch % anchor_batch == 0:
        repeats = user_batch // anchor_batch
        return anchor_x.repeat(repeats, *([1] * (anchor_x.dim() - 1)))
    if anchor_batch % user_batch == 0:
        return anchor_x[:user_batch]
    return None


def resolve_adapter_anchor_input(x, state, layer_index, transformer_options):
    """Replace cond rows with fixed-anchor hidden states for one attention layer."""
    if isinstance(transformer_options, dict) and transformer_options.get(
        "multigpu_thread_device"
    ) is not None:
        # The multigpu sampler invokes one shared model-options wrapper from
        # concurrent device workers.  A single mutable Anchor cache cannot be
        # safely shared between those workers, so keep the user Q path intact.
        return x
    if not state.get("artist_anchor_q", False):
        return x
    if state.get("_anchor_failed", False):
        return x
    if state.get("_adapter_anchor_failed", False):
        return x
    if state.get("_embedding_mixer_failed", False):
        return x
    if not in_stabilizer_window(state):
        return x

    threshold = int(state.get(
        "anchor_deep_layer_threshold",
        ANCHOR_LAYER_THRESHOLD_DISABLED,
    ))
    if threshold >= 0 and layer_index >= threshold:
        return x

    anchor_x = state.get("_anchor_cache", {}).get(layer_index)
    if anchor_x is None:
        return x
    anchor_x = _match_anchor_batch(anchor_x, x)
    if anchor_x is None:
        return x
    anchor_x = anchor_x.to(device=x.device, dtype=x.dtype)

    user_blend = max(
        0.0,
        min(1.0, float(state.get("anchor_user_blend", 0.0))),
    )
    anchored = (
        user_blend * x + (1.0 - user_blend) * anchor_x
        if user_blend > 0.0 else anchor_x
    )

    options = transformer_options or {}
    cond_or_uncond = options.get("cond_or_uncond")
    mask = resolve_mask(cond_or_uncond, int(x.shape[0]), False, state)
    row_mask = torch.tensor(
        mask,
        device=x.device,
        dtype=torch.bool,
    ).view(x.shape[0], *([1] * (x.dim() - 1)))
    return torch.where(row_mask, anchored, x)


class AdapterAnchorQForwardPatch:
    """Patch one cross-attention forward without adding another artist branch."""

    _anima_adapter_anchor_q_forward_patch = True

    def __init__(self, original_forward, state, layer_index):
        self.original_forward = original_forward
        self.state = state
        self.layer_index = int(layer_index)
        self._anima_mixer_state = state

    def rebind_state(
        self,
        state,
        original_forward=None,
        original_module=None,
    ):
        """Create a clone-local patch with the clone's original forward."""
        del original_module  # kept for a common object-patch rebind signature
        return type(self)(
            self.original_forward if original_forward is None else original_forward,
            state,
            self.layer_index,
        )

    def __call__(self, x, context=None, rope_emb=None, transformer_options=None):
        if self.state.get("_in_anchor_run", False):
            cache = self.state.setdefault("_anchor_cache", {})
            cache[self.layer_index] = x.detach().clone()
            return self.original_forward(
                x,
                context,
                rope_emb=rope_emb,
                transformer_options=transformer_options,
            )

        q_input = x
        try:
            q_input = resolve_adapter_anchor_input(
                x,
                self.state,
                self.layer_index,
                transformer_options,
            )
        except BaseException as error:
            if should_reraise(error):
                clear_mixer_run_state(self.state, interrupted=True)
                raise
            if not self.state.get("_warned_adapter_anchor_failure", False):
                logger.exception(
                    "[AnimaAdapterAnchorQ] Q replacement failed; the user Q "
                    "will be used: %s",
                    error,
                )
                self.state["_warned_adapter_anchor_failure"] = True
            self.state["_adapter_anchor_failed"] = True

        return self.original_forward(
            q_input,
            context,
            rope_emb=rope_emb,
            transformer_options=transformer_options,
        )


def make_adapter_anchor_q_forward_patch(original_forward, state, layer_index):
    return AdapterAnchorQForwardPatch(original_forward, state, layer_index)
