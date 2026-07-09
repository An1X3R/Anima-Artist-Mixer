# Anima-Artist-Mixer

A ComfyUI custom node for **multi-artist mixing** on Anima models. It encodes each artist separately and mixes the resulting conditionings inside Anima's cross-attention layers, avoiding the prompt-side artist interference caused by Anima's LLM text encoder.

## What It Does

Anima uses an LLM-based text encoder. When several artist tags are placed in one prompt, the encoder contextualizes them together and the styles can blur or interfere. This plugin instead:

1. Splits the artist chain into individual artists.
2. Encodes each artist with the same base prompt.
3. Patches Anima cross-attention on a cloned model.
4. Mixes the artist outputs with selectable strategies.

The normal workflow still uses three main nodes:

- `Anima Artist Pack (Split + Encode)`
- `Anima Artist Cross-Attn (v26 fixed)`
- `Anima Artist Options (Advanced)`

There is also an optional `Anima Artist Structure Guard` node for object/composition stability experiments.
`Anima Artist Style Balance` can be used when different seeds make different artists dominate the mix.

## Installation

Clone or download this repository into your ComfyUI `custom_nodes` directory:

```text
ComfyUI/custom_nodes/Anima-Artist-Mixer/
```

Restart ComfyUI. No extra Python dependencies are required.

## Requirements

- Anima model only.
- Use the same CLIP/text loader that Anima's normal workflow uses.
- Inference only.

The plugin checks for Anima's `preprocess_text_embeds` path and will reject unsupported model structures.

## Quick Start

1. Connect your CLIP loader to `Anima Artist Pack`.
2. Put artists in the top artist-chain text box, separated by commas or newlines.
3. Put your main positive prompt in the base prompt text box.
4. Connect `artist_pack` to `Anima Artist Cross-Attn`.
5. Connect your Anima model to `Anima Artist Cross-Attn`.
6. Send the patched model to KSampler.
7. Send the `base_prompt` output to KSampler positive conditioning.

Recommended starting point:

```text
combine_mode = output_avg
fusion_mode  = interpolate
strength     = 1.0
normalize_weights = true
```

For stronger style:

```text
fusion_mode = base_preserve
strength    = 1.2 - 1.8
```

## Artist Weights

Artist-chain entries can use normal prompt weights and injection-layer weights:

```text
wlop
(krenz:0.8)
1.2::sakimichan
sakimichan::1.2
```

Notes:

- `(artist:1.2)` is applied before CLIP/text encoding.
- `1.2::artist` and `artist::1.2` are linear artist-mixing weights.
- When `normalize_weights` is enabled, explicit `::weight` values are treated as relative ratios.
- When `normalize_weights` is disabled, weights act as direct multipliers.

Example:

```text
1::wlop, 2::sakimichan
```

With `normalize_weights=true`, this becomes a 1:2 relative mix, not a 3x amplification.

## Cross-Attention Node

### combine_mode

- `output_avg`: runs each artist separately and averages outputs. Usually the best default.
- `concat`: concatenates artist conditionings before attention. Faster for many artists, but often less controlled.
- `lowrank_avg`: stabilized averaging that constrains multi-artist deltas using a low-rank projection.

### fusion_mode

- `interpolate`: blends base and artist outputs directly.
- `base_preserve`: removes the artist delta component that points along the base output direction, usually preserving subject/composition better.
- `concat_with_base`: experimental and currently not the recommended path.

### strength

- `0.0`: pure base.
- `1.0`: normal artist mix.
- `>1.0`: extrapolates style strength. Useful, but can damage structure if pushed too high.

### apply_to_uncond and uncond_strength

`apply_to_uncond` is off by default. Leave it off for stable CFG behavior.

When `apply_to_uncond` is enabled, `uncond_strength` controls how much artist injection is applied to uncond rows:

- `0.0`: no artist injection on uncond rows.
- `0.15 - 0.35`: weak uncond style influence.
- `0.4 - 0.65`: stronger experimental influence.
- `1.0`: old full-uncond injection behavior.

## Advanced Options

`Anima Artist Options (Advanced)` exposes:

- block range: `start_block`, `end_block`
- sampling range: `start_percent`, `end_percent`
- `normalize_weights`
- `artist_ema_alpha`
- `lowrank_k`
- `artist_static_capture`
- `static_capture_k`
- `artist_anchor_q`
- `anchor_seed_list`
- `anchor_seeds_count`
- `anchor_user_blend`
- `anchor_deep_layer_threshold`
- `stabilizer_end_percent`
- optional `layer_filter`

The advanced node intentionally keeps its original widget order for workflow compatibility. New experimental controls are placed in separate helper nodes.

It also outputs `anchor_seeds_used`, a text list of the anchor seeds that will be used. If `anchor_seed_list` is empty, this shows the built-in seeds selected by `anchor_seeds_count`; if `anchor_seed_list` is filled, it shows the parsed manual list.

## Structure Guard

`Anima Artist Structure Guard` is optional. Connect it like this:

```text
Anima Artist Options (Advanced) -> Anima Artist Structure Guard -> Anima Artist Cross-Attn
```

It adds two controls:

- `structure_preserve`: pushes `interpolate` deltas toward the safer `base_preserve` direction.
- `delta_norm_cap`: limits the artist delta magnitude relative to the base attention output.

Both default to `0.0`, which preserves the old behavior.

Suggested tests:

```text
interpolate:
structure_preserve = 0.25 - 0.50
delta_norm_cap     = 1.25 - 1.75

base_preserve:
structure_preserve = 0.0
delta_norm_cap     = 1.0 - 1.5
```

If an existing workflow suddenly produces bad structure after an update, recreate the `Anima Artist Options (Advanced)` node once so ComfyUI refreshes its widget mapping.

## Style Balance

`Anima Artist Style Balance` is optional. Connect it like other option helper nodes:

```text
Anima Artist Options (Advanced) -> Anima Artist Style Balance -> Anima Artist Cross-Attn
```

It adds:

- `style_balance`: reduces seed-to-seed artist dominance drift by matching each artist's output volume before user weights are applied.

This does not replace `::artist::weight` or `1.2::artist`. The balance step happens first, then your artist weights are still applied normally.

Suggested tests:

```text
style_balance = 0.25 - 0.35  # light
style_balance = 0.45 - 0.60  # stronger
```

Very high values can make different artists feel more averaged.

## Manual Anchor Seeds

When `artist_anchor_q` is enabled, `anchor_seed_list` can pin the anchor pass to seeds you choose:

```text
anchor_seed_list = 12345,67890
```

If `anchor_seed_list` is empty, `anchor_seeds_count` uses the built-in anchor seeds as before. If `anchor_seed_list` is filled, `anchor_seeds_count` is ignored. You can also enter a single seed to lock the style reference to one selected result.

## Recent Fixes

This version includes structural and runtime fixes inspired by PR #4 while keeping the public node set conservative:

- Split implementation into `anima_mixer/` modules.
- Patch `cross_attn.forward` instead of replacing the whole module.
- Preserve disabled-node behavior by returning the unpatched model.
- Fix CFG cond/uncond row masking for batched sampling.
- Make explicit `::weight` respect `normalize_weights` again.
- Reset runtime caches between sampling runs.
- Improve anchor/static/EMA cache keys.
- Avoid zero-padding uncond rows in `concat_with_base`.
- Reraise OOM and Comfy interrupt exceptions instead of silently disabling layers.
- Add optional structure-guard controls without changing old advanced-option widget order.

## Caveats

This plugin cannot make Anima artist mixing as lossless as SDXL artist chains. Anima's LLM encoder and adapter are highly non-linear, so any cross-attention mixing can still affect composition or object structure. The goal is to make the tradeoff controllable and debuggable.

## License

MIT License.
