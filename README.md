# Anima-Artist-Mixer

A ComfyUI custom node for **multi-artist mixing** on Anima models. It encodes each artist separately, then mixes either cross-attention outputs or post-adapter embeddings, avoiding the prompt-side artist interference caused by Anima's LLM text encoder.

## New in 26.8.2: Automatic Adapter Anchor Seeds

The Adapter Mixer's Q-only Anchor once again accepts an empty `anchor_seed_list`. In that mode, `anchor_seeds_count` generates fresh random anchor seeds on every execution so users can explore different style references. Entering one or more fixed seeds still provides repeatable cross-run stabilization and allows `once` or `warm_cache` data to be reused while the cache key remains unchanged.

## New in 26.8.1: Post-Adapter Artist Mixing

### Special thanks [sparrow]（ https://github.com/spawner1145 )
The idea and formula of Post Adapter Artist Mixing were proposed by him

`Anima Artist Adapter Mixer (Experimental)` is the headline addition in 26.8.1. It moves artist mixing to the post-LLMAdapter context and performs the projection once at the model boundary instead of running the established artist-output mix inside every patched cross-attention layer.

- In current Anima testing, this path delivered close to twice the generation throughput of the established Cross-Attn path while keeping visual quality close. Actual gains depend on artist count, resolution, sampler, and hardware.
- The default `base_anchored` alignment keeps every real base and artist Adapter row. It aligns rows with T5 token IDs without pooling, truncating, or replacing the artist's Qwen source embedding and T5 target sequence.
- Optional Q-only Anchor accepts either automatically generated or fixed seeds for cross-seed style control. `warm_cache` can spend extra time on the first complete run, retain bounded CPU Q keyframes, and reuse them for later sampler seeds when fixed seeds keep the cache key stable; `adaptive_q` keeps the most informative keyframes.
- The established `Anima Artist Cross-Attn` node remains available and unchanged as the compatibility path. The two mixers are alternatives and must not be chained.

Shortest Adapter workflow:

```text
Anima Artist Pack -> Anima Artist Adapter Mixer -> KSampler model
                                  |
                                  +-> base_prompt -> KSampler positive
```

Start with `alignment_mode=base_anchored`, `strength=1.0`, `normalize_weights=true`, and `apply_to_uncond=false`. See [Experimental Adapter Path](#experimental-adapter-path) for alignment, Anchor-Q, warm-cache, and parameter details.

## What It Does

Anima uses an LLM-based text encoder. When several artist tags are placed in one prompt, the encoder contextualizes them together and the styles can blur or interfere. This plugin instead:

1. Splits the artist chain into individual artists.
2. Encodes each artist with the same base prompt.
3. Mixes artists through either the established cross-attention path or an experimental post-adapter path.
4. Patches only a cloned model, leaving the input model unchanged.

The normal workflow still uses three main nodes:

- `Anima Artist Pack (Split + Encode)`
- `Anima Artist Cross-Attn (v26 fixed)`
- `Anima Artist Options (Advanced)`

`Anima Artist Adapter Mixer (Experimental)` is an alternative to the Cross-Attn node. It performs one perpendicular projection in LLMAdapter embedding space and does not patch individual attention layers.

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

## Experimental Adapter Path

Replace `Anima Artist Cross-Attn` with `Anima Artist Adapter Mixer` to test the decoupled path:

```text
Anima Artist Pack -> Anima Artist Adapter Mixer -> KSampler
                                     |
                                     +-> base_prompt -> positive

Anima Artist Options (Advanced) -> advanced_options (optional Q-only Anchor)
```

Recommended first test:

```text
strength         = 1.0
normalize_weights = true
alignment_mode   = base_anchored
apply_to_uncond  = false
```

The Adapter Mixer computes, per token:

```text
mixed = base + strength * perpendicular(sum(weight_i * artist_i) - base, base)
```

`base_anchored` is the default. Every artist keeps its complete Qwen source embedding, its own T5 target IDs, and its own T5 weights through LLMAdapter. After the Adapter, the node finds the base prompt's T5 token sequence inside each artist sequence, places matching base rows on shared anchors, and places every unmatched artist row in gap slots. Exact suffix matching is used when possible; an LCS fallback handles tokenizer-boundary differences. No real base or artist token row is pooled, truncated, or overwritten.

`shared_base_ids` remains as the older A/B mode. It gives every artist Adapter pass the base prompt's T5 target grid, so shapes and positions match directly, but the artist's original T5 target sequence and artist-specific T5 weights are replaced. It is cheaper to reason about, but less information-preserving.

This is T5-token-guided alignment after LLMAdapter, not padding of Qwen embeddings. Anima itself still zero-pads Adapter outputs shorter than 512 rows; those native zero rows are not treated as prompt tokens. `base_anchored` modifies cond rows only because the model wrapper does not receive the negative prompt's T5 IDs and therefore cannot align uncond rows honestly.

For stronger cross-seed stability, connect `Anima Artist Options (Advanced)` to the Adapter Mixer's optional `advanced_options` input and enable `artist_anchor_q`. Leave `anchor_seed_list` empty to generate `anchor_seeds_count` fresh random references on every execution, or enter fixed values such as `42,12345` for repeatable stabilization. The selected seeds are averaged, and the Adapter anchor pass uses the same mixed post-Adapter context as the real denoising pass. The old per-artist attention mixer is not run a second time. `anchor_user_blend`, `anchor_deep_layer_threshold`, `stabilizer_end_percent`, `anchor_refresh_mode`, `anchor_cache_points`, and `anchor_keyframe_mode` apply; the other advanced mixing controls are ignored.

`anchor_refresh_mode=once` keeps the low-cost legacy timing: one start-sigma Q snapshot is reused throughout sampling and across later executions while its cache key remains valid. `warm_cache` runs the selected anchor seeds at every active sigma during the first complete sampling run, keeps `anchor_cache_points` averaged Q keyframes in CPU RAM, and linearly interpolates them on later runs. The default is 8 points. `anchor_keyframe_mode=uniform_sigma` retains evenly spaced sigma frames. `adaptive_q` observes every warmup sigma and keeps the bounded set whose sampled Q trajectory has the greatest interpolation error; it adds CPU transfer during the first warmup but has the same later-run model-forward count. Changing only the KSampler seed then needs no anchor model passes when the anchor seed list is fixed. Prompt/context, artist mix, resolution/batch shape, anchor seeds, cache-point count, keyframe mode, or stabilizer range changes rebuild the cache. Automatic mode intentionally generates a new seed list for each execution, so it also starts a new `once` or `warm_cache` cycle. The cache is session-only and is cleared by a ComfyUI restart. First-run time scales with the number of selected anchor seeds; later runs with fixed seeds still transfer cached Q keyframes from CPU but do not execute the anchor model.

Do not chain Adapter Mixer and Cross-Attn Mixer on the same model. They remain alternative artist-mixing algorithms. Q-only Anchor already patches the Adapter Mixer's attention Q when enabled; chaining the full Cross-Attn node would inject the artist set twice.

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
- `anchor_refresh_mode` (Adapter Mixer only)
- `anchor_cache_points` (Adapter Mixer only)
- optional `layer_filter`
- optional `anchor_keyframe_mode` (Adapter Mixer only)

The advanced node preserves its original widget order for workflow compatibility. Compatibility-safe additions such as `anchor_keyframe_mode` are appended after the existing fields; unrelated experimental controls remain in separate helper nodes.

It also outputs `anchor_seeds_used`, a text list of the anchor seeds that will be used. If `anchor_seed_list` is empty, a new list of KSampler-range 64-bit seeds is generated on every execution according to `anchor_seeds_count`. If `anchor_seed_list` is filled, it shows and uses the parsed manual list without randomizing it.

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

## Anchor Seeds: Automatic or Fixed

When `artist_anchor_q` is enabled, `anchor_seed_list` can pin the anchor pass to seeds you choose:

```text
anchor_seed_list = 12345,67890
```

If `anchor_seed_list` is empty, `anchor_seeds_count` controls how many fresh random anchor seeds are generated for each execution. If `anchor_seed_list` is filled, `anchor_seeds_count` is ignored. You can also enter a single seed to lock the style reference to one selected result.

The Adapter Mixer accepts both modes. Automatic seeds preserve a way to search for a useful style reference, but the new list causes `once` and `warm_cache` to rebuild on the next execution. Fixed seeds preserve the chosen reference and permit cross-execution cache reuse. Multiple seeds are averaged before either refresh mode is applied. Start with one seed to limit warmup cost; use two or more only when a single reference seed carries too much of its own composition bias.

`warm_cache` stores only the averaged result, not a separate copy per seed. RAM usage still scales with resolution, active anchor layers, and `anchor_cache_points`. Eight full-layer keyframes can require several GiB at 1024-class resolutions; lower `anchor_cache_points` or a finite `anchor_deep_layer_threshold` reduces that cost. `adaptive_q` temporarily copies each observed warmup frame to CPU for scoring, then immediately prunes back to the configured bound.

## 26.8.1 Release Highlights

This version includes the new post-Adapter path plus structural and runtime fixes inspired by PR #4 while keeping the established Cross-Attn path compatible:

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
- Fix FP16 `base_preserve` projection NaNs and batched Anchor condition selection.
- Reuse compatible Anima Q projections across artists after a runtime equivalence check.
- Automatically split artist batches according to currently available VRAM.
- Add an experimental post-adapter mixer that uses a model-level context wrapper instead of per-layer attention patches.
- Add `shared_base_ids` alignment so every artist Adapter output uses the same T5 target-token grid.
- Add lossless `base_anchored` alignment that preserves every real base and artist Adapter row; keep `shared_base_ids` as an A/B mode and remove the unsafe `pad_longest` UI mode.
- Add optional Q-only Anchor for the Adapter Mixer, using selected cross-attention anchor seeds without running the old artist mixer twice.
- Make Adapter Anchor-Q reference the mixed post-Adapter context and add a session-level sigma-keyframe warm cache for later sampler seeds.
- Cache the final projected Adapter context, avoid per-step GPU value fingerprints, and add bounded adaptive Q keyframe selection.

## Caveats

This plugin cannot make Anima artist mixing as lossless as SDXL artist chains. Anima's LLM encoder and adapter are highly non-linear, so any cross-attention mixing can still affect composition or object structure. The goal is to make the tradeoff controllable and debuggable.

## License

Copyright (c) 2026 An1X3R and 汐浮尘.

Starting with version 26.8.1, this project is licensed under the **GNU General Public License v3.0**. See [LICENSE](LICENSE) for the complete terms. GPLv3 permits commercial use, but distribution of covered modified or combined versions must preserve the GPLv3 freedoms and provide the corresponding source as required by the license.

Versions published before 26.8.1 remain available under the MIT License that accompanied those releases. The GPLv3 change does not revoke rights already granted for those historical versions.
