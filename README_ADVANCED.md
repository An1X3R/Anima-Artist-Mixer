# Anima-Artist-Mixer

## Introduction

This is a ComfyUI custom node that provides **multi-artist mixing** for the Anima model. It offers an established cross-attention output path and an experimental post-LLMAdapter embedding path, sidestepping the interference that LLM-based text encoders suffer from when multiple artist tags coexist in a single prompt.

The companion `AnimaArtistPack` node provides a one-shot experience: write your artist list in one text box (comma or newline separated) and your main prompt in another. The node automatically splits, encodes, and packages everything for downstream use.

This README documents the current split-module architecture, including layered cross-seed stabilization (EMA / lowrank / static-capture / anchor-Q), CFG-style strength extrapolation, linear injection-layer weight syntax, and the optional Structure Guard helper.

## 26.8.1 Headline: Post-Adapter Mixer

`AnimaArtistAdapterMixer` is the major new experimental path. It mixes post-LLMAdapter context once at the model boundary, uses information-preserving `base_anchored` token-row alignment by default, and leaves individual cross-attention layers untouched unless optional Q-only Anchor is enabled. In current Anima testing it approached twice the throughput of the established per-layer artist-output path while retaining similar visual quality; results vary by workflow and hardware.

The Adapter Mixer and `AnimaArtistCrossAttn` are alternative algorithms. Do not chain them. See [AnimaArtistAdapterMixer](#animaartistadaptermixer-experimental-cross-attention-decoupled) for the full contract, mathematical limits, accepted advanced parameters, manual Anchor seeds, warm-cache behavior, and adaptive Q keyframes.

## What problem it solves

Anima uses an LLM as its text encoder (unlike SDXL's CLIP). LLM encoders are heavily **contextualized** — every token's embedding fuses semantics from surrounding tokens. This has a direct consequence:

- Single artist tag: the LLM produces a conditioning that captures that artist's style accurately. Works well.
- Multiple artist tags together: the artist tags' embeddings interfere with each other, and the resulting conditioning ends up looking like neither A nor B but a "squeezed-together" middle ground.

This node encodes each artist as a **separate** conditioning, bypassing the interference at the encoding stage, then mixes them inside the model at the cross-attention level using selectable strategies. Mixing happens in an already-stabilized feature space, where it's far more controllable than mixing at the prompt level.

## How it works

### Anima's structure

Anima = MiniTrainDIT backbone + LLMAdapter text adapter. Text flows like this:

```
Prompt
  → LLM encoder (Qwen, etc.)
  → raw embedding (1, T, 1024)
  → LLMAdapter (6-layer transformer, adapts LLM output to DiT's expected distribution)
  → processed (1, max(T5 tokens, 512), 1024), right-padded to at least 512
  → consumed as K/V by every DiT block's cross-attention
```

The DiT backbone has 28 blocks total, each with its own independent cross-attention layer. The same text conditioning is consumed 28 times across these layers.

### Injection mechanism

The node patches only `diffusion_model.blocks[i].cross_attn.forward` using ComfyUI's `add_object_patch` API. The original attention module stays in the model tree, so weight paths remain unchanged and ComfyUI can restore or unload the model normally.

Each artist conditioning is lazily run through the LLMAdapter on its first forward call (when the model is already on GPU), producing a processed embedding that's cached for reuse across sampling steps. Anima right-pads outputs shorter than 512 rows but does not truncate outputs longer than 512.

Each layer's injection is wrapped in exception isolation: if a single layer's injection fails, only that layer falls back to the original cross-attention; other layers continue working normally.

### CFG compatibility

ComfyUI batches cond and uncond into a single `batch=2` forward, with `transformer_options["cond_or_uncond"]` marking each row. This node injects only into the cond rows by default; uncond rows keep their original base context, so CFG guidance is preserved naturally.

`apply_to_uncond` can also inject the artist mix into uncond rows. This is experimental: it can produce striking results, but it also changes CFG's reference branch and can destabilize structure. `uncond_strength` controls that uncond-side injection. `1.0` matches the old full-uncond behavior, while lower values such as `0.15~0.35` give a weaker influence.

## The cross-seed instability problem (and how v24 addresses it)

In multi-artist setups, the same prompt with different seeds tends to produce **noticeably different style mixes** — sometimes wlop dominates, other times sakimichan does, even though their weights are equal. This is structural, not a bug:

- The cross-attn Q comes from base hidden state, which is seed-driven
- For each seed, attention picks slightly different artist token weights
- Across seeds, the "dominant artist" can flip

v24 layers four optional stabilizers, ordered from light to heavy (configured in `AnimaArtistOptions`):

1. **artist_ema_alpha**  — temporal EMA smoothing across sampling steps
2. **combine_mode = lowrank_avg + lowrank_k**  — SVD low-rank constraint on multi-artist deltas
3. **artist_static_capture + static_capture_k**  — freeze artist attention after the first K steps
4. **artist_anchor_q**  — replace user-seed Q with a fixed-seed anchor's Q (most aggressive, ~fully decouples cross-seed)

These can be combined freely. None of them are on by default — leaving them off gives the original v17-equivalent behavior.

## Mathematical limits of artist mixing

Up front: **this node cannot achieve the near-lossless artist mixing that SDXL does.**

SDXL's CLIP encoder produces approximately linearly composable per-token features. Anima's LLM + LLMAdapter output is **strongly non-linear** — any mixing strategy introduces distortion. What this node does is make distortion as controllable as possible and avoid the worst failure modes, not eliminate it.

In practice:
- Style-similar artists tend to mix well
- Style-divergent artists may "regress to the mean", landing in a compromise that resembles neither A nor B. This is more pronounced with weight normalization on (the default), since features get averaged after being normalized to relative proportions
- Extreme weight ratios (e.g. `"1.0, 0.05"`) typically collapse back to the dominant artist's pure style
- v24's `lowrank_avg` (k=1) deliberately accepts more "regression to the mean" in exchange for cross-seed stability — good for production, less suited to experimental style exploration

## Requirements

- **Anima model only**. Depends on Anima's built-in `LLMAdapter` (`preprocess_text_embeds`); plain MiniTrainDIT or other DiTs won't work
- Must use the **CLIP loader compatible with Anima's text-encoding workflow** (i.e. one whose tokens carry `t5xxl_ids`). `AnimaArtistPack` calls `clip.encode_from_tokens_scheduled` internally
- Inference path only, no training support

## Installation

Clone or download into your ComfyUI `custom_nodes` directory:

```
ComfyUI/custom_nodes/<this-plugin-folder>/
```

Restart ComfyUI. No extra dependencies.

## Quick start

```
                          ┌──► artist_pack ──► AnimaArtistCrossAttn ──► MODEL ──► KSampler
[Load CLIP] ─► CLIP ──────┤                              │                          │
                          │                              └──► base_prompt ──► (positive)
                          │
                          └──► CLIPTextEncode (Negative) ──► (negative)

[Load Anima Model] ──► MODEL ──► AnimaArtistCrossAttn

(optional) AnimaArtistOptions ──► advanced_options ──► AnimaArtistCrossAttn
```

Experimental decoupled alternative:

```
                          ┌──► artist_pack ──► AnimaArtistAdapterMixer ──► MODEL ──► KSampler
[Load CLIP] ─► CLIP ──────┤                                  │
                          │                                  └──► base_prompt ──► (positive)
                          └──► CLIPTextEncode (Negative) ──► (negative)

[Load Anima Model] ──► MODEL ──► AnimaArtistAdapterMixer
```

Key points:
- Write your artist chain in `AnimaArtistPack`'s top text box (comma or newline separated)
- Write your main prompt in the bottom text box
- Connect `AnimaArtistCrossAttn`'s `base_prompt` output directly to KSampler's positive input
- Encode the negative prompt independently with `CLIPTextEncode`; it does not go through this plugin
- Advanced controls (layer range, sampling-step range, stabilizers) come via the optional `AnimaArtistOptions` node

## Parameters

### AnimaArtistPack (artist chain split + encode)

| Parameter | Type | Description |
|---|---|---|
| `clip` | CLIP | Anima-compatible CLIP |
| `artist_chain` | STRING (multiline) | Artist chain. Comma or newline separated. Supports both CLIP weighting `(wlop:1.2)` and the new injection-layer weight `::wlop::1.5`, including stacking |
| `base_prompt` | STRING (multiline, optional) | Main prompt. Leave empty to encode artists alone |

Outputs `ANIMA_PACK`, an internal struct holding each artist's separately-encoded conditioning, the artist label list, the parsed per-artist weights, and a separately-encoded conditioning for the bare base prompt.

How it works internally: the node splits `artist_chain` into N artist names, parses any `::name::weight` syntax to extract per-artist injection weights (which are stripped before CLIP encoding), and encodes each as `<artist_name>\n<base_prompt>` (Anima's recommended format: artist first, newline, then main prompt). It also encodes a clean copy of `base_prompt` alone for use as KSampler's positive conditioning.

### AnimaArtistCrossAttn (main node)

| Parameter | Type | Description |
|---|---|---|
| `model` | MODEL | Anima model |
| `artist_pack` | ANIMA_PACK | Output from `AnimaArtistPack` |
| `combine_mode` | enum | How multiple artists are merged: `output_avg` (recommended) / `concat` / `lowrank_avg` (cross-seed-stable) |
| `fusion_mode` | enum | How merged artists act on the main prompt: `interpolate` (recommended) / `concat_with_base` / `base_preserve` |
| `strength` | FLOAT 0~4 | Overall artist contribution. 0~1 = interpolation, 1~4 = CFG-style extrapolation (style amplified) |
| `enabled` | BOOLEAN | Master switch |
| `apply_to_uncond` | BOOLEAN | Default False. Enables artist injection on uncond rows |
| `uncond_strength` | FLOAT 0~1 | Only used when `apply_to_uncond=True`. 0 = no uncond injection, 1 = old full-uncond behavior |
| `advanced_options` | ANIMA_OPTS | Optional advanced controls |

Outputs:
- `model`: model with artist mixing patched in. Connect to KSampler's `model` input
- `base_prompt`: the bare base-prompt conditioning from `artist_pack`. Connect to KSampler's positive input

### AnimaArtistAdapterMixer (experimental, cross-attention-decoupled)

This node is an alternative to `AnimaArtistCrossAttn`, not an add-on. It mixes the post-LLMAdapter context once at the model boundary. Individual `cross_attn.forward` methods remain untouched unless the optional Q-only Anchor is enabled.

For post-adapter base embedding `B` and artist embeddings `A_i`, it applies the projection independently to every token row:

```
C = B + strength * proj_perpendicular(sum_i(weight_i * A_i) - B, B)
```

The mixed `C` is then used as the normal model context by every DiT block.

| Parameter | Description |
|---|---|
| `model` | Anima model. The node only requires a callable `preprocess_text_embeds` Adapter interface |
| `artist_pack` | Output from `AnimaArtistPack` |
| `strength` | Perpendicular artist-delta strength, range 0~4 |
| `normalize_weights` | Normalize `::weight` values to relative ratios |
| `alignment_mode` | `base_anchored` (default) or `shared_base_ids` (older A/B mode) |
| `enabled` | Return the original model unchanged when disabled |
| `apply_to_uncond` | Also mix artist context into CFG uncond rows |
| `uncond_strength` | Uncond-side scale when `apply_to_uncond=True` |
| `advanced_options` | Optional `AnimaArtistOptions` payload. Only Q-only Anchor controls are consumed |

`base_anchored` runs every artist through LLMAdapter with that artist conditioning's complete Qwen source embedding, own `t5xxl_ids`, and own T5 weights. It then uses T5 token IDs to build a common post-Adapter canvas. The full base token sequence is searched inside each artist sequence, preferring an exact suffix/substring match. Matching base tokens share anchor rows. Tokens before, between, or after those anchors are placed in explicit gap rows, so every real Adapter output row is retained exactly once. If tokenization prevents an exact match, a longest-common-subsequence fallback preserves all unmatched rows and reports the reduced match quality in the log.

`shared_base_ids` is retained for controlled comparison. Each artist keeps its separately encoded Qwen source, but every LLMAdapter pass uses the base prompt's T5 IDs and weights. This creates a common grid without scattering, at the cost of replacing the artist conditioning's original target sequence and artist-specific T5 weighting.

Neither mode pads Qwen. The alignment is performed after LLMAdapter and is guided by T5 target IDs. Anima's own minimum-512 zero padding remains, but native padding rows are excluded from token matching. `base_anchored` changes cond rows only: the wrapper does not receive negative-prompt T5 IDs, so applying the same remap to uncond would risk dropping or misaligning negative-prompt semantics.

Q-only Anchor can be enabled by connecting `AnimaArtistOptions`, setting `artist_anchor_q=True`, and entering a fixed manual `anchor_seed_list`. Cross-attention itself has no random generator: each selected seed creates a fixed anchor latent, and multiple seeds are averaged before their per-layer hidden states become Q inputs. The Adapter anchor pre-run receives the same mixed post-Adapter context as the real denoising pass. That mixed context remains the only K/V input, and the old per-artist attention branches are not executed. Q replacement is cond-only; uncond keeps the current sampling hidden state.

The Adapter path reads `anchor_user_blend`, `anchor_deep_layer_threshold`, `stabilizer_end_percent`, `anchor_refresh_mode`, `anchor_cache_points`, and `anchor_keyframe_mode`. It deliberately rejects an empty/non-manual anchor seed because `AnimaArtistOptions` generates fresh random anchors when the field is empty. Start with `anchor_user_blend=0.0`, `anchor_deep_layer_threshold=-1`, and `stabilizer_end_percent=1.0` to measure maximum stability. Then use a higher user blend or a finite deep-layer threshold if composition becomes too constrained.

Adapter Anchor refresh modes:

- `once` (default): capture one start-sigma Q snapshot. It is reused for every later sigma and for later executions while shape, context, and seeds remain unchanged.
- `warm_cache`: during the first complete sampling run, execute the selected anchor seeds at every active sigma. Average their hidden states immediately, retain only `anchor_cache_points` CPU keyframes, and use sigma interpolation on later runs. Once warm, changing only the KSampler seed performs no anchor model forward. `uniform_sigma` keeps evenly spaced sigma frames. `adaptive_q` scores sampled Q-trajectory interpolation error and continuously prunes to the configured capacity, preserving both endpoints.

The warm cache is owned by the patched model object and lasts for the current ComfyUI session. It is rebuilt when the mixed context, image/batch shape, manual seed list, cache-point count, keyframe mode, deep-layer threshold, or stabilizer range changes. Restarting ComfyUI also clears it. System-RAM usage scales with spatial token count, cached layers, and keyframe count; only the seed average is stored, so adding more reference seeds increases first-run compute but not cache size. A 1024-class, full-layer Anima cache can require several GiB with eight points. `adaptive_q` copies each first-run candidate to CPU for scoring but immediately removes the least useful interior frame, so retained memory stays bounded. Interpolation also introduces small CPU-to-GPU transfer and arithmetic cost, but no additional model pass after warmup.

The Adapter Mixer:

- does not patch individual cross-attention modules unless Q-only Anchor is enabled
- computes and caches each artist's post-adapter embedding once per device and dtype
- caches the final projected context by conditioning identity/version and CFG row layout, avoiding per-step projection and GPU value fingerprints
- preserves every real base and artist token row in `base_anchored`; it performs no pooling
- preserves pre-existing model wrappers through chain-safe delegation
- modifies cond rows only by default
- does not use start/end block ranges, sampling ranges, EMA, low-rank output averaging, static capture, Structure Guard, or Style Balance

Do not connect `AnimaArtistCrossAttn` and `AnimaArtistAdapterMixer` in series. Use separate branches from the same unpatched model for controlled A/B comparisons.

### AnimaArtistOptions (advanced)

Not connecting this node = default behavior. Connecting it makes its settings take effect.

Outputs:
- `advanced_options`: connect to `AnimaArtistCrossAttn`, or to `AnimaArtistAdapterMixer` for its Q-only Anchor subset
- `anchor_seeds_used`: text list of the anchor seeds that will be used. The Cross-Attn Mixer can generate fresh seeds when `anchor_seed_list` is empty; Adapter Q-only Anchor requires a manual list

| Parameter | Description |
|---|---|
| `start_block` / `end_block` | Inject only on DiT blocks in `[start, end]`. `end_block = -1` means up to the last block |
| `start_percent` / `end_percent` | Inject only during sampling progress in `[start, end]`. `0.0` = sampling start, `1.0` = end |
| `normalize_weights` | True: weights are normalized to relative proportions, including explicit `::weight` values. False: weights act as independent strength multipliers |
| `layer_filter` | Advanced layer-selection string (overrides start_block/end_block). Example: `"0,3,5-10,-1"` |
| `artist_ema_alpha` | Temporal EMA on artist attention output across steps. 0 = off |
| `lowrank_k` | SVD truncation rank for `lowrank_avg`. 1 = most stable |
| `artist_static_capture` | Freeze artist attention after `static_capture_k` warmup steps |
| `static_capture_k` | Number of warmup steps before freezing. Default 6, range 1~12 |
| `artist_anchor_q` | Replace user-seed Q with a fixed-seed anchor's Q. The strongest cross-seed stabilizer |
| `anchor_seed_list` | Optional manual anchor seeds, e.g. `12345,67890`. When filled, `anchor_seeds_count` is ignored |
| `anchor_seeds_count` | Number of anchor seeds to average. Default 1, range 1~4 |
| `anchor_user_blend` | Blend ratio between anchor Q and user Q. 0 = pure anchor, 1 = pure user |
| `anchor_deep_layer_threshold` | Use anchor for shallow layers `[0, N)`, user Q for deep layers `[N, end]`. -1 disables |
| `anchor_refresh_mode` | Adapter Mixer only: `once` or session-level `warm_cache` sigma keyframes |
| `anchor_cache_points` | Adapter Mixer only: `warm_cache` CPU keyframes. Default 8, range 2~12 |
| `anchor_keyframe_mode` | Adapter Mixer only: `uniform_sigma` or bounded `adaptive_q` selection for `warm_cache` |

### Optional Structure Guard

`AnimaArtistStructureOptions` is a separate helper node. It is intentionally not part of `AnimaArtistOptions`, so existing ComfyUI workflows keep their original widget order.

Connect it as:

```
AnimaArtistOptions -> AnimaArtistStructureOptions -> AnimaArtistCrossAttn
```

| Parameter | Description |
|---|---|
| `structure_preserve` | Moves `interpolate` deltas toward the safer `base_preserve` direction. `0.0` keeps old behavior |
| `delta_norm_cap` | Caps artist delta magnitude relative to base attention output. `0.0` disables the cap |

For object-heavy prompts, try `structure_preserve=0.25~0.50` with `interpolate`, or `delta_norm_cap=1.0~1.5` with `base_preserve`.

### Optional Style Balance

`AnimaArtistStyleBalance` is a separate helper node for reducing seed-to-seed artist dominance drift.

Connect it as:

```
AnimaArtistOptions -> AnimaArtistStyleBalance -> AnimaArtistCrossAttn
```

| Parameter | Description |
|---|---|
| `style_balance` | Matches each artist output's volume before user weights are applied. `0.0` keeps old behavior |

This does not replace injection weights. For example, `1::wlop, 2::sakimichan` still applies a 1:2 user weight ratio after the balance step. Try `0.25~0.35` for light stabilization and `0.45~0.60` for stronger stabilization.

## Core concepts

### Post-adapter projection versus output projection

The established `base_preserve` path applies perpendicular projection after attention:

```
base_out   = Attention(Q, base_context)
artist_out = sum_i(weight_i * Attention(Q, artist_context_i))
output     = base_out + strength * proj_perpendicular(artist_out - base_out, base_out)
```

The Adapter Mixer applies the same geometric form before attention:

```
mixed_context = base_context + strength *
                proj_perpendicular(artist_context - base_context, base_context)
output = Attention(Q, mixed_context)
```

These are deliberately separate algorithms. K/V projection, K normalization, and attention softmax are nonlinear as a whole, so mixing contexts before attention is not mathematically equivalent to mixing outputs after attention. The Adapter path trades that direct output-space control for much lower repeated attention cost and weaker coupling to a particular DiT block layout.

### combine_mode: how multiple artists are merged

#### `output_avg` (recommended default)

Each artist runs through cross-attention **independently**, producing N outputs that are then weighted-averaged:

```
out = sum_i (w_i * cross_attn(x, K_i, V_i))
```

Each softmax is computed independently over its own K, V — artists don't compete for attention budget. Mathematically the cleanest mixing strategy. Cost: cross-attention forwards = number of artists.

#### `concat`

Concatenates artist conditionings along the token dimension:

```
K/V = [artist 1's 512 tokens, artist 2's 512 tokens, ...]
out = cross_attn(x, K_concat, V_concat)
```

Single cross-attention call, but all artists compete in the same softmax. The padding zero-vectors at the tail of LLMAdapter outputs are naturally suppressed by attention (no manual masking needed).

Pros: single forward, fast. Cons: attention is shared across artists, typically less expressive than output_avg.

#### `lowrank_avg` (v22, cross-seed stable)

A stabilized variant of `output_avg`. Each artist's attention output minus the base output gives a delta tensor; the N delta tensors are stacked into a matrix `D ∈ ℝ^(N × M)`. SVD truncates `D` to its top-k principal components before reconstructing and weighted-averaging:

```
delta_i = cross_attn(x, K_i, V_i) - base_out
D = stack(delta_i)
D_lowrank = SVD_truncate(D, k)
artist_total = base_out + sum_i (w_i * D_lowrank[i])
```

Why it stabilizes cross-seed: the seed-flipping "dominant artist" effect is largely high-rank noise in `D`. Truncating to top-k strips that noise. `k=1` (the default) projects all artists onto a single shared direction, giving maximum stability at the cost of artists looking more homogeneous. `k=2` or `3` keeps small per-artist differentiation. `k >= N` is mathematically equivalent to `output_avg` (no projection).

> Earlier versions had `mean` and `weighted_sum` modes (per-position weighted average over LLMAdapter outputs). They were removed: position-i in different artists carries different semantics, so element-wise averaging causes K/V semantic misalignment and inevitably produces broken images. A `replace` mode was also removed: it discards the main prompt's role in cross-attention entirely, severely degrading prompt adherence.

### fusion_mode: how the merged artist acts on the main prompt

#### `interpolate` (recommended with output_avg)

Base and artist each run cross-attention once, then outputs are linearly interpolated by `strength`:

```
out = base_out * (1 - strength) + artist_out * strength
```

With `strength` in `[0, 1]`, this is strict interpolation (`strength=0` = pure base, `strength=1` = pure artist). With `strength` in `(1, 4]`, it becomes CFG-style extrapolation: `out = base_out + strength * (artist_out - base_out)` — the artist's deviation from base is amplified, producing a stronger style. Smooth transitions, minimal style drift. Cost: one extra base forward per layer.

#### `concat_with_base`

Cross-attention's K/V becomes `[base_tokens, artist_tokens]`, letting attention see both base and artist:

```
K = [K_base, K_artist]
V = [V_base, V_artist]
out = cross_attn(x, K, V)
```

The softmax decides per-pixel-position whether to attend to base or artist. With `strength < 1`, the result is mixed once more with a pure-base output.

Pros: base prompt stays in the attention computation, so prompt adherence is best preserved. Artist still dominates style, but with the lightest drift.

#### `base_preserve` (v17)

Decomposes the artist contribution into components parallel and perpendicular to the base output, and only injects the perpendicular component:

```
delta = artist_total - base_out
delta_perp = delta - proj_to_base(delta)
out = base_out + strength * delta_perp
```

The base direction is left untouched; the artist can only add a sideways offset. Mild style impact, good for keeping the main prompt in firm control. Useful when `interpolate` shifts composition more than desired.

## Cross-seed stabilizers in detail

All four stabilizers are off by default. Enable progressively from light to heavy.

### artist_ema_alpha (lightest)

Applies a per-layer exponential moving average to the artist attention output across sampling steps:

```
artist_total_t = alpha * artist_total_{t-1} + (1 - alpha) * artist_total_t
```

Reasoning: cross-step jitter in artist attention often comes from the dominant artist flipping under shifting cross-attn QK match. EMA smooths this over time. Effective mainly on `interpolate` and `base_preserve` (not `concat_with_base`, where the artist isn't isolated).

Range 0~0.95. 0.3~0.5 is a light effect, 0.5~0.8 is medium-heavy. Higher values can lag the style behind the base content.

Cache resets when sigma jumps up (i.e. a new sampling run begins).

### lowrank_avg + lowrank_k (medium)

See `combine_mode = lowrank_avg` above. The SVD constraint is permanent (every step), unlike EMA which only smooths over time. More aggressive but more uniform-feeling result.

### artist_static_capture + static_capture_k (heavy, also a perf win)

During the first K sampling steps, the artist attention output is computed and accumulated. After step K, the average is **frozen** and reused for all remaining steps. Subsequent steps skip the artist cross-attention entirely.

Why this works: after a few denoising steps, the user hidden state has stabilized enough that further per-step recomputation of artist attention only adds jitter without adding style. Freezing decouples style from later-step content fluctuations.

Side benefit: steps after K skip N artist cross-attention forwards each, giving 30~50% wall-clock speedup at typical K=6, N=5~7.

`K` ranges 1~12, default 6. Lower K = earlier freeze = more stable but earlier commitment to a possibly-suboptimal style estimate. Higher K = later freeze = closer to non-frozen behavior.

Cache resets on sigma jump. Mutually exclusive with EMA (which becomes a no-op once frozen).

### artist_anchor_q (heaviest, true cross-seed decoupling)

Root cause of cross-seed style drift: the cross-attn Q comes from base hidden state, which is seed-driven. v23 fully addresses this by **replacing the Q's source with a fixed-seed anchor's hidden state**:

1. On first invocation, the plugin runs a single-step "anchor pass" using a fixed seed (default 42) with the user's prompt context
2. Each layer's pre-cross-attn hidden state is captured during this anchor pass and cached
3. During real sampling, when computing artist attention, Q is sourced from the anchor's cached hidden state instead of the user's current hidden state
4. The base attention still uses user Q (so base content adapts to user seed normally)

Result: artist attention is identical across all seeds for the same prompt + resolution. Cross-seed style drift drops to near-zero.

Cache key is `(x.shape, id(context))` — same prompt + same resolution reuses the anchor for free across seeds. Different prompt or different resolution triggers a fresh anchor pass.

First-time cost: ~1 extra step worth of forward time for the anchor pass. After that, zero overhead per seed.

**Sub-options for finer control**:

- `anchor_seed_list` (optional): manually specify fixed anchor seeds, e.g. `12345,67890`. If filled, this list is used directly and `anchor_seeds_count` is ignored. A single seed can lock the style reference to one selected result.
- `anchor_seeds_count` (1~4, default 1): when `anchor_seed_list` is empty, generates N new random seeds per execution, runs N anchor passes, and averages their hidden states. This avoids permanently biasing all runs toward one built-in seed set. Cost scales linearly with N.
- `anchor_user_blend` (0~1, default 0): blends anchor Q with user Q. 0 = pure anchor (max stability), 1 = pure user (equivalent to disabling anchor). Useful if pure anchor produces brushwork that looks slightly disconnected from the actual content.
- `anchor_deep_layer_threshold` (-1~64, default -1 = disabled): when set to N, layers `[0, N)` use anchor Q (style stability) while layers `[N, end]` use user Q (content fidelity). Based on the principle that early DiT blocks set style and late blocks add detail.

Mutually exclusive with `artist_static_capture` (anchor takes priority, with a warn log).

## Recommended combinations

### Daily use, no stabilizers

```
combine_mode = output_avg
fusion_mode  = interpolate
strength     = 1.0
```

### Light cross-seed improvement

```
combine_mode      = output_avg
fusion_mode       = interpolate
strength          = 1.0
artist_ema_alpha  = 0.4
```

### Strong cross-seed stability + style amplification

```
combine_mode      = lowrank_avg
lowrank_k         = 1
fusion_mode       = interpolate
strength          = 2.0
artist_static_capture = True
static_capture_k  = 6
```

### Maximum cross-seed stability (production)

```
combine_mode      = output_avg
fusion_mode       = interpolate
strength          = 1.5
artist_anchor_q   = True
anchor_seeds_count = 1
anchor_user_blend = 0.0
```

To control individual artist strength within the chain, use either weight syntax inside `artist_chain`:

```
wlop, ::sakimichan::1.2, (krenz:0.7)
```

## Performance notes

### Computational cost

In `output_avg` and `lowrank_avg` modes, each artist still needs an independent attention softmax plus the base attention. This is mathematically necessary:

```
sum_i (w_i * softmax(Q @ K_i^T / √d) @ V_i)
```

Each softmax must be computed independently over its own K, V. Merging them into one shared softmax would degrade the semantics to `concat` mode. On compatible Anima attention modules, the plugin now reuses the shared Q projection and only repeats the artist-specific K/V and softmax work. The first compatible module is checked against the original path at runtime before this optimization is enabled.

Artist batching is VRAM-aware. The plugin checks available VRAM once per sampling shape and automatically chooses how many artists to process together. It uses full parallel batching when memory is plentiful and smaller chunks when memory is tight. Q remains shared across chunks, including the one-artist-per-chunk fallback. This changes execution order only; artist weighting and fusion formulas remain unchanged.

### Approximate timing (30 steps, varies by GPU)

| Configuration | Relative time |
|---|---|
| 1 artist | 1.0x (baseline) |
| 4 artists | ~1.4x |
| 8 artists | ~1.7x |
| 5 artists + `artist_static_capture` (K=6) | ~1.1x |
| 5 artists + `artist_anchor_q` (cached, 2nd seed onward) | ~1.05x |

**More artists means more time** — but `artist_static_capture` and `artist_anchor_q` largely amortize this away after the warmup steps.

### Strongly recommended: use layer range and step range to reduce cost

After connecting `AnimaArtistOptions`, you can **dramatically cut generation time** with usually minimal quality loss:

- **Layer range** (`start_block / end_block` or `layer_filter`): inject only on specific DiT blocks. `0..13` (front half) cuts time roughly in half. Artist style is mostly determined by early blocks, so the loss is usually acceptable
- **Sampling-step range** (`start_percent / end_percent`): inject only during a portion of sampling. `0.0..0.5` (first half) similarly cuts time, since artist style is mostly absorbed during early sampling

Both can be **combined**: "front-half layers + front-half sampling" can bring 8-artist scenarios back to near-single-artist timing. This is the most effective optimization for multi-artist setups, and stacks with `artist_static_capture` / `artist_anchor_q`.

## How to write the artist chain

### Recommended format: artist on top, main prompt separate

The two text boxes of `AnimaArtistPack` have distinct roles:

```
artist_chain (top box):
  wlop
  ::sakimichan::1.2
  (krenz:0.7)

base_prompt (bottom box):
  masterpiece, 1girl, standing, in a forest, ...
```

Internally the node concatenates each as `<artist_name>\n<base_prompt>` before encoding — Anima's empirically most stable format. You don't need to repeat artist names in the main prompt.

### Two layers of weighting

There are **two independent** weighting points:

1. **CLIP weighting** (`(name:1.2)` syntax): scales token embeddings before they pass through the LLMAdapter (a non-linear 6-layer transformer). Outcome isn't strictly predictable but stays close to the LLM's natural output distribution. Same as SD/A1111 syntax.
2. **Injection-layer weighting** (`::name::1.5` syntax, v24): scales the artist's contribution at the cross-attention output stage. Linear and predictable: `::name::2.0` makes that artist's relative contribution exactly twice as strong as a default-weight artist.

They can be **stacked**: `::(wlop:1.1)::0.8` applies CLIP weight 1.1 first, then injection weight 0.8.

When `normalize_weights` is enabled, explicit `::weight` values are treated as relative ratios. For example, `1::wlop, 2::sakimichan` becomes a 1:2 relative mix. When `normalize_weights` is disabled, explicit weights are honored as direct multipliers.

Global artist contribution is controlled by `AnimaArtistCrossAttn`'s `strength` (independent of per-artist weights).

### Important note on heavy weights

Do **not** push both `strength` and direct, unnormalized `::weight` values high at once when you have many artists. With `normalize_weights=False`, cross-attn output magnitude is roughly `strength * sum(::weight)`, which becomes unstable past ~6-8x baseline. Symptoms: oversaturation, noise patches, or fully broken output. Default values (`strength=1`, `normalize_weights=True`) are always safe.

### When `normalize_weights = False`

Default `normalize_weights = True`: in `output_avg`, N artists' weights are normalized to `1/N` each, so **total contribution always equals 1**.

With normalization off and no `::weight` used: each artist contributes at its raw weight (default 1.0). **Total contribution = N**, which exceeds the model's training distribution and produces pure noise.

The node intercepts dangerous unnormalized configurations:

| Artist count + normalize=False | Behavior |
|---|---|
| 1 artist | Normal (equivalent to normalized) |
| 2~3 artists | Warning, but allowed (may overexpose) |
| 4+ artists | **Hard error**, with three suggested fixes |

If you actually want "one artist weakened", the **recommended approach is to keep normalize_weights=True** and use either CLIP weighting `(name:0.3)` or injection weighting `::name::0.3` to lower a specific artist:

```
wlop, ::krenz::0.3
```

This keeps wlop dominant with a krenz accent, without breaking total-contribution stability.

## Advanced layer/step controls

### Layer range (`start_block` / `end_block`)

DiT blocks at different depths correspond to different semantic levels: early blocks affect overall composition and style, later blocks affect detail and texture. For example:

- `0..13` (front half): artist dominates composition; details are filled in by the model
- `14..27` (back half): only inject into detail layers; composition follows the main prompt

### `layer_filter` (more flexible layer selection)

A string with **higher priority than `start_block / end_block`** (overrides them when set). Syntax:

- Comma-separated indices: `"0,3,7"`
- Ranges with hyphen: `"5-10"`
- Negative indices (counted from end): `"-1"` = last block
- Mix: `"0,3,5-10,-1"`

Useful for non-contiguous patterns like "early + last only" or interval injection experiments.

### Sampling-step range (`start_percent` / `end_percent`)

Different sampling stages determine different image content (high sigma = composition, low sigma = texture refinement). For example:

- `0.0..0.5`: inject only in the first half; artist sets the overall layout, then the model is free to refine details
- `0.3..1.0`: skip the very early steps to avoid the artist pushing composition too hard

Implementation detail: the node uses `set_model_unet_function_wrapper` to capture the current sigma at each `apply_model` call, then maps user-set percent ranges to sigma ranges via `model_sampling.percent_to_sigma()`.

## Known issues

### Adapter target-token alignment

`base_anchored` preserves all real rows, but token-ID equality is only a structural alignment cue. LLMAdapter embeddings are contextual: a base token encoded with an artist prefix is not numerically or semantically identical to the same token encoded in the bare base prompt. The perpendicular projection also happens before K/V projection and softmax, so it cannot guarantee output-space orthogonality.

Exact suffix matching is expected for the plugin's normal `<artist>\n<base_prompt>` encoding. If the tokenizer changes boundary tokens, LCS fallback is used and the log reports the match count. Every unmatched row is retained, but weak matching means weaker row-to-row semantic correspondence. `pad_longest` is no longer exposed because equal tensor lengths alone do not align token meaning.

### `model_function_wrapper` chain conflicts

When sampling-step range or any cross-seed stabilizer is enabled, this node uses `set_model_unet_function_wrapper` to capture per-step sigma. The implementation is chain-safe — it preserves and forwards calls to any pre-existing wrapper.

However, if another custom node connected **after** this one sets a wrapper without chain-safety (overwriting blindly), the sigma capture is lost, and step-range / EMA-reset / static-capture-reset all silently degrade.

Diagnosis: reset `start_percent / end_percent` to 0.0 / 1.0 and disable all stabilizers; if behavior recovers, another node is overwriting the wrapper.

## Acknowledgements

Special thanks to **汐浮尘** for co-development, testing, and design contributions during the development of this node. The `AnimaArtistPack` split-and-encode design comes from their improvement.

## License

Copyright (c) 2026 An1X3R and 汐浮尘.

Starting with version 26.8.1, this project is licensed under the **GNU General Public License v3.0**. See [LICENSE](LICENSE) for the complete terms. Versions published before 26.8.1 remain under the MIT License included with those historical releases.
