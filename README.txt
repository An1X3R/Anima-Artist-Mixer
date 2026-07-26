This plugin solves the problem of Anima not handling artist chains well.
This plugin provides the established Cross-Attn path plus an experimental post-adapter path.

Anima Artist Pack
Node for inputting artist chain and prompt, here referred to as artist and base_prompt
1. This node has two text boxes (top and bottom), one clip input and one custom-type artist_pack output
2. The top text box is for artists, the bottom is for base_prompt. base_prompt does not include negative prompts
3. Artist writing follows the general Anima conventions. Additionally, when writing artists in this node, it's recommended to put one artist per line
4. Two weight syntaxes are supported (they can coexist):
	(artist:1.5)   bracket syntax, applied at the CLIP encoding stage, non-linear, same as SD/A1111
	::artist::1.5  the new injection-layer weight, applied at the multi-artist fusion stage, linear and predictable
	The two can be stacked, e.g. ::(artist:1.1)::0.8
	When normalize_weights is enabled, ::weight values are treated as relative ratios
	When normalize_weights is disabled, ::weight values act as direct multipliers
	weight range is 0.0~4.0, default 1.0

Anima Artist Cross-Attn
The node that actually does the mixing
1. This node has three input ports: a required model input, a required artist_pack input, and an optional advanced_options input
It has two output ports: a model output and a base_prompt output
2. combine_mode determines when/how multiple artists are mixed
	output_avg theoretically represents each artist's traits better, but compute scales with artist count
	concat theoretically mixes better, with compute mostly independent of artist count
	lowrank_avg is a stabilized version of output_avg, applying SVD low-rank constraint to multi-artist deltas, more stable across seeds, controlled by lowrank_k
	default is output_avg
3. fusion_mode determines how the artist mix interacts with base_prompt
	Performance-wise, concat_with_base gets slower as artist count grows; interpolate is much less affected
	interpolate theoretically does not impact base_prompt's content or composition
	concat_with_base theoretically fuses base_prompt and artists more thoroughly
	base_preserve only lets artists influence base from the side, leaving base's direction untouched, mild style
	The first two are usually similar in practice, default is interpolate (less affected by multi-artist scaling)
4. strength controls the artist/base_prompt mixing ratio, usually set to 1
	0.0~1.0 interpolation mode, 0=pure base, 1=pure artist
	1.0~4.0 extrapolation mode, style is amplified, equivalent to "stronger style"
	Recommended range 1.5~2.5, >3 easily oversaturates
	Don't push both strength and ::weight high at the same time when artist count is large
5. enabled toggles the node on/off
6. apply_to_uncond toggles mixing the artist chain into uncond/negative rows
	uncond_strength controls how much artist injection uncond rows receive when enabled
	0.0=no uncond injection, 0.15~0.35=weak influence, 1.0=old full-uncond behavior
7. Output wiring: the model output goes directly into KSampler; base_prompt goes to KSampler's positive (positive conditioning)

Anima Artist Adapter Mixer (Experimental)
An alternative to Anima Artist Cross-Attn. Do not connect both in series.
1. Mixes LLMAdapter output embeddings once at the model boundary instead of patching every cross-attention layer
2. Formula: mixed = base + strength * perpendicular(weighted_artists - base, base)
3. base_anchored is the default alignment mode
	Each artist keeps its complete Qwen source, own T5 target IDs, and own T5 weights through LLMAdapter
	After Adapter, matching base T5 tokens share anchor rows and every unmatched artist token gets a gap row
	Exact suffix matching is preferred; LCS is only a tokenizer-boundary fallback
	No real base or artist token row is pooled or truncated
4. shared_base_ids is retained as the older A/B mode
	It uses the base T5 target grid for every artist, replacing each artist's original target sequence
5. Alignment is guided by T5 IDs after Adapter; Qwen embeddings are not padded by this plugin
6. base_anchored keeps uncond rows unchanged because negative T5 IDs are unavailable for honest alignment
7. Optional Q-only Anchor:
	Connect Anima Artist Options(Advanced) to the Adapter Mixer's advanced_options input
	Enable artist_anchor_q and enter fixed manual anchor_seed_list values such as 42,12345
	Selected seeds are averaged and the anchor pre-run uses the mixed post-Adapter context
	anchor_user_blend, anchor_deep_layer_threshold, stabilizer_end_percent, anchor_refresh_mode, anchor_cache_points, and anchor_keyframe_mode also apply
	An empty seed list is rejected because it changes the anchor between executions
	Q-only Anchor changes cond Q but does not run the old per-artist attention mixer again
8. Anchor refresh modes:
	once is the default and reuses one start-sigma Q snapshot
	warm_cache runs anchor passes across the first complete sampling run, stores averaged CPU Q keyframes, and reuses them for later KSampler seeds
	anchor_cache_points controls retained keyframes; default 8, range 2~12
	anchor_keyframe_mode=uniform_sigma keeps evenly spaced sigma frames
	anchor_keyframe_mode=adaptive_q keeps the bounded frames with the largest sampled Q-trajectory interpolation error; first warmup uses more CPU transfer
	The warm cache is session-only and rebuilds when context, shape, seeds, or relevant settings change
9. Layer/step ranges, EMA, lowrank_avg, static capture, Structure Guard, and Style Balance remain Cross-Attn-only
10. Output wiring is the same as Cross-Attn: model to KSampler model, base_prompt to positive

Anima Artist Options(Advanced)
Provides advanced settings for users
1. This node outputs advanced_options plus anchor_seeds_used
	advanced_options connects to Anima Artist Cross-Attn or Anima Artist Adapter Mixer
	anchor_seeds_used is a text list of the seeds used by anchor_q
2. Basic settings: layer range, sampling-progress range, normalize toggle, layer_filter (custom layer selection)
3. Stability-related parameters (for resolving multi-artist cross-seed style drift):
	artist_ema_alpha             cross-step EMA smoothing, 0=off, 0.3~0.5 light, 0.5~0.8 medium-heavy
	lowrank_k                    only effective when combine_mode=lowrank_avg, 1=most stable, 2~3 keeps small per-artist differences
	artist_static_capture        accumulate artist attention for the first K steps then freeze, 30~50% performance gain
	static_capture_k             the K above, default 6, range 1~12
	artist_anchor_q              use a fixed-seed anchor instead of user-seed Q, the most stable approach across seeds
	anchor_seed_list             optional manual anchor seeds, e.g. 12345,67890; when filled, anchor_seeds_count is ignored
	anchor_seeds_count           number of anchor seeds, default 1, range 1~4
	anchor_user_blend            anchor / user-x blend ratio, 0=pure anchor, 1=pure user x
	anchor_deep_layer_threshold  shallow layers use anchor for stable style, deep layers use user x for fine brushwork, -1=disabled
	anchor_refresh_mode          [Adapter Mixer only] once=single snapshot, warm_cache=session reuse
	anchor_cache_points          [Adapter Mixer only] warm_cache CPU keyframes, default 8, range 2~12
	anchor_keyframe_mode         [Adapter Mixer only] uniform_sigma or bounded adaptive_q selection
4. Stability tools are usually enabled progressively from light to heavy: try ema first → if not enough, switch to lowrank_avg → still not enough, enable static_capture → still not enough, enable anchor_q
5. Layer range can shorten generation time, but with some quality impact, advanced users can tune it themselves

Anima Artist Structure Guard
Optional helper node for object/composition stability experiments
1. Connect it between Anima Artist Options(Advanced) and Anima Artist Cross-Attn
2. structure_preserve pushes interpolate deltas toward the safer base_preserve direction
3. delta_norm_cap limits artist delta magnitude relative to the base attention output
4. Both default to 0.0, which keeps old behavior unchanged

Anima Artist Style Balance
Optional helper node for reducing seed-to-seed artist dominance drift
1. Connect it between Anima Artist Options(Advanced) and Anima Artist Cross-Attn
2. style_balance matches each artist's output volume before user weights are applied
3. ::artist::weight still works normally after style_balance
4. Try 0.25~0.35 for light stabilization, 0.45~0.60 for stronger stabilization

License
Starting with version 26.8.1, this project is licensed under GNU General Public License v3.0.
Versions published before 26.8.1 remain under their original MIT License.
