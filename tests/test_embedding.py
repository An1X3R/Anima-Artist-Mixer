import unittest
from unittest import mock

import torch

from anima_mixer.anchor import (
    make_sigma_capture,
    _finalize_anchor_trajectory,
    _new_anchor_trajectory,
    _record_trajectory_step,
    _store_trajectory_frame,
)
from anima_mixer.alignment import (
    align_artist_embeddings,
    align_base_context,
    build_base_anchored_plan,
)
from anima_mixer.constants import (
    ALIGN_BASE_ANCHORED,
    ALIGN_SHARED_BASE_IDS,
    ANCHOR_CACHE_POINTS_DEFAULT,
    ANCHOR_CACHE_POINTS_MAX,
    ANCHOR_KEYFRAME_ADAPTIVE_Q,
)
from anima_mixer.embedding import (
    build_artist_embedding_sum,
    make_adapter_embedding_wrapper,
    mix_projected_context,
    pad_embeddings_to_longest,
    weighted_embedding_sum,
)
from anima_mixer.adapter_anchor import make_adapter_anchor_q_forward_patch
from anima_mixer.nodes_embedding import AnimaArtistAdapterMixer
from anima_mixer.nodes_ui import AnimaArtistOptions


class FakeAdapterModel:
    def __init__(self):
        self.seen_ids = []

    def preprocess_text_embeds(self, raw, ids, t5xxl_weights=None):
        self.seen_ids.append(ids.detach().cpu().clone())
        length = int(ids.shape[-1])
        marker = raw[:, :1, :].to(torch.float32)
        output = marker.expand(raw.shape[0], length, raw.shape[-1]).clone()
        if t5xxl_weights is not None:
            output = output * t5xxl_weights
        return output


class FakeCrossAttention:
    context_dim = 2

    def __init__(self):
        self.calls = []

    def forward(self, x, context=None, rope_emb=None, transformer_options=None):
        self.calls.append({
            "x": x.detach().clone(),
            "context": context.detach().clone(),
            "transformer_options": dict(transformer_options or {}),
        })
        return x


class FakeBlock:
    def __init__(self):
        self.cross_attn = FakeCrossAttention()


class FakeAnchorAdapterModel(FakeAdapterModel):
    def __init__(self):
        super().__init__()
        self.blocks = [FakeBlock()]


class FakeModelPatcher:
    def __init__(self, diffusion_model):
        self.diffusion_model = diffusion_model
        self.model_options = {}
        self.object_patches = {}

    def get_model_object(self, name):
        if name != "diffusion_model":
            raise KeyError(name)
        return self.diffusion_model

    def clone(self):
        cloned = FakeModelPatcher(self.diffusion_model)
        cloned.model_options = dict(self.model_options)
        cloned.object_patches = dict(self.object_patches)
        return cloned

    def set_model_unet_function_wrapper(self, wrapper):
        self.model_options["model_function_wrapper"] = wrapper

    def add_object_patch(self, path, value):
        self.object_patches[path] = value


def make_state(dm, alignment_mode=ALIGN_BASE_ANCHORED):
    state = {
        "dm_ref": dm,
        "labels": ["a", "b"],
        "raws": [
            torch.tensor([[[1.0, 2.0]]]),
            torch.tensor([[[3.0, 4.0]]]),
        ],
        "ids_list": [
            torch.tensor([101, 10, 11, 12]),
            torch.tensor([201, 202, 10, 11, 12]),
        ],
        "t5_weights_list": [None, None],
        "base_ids": torch.tensor([10, 11, 12]),
        "base_t5_weights": None,
        "user_weights": [1.0, 1.0],
        "normalize_weights": True,
        "alignment_mode": alignment_mode,
        "strength": 1.0,
        "apply_to_uncond": False,
        "uncond_strength": 0.0,
        "_artist_embedding_cache": {},
        "_embedding_mixer_failed": False,
        "_warned_embedding_failure": False,
        "_warned_no_context": False,
        "_warned": False,
    }
    state["alignment_plan"] = (
        build_base_anchored_plan(state["base_ids"], state["ids_list"])
        if alignment_mode == ALIGN_BASE_ANCHORED else None
    )
    return state


class TokenAlignmentTests(unittest.TestCase):
    def test_exact_suffix_alignment_preserves_every_row(self):
        plan = build_base_anchored_plan(
            torch.tensor([10, 11, 12]),
            [
                torch.tensor([101, 10, 11, 12]),
                torch.tensor([201, 202, 10, 11, 12]),
            ],
        )
        self.assertEqual(plan["length"], 5)
        self.assertEqual(plan["base_positions"], (2, 3, 4))
        self.assertEqual(plan["artist_positions"][0], (1, 2, 3, 4))
        self.assertEqual(plan["artist_positions"][1], (0, 1, 2, 3, 4))
        self.assertEqual(plan["methods"], ("exact", "exact"))

    def test_lcs_fallback_keeps_unmatched_tokens_in_gaps(self):
        plan = build_base_anchored_plan(
            torch.tensor([10, 11, 12]),
            [torch.tensor([101, 10, 99, 12])],
        )
        self.assertEqual(plan["methods"], ("lcs",))
        self.assertEqual(plan["matched_counts"], (2,))
        positions = plan["artist_positions"][0]
        self.assertEqual(len(positions), 4)
        self.assertEqual(len(set(positions)), 4)
        self.assertTrue(all(0 <= position < plan["length"] for position in positions))

    def test_scatter_preserves_values_and_aligns_base(self):
        plan = build_base_anchored_plan(
            torch.tensor([10, 11]),
            [torch.tensor([90, 10, 11]), torch.tensor([80, 81, 10, 11])],
        )
        artists = align_artist_embeddings([
            torch.tensor([[[1.0], [2.0], [3.0]]]),
            torch.tensor([[[4.0], [5.0], [6.0], [7.0]]]),
        ], plan)
        base = align_base_context(
            torch.tensor([[[8.0], [9.0], [0.0], [0.0]]]),
            plan,
        )
        self.assertEqual(artists[0][artists[0] != 0].tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(artists[1][artists[1] != 0].tolist(), [4.0, 5.0, 6.0, 7.0])
        self.assertEqual(base[base != 0].tolist(), [8.0, 9.0])


class OptionNodeTests(unittest.TestCase):
    @mock.patch(
        "anima_mixer.nodes_ui.secrets.randbelow",
        side_effect=[11, 11, 22, 33],
    )
    def test_empty_seed_list_generates_requested_unique_seed_count(self, randbelow):
        options, seeds_used = AnimaArtistOptions().build(
            start_block=0,
            end_block=-1,
            start_percent=0.0,
            end_percent=1.0,
            normalize_weights=True,
            artist_anchor_q=True,
            anchor_seed_list="",
            anchor_seeds_count=3,
        )

        self.assertEqual(seeds_used, "11,22,33")
        self.assertEqual(options["anchor_seed_list"], "11,22,33")
        self.assertFalse(options["anchor_seed_list_is_manual"])
        self.assertEqual(options["anchor_seeds_count"], 3)
        self.assertEqual(randbelow.call_count, 4)

    def test_warm_cache_options_keep_manual_seed_list(self):
        options, seeds_used = AnimaArtistOptions().build(
            start_block=0,
            end_block=-1,
            start_percent=0.0,
            end_percent=1.0,
            normalize_weights=True,
            artist_anchor_q=True,
            anchor_seed_list="42,12345",
            anchor_refresh_mode="warm_cache",
            anchor_cache_points=99,
            anchor_keyframe_mode=ANCHOR_KEYFRAME_ADAPTIVE_Q,
        )
        self.assertEqual(seeds_used, "42,12345")
        self.assertTrue(options["anchor_seed_list_is_manual"])
        self.assertEqual(options["anchor_refresh_mode"], "warm_cache")
        self.assertEqual(options["anchor_cache_points"], ANCHOR_CACHE_POINTS_MAX)
        self.assertEqual(options["anchor_keyframe_mode"], ANCHOR_KEYFRAME_ADAPTIVE_Q)

    def test_new_nodes_default_to_eight_anchor_cache_points(self):
        inputs = AnimaArtistOptions.INPUT_TYPES()
        points = inputs["required"]["anchor_cache_points"]
        self.assertEqual(ANCHOR_CACHE_POINTS_DEFAULT, 8)
        self.assertEqual(points[1]["default"], 8)
        for section, name in (
            ("required", "anchor_refresh_mode"),
            ("required", "anchor_cache_points"),
            ("optional", "anchor_keyframe_mode"),
        ):
            self.assertIn("Adapter Mixer only", inputs[section][name][1]["tooltip"])


class AnchorKeyframeSelectionTests(unittest.TestCase):
    def test_uniform_sigma_keeps_target_crossings(self):
        state = {
            "anchor_cache_points": 3,
            "anchor_keyframe_mode": "uniform_sigma",
            "anchor_deep_layer_threshold": -1,
            "stabilizer_min_sigma": 0.0,
            "anchor_log_name": "test",
        }
        trajectory = _new_anchor_trajectory(state, ("test",), 1.0)
        state["_anchor_trajectory"] = trajectory
        for sigma in (1.0, 0.75, 0.5, 0.25, 0.1):
            state["_anchor_cache"] = {0: torch.full((1, 1, 2), sigma)}
            _record_trajectory_step(state, trajectory, sigma)
        _finalize_anchor_trajectory(state)

        self.assertTrue(trajectory["ready"])
        self.assertEqual(
            [frame["sigma"] for frame in trajectory["frames"]],
            [0.1, 0.5, 1.0],
        )

    def test_adaptive_q_keeps_endpoints_and_nonlinear_middle(self):
        state = {
            "anchor_cache_points": 3,
            "anchor_keyframe_mode": ANCHOR_KEYFRAME_ADAPTIVE_Q,
            "anchor_deep_layer_threshold": -1,
            "stabilizer_min_sigma": 0.0,
        }
        trajectory = _new_anchor_trajectory(state, ("test",), 1.0)
        samples = (
            (1.0, 0.0),
            (0.75, 0.25),
            (0.5, 3.0),
            (0.25, 0.75),
            (0.0, 1.0),
        )
        for sigma, value in samples:
            cache = {0: torch.full((1, 2, 4), value)}
            self.assertTrue(_store_trajectory_frame(state, trajectory, sigma, cache))

        self.assertEqual(
            [frame["sigma"] for frame in trajectory["frames"]],
            [0.0, 0.5, 1.0],
        )
        self.assertEqual(trajectory["observed_frames"], len(samples))
        self.assertEqual(trajectory["pruned_frames"], 2)
        self.assertEqual(len(trajectory["frames"]), 3)
        self.assertEqual(
            trajectory["bytes"],
            3 * torch.full((1, 2, 4), 0.0).numel() * 4,
        )


class EmbeddingMathTests(unittest.TestCase):
    def test_zero_padding_uses_longest_sequence(self):
        first = torch.ones((1, 2, 3))
        second = torch.full((1, 4, 3), 2.0)
        padded = pad_embeddings_to_longest([first, second])
        self.assertEqual([tuple(item.shape) for item in padded], [(1, 4, 3), (1, 4, 3)])
        self.assertTrue(torch.equal(padded[0][:, 2:], torch.zeros((1, 2, 3))))

    def test_weighted_sum_normalizes_relative_weights(self):
        first = torch.ones((1, 1, 2))
        second = torch.full((1, 1, 2), 4.0)
        mixed = weighted_embedding_sum([first, second], [1.0, 3.0], normalize=True)
        self.assertTrue(torch.allclose(mixed, torch.full_like(mixed, 3.25)))

    def test_projection_is_per_token_and_preserves_uncond(self):
        base = torch.tensor([
            [[1.0, 0.0], [0.0, 2.0]],
            [[1.0, 0.0], [0.0, 2.0]],
        ])
        artist = torch.tensor([[[1.0, 4.0], [3.0, 2.0]]])
        mixed = mix_projected_context(
            base,
            artist,
            strengths=[1.0, 1.0],
            mask=[True, False],
        )
        expected_cond = torch.tensor([[1.0, 4.0], [3.0, 2.0]])
        self.assertTrue(torch.allclose(mixed[0], expected_cond))
        self.assertTrue(torch.equal(mixed[1], base[1]))

        delta = mixed[0] - base[0]
        dot = (delta * base[0]).sum(dim=-1)
        self.assertTrue(torch.allclose(dot, torch.zeros_like(dot), atol=1e-6))

    def test_partial_uncond_strength(self):
        base = torch.tensor([
            [[1.0, 0.0]],
            [[1.0, 0.0]],
        ])
        artist = torch.tensor([[[1.0, 2.0]]])
        mixed = mix_projected_context(
            base,
            artist,
            strengths=[1.0, 0.25],
            mask=[True, True],
        )
        self.assertTrue(torch.allclose(mixed[0], torch.tensor([[1.0, 2.0]])))
        self.assertTrue(torch.allclose(mixed[1], torch.tensor([[1.0, 0.5]])))


class EmbeddingAdapterTests(unittest.TestCase):
    def test_shared_base_ids_use_one_target_grid(self):
        dm = FakeAdapterModel()
        state = make_state(dm, ALIGN_SHARED_BASE_IDS)
        ref = torch.zeros((2, 3, 2))
        artist_sum = build_artist_embedding_sum(state, ref)

        self.assertEqual(tuple(artist_sum.shape), (1, 3, 2))
        self.assertEqual(len(dm.seen_ids), 2)
        for seen in dm.seen_ids:
            self.assertTrue(torch.equal(seen.flatten(), state["base_ids"]))
        self.assertTrue(torch.allclose(
            artist_sum,
            torch.tensor([[[2.0, 3.0], [2.0, 3.0], [2.0, 3.0]]]),
        ))

    def test_base_anchored_keeps_artist_specific_ids_and_aligns_rows(self):
        dm = FakeAdapterModel()
        state = make_state(dm, ALIGN_BASE_ANCHORED)
        ref = torch.zeros((1, 5, 2))
        artist_sum = build_artist_embedding_sum(state, ref)

        self.assertEqual(tuple(artist_sum.shape), (1, 5, 2))
        self.assertEqual([ids.numel() for ids in dm.seen_ids], [4, 5])
        self.assertTrue(torch.allclose(
            artist_sum,
            torch.tensor([[
                [1.5, 2.0],
                [2.0, 3.0],
                [2.0, 3.0],
                [2.0, 3.0],
                [2.0, 3.0],
            ]]),
        ))

    def test_wrapper_replaces_cond_context_and_chains(self):
        dm = FakeAdapterModel()
        state = make_state(dm, ALIGN_BASE_ANCHORED)
        state["raws"] = [torch.tensor([[[1.0, 2.0]]])]
        state["ids_list"] = [torch.tensor([99, 10, 11, 12])]
        state["t5_weights_list"] = [None]
        state["user_weights"] = [1.0]
        state["alignment_plan"] = build_base_anchored_plan(
            state["base_ids"], state["ids_list"]
        )

        observed = {}

        def apply_model(input_tensor, timestep, **c):
            observed["context"] = c["c_crossattn"]
            return input_tensor

        def previous_wrapper(model_function, options):
            observed["previous_called"] = True
            return model_function(
                options["input"],
                options["timestep"],
                **options["c"],
            )

        wrapper = make_adapter_embedding_wrapper(state, previous_wrapper)
        base = torch.tensor([
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
        ])
        input_tensor = torch.zeros((2, 1))
        options = {
            "input": input_tensor,
            "timestep": torch.ones((2,)),
            "c": {
                "c_crossattn": base,
                "transformer_options": {"cond_or_uncond": [0, 1]},
            },
            "cond_or_uncond": [0, 1],
        }

        output = wrapper(apply_model, options)
        self.assertIs(output, input_tensor)
        self.assertTrue(observed["previous_called"])
        self.assertTrue(torch.allclose(
            observed["context"][0],
            torch.tensor([
                [1.0, 2.0],
                [1.0, 2.0],
                [1.0, 2.0],
                [1.0, 2.0],
            ]),
        ))
        self.assertTrue(torch.equal(observed["context"][1], base[1]))

    def test_wrapper_reuses_mixed_context_and_invalidates_inplace_changes(self):
        dm = FakeAdapterModel()
        state = make_state(dm, ALIGN_SHARED_BASE_IDS)
        state["raws"] = [torch.tensor([[[1.0, 2.0]]])]
        state["ids_list"] = [torch.tensor([99, 10, 11, 12])]
        state["t5_weights_list"] = [None]
        state["user_weights"] = [1.0]
        observed = []

        def apply_model(input_tensor, timestep, **c):
            observed.append(c["c_crossattn"])
            return input_tensor

        wrapper = make_adapter_embedding_wrapper(state, None)
        base = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]])
        options = {
            "input": torch.zeros((1, 1)),
            "timestep": torch.ones((1,)),
            "c": {
                "c_crossattn": base,
                "transformer_options": {"cond_or_uncond": [0]},
            },
            "cond_or_uncond": [0],
        }

        wrapper(apply_model, options)
        wrapper(apply_model, options)
        self.assertIs(observed[0], observed[1])
        first_key = state["_mixed_context_cache"]["key"]

        observed[1][0, 0, 0].add_(1.0)
        wrapper(apply_model, options)
        self.assertIsNot(observed[1], observed[2])
        self.assertEqual(first_key, state["_mixed_context_cache"]["key"])

        base[0, 0, 0].add_(1.0)
        wrapper(apply_model, options)
        self.assertIsNot(observed[2], observed[3])
        self.assertNotEqual(first_key, state["_mixed_context_cache"]["key"])

    def test_node_patches_clone_and_disabled_path_is_identity(self):
        dm = FakeAdapterModel()
        model = FakeModelPatcher(dm)
        base_conditioning = [[
            torch.tensor([[[0.0, 0.0]]]),
            {"t5xxl_ids": torch.tensor([10, 11, 12])},
        ]]
        artist_pack = {
            "conditionings": [[[
                torch.tensor([[[1.0, 2.0]]]),
                {"t5xxl_ids": torch.tensor([20, 21, 10, 11, 12])},
            ]]],
            "labels": ["artist"],
            "weights": [1.0],
            "base_conditioning": base_conditioning,
        }
        node = AnimaArtistAdapterMixer()

        disabled_model, disabled_base = node.patch(
            model,
            artist_pack,
            strength=1.0,
            normalize_weights=True,
            alignment_mode=ALIGN_BASE_ANCHORED,
            enabled=False,
            apply_to_uncond=False,
            uncond_strength=0.0,
        )
        self.assertIs(disabled_model, model)
        self.assertIs(disabled_base, base_conditioning)

        patched_model, patched_base = node.patch(
            model,
            artist_pack,
            strength=1.0,
            normalize_weights=True,
            alignment_mode=ALIGN_BASE_ANCHORED,
            enabled=True,
            apply_to_uncond=False,
            uncond_strength=0.0,
        )
        self.assertIsNot(patched_model, model)
        self.assertIs(patched_base, base_conditioning)
        self.assertIn("model_function_wrapper", patched_model.model_options)

        seen = {}

        def apply_model(input_tensor, timestep, **c):
            seen["context"] = c["c_crossattn"]
            return input_tensor

        wrapper = patched_model.model_options["model_function_wrapper"]
        base_context = torch.tensor([[
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]])
        input_tensor = torch.zeros((1, 1))
        output = wrapper(apply_model, {
            "input": input_tensor,
            "timestep": torch.ones((1,)),
            "c": {
                "c_crossattn": base_context,
                "transformer_options": {"cond_or_uncond": [0]},
            },
            "cond_or_uncond": [0],
        })
        self.assertIs(output, input_tensor)
        self.assertTrue(torch.allclose(
            seen["context"],
            torch.tensor([[
                [1.0, 2.0],
                [1.0, 2.0],
                [1.0, 2.0],
                [1.0, 2.0],
                [1.0, 2.0],
            ]]),
        ))

    def test_base_anchored_keeps_uncond_even_when_old_workflow_enables_it(self):
        dm = FakeAdapterModel()
        model = FakeModelPatcher(dm)
        base_conditioning = [[
            torch.tensor([[[0.0, 0.0]]]),
            {"t5xxl_ids": torch.tensor([10, 11])},
        ]]
        artist_pack = {
            "conditionings": [[[
                torch.tensor([[[1.0, 2.0]]]),
                {"t5xxl_ids": torch.tensor([20, 10, 11])},
            ]]],
            "labels": ["artist"],
            "weights": [1.0],
            "base_conditioning": base_conditioning,
        }
        patched_model, _ = AnimaArtistAdapterMixer().patch(
            model,
            artist_pack,
            strength=1.0,
            normalize_weights=True,
            alignment_mode=ALIGN_BASE_ANCHORED,
            enabled=True,
            apply_to_uncond=True,
            uncond_strength=1.0,
        )

        observed = {}

        def apply_model(input_tensor, timestep, **c):
            observed["context"] = c["c_crossattn"]
            return input_tensor

        base = torch.tensor([
            [[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
        ])
        input_tensor = torch.zeros((2, 1))
        wrapper = patched_model.model_options["model_function_wrapper"]
        wrapper(apply_model, {
            "input": input_tensor,
            "timestep": torch.ones((2,)),
            "c": {
                "c_crossattn": base,
                "transformer_options": {"cond_or_uncond": [0, 1]},
            },
            "cond_or_uncond": [0, 1],
        })

        self.assertFalse(torch.equal(observed["context"][0], base[0]))
        self.assertTrue(torch.equal(observed["context"][1], base[1]))

    @mock.patch(
        "anima_mixer.nodes_ui.secrets.randbelow",
        side_effect=[987654321, 123456789],
    )
    def test_adapter_q_anchor_accepts_automatically_generated_seeds(self, randbelow):
        dm = FakeAnchorAdapterModel()
        model = FakeModelPatcher(dm)
        artist_pack = {
            "conditionings": [[[
                torch.tensor([[[1.0, 2.0]]]),
                {"t5xxl_ids": torch.tensor([20, 10, 11])},
            ]]],
            "labels": ["artist"],
            "weights": [1.0],
            "base_conditioning": [[
                torch.tensor([[[0.0, 0.0]]]),
                {"t5xxl_ids": torch.tensor([10, 11])},
            ]],
        }

        advanced_options, seeds_used = AnimaArtistOptions().build(
            start_block=0,
            end_block=-1,
            start_percent=0.0,
            end_percent=1.0,
            normalize_weights=True,
            artist_anchor_q=True,
            anchor_seed_list="",
            anchor_seeds_count=2,
        )
        patched_model, _ = AnimaArtistAdapterMixer().patch(
            model,
            artist_pack,
            strength=1.0,
            normalize_weights=True,
            alignment_mode=ALIGN_BASE_ANCHORED,
            enabled=True,
            apply_to_uncond=False,
            uncond_strength=0.0,
            advanced_options=advanced_options,
        )

        patch_path = "diffusion_model.blocks.0.cross_attn.forward"
        attention_patch = patched_model.object_patches[patch_path]
        self.assertEqual(seeds_used, "987654321,123456789")
        self.assertFalse(advanced_options["anchor_seed_list_is_manual"])
        self.assertEqual(
            attention_patch.state["anchor_seed_list"],
            [987654321, 123456789],
        )
        self.assertEqual(randbelow.call_count, 2)

    def test_adapter_q_anchor_reuses_fixed_cond_q_and_preserves_uncond_q(self):
        dm = FakeAnchorAdapterModel()
        model = FakeModelPatcher(dm)
        artist_pack = {
            "conditionings": [[[
                torch.tensor([[[1.0, 2.0]]]),
                {"t5xxl_ids": torch.tensor([20, 10, 11])},
            ]]],
            "labels": ["artist"],
            "weights": [1.0],
            "base_conditioning": [[
                torch.tensor([[[0.0, 0.0]]]),
                {"t5xxl_ids": torch.tensor([10, 11])},
            ]],
        }
        patched_model, _ = AnimaArtistAdapterMixer().patch(
            model,
            artist_pack,
            strength=1.0,
            normalize_weights=True,
            alignment_mode=ALIGN_BASE_ANCHORED,
            enabled=True,
            apply_to_uncond=False,
            uncond_strength=0.0,
            advanced_options={
                "artist_anchor_q": True,
                "anchor_seed_list": "123456789",
                "anchor_seed_list_is_manual": True,
                "anchor_user_blend": 0.0,
                "anchor_deep_layer_threshold": -1,
                "stabilizer_end_percent": 1.0,
            },
        )

        patch_path = "diffusion_model.blocks.0.cross_attn.forward"
        attention_patch = patched_model.object_patches[patch_path]
        wrapper = patched_model.model_options["model_function_wrapper"]
        base_context = torch.tensor([
            [[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
            [[7.0, 8.0], [9.0, 10.0], [0.0, 0.0]],
        ])

        def apply_model(input_tensor, timestep, **c):
            return attention_patch(
                input_tensor,
                c["c_crossattn"],
                transformer_options=c.get("transformer_options"),
            )

        def run(user_x):
            return wrapper(apply_model, {
                "input": user_x,
                "timestep": torch.ones((2,)),
                "c": {
                    "c_crossattn": base_context,
                    "transformer_options": {"cond_or_uncond": [0, 1]},
                },
                "cond_or_uncond": [0, 1],
            })

        first_user_x = torch.full((2, 2, 2), 10.0)
        first_user_x[1].fill_(20.0)
        first_output = run(first_user_x)
        self.assertEqual(len(dm.blocks[0].cross_attn.calls), 2)

        second_user_x = torch.full((2, 2, 2), 30.0)
        second_user_x[1].fill_(40.0)
        second_output = run(second_user_x)
        self.assertEqual(len(dm.blocks[0].cross_attn.calls), 3)

        self.assertTrue(torch.equal(first_output[0], second_output[0]))
        self.assertFalse(torch.equal(first_output[0], first_user_x[0]))
        self.assertTrue(torch.equal(first_output[1], first_user_x[1]))
        self.assertTrue(torch.equal(second_output[1], second_user_x[1]))
        self.assertEqual(attention_patch.state["anchor_seed_list"], [123456789])

        first_mixed_context = dm.blocks[0].cross_attn.calls[1]["context"]
        second_mixed_context = dm.blocks[0].cross_attn.calls[2]["context"]
        self.assertTrue(torch.equal(first_mixed_context, second_mixed_context))
        self.assertFalse(torch.equal(first_mixed_context, base_context))
        anchor_context = dm.blocks[0].cross_attn.calls[0]["context"]
        self.assertTrue(torch.equal(anchor_context[0], first_mixed_context[0]))
        self.assertFalse(torch.equal(anchor_context[0], base_context[0]))

    def test_adapter_warm_cache_reuses_sigma_trajectory_and_invalidates_context(self):
        dm = FakeAnchorAdapterModel()
        model = FakeModelPatcher(dm)
        artist_pack = {
            "conditionings": [[[
                torch.tensor([[[1.0, 2.0]]]),
                {"t5xxl_ids": torch.tensor([20, 10, 11])},
            ]]],
            "labels": ["artist"],
            "weights": [1.0],
            "base_conditioning": [[
                torch.tensor([[[0.0, 0.0]]]),
                {"t5xxl_ids": torch.tensor([10, 11])},
            ]],
        }
        patched_model, _ = AnimaArtistAdapterMixer().patch(
            model,
            artist_pack,
            strength=1.0,
            normalize_weights=True,
            alignment_mode=ALIGN_BASE_ANCHORED,
            enabled=True,
            apply_to_uncond=False,
            uncond_strength=0.0,
            advanced_options={
                "artist_anchor_q": True,
                "anchor_seed_list": "42,12345",
                "anchor_seed_list_is_manual": True,
                "anchor_user_blend": 0.0,
                "anchor_deep_layer_threshold": -1,
                "stabilizer_end_percent": 1.0,
                "anchor_refresh_mode": "warm_cache",
                "anchor_cache_points": 3,
                "anchor_keyframe_mode": ANCHOR_KEYFRAME_ADAPTIVE_Q,
            },
        )

        patch_path = "diffusion_model.blocks.0.cross_attn.forward"
        attention_patch = patched_model.object_patches[patch_path]
        wrapper = patched_model.model_options["model_function_wrapper"]
        base_context = torch.tensor([
            [[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
            [[7.0, 8.0], [9.0, 10.0], [0.0, 0.0]],
        ])

        def apply_model(input_tensor, timestep, **c):
            return attention_patch(
                input_tensor,
                c["c_crossattn"],
                transformer_options=c.get("transformer_options"),
            )

        def run(sigma, value, context=base_context):
            user_x = torch.full((2, 2, 2), float(value))
            user_x[1].fill_(float(value) + 100.0)
            output = wrapper(apply_model, {
                "input": user_x,
                "timestep": torch.full((2,), float(sigma)),
                "c": {
                    "c_crossattn": context,
                    "transformer_options": {"cond_or_uncond": [0, 1]},
                },
                "cond_or_uncond": [0, 1],
            })
            return user_x, output

        first_outputs = []
        with mock.patch("anima_mixer.anchor.context_fingerprint") as fingerprint:
            for sigma, value in zip((1.0, 0.5, 0.1), (10.0, 20.0, 30.0)):
                _user_x, output = run(sigma, value)
                first_outputs.append(output)
        fingerprint.assert_not_called()

        # Two manual seeds produce two anchor calls plus one user call per sigma.
        self.assertEqual(len(dm.blocks[0].cross_attn.calls), 9)
        trajectory = attention_patch.state["_anchor_trajectory"]
        self.assertFalse(trajectory["ready"])

        second_outputs = []
        for sigma, value in zip(
            (1.0, 0.75, 0.25, 0.1),
            (40.0, 50.0, 60.0, 70.0),
        ):
            user_x, output = run(sigma, value)
            second_outputs.append(output)
            self.assertTrue(torch.equal(output[1], user_x[1]))

        # The second run adds only its four user calls, including interpolated
        # 0.75/0.25 sigmas; no anchor model pass runs.
        self.assertEqual(len(dm.blocks[0].cross_attn.calls), 13)
        trajectory = attention_patch.state["_anchor_trajectory"]
        self.assertTrue(trajectory["ready"])
        self.assertEqual(trajectory["keyframe_mode"], ANCHOR_KEYFRAME_ADAPTIVE_Q)
        self.assertEqual(len(trajectory["frames"]), 3)
        self.assertGreater(trajectory["bytes"], 0)
        self.assertTrue(all(
            hidden.device.type == "cpu"
            for frame in trajectory["frames"]
            for hidden in frame["layers"].values()
        ))
        for second in second_outputs:
            self.assertTrue(
                torch.allclose(
                    first_outputs[0][0],
                    second[0],
                    rtol=1e-6,
                    atol=1e-6,
                ),
            )

        changed_context = base_context.clone()
        changed_context[0, 0, 0] = 5.0
        run(1.0, 80.0, context=changed_context)
        self.assertEqual(len(dm.blocks[0].cross_attn.calls), 16)
        self.assertFalse(attention_patch.state["_anchor_trajectory"]["ready"])


class EmbeddingLengthRegressionTests(unittest.TestCase):
    def test_expanding_cfg_splits_conditions_and_restores_batch_order(self):
        for markers in ([0, 1], [1, 0], [0, 1, 0], [1, 0, 1], [0, 0, 1]):
            for context_key in ("c_crossattn", "context"):
                with self.subTest(markers=markers, context_key=context_key):
                    dm = FakeAdapterModel()
                    state = make_state(dm)
                    batch_size = len(markers) * 2
                    base = torch.arange(batch_size * 8, dtype=torch.float32).reshape(
                        batch_size, 4, 2,
                    ) + 1.0
                    user_x = torch.arange(batch_size * 2, dtype=torch.float32).reshape(
                        batch_size, 2,
                    )
                    timestep = torch.ones(batch_size)
                    extra = torch.arange(batch_size, dtype=torch.float32).reshape(-1, 1)
                    singleton = torch.ones((1, 2))
                    sigmas = torch.ones(2)
                    sample_sigmas = torch.arange(batch_size, dtype=torch.float32)
                    patches = {"test": [sample_sigmas]}
                    uuids = [f"condition-{index}" for index in range(len(markers))]
                    observed = []

                    def apply_model(input_tensor, model_timestep, **c):
                        return input_tensor + c["y"] + c[context_key].sum((1, 2))[:, None]

                    def previous_wrapper(model_function, options):
                        observed.append(options)
                        return model_function(
                            options["input"], options["timestep"], **options["c"],
                        )

                    wrapper = make_adapter_embedding_wrapper(state, previous_wrapper)
                    options = {
                        "input": user_x,
                        "timestep": timestep,
                        "cond_or_uncond": markers,
                        "c": {
                            context_key: base,
                            "y": extra,
                            "control": {"input": [extra, None], "output": (extra,)},
                            "singleton": singleton,
                            "scalar": torch.tensor(3.0),
                            "transformer_options": {
                                "cond_or_uncond": markers,
                                "uuids": uuids,
                                "sigmas": sigmas,
                                "sample_sigmas": sample_sigmas,
                                "patches": patches,
                            },
                        },
                    }
                    output = wrapper(apply_model, options)
                    mask = [marker == 0 for marker in markers for _ in range(2)]
                    artist_sum = build_artist_embedding_sum(state, base)
                    expected_context = mix_projected_context(
                        align_base_context(base, state["alignment_plan"]),
                        artist_sum, [1.0] * batch_size, mask, fallback_base=base,
                    )
                    expected = user_x + extra
                    for row, should_mix in enumerate(mask):
                        context = expected_context[row] if should_mix else base[row]
                        expected[row] += context.sum()
                    self.assertTrue(torch.equal(output, expected))
                    self.assertEqual(len(observed), 2)
                    self.assertFalse(state["_embedding_mixer_failed"])
                    for group, should_mix in zip(observed, (True, False)):
                        chunks = [
                            index for index, marker in enumerate(markers)
                            if (marker == 0) == should_mix
                        ]
                        rows = [row for chunk in chunks for row in (2 * chunk, 2 * chunk + 1)]
                        group_c = group["c"]
                        group_transformer = group_c["transformer_options"]
                        self.assertTrue(torch.equal(group["input"], user_x[rows]))
                        self.assertTrue(torch.equal(group["timestep"], timestep[rows]))
                        self.assertTrue(torch.equal(group_c["y"], extra[rows]))
                        self.assertTrue(torch.equal(group_c["control"]["input"][0], extra[rows]))
                        self.assertTrue(torch.equal(group_c["control"]["output"][0], extra[rows]))
                        self.assertIsNone(group_c["control"]["input"][1])
                        self.assertIs(group_c["singleton"], singleton)
                        self.assertEqual(group_c["scalar"].item(), 3.0)
                        expected_markers = [markers[index] for index in chunks]
                        self.assertEqual(group["cond_or_uncond"], expected_markers)
                        self.assertEqual(group_transformer["cond_or_uncond"], expected_markers)
                        self.assertEqual(group_transformer["uuids"], [uuids[index] for index in chunks])
                        self.assertIs(group_transformer["sigmas"], sigmas)
                        self.assertIs(group_transformer["sample_sigmas"], sample_sigmas)
                        self.assertIs(group_transformer["patches"], patches)
                        self.assertEqual(group_c[context_key].shape[1], 5 if should_mix else 4)
                        if not should_mix:
                            self.assertTrue(torch.equal(group_c[context_key], base[rows]))
                    self.assertIs(options["c"][context_key], base)
                    self.assertIs(options["c"]["transformer_options"]["cond_or_uncond"], markers)

    def test_native_512_boundary_only_splits_expanding_batches(self):
        for base_count in (504, 505, 506, 512, 525, 651, 1024):
            with self.subTest(base_count=base_count):
                dm = FakeAdapterModel()
                state = make_state(dm)
                state["base_ids"] = torch.arange(base_count) + 10
                state["ids_list"] = [
                    torch.cat((torch.arange(prefix) + 10000, state["base_ids"]))
                    for prefix in (7, 5)
                ]
                state["alignment_plan"] = build_base_anchored_plan(
                    state["base_ids"], state["ids_list"],
                )
                context_length = max(512, base_count)
                base = torch.ones((2, context_length, 2))
                observed = []

                def apply_model(input_tensor, timestep, **c):
                    observed.append(c["c_crossattn"])
                    return input_tensor

                wrapper = make_adapter_embedding_wrapper(state, None)
                user_x = torch.arange(2, dtype=torch.float32).reshape(2, 1)
                output = wrapper(apply_model, {
                    "input": user_x,
                    "timestep": torch.ones(2),
                    "c": {"c_crossattn": base},
                    "cond_or_uncond": [0, 1],
                })
                self.assertTrue(torch.equal(output, user_x))
                self.assertFalse(state["_embedding_mixer_failed"])
                expanding = base_count + 7 > context_length
                self.assertEqual(len(observed), 2 if expanding else 1)
                if expanding:
                    self.assertEqual(observed[0].shape[1], base_count + 7)
                    self.assertTrue(torch.equal(observed[1], base[1:]))
                else:
                    self.assertEqual(observed[0].shape[1], 512)
                    self.assertTrue(torch.equal(observed[0][1], base[1]))

    def test_long_positive_and_short_negative_calls_do_not_disable_mixer(self):
        for negative_first in (True, False):
            with self.subTest(negative_first=negative_first):
                dm = FakeAdapterModel()
                state = make_state(dm)
                state["base_ids"] = torch.arange(525) + 10
                state["ids_list"] = [
                    torch.cat((torch.arange(prefix) + 10000, state["base_ids"]))
                    for prefix in (7, 5)
                ]
                state["alignment_plan"] = build_base_anchored_plan(
                    state["base_ids"], state["ids_list"],
                )
                observed = []

                def apply_model(input_tensor, timestep, **c):
                    observed.append(c["c_crossattn"])
                    return input_tensor

                wrapper = make_adapter_embedding_wrapper(state, None)
                positive = torch.ones((1, 525, 2))
                negative = torch.full((1, 512, 2), 7.0)
                markers = [1, 0, 1, 0] if negative_first else [0, 1, 0, 1]
                for marker in markers:
                    source = negative if marker else positive
                    before_builds = len(dm.seen_ids)
                    user_x = torch.zeros((1, 1))
                    output = wrapper(apply_model, {
                        "input": user_x,
                        "timestep": torch.ones(1),
                        "c": {"c_crossattn": source},
                        "cond_or_uncond": [marker],
                    })
                    self.assertIs(output, user_x)
                    self.assertFalse(state["_embedding_mixer_failed"])
                    if marker:
                        self.assertIs(observed[-1], negative)
                        self.assertEqual(len(dm.seen_ids), before_builds)
                    else:
                        self.assertEqual(observed[-1].shape[1], 532)
                        self.assertFalse(torch.equal(observed[-1][:, :525], positive))
                self.assertEqual(len(dm.seen_ids), 2)

    def test_successful_paths_keep_exact_context_and_single_dispatch(self):
        for mode in (ALIGN_BASE_ANCHORED, ALIGN_SHARED_BASE_IDS):
            for markers in ([0, 1], [1, 0], [0], [1]):
                with self.subTest(mode=mode, markers=markers):
                    state = make_state(FakeAdapterModel(), mode)
                    batch_size = len(markers)
                    base = torch.arange(batch_size * 12, dtype=torch.float32).reshape(
                        batch_size, 6, 2,
                    ) + 1.0
                    observed = []

                    def apply_model(input_tensor, timestep, **c):
                        observed.append(c["c_crossattn"])
                        return input_tensor

                    wrapper = make_adapter_embedding_wrapper(state, None)
                    user_x = torch.ones((batch_size, 1))
                    output = wrapper(apply_model, {
                        "input": user_x,
                        "timestep": torch.ones(batch_size),
                        "c": {"c_crossattn": base},
                        "cond_or_uncond": markers,
                    })
                    artist_sum = build_artist_embedding_sum(state, base)
                    projection = (
                        align_base_context(base, state["alignment_plan"])
                        if mode == ALIGN_BASE_ANCHORED else base
                    )
                    expected = mix_projected_context(
                        projection, artist_sum, [1.0] * batch_size,
                        [marker == 0 for marker in markers], fallback_base=base,
                    )
                    self.assertIs(output, user_x)
                    self.assertEqual(len(observed), 1)
                    self.assertTrue(torch.equal(observed[0], expected))
                    self.assertFalse(state["_embedding_mixer_failed"])

    def test_shared_base_ids_uncond_padding_is_not_changed(self):
        state = make_state(FakeAdapterModel(), ALIGN_SHARED_BASE_IDS)
        state["base_ids"] = torch.arange(5)
        source = torch.ones((1, 4, 2))
        observed = []

        def apply_model(input_tensor, timestep, **c):
            observed.append(c["c_crossattn"])
            return input_tensor

        make_adapter_embedding_wrapper(state, None)(apply_model, {
            "input": torch.zeros((1, 1)),
            "timestep": torch.ones(1),
            "c": {"c_crossattn": source},
            "cond_or_uncond": [1],
        })
        self.assertEqual(observed[0].shape[1], 5)
        self.assertTrue(torch.equal(observed[0][:, :4], source))
        self.assertEqual(observed[0][:, 4:].count_nonzero().item(), 0)
        self.assertFalse(state["_embedding_mixer_failed"])

    def test_split_anchor_reuse_and_warm_cache_ignore_negative_calls(self):
        for refresh_mode in ("once", "warm_cache"):
            for markers in ([0, 1], [1, 0], [0, 1, 0]):
                with self.subTest(refresh_mode=refresh_mode, markers=markers):
                    dm = FakeAnchorAdapterModel()
                    state = make_state(dm)
                    state.update({
                        "artist_anchor_q": True,
                        "anchor_seed_list": [42],
                        "anchor_refresh_mode": refresh_mode,
                        "anchor_cache_points": 3,
                        "stabilizer_min_sigma": 0.0,
                        "_identity_context_signature": True,
                    })
                    attention_patch = make_adapter_anchor_q_forward_patch(
                        dm.blocks[0].cross_attn.forward, state, 0,
                    )
                    wrapper = make_adapter_embedding_wrapper(
                        state, make_sigma_capture(state, None),
                    )
                    batch_size = len(markers)
                    base = torch.ones((batch_size, 4, 2))
                    anchor_calls = []

                    def apply_model(input_tensor, timestep, **c):
                        if state.get("_in_anchor_run", False):
                            anchor_calls.append(c["c_crossattn"])
                        return attention_patch(
                            input_tensor, c["c_crossattn"],
                            transformer_options=c.get("transformer_options"),
                        )

                    def run(sigma, value):
                        user_x = torch.full((batch_size, 2, 2), float(value))
                        output = wrapper(apply_model, {
                            "input": user_x,
                            "timestep": torch.full((batch_size,), sigma),
                            "c": {
                                "c_crossattn": base,
                                "transformer_options": {"cond_or_uncond": markers},
                            },
                            "cond_or_uncond": markers,
                        })
                        for row, marker in enumerate(markers):
                            if marker:
                                self.assertTrue(torch.equal(output[row], user_x[row]))
                            else:
                                self.assertFalse(torch.equal(output[row], user_x[row]))
                        trajectory = state.get("_anchor_trajectory")
                        cache = state["_anchor_cache"]
                        negative = base[:1]
                        negative_x = torch.full((1, 2, 2), 99.0)
                        negative_output = wrapper(apply_model, {
                            "input": negative_x,
                            "timestep": torch.tensor([sigma]),
                            "c": {
                                "c_crossattn": negative,
                                "transformer_options": {"cond_or_uncond": [1]},
                            },
                            "cond_or_uncond": [1],
                        })
                        self.assertTrue(torch.equal(negative_output, negative_x))
                        self.assertIs(state.get("_anchor_trajectory"), trajectory)
                        self.assertIs(state["_anchor_cache"], cache)
                        self.assertFalse(state["_embedding_mixer_failed"])
                        return output

                    first_outputs = [run(sigma, value) for sigma, value in (
                        (1.0, 10.0), (0.5, 20.0), (0.1, 30.0),
                    )]
                    self.assertEqual(len(anchor_calls), 3 if refresh_mode == "warm_cache" else 1)
                    if refresh_mode == "warm_cache":
                        trajectory = state["_anchor_trajectory"]
                        for sigma, value in ((1.0, 40.0), (0.5, 50.0), (0.1, 60.0)):
                            output = run(sigma, value)
                            for row, marker in enumerate(markers):
                                if marker == 0:
                                    self.assertTrue(torch.equal(output[row], first_outputs[0][row]))
                        self.assertIs(state["_anchor_trajectory"], trajectory)
                        self.assertTrue(trajectory["ready"])
                        self.assertEqual(len(anchor_calls), 3)



if __name__ == "__main__":
    unittest.main()
