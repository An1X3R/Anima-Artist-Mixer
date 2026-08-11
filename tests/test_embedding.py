import unittest
from unittest import mock

import torch

from anima_mixer.anchor import (
    _evenly_spaced_positions,
    _finalize_anchor_trajectory,
    make_sigma_capture,
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
from anima_mixer.nodes_embedding import AnimaArtistAdapterMixer
from anima_mixer.nodes_ui import AnimaArtistOptions
from anima_mixer.patching import (
    _make_mixer_cleanup_callback,
    _rebind_mixer_object_patches,
    begin_mixer_execution,
    call_with_mixer_owner,
    clear_mixer_run_state,
    reset_run_state,
)
from anima_mixer.adapter_anchor import make_adapter_anchor_q_forward_patch


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
        self.callbacks = {
            "on_clone": {},
            "on_pre_run": {},
            "on_cleanup": {},
        }

    def get_model_object(self, name):
        if name != "diffusion_model":
            raise KeyError(name)
        return self.diffusion_model

    def clone(self):
        cloned = FakeModelPatcher(self.diffusion_model)
        cloned.model_options = dict(self.model_options)
        cloned.object_patches = dict(self.object_patches)
        cloned.callbacks = {
            call_type: {
                key: list(callbacks)
                for key, callbacks in groups.items()
            }
            for call_type, groups in self.callbacks.items()
        }
        for callback in self.get_all_callbacks("on_clone"):
            callback(self, cloned)
        return cloned

    def set_model_unet_function_wrapper(self, wrapper):
        self.model_options["model_function_wrapper"] = wrapper

    def add_object_patch(self, path, value):
        self.object_patches[path] = value

    def get_callbacks(self, call_type, key):
        return self.callbacks.get(call_type, {}).get(key, [])

    def get_all_callbacks(self, call_type):
        return [
            callback
            for callbacks in self.callbacks.get(call_type, {}).values()
            for callback in callbacks
        ]

    def add_callback_with_key(self, call_type, key, callback):
        self.callbacks.setdefault(call_type, {}).setdefault(key, []).append(callback)

    def remove_callbacks_with_key(self, call_type, key):
        self.callbacks.setdefault(call_type, {}).pop(key, None)

    def pre_run(self):
        for callback in self.get_all_callbacks("on_pre_run"):
            callback(self)

    def cleanup(self):
        for callback in self.get_all_callbacks("on_cleanup"):
            callback(self)


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
    def test_large_cache_positions_do_not_cross_upper_bound(self):
        length = 19_808_256
        count = 256

        positions = _evenly_spaced_positions(length, count, torch.device("cpu"))
        expected = torch.linspace(
            0,
            length - 1,
            steps=count,
            dtype=torch.float64,
        ).round().to(torch.long)

        self.assertTrue(torch.equal(positions, expected))
        self.assertEqual(positions[0].item(), 0)
        self.assertEqual(positions[-1].item(), length - 1)
        self.assertTrue(torch.all(positions[1:] >= positions[:-1]))

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
    def test_cuda_runtime_error_is_not_silently_downgraded(self):
        class FailingAdapter(FakeAdapterModel):
            def preprocess_text_embeds(self, raw, ids, t5xxl_weights=None):
                raise RuntimeError("CUDA error: an illegal memory access was encountered")

        state = make_state(FailingAdapter(), ALIGN_SHARED_BASE_IDS)
        wrapper = make_adapter_embedding_wrapper(state, None)
        options = {
            "input": torch.zeros((1, 1)),
            "timestep": torch.ones((1,)),
            "c": {
                "c_crossattn": torch.zeros((1, 3, 2)),
                "transformer_options": {"cond_or_uncond": [0]},
            },
            "cond_or_uncond": [0],
        }

        with self.assertRaisesRegex(RuntimeError, "illegal memory access"):
            wrapper(lambda *_args, **_kwargs: torch.zeros((1, 1)), options)

        self.assertEqual(state["_artist_embedding_cache"], {})
        self.assertFalse(state["_run_active"])

    def test_adapter_async_cuda_error_is_surfaced_before_main_model(self):
        state = make_state(FakeAdapterModel(), ALIGN_SHARED_BASE_IDS)
        wrapper = make_adapter_embedding_wrapper(state, None)
        options = {
            "input": torch.zeros((1, 1)),
            "timestep": torch.ones((1,)),
            "c": {
                "c_crossattn": torch.zeros((1, 3, 2)),
                "transformer_options": {"cond_or_uncond": [0]},
            },
            "cond_or_uncond": [0],
        }

        with mock.patch(
            "anima_mixer.embedding.torch.cuda.is_available",
            return_value=True,
        ), mock.patch(
            "anima_mixer.embedding.torch.cuda.synchronize",
            side_effect=RuntimeError("CUDA error: launch failure"),
        ) as synchronize:
            with self.assertRaisesRegex(RuntimeError, "launch failure"):
                wrapper(lambda *_args, **_kwargs: torch.zeros((1, 1)), options)

        # The first call is the post-Adapter boundary; abort cleanup performs a
        # second best-effort device synchronization while preserving the first
        # CUDA exception.
        self.assertEqual(synchronize.call_count, 2)
        self.assertEqual(synchronize.call_args_list[0], mock.call(None))
        self.assertEqual(state["_artist_embedding_cache"], {})
        self.assertFalse(state["_run_active"])

    def test_nonfinite_adapter_embedding_aborts_instead_of_sampling(self):
        class NonFiniteAdapter(FakeAdapterModel):
            def preprocess_text_embeds(self, raw, ids, t5xxl_weights=None):
                output = super().preprocess_text_embeds(
                    raw,
                    ids,
                    t5xxl_weights=t5xxl_weights,
                )
                output[..., 0] = float("nan")
                return output

        state = make_state(NonFiniteAdapter(), ALIGN_SHARED_BASE_IDS)
        wrapper = make_adapter_embedding_wrapper(state, None)
        options = {
            "input": torch.zeros((1, 1)),
            "timestep": torch.ones((1,)),
            "c": {
                "c_crossattn": torch.zeros((1, 3, 2)),
                "transformer_options": {"cond_or_uncond": [0]},
            },
            "cond_or_uncond": [0],
        }

        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            wrapper(lambda *_args, **_kwargs: torch.zeros((1, 1)), options)

        self.assertEqual(state["_artist_embedding_cache"], {})
        self.assertFalse(state["_run_active"])

    def test_interrupt_in_begin_boundary_clears_mixer_state(self):
        dm = FakeAdapterModel()
        state = make_state(dm, ALIGN_BASE_ANCHORED)
        state.update({
            "_artist_embedding_cache": {"stale": torch.ones(1)},
            "_mixed_context_cache": {"mixed": torch.ones(1)},
            "_run_active": True,
            "_run_call_count": 2,
        })

        class Interrupt(BaseException):
            pass

        original = Interrupt("begin-stop")
        wrapper = make_adapter_embedding_wrapper(state, None)
        options = {
            "input": torch.zeros((1, 1)),
            "timestep": torch.ones((1,)),
            "c": {},
        }
        with mock.patch(
            "anima_mixer.embedding.begin_mixer_execution",
            side_effect=original,
        ):
            with self.assertRaises(Interrupt) as raised:
                wrapper(lambda *_args, **_kwargs: None, options)

        self.assertIs(raised.exception, original)
        self.assertEqual(state["_artist_embedding_cache"], {})
        self.assertIsNone(state["_mixed_context_cache"])
        self.assertFalse(state["_run_active"])

    def test_interrupt_in_no_context_early_path_clears_mixer_state(self):
        dm = FakeAdapterModel()
        state = make_state(dm, ALIGN_SHARED_BASE_IDS)
        state.update({
            "_artist_embedding_cache": {"stale": torch.ones(1)},
            "_mixed_context_cache": {"mixed": torch.ones(1)},
            "_run_active": True,
        })

        class Interrupt(BaseException):
            pass

        original = Interrupt("no-context-stop")
        wrapper = make_adapter_embedding_wrapper(state, None)

        def underlying(*_args, **_kwargs):
            raise original

        with self.assertRaises(Interrupt) as raised:
            wrapper(underlying, {
                "input": torch.zeros((1, 1)),
                "timestep": torch.ones((1,)),
                "c": {},
            })

        self.assertIs(raised.exception, original)
        self.assertEqual(state["_artist_embedding_cache"], {})
        self.assertIsNone(state["_mixed_context_cache"])
        self.assertFalse(state["_run_active"])

    def test_interrupt_in_failed_fast_path_clears_mixer_state(self):
        state = make_state(FakeAdapterModel(), ALIGN_SHARED_BASE_IDS)
        state.update({
            "_embedding_mixer_failed": True,
            "_artist_embedding_cache": {"stale": torch.ones(1)},
            "_mixed_context_cache": {"mixed": torch.ones(1)},
            "_run_active": True,
        })

        class Interrupt(BaseException):
            pass

        original = Interrupt("failed-fast-stop")
        wrapper = make_adapter_embedding_wrapper(state, None)

        def underlying(*_args, **_kwargs):
            raise original

        with self.assertRaises(Interrupt) as raised:
            wrapper(underlying, {
                "input": torch.zeros((1, 1)),
                "timestep": torch.ones((1,)),
                "c": {"c_crossattn": torch.zeros((1, 2, 2))},
            })

        self.assertIs(raised.exception, original)
        self.assertEqual(state["_artist_embedding_cache"], {})
        self.assertIsNone(state["_mixed_context_cache"])
        self.assertFalse(state["_run_active"])

    def test_interrupt_in_multigpu_dispatch_clears_mixer_state(self):
        dm = FakeAdapterModel()
        state = make_state(dm, ALIGN_SHARED_BASE_IDS)
        state.update({
            "_artist_embedding_cache": {"stale": torch.ones(1)},
            "_mixed_context_cache": {"mixed": torch.ones(1)},
            "_run_active": True,
        })

        class Interrupt(BaseException):
            pass

        original = Interrupt("multigpu-stop")
        wrapper = make_adapter_embedding_wrapper(state, None)

        class Patcher:
            model_options = {}

        class Runner:
            current_patcher = Patcher()

            def apply_model(self, *_args, **_kwargs):
                raise original

        Runner.current_patcher.model_options = {
            "model_function_wrapper": lambda *_args, **_kwargs: (_ for _ in ()).throw(original),
        }
        with self.assertRaises(Interrupt) as raised:
            wrapper(Runner().apply_model, {
                "input": torch.zeros((1, 1)),
                "timestep": torch.ones((1,)),
                "c": {
                    "c_crossattn": torch.zeros((1, 2, 2)),
                    "transformer_options": {
                        "multigpu_thread_device": torch.device("cpu"),
                    },
                },
            })

        self.assertIs(raised.exception, original)
        self.assertEqual(state["_artist_embedding_cache"], {})
        self.assertIsNone(state["_mixed_context_cache"])
        self.assertFalse(state["_run_active"])

    def test_abort_then_silent_t5_weight_change_rebuilds_artist_embedding(self):
        dm = FakeAdapterModel()
        state = make_state(dm, ALIGN_BASE_ANCHORED)
        state["t5_weights_list"] = [torch.ones(4), torch.ones(5)]
        state["user_weights"] = [1.0, 1.0]
        state["_cache_namespace"] = ("same-node",)
        state["_artist_embedding_cache"] = {}
        ref = torch.zeros((1, 5, 2))
        first = build_artist_embedding_sum(state, ref)
        first_signature = state["_runtime_input_signature"]

        class Interrupt(BaseException):
            pass

        clear_mixer_run_state(state, interrupted=True)
        state["t5_weights_list"][0].data.mul_(2.0)
        self.assertEqual(int(state["t5_weights_list"][0]._version), 0)

        begin_mixer_execution(
            state,
            None,
            torch.ones((1,)),
            explicit_run_start=False,
        )
        second = build_artist_embedding_sum(state, ref)
        self.assertNotEqual(first_signature, state["_runtime_input_signature"])
        self.assertFalse(torch.equal(first, second))
        self.assertEqual(len(dm.seen_ids), 4)

    def test_abort_then_replaced_prompt_tensors_rebuilds_alignment_at_same_sigma(self):
        dm = FakeAdapterModel()
        state = make_state(dm, ALIGN_BASE_ANCHORED)
        state["_cache_namespace"] = ("same-wrapper",)
        observed = []

        def apply_model(input_tensor, timestep, **c):
            observed.append(c["c_crossattn"])
            return input_tensor

        wrapper = make_adapter_embedding_wrapper(state, None)

        def run(context):
            wrapper(apply_model, {
                "input": torch.zeros((1, 1)),
                "timestep": torch.ones((1,)),
                "c": {
                    "c_crossattn": context,
                    "transformer_options": {"cond_or_uncond": [0]},
                },
                "cond_or_uncond": [0],
            })

        old_raw = state["raws"][0]
        run(torch.zeros((1, 3, 2)))
        old_plan = state["alignment_plan"]
        old_signature = state["_runtime_input_signature"]
        old_mixed = observed[-1]
        state["_anchor_cache"] = {0: torch.ones((1, 1, 2))}
        state["_anchor_cache_key"] = ("old-anchor",)
        state["_anchor_trajectory"] = {"ready": True, "frames": []}
        state["individuals"] = [torch.ones((1, 4, 2))]
        state["real_lens"] = [4]
        state["_ctx_fp_memo"] = {123: ("old",)}

        clear_mixer_run_state(state, interrupted=True)
        # ComfyUI may still deliver its normal cleanup callback while the
        # interrupted wrapper unwinds.  It must not erase the abort fingerprint
        # used to compare the next prompt.
        _make_mixer_cleanup_callback(state)(None)
        state["raws"] = [
            torch.tensor([[[7.0, 8.0]]]),
            torch.tensor([[[9.0, 10.0]]]),
        ]
        state["base_ids"] = torch.tensor([10, 11, 12, 13])
        state["ids_list"] = [
            torch.tensor([301, 10, 11, 12, 13]),
            torch.tensor([401, 402, 10, 11, 12, 13]),
        ]
        state["t5_weights_list"] = [None, None]

        # The first sigma is deliberately identical to the interrupted run.
        new_context = torch.zeros((1, 4, 2))
        run(new_context)

        expected_plan = build_base_anchored_plan(
            state["base_ids"], state["ids_list"]
        )
        self.assertIsNot(state["alignment_plan"], old_plan)
        self.assertEqual(state["alignment_plan"], expected_plan)
        self.assertNotEqual(state["_runtime_input_signature"], old_signature)
        self.assertEqual(
            state["_runtime_input_signature"][0][0][0][0],
            id(state["raws"][0]),
        )
        self.assertNotIn(id(old_raw), state["_execution_value_fp_memo"])
        self.assertIn(id(new_context), state["_execution_value_fp_memo"])
        self.assertEqual(state["_ctx_fp_memo"], {})
        self.assertEqual(state["_anchor_cache"], {})
        self.assertIsNone(state["_anchor_cache_key"])
        self.assertIsNone(state["_anchor_trajectory"])
        self.assertIsNone(state["individuals"])
        self.assertIsNone(state["real_lens"])
        self.assertIsNot(observed[-1], old_mixed)
        self.assertEqual(tuple(observed[-1].shape), (1, 6, 2))
        self.assertIs(state["_mixed_context_cache"]["mixed"], observed[-1])
        self.assertEqual(len(dm.seen_ids), 4)

    def test_completed_run_then_reused_wrapper_rebuilds_changed_alignment(self):
        dm = FakeAdapterModel()
        state = make_state(dm, ALIGN_BASE_ANCHORED)
        state["_cache_namespace"] = ("same-wrapper-clean",)
        wrapper = make_adapter_embedding_wrapper(state, None)

        def apply_model(input_tensor, _timestep, **_kwargs):
            return input_tensor

        options = {
            "input": torch.zeros((1, 1)),
            "timestep": torch.ones((1,)),
            "c": {
                "c_crossattn": torch.zeros((1, 3, 2)),
                "transformer_options": {"cond_or_uncond": [0]},
            },
            "cond_or_uncond": [0],
        }
        wrapper(apply_model, options)
        old_plan = state["alignment_plan"]
        _make_mixer_cleanup_callback(state)(None)

        state["raws"] = [
            torch.tensor([[[7.0, 8.0]]]),
            torch.tensor([[[9.0, 10.0]]]),
        ]
        state["base_ids"] = torch.tensor([10, 11, 12, 13])
        state["ids_list"] = [
            torch.tensor([301, 10, 11, 12, 13]),
            torch.tensor([401, 402, 10, 11, 12, 13]),
        ]
        wrapper(apply_model, {
            **options,
            "c": {
                **options["c"],
                "c_crossattn": torch.zeros((1, 4, 2)),
            },
        })

        self.assertIsNot(state["alignment_plan"], old_plan)
        self.assertEqual(
            state["alignment_plan"],
            build_base_anchored_plan(state["base_ids"], state["ids_list"]),
        )

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

    def test_wrapper_clears_live_context_on_next_sampling_run(self):
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

        def call(sigma):
            wrapper(apply_model, {
                "input": torch.zeros((1, 1)),
                "timestep": torch.full((1,), sigma),
                "c": {
                    "c_crossattn": base,
                    "transformer_options": {"cond_or_uncond": [0]},
                },
                "cond_or_uncond": [0],
            })

        call(1.0)
        call(0.5)
        self.assertIs(observed[0], observed[1])

        # The next high sigma starts a new sampler pass. It may retain a
        # model-stable artist embedding, but never a live mixed-context tensor.
        call(1.0)
        self.assertIsNot(observed[1], observed[2])

    def test_wrapper_discards_model_bound_cache_when_patcher_changes(self):
        dm = FakeAdapterModel()
        state = make_state(dm, ALIGN_SHARED_BASE_IDS)
        state["raws"] = [torch.tensor([[[1.0, 2.0]]])]
        state["ids_list"] = [torch.tensor([99, 10, 11, 12])]
        state["t5_weights_list"] = [None]
        state["user_weights"] = [1.0]
        observed = []

        class Patcher:
            pass

        class Runner:
            def __init__(self):
                self.current_patcher = Patcher()

            def apply_model(self, input_tensor, timestep, **c):
                observed.append(c["c_crossattn"])
                return input_tensor

        runner = Runner()
        wrapper = make_adapter_embedding_wrapper(state, None)
        base = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]])

        def call():
            wrapper(runner.apply_model, {
                "input": torch.zeros((1, 1)),
                "timestep": torch.ones((1,)),
                "c": {
                    "c_crossattn": base,
                    "transformer_options": {"cond_or_uncond": [0]},
                },
                "cond_or_uncond": [0],
            })

        call()
        self.assertEqual(len(dm.seen_ids), 1)
        self.assertEqual(len(state["_artist_embedding_cache"]), 1)
        state["_anchor_trajectory"] = {"ready": True}

        # This models a wrapper surviving ModelPatcher.clone(), for example
        # after a changed LoRA or another model-optimization node. Old adapter
        # output and Anchor-Q state cannot be used under the new patcher.
        runner.current_patcher = Patcher()
        call()
        self.assertEqual(len(dm.seen_ids), 2)
        self.assertIsNone(state["_anchor_trajectory"])
        self.assertIsNot(observed[0], observed[1])

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

    @mock.patch(
        "anima_mixer.nodes_embedding.secrets.randbelow",
        side_effect=[987654321, 123456789],
    )
    def test_adapter_q_anchor_generates_seeds_when_options_payload_is_blank(self, randbelow):
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
                "anchor_seed_list": "",
                "anchor_seeds_count": 2,
            },
        )
        patch_path = "diffusion_model.blocks.0.cross_attn.forward"
        self.assertEqual(
            patched_model.object_patches[patch_path].state["anchor_seed_list"],
            [987654321, 123456789],
        )
        self.assertEqual(randbelow.call_count, 2)

    def test_repatch_replaces_previous_adapter_wrapper_and_anchor_patch(self):
        dm = FakeAnchorAdapterModel()
        model = FakeModelPatcher(dm)
        node = AnimaArtistAdapterMixer()

        def pack(marker):
            return {
                "conditionings": [[[
                    torch.tensor([[[float(marker), 2.0]]]),
                    {"t5xxl_ids": torch.tensor([20, 10, 11])},
                ]]],
                "labels": [f"artist-{marker}"],
                "weights": [1.0],
                "base_conditioning": [[
                    torch.tensor([[[0.0, 0.0]]]),
                    {"t5xxl_ids": torch.tensor([10, 11])},
                ]],
                "base_prompt": f"base-{marker}",
            }

        advanced = {
            "artist_anchor_q": True,
            "anchor_seed_list": "42",
            "anchor_seed_list_is_manual": True,
        }
        first, _ = node.patch(
            model,
            pack(1),
            1.0,
            True,
            ALIGN_BASE_ANCHORED,
            True,
            False,
            0.0,
            advanced,
        )
        first_wrapper = first.model_options["model_function_wrapper"]
        first_patch = first.object_patches[
            "diffusion_model.blocks.0.cross_attn.forward"
        ]
        first_patch.state["_anchor_trajectory"] = {"ready": True}

        second, _ = node.patch(
            first,
            pack(2),
            1.0,
            True,
            ALIGN_BASE_ANCHORED,
            True,
            False,
            0.0,
            advanced,
        )
        second_wrapper = second.model_options["model_function_wrapper"]
        second_patch = second.object_patches[
            "diffusion_model.blocks.0.cross_attn.forward"
        ]
        first_sigma_wrapper = first_wrapper._anima_adapter_mixer_previous
        second_sigma_wrapper = second_wrapper._anima_adapter_mixer_previous

        self.assertIsNot(second_wrapper, first_wrapper)
        self.assertIsNot(second_sigma_wrapper, first_sigma_wrapper)
        self.assertTrue(second_sigma_wrapper._anima_adapter_anchor_sigma_wrapper)
        self.assertIsNone(second_sigma_wrapper._anima_adapter_anchor_sigma_previous)
        self.assertIsNot(second_patch, first_patch)
        self.assertIsNone(second_patch.state["_anchor_trajectory"])
        self.assertNotEqual(
            first_patch.state["_cache_namespace"],
            second_patch.state["_cache_namespace"],
        )

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


class MixerLifecycleRegressionTests(unittest.TestCase):
    def _artist_pack(self):
        return {
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

    def test_cleanup_releases_run_bound_tensors_and_flags(self):
        state = make_state(FakeAdapterModel())
        state.update({
            "_artist_embedding_cache": {"artist": torch.ones(1)},
            "_mixed_context_cache": {"mixed": torch.ones(1)},
            "_anchor_cache": {0: torch.ones(1)},
            "_anchor_cache_key": ("anchor",),
            "_anchor_trajectory": {"ready": True},
            "_anchor_last_sigma": 0.5,
            "_in_anchor_run": True,
            "individuals": [torch.ones(1)],
            "real_lens": [1],
            "_ctx_fp_memo": {1: "fp"},
            "_execution_value_fp_memo": {2: "fp"},
            "_runtime_input_signature": ("inputs",),
            "_run_last_sigma": 0.5,
            "_run_call_count": 3,
            "_run_active": True,
            "_mixer_run_start_pending": True,
            "_adapter_mixer_run_start": True,
            "_adapter_mixer_finalize_warm_cache": True,
        })

        _make_mixer_cleanup_callback(state)(None)

        self.assertEqual(state["_artist_embedding_cache"], {})
        self.assertIsNone(state["_mixed_context_cache"])
        self.assertEqual(state["_anchor_cache"], {})
        self.assertIsNone(state["_anchor_cache_key"])
        self.assertIsNone(state["_anchor_trajectory"])
        self.assertFalse(state["_in_anchor_run"])
        self.assertIsNone(state["individuals"])
        self.assertIsNone(state["real_lens"])
        self.assertFalse(state["_run_active"])
        self.assertFalse(state["_mixer_run_start_pending"])
        self.assertIsNone(state["_adapter_mixer_run_start"])
        self.assertFalse(state["_adapter_mixer_finalize_warm_cache"])
        self.assertEqual(state["_run_call_count"], 0)
        self.assertTrue(state["_last_run_had_calls"])

        # Interrupt unwinding may call the same release helper more than once.
        clear_mixer_run_state(state)
        self.assertEqual(state["_artist_embedding_cache"], {})
        self.assertFalse(state["_run_active"])

    def test_abort_boundary_releases_state_and_preserves_base_exception(self):
        state = make_state(FakeAdapterModel())
        state.update({
            "_artist_embedding_cache": {"artist": torch.ones(1)},
            "_mixed_context_cache": {"mixed": torch.ones(1)},
            "_anchor_cache": {0: torch.ones(1)},
            "_run_active": True,
            "_run_call_count": 2,
        })

        class Interrupt(BaseException):
            pass

        original = Interrupt("stop")

        def previous_wrapper(_apply_model, _options):
            raise original

        wrapper = make_adapter_embedding_wrapper(state, previous_wrapper)
        options = {
            "input": torch.zeros(1, 1, 2),
            "timestep": torch.tensor([0.5]),
            "c": {"context": torch.zeros(1, 3, 2)},
        }
        with mock.patch(
            "anima_mixer.patching.torch.cuda.is_available",
            return_value=True,
        ), mock.patch(
            "anima_mixer.patching.torch.cuda.synchronize",
        ) as synchronize:
            with self.assertRaises(Interrupt) as raised:
                wrapper(lambda *_args, **_kwargs: None, options)

        self.assertIs(raised.exception, original)
        self.assertEqual(state["_artist_embedding_cache"], {})
        self.assertIsNone(state["_mixed_context_cache"])
        self.assertEqual(state["_anchor_cache"], {})
        self.assertFalse(state["_run_active"])
        self.assertEqual(synchronize.call_count, 2)
        self.assertEqual(synchronize.call_args_list[0], mock.call(None))
        self.assertEqual(synchronize.call_args_list[1], mock.call())

    def test_cleanup_keeps_only_finalized_cpu_warm_trajectory(self):
        state = make_state(FakeAdapterModel())
        state["anchor_refresh_mode"] = "warm_cache"
        state["_anchor_trajectory"] = {
            "ready": True,
            "frames": [{"sigma": 1.0, "layers": {}}],
            "last_cache": {0: torch.ones(1)},
            "active_sigma": 1.0,
            "active_device": torch.device("cpu"),
            "active_dtype": torch.float32,
        }

        _make_mixer_cleanup_callback(state)(None)

        trajectory = state["_anchor_trajectory"]
        self.assertIsNotNone(trajectory)
        self.assertTrue(trajectory["ready"])
        self.assertEqual(trajectory["frames"], [{"sigma": 1.0, "layers": {}}])
        self.assertIsNone(trajectory["last_cache"])
        self.assertIsNone(trajectory["active_sigma"])
        self.assertIsNone(trajectory["active_device"])
        self.assertIsNone(trajectory["active_dtype"])

    def test_clone_rebinds_wrapper_and_anchor_patch_state(self):
        dm = FakeAnchorAdapterModel()
        model = FakeModelPatcher(dm)
        patched, _ = AnimaArtistAdapterMixer().patch(
            model,
            self._artist_pack(),
            strength=1.0,
            normalize_weights=True,
            alignment_mode=ALIGN_BASE_ANCHORED,
            enabled=True,
            apply_to_uncond=False,
            uncond_strength=0.0,
            advanced_options={
                "artist_anchor_q": True,
                "anchor_seed_list": "42",
                "anchor_seed_list_is_manual": True,
            },
        )
        path = "diffusion_model.blocks.0.cross_attn.forward"
        original_patch = patched.object_patches[path]
        original_wrapper = patched.model_options["model_function_wrapper"]

        cloned = patched.clone()
        cloned_patch = cloned.object_patches[path]
        cloned_wrapper = cloned.model_options["model_function_wrapper"]

        self.assertIsNot(cloned_patch.state, original_patch.state)
        self.assertIsNot(cloned_wrapper, original_wrapper)
        self.assertIs(
            cloned_wrapper._anima_adapter_mixer_state,
            cloned_patch.state,
        )
        cloned_patch.state["_embedding_mixer_failed"] = True
        self.assertFalse(original_patch.state["_embedding_mixer_failed"])

    def test_clone_rebinds_mixer_hidden_inside_external_wrapper(self):
        dm = FakeAdapterModel()
        model = FakeModelPatcher(dm)
        patched, _ = AnimaArtistAdapterMixer().patch(
            model,
            self._artist_pack(),
            strength=1.0,
            normalize_weights=True,
            alignment_mode=ALIGN_BASE_ANCHORED,
            enabled=True,
            apply_to_uncond=False,
            uncond_strength=0.0,
        )
        original_mixer = patched.model_options["model_function_wrapper"]
        original_state = original_mixer._anima_mixer_state

        def opaque_external_wrapper(apply_model, options):
            return original_mixer(apply_model, options)

        patched.model_options["model_function_wrapper"] = opaque_external_wrapper
        cloned = patched.clone()
        registry = cloned.model_options.get("_anima_mixer_clone_wrappers", {})
        rebound = registry.get(id(original_state))

        self.assertIsNotNone(rebound)
        self.assertIsNot(rebound._anima_mixer_state, original_state)
        original_state["_artist_embedding_cache"] = {"source": torch.ones(1)}

        class Runner:
            def __init__(self, patcher):
                self.current_patcher = patcher

            def apply_model(self, input_tensor, timestep, **_kwargs):
                return input_tensor

        runner = Runner(cloned)
        cloned.model_options["model_function_wrapper"](runner.apply_model, {
            "input": torch.zeros((1, 1)),
            "timestep": torch.ones((1,)),
            "c": {
                "c_crossattn": torch.ones((1, 2, 2)),
                "transformer_options": {"cond_or_uncond": [0]},
            },
            "cond_or_uncond": [0],
        })

        self.assertEqual(
            list(original_state["_artist_embedding_cache"]),
            ["source"],
        )
        self.assertTrue(rebound._anima_mixer_state["_artist_embedding_cache"])

        second_clone = cloned.clone()
        second_rebound = second_clone.model_options[
            "_anima_mixer_clone_wrappers"
        ][id(original_state)]
        self.assertIsNot(
            second_rebound._anima_mixer_state,
            rebound._anima_mixer_state,
        )
        second_rebound._anima_mixer_state["_execution_value_fp_memo"] = {
            "second": "clone",
        }
        self.assertNotIn(
            "second",
            rebound._anima_mixer_state.get("_execution_value_fp_memo", {}),
        )

    def test_new_mixer_supersedes_older_mixer_hidden_by_external_wrapper(self):
        dm = FakeAdapterModel()
        model = FakeModelPatcher(dm)
        first, _ = AnimaArtistAdapterMixer().patch(
            model,
            self._artist_pack(),
            strength=1.0,
            normalize_weights=True,
            alignment_mode=ALIGN_BASE_ANCHORED,
            enabled=True,
            apply_to_uncond=False,
            uncond_strength=0.0,
        )
        first_wrapper = first.model_options["model_function_wrapper"]
        first_state = first_wrapper._anima_mixer_state

        # Optimization/control wrappers can hide the existing Mixer from the
        # top-level unwrapping performed when a later sampling stage installs
        # its own Mixer.
        def opaque_external_wrapper(apply_model, options):
            return first_wrapper(apply_model, options)

        first.model_options["model_function_wrapper"] = opaque_external_wrapper
        second_pack = self._artist_pack()
        second_pack["conditionings"][0][0][0] = torch.tensor([[[5.0, 6.0]]])
        second, _ = AnimaArtistAdapterMixer().patch(
            first,
            second_pack,
            strength=1.0,
            normalize_weights=True,
            alignment_mode=ALIGN_BASE_ANCHORED,
            enabled=True,
            apply_to_uncond=False,
            uncond_strength=0.0,
        )
        second_wrapper = second.model_options["model_function_wrapper"]
        second_state = second_wrapper._anima_mixer_state
        rebound_first = second.model_options[
            "_anima_mixer_clone_wrappers"
        ][id(first_state)]
        rebound_first_state = rebound_first._anima_mixer_state

        class Runner:
            def __init__(self, patcher):
                self.current_patcher = patcher

            def apply_model(self, _input, _timestep, **kwargs):
                return kwargs["c_crossattn"]

        runner = Runner(second)
        second.pre_run()
        output = second_wrapper(runner.apply_model, {
            "input": torch.zeros((1, 1)),
            "timestep": torch.ones((1,)),
            "c": {
                "c_crossattn": torch.ones((1, 2, 2)),
                "transformer_options": {"cond_or_uncond": [0]},
            },
            "cond_or_uncond": [0],
        })

        self.assertEqual(tuple(output.shape), (1, 3, 2))
        self.assertEqual(len(dm.seen_ids), 1)
        self.assertTrue(second_state["_run_active"])
        self.assertFalse(rebound_first_state.get("_run_active", False))

        # A further downstream clone must prune the superseded lifecycle
        # callbacks instead of accumulating one inactive callback set per
        # sampling stage.
        third = second.clone()
        self.assertEqual(len(third.get_all_callbacks("on_pre_run")), 1)

    def test_superseding_mixer_keeps_source_branch_independently_active(self):
        dm = FakeAdapterModel()
        model = FakeModelPatcher(dm)
        first, _ = AnimaArtistAdapterMixer().patch(
            model,
            self._artist_pack(),
            strength=1.0,
            normalize_weights=True,
            alignment_mode=ALIGN_BASE_ANCHORED,
            enabled=True,
            apply_to_uncond=False,
            uncond_strength=0.0,
        )
        first_wrapper = first.model_options["model_function_wrapper"]

        def opaque_external_wrapper(apply_model, options):
            return first_wrapper(apply_model, options)

        first.model_options["model_function_wrapper"] = opaque_external_wrapper
        second_pack = self._artist_pack()
        second_pack["conditionings"][0][0][0] = torch.tensor([[[5.0, 6.0]]])
        second, _ = AnimaArtistAdapterMixer().patch(
            first,
            second_pack,
            strength=1.0,
            normalize_weights=True,
            alignment_mode=ALIGN_BASE_ANCHORED,
            enabled=True,
            apply_to_uncond=False,
            uncond_strength=0.0,
        )

        class Runner:
            def __init__(self, patcher):
                self.current_patcher = patcher

            def apply_model(self, _input, _timestep, **kwargs):
                return kwargs["c_crossattn"]

        options = {
            "input": torch.zeros((1, 1)),
            "timestep": torch.ones((1,)),
            "c": {
                "c_crossattn": torch.ones((1, 2, 2)),
                "transformer_options": {"cond_or_uncond": [0]},
            },
            "cond_or_uncond": [0],
        }

        second_runner = Runner(second)
        second.pre_run()
        second.model_options["model_function_wrapper"](
            second_runner.apply_model,
            options,
        )
        second.cleanup()

        first_runner = Runner(first)
        first.pre_run()
        first.model_options["model_function_wrapper"](
            first_runner.apply_model,
            options,
        )

        # One Adapter call for each branch: installing the downstream Mixer
        # neither double-runs it nor globally disables the source branch.
        self.assertEqual(len(dm.seen_ids), 2)

    def test_pre_run_boundary_resets_even_when_first_sigma_is_identical(self):
        dm = FakeAdapterModel()
        model = FakeModelPatcher(dm)
        patched, _ = AnimaArtistAdapterMixer().patch(
            model,
            self._artist_pack(),
            strength=1.0,
            normalize_weights=True,
            alignment_mode=ALIGN_BASE_ANCHORED,
            enabled=True,
            apply_to_uncond=False,
            uncond_strength=0.0,
        )
        wrapper = patched.model_options["model_function_wrapper"]
        state = wrapper._anima_adapter_mixer_state

        def apply_model(input_tensor, timestep, **_kwargs):
            return input_tensor

        options = {
            "input": torch.zeros((1, 1)),
            "timestep": torch.ones((1,)),
            "c": {
                "c_crossattn": torch.ones((1, 2, 2)),
                "transformer_options": {"cond_or_uncond": [0]},
            },
            "cond_or_uncond": [0],
        }
        patched.pre_run()
        wrapper(apply_model, options)
        first_execution = state["_execution_index"]
        patched.cleanup()
        patched.pre_run()
        wrapper(apply_model, options)

        self.assertEqual(state["_execution_index"], first_execution + 1)

    def test_pre_run_owner_survives_shared_current_patcher_drift(self):
        dm = FakeAdapterModel()
        model = FakeModelPatcher(dm)
        patched, _ = AnimaArtistAdapterMixer().patch(
            model,
            self._artist_pack(),
            strength=1.0,
            normalize_weights=True,
            alignment_mode=ALIGN_BASE_ANCHORED,
            enabled=True,
            apply_to_uncond=False,
            uncond_strength=0.0,
        )
        wrapper = patched.model_options["model_function_wrapper"]
        state = wrapper._anima_adapter_mixer_state

        class SharedModel:
            def __init__(self):
                self.current_patcher = patched

            def apply_model(self, _input, _timestep, **kwargs):
                return kwargs["c_crossattn"]

        shared_model = SharedModel()
        options = {
            "input": torch.zeros((1, 1)),
            "timestep": torch.ones((1,)),
            "c": {
                "c_crossattn": torch.ones((1, 2, 2)),
                "transformer_options": {"cond_or_uncond": [0]},
            },
            "cond_or_uncond": [0],
        }

        patched.pre_run()
        expected_owner = ("patcher", id(patched))
        self.assertEqual(state["_model_owner_token"], expected_owner)

        # Several ModelPatchers can share one BaseModel.  A later pre_run on a
        # sibling clone overwrites BaseModel.current_patcher even though this
        # wrapper/options pair still belongs to the original sampling pass.
        sibling = FakeModelPatcher(dm)
        sibling.model_options = dict(patched.model_options)
        shared_model.current_patcher = sibling
        wrapper(shared_model.apply_model, options)

        self.assertEqual(state["_model_owner_token"], expected_owner)

    def test_shared_current_patcher_drift_does_not_route_adapter_or_forward_to_sibling(self):
        main_dm = FakeAdapterModel()
        sibling_dm = FakeAdapterModel()
        state = make_state(main_dm)
        state.update({
            "_adapter_mixer_instance_token": "main-mixer",
            "_adapter_mixer_selected_for_run": True,
            "_run_active": True,
        })

        class Owner:
            def __init__(self, diffusion_model):
                self.diffusion_model = diffusion_model
                self.model = None

            def get_model_object(self, name):
                if name != "diffusion_model":
                    raise KeyError(name)
                return self.diffusion_model

        main_owner = Owner(main_dm)
        sibling_owner = Owner(sibling_dm)

        class SharedModel:
            def __init__(self):
                self.current_patcher = sibling_owner
                self.seen_patchers = []

            def apply_model(self, _input, _timestep, **kwargs):
                self.seen_patchers.append(self.current_patcher)
                return kwargs["c_crossattn"]

        shared_model = SharedModel()
        main_owner.model = shared_model
        sibling_owner.model = shared_model
        state["_model_owner_token"] = ("patcher", id(main_owner))
        state["_model_owner_ref"] = main_owner
        wrapper = make_adapter_embedding_wrapper(state, None)
        context = torch.ones((1, 3, 2))

        output = wrapper(shared_model.apply_model, {
            "input": torch.zeros((1, 1)),
            "timestep": torch.ones((1,)),
            "c": {
                "c_crossattn": context,
                "transformer_options": {"cond_or_uncond": [0]},
            },
            "cond_or_uncond": [0],
        })

        self.assertEqual(len(main_dm.seen_ids), 2)
        self.assertEqual(len(sibling_dm.seen_ids), 0)
        self.assertEqual(shared_model.seen_patchers, [main_owner])
        self.assertIs(shared_model.current_patcher, sibling_owner)
        self.assertEqual(tuple(output.shape), (1, 5, 2))

    def test_owner_scope_restores_pointer_after_interrupt(self):
        state = {}

        class Owner:
            def __init__(self):
                self.model = None

        class SharedModel:
            def __init__(self):
                self.current_patcher = None

            def apply_model(self, *_args, **_kwargs):
                raise AssertionError("callback should not be used in this test")

        owner = Owner()
        sibling = Owner()
        shared_model = SharedModel()
        owner.model = shared_model
        sibling.model = shared_model
        shared_model.current_patcher = sibling
        state["_model_owner_ref"] = owner

        class Interrupt(BaseException):
            pass

        def abort():
            raise Interrupt("stop")

        with self.assertRaises(Interrupt):
            call_with_mixer_owner(
                state,
                shared_model.apply_model,
                abort,
            )

        self.assertIs(shared_model.current_patcher, sibling)

    def test_inactive_parent_clone_is_not_reactivated_by_shared_patcher(self):
        dm = FakeAdapterModel()
        state = make_state(dm)
        state["_adapter_mixer_instance_token"] = "same-mixer"
        state["_adapter_mixer_selected_for_run"] = False
        wrapper = make_adapter_embedding_wrapper(state, None)

        class Patcher:
            model_options = {
                "_anima_adapter_mixer_active_token": "same-mixer",
            }

        class SharedModel:
            current_patcher = Patcher()

            def apply_model(self, _input, _timestep, **kwargs):
                return kwargs["c_crossattn"]

        shared_model = SharedModel()
        context = torch.ones((1, 3, 2))
        output = wrapper(shared_model.apply_model, {
            "input": torch.zeros((1, 1)),
            "timestep": torch.ones((1,)),
            "c": {
                "c_crossattn": context,
                "transformer_options": {"cond_or_uncond": [0]},
            },
            "cond_or_uncond": [0],
        })

        self.assertIs(output, context)
        self.assertEqual(dm.seen_ids, [])

    def test_cross_attention_sigma_fallback_binds_timestep_before_begin(self):
        state = make_state(FakeAdapterModel())
        state["artist_anchor_q"] = False
        wrapper = make_sigma_capture(state, None)
        input_tensor = torch.zeros((1, 1))

        def apply_model(input_value, timestep, **_kwargs):
            return input_value

        for sigma in (1.0, 0.8, 0.6, 1.0):
            output = wrapper(apply_model, {
                "input": input_tensor,
                "timestep": torch.full((1,), sigma),
                "c": {"c_crossattn": torch.ones((1, 2, 2))},
            })
            self.assertIs(output, input_tensor)
        self.assertEqual(state["_execution_index"], 2)
        self.assertAlmostEqual(state["_run_last_sigma"], 1.0)

    def test_cross_attention_pre_run_counts_first_direct_sigma_call(self):
        state = make_state(FakeAdapterModel())
        state["artist_anchor_q"] = False
        wrapper = make_sigma_capture(state, None)
        input_tensor = torch.zeros((1, 1))

        class Owner:
            pass

        class Runner:
            def __init__(self):
                self.current_patcher = Owner()

            def apply_model(self, input_value, timestep, **_kwargs):
                return input_value

        runner = Runner()
        begin_mixer_execution(
            state,
            runner.apply_model,
            None,
            owner=runner.current_patcher,
            explicit_run_start=True,
        )
        wrapper(runner.apply_model, {
            "input": input_tensor,
            "timestep": torch.ones((1,)),
            "c": {"c_crossattn": torch.ones((1, 2, 2))},
        })
        self.assertEqual(state["_run_call_count"], 1)
        self.assertAlmostEqual(state["_run_last_sigma"], 1.0)

    def test_weight_patch_identity_change_clears_artist_cache_on_shared_model(self):
        dm = FakeAdapterModel()
        state = make_state(dm)

        class Owner:
            def __init__(self):
                self.patches_uuid = "lora-a"
                self.patches = {"adapter.weight": []}
                self.model = type("SharedModel", (), {
                    "current_weight_patches_uuid": "lora-a",
                })()

        owner = Owner()
        begin_mixer_execution(
            state,
            None,
            None,
            owner=owner,
            explicit_run_start=True,
        )
        state["_artist_embedding_cache"] = {"old-lora": torch.ones(1)}
        state["individuals"] = [torch.ones(1)]

        # This models a normal LoRA clone: the BaseModel is shared, its
        # ModelPatcher remains the same object, but add_patches rerolls UUID.
        owner.patches_uuid = "lora-b"
        owner.model.current_weight_patches_uuid = "lora-b"
        begin_mixer_execution(state, None, None, owner=owner)

        self.assertEqual(state["_artist_embedding_cache"], {})
        self.assertIsNone(state["individuals"])
        self.assertEqual(
            state["_model_weight_patch_identity"],
            ("lora-b", "lora-b", 1),
        )

    def test_multigpu_shared_wrapper_dispatches_to_worker_state(self):
        main_dm = FakeAdapterModel()
        worker_dm = FakeAdapterModel()
        main_state = make_state(main_dm, ALIGN_SHARED_BASE_IDS)
        worker_state = make_state(worker_dm, ALIGN_SHARED_BASE_IDS)
        main_state["raws"] = worker_state["raws"] = [
            torch.tensor([[[1.0, 2.0]]]),
        ]
        main_state["ids_list"] = worker_state["ids_list"] = [
            torch.tensor([10, 11, 12]),
        ]
        main_state["t5_weights_list"] = worker_state["t5_weights_list"] = [None]
        main_state["user_weights"] = worker_state["user_weights"] = [1.0]

        main_sigma = make_sigma_capture(main_state, None)
        worker_sigma = make_sigma_capture(worker_state, None)
        main_wrapper = make_adapter_embedding_wrapper(main_state, main_sigma)
        worker_wrapper = make_adapter_embedding_wrapper(worker_state, worker_sigma)

        class WorkerModel:
            def __init__(self, patcher):
                self.current_patcher = patcher

            def apply_model(self, input_value, timestep, **kwargs):
                return kwargs["c_crossattn"]

        class WorkerPatcher:
            def __init__(self):
                self.model_options = {"model_function_wrapper": worker_wrapper}

        worker_patcher = WorkerPatcher()
        worker_model = WorkerModel(worker_patcher)
        context = torch.zeros((1, 3, 2))
        output = main_wrapper(worker_model.apply_model, {
            "input": torch.zeros((1, 1)),
            "timestep": torch.ones((1,)),
            "c": {
                "c_crossattn": context,
                "transformer_options": {
                    "multigpu_thread_device": torch.device("cuda", 1),
                    "cond_or_uncond": [0],
                },
            },
            "cond_or_uncond": [0],
        })

        self.assertEqual(len(main_dm.seen_ids), 0)
        self.assertEqual(len(worker_dm.seen_ids), 1)
        self.assertEqual(tuple(output.shape), tuple(context.shape))

    def test_clone_rebind_resolves_forward_from_fresh_model(self):
        old_dm = FakeAnchorAdapterModel()
        new_dm = FakeAnchorAdapterModel()
        old_state = make_state(old_dm)
        new_state = make_state(new_dm)

        class Model:
            def __init__(self, dm):
                self.diffusion_model = dm

        class Patcher:
            def __init__(self, dm):
                self.model = Model(dm)
                self.object_patches = {}

        patcher = Patcher(new_dm)
        path = "diffusion_model.blocks.0.cross_attn.forward"
        patcher.object_patches[path] = make_adapter_anchor_q_forward_patch(
            old_dm.blocks[0].cross_attn.forward,
            old_state,
            0,
        )
        _rebind_mixer_object_patches(patcher, old_state, new_state)
        rebound = patcher.object_patches[path]
        self.assertIs(rebound.original_forward.__self__, new_dm.blocks[0].cross_attn)

    def test_silent_data_write_invalidates_artist_embedding_cache(self):
        dm = FakeAdapterModel()
        state = make_state(dm)
        ref_context = torch.zeros((1, 5, 2))
        first = build_artist_embedding_sum(state, ref_context)
        version = int(state["raws"][0]._version)
        state["raws"][0].data.add_(10.0)
        self.assertEqual(int(state["raws"][0]._version), version)

        reset_run_state(state)
        second = build_artist_embedding_sum(state, ref_context)
        self.assertFalse(torch.equal(first, second))
        self.assertEqual(len(state["_artist_embedding_cache"]), 2)

    def test_explicit_pre_run_refreshes_silent_prompt_edit_without_cleanup(self):
        dm = FakeAdapterModel()
        state = make_state(dm)
        ref_context = torch.zeros((1, 5, 2))

        class Owner:
            pass

        owner = Owner()
        begin_mixer_execution(
            state,
            None,
            None,
            owner=owner,
            explicit_run_start=True,
        )
        first = build_artist_embedding_sum(state, ref_context)
        self.assertEqual(len(dm.seen_ids), 2)

        # ModelPatcher cleanup is expected in the normal path, but an outer
        # wrapper can unwind before it runs.  Simulate an in-place conditioning
        # edit and enter the next explicit pre_run directly.
        state["raws"][0].data.add_(10.0)
        begin_mixer_execution(
            state,
            None,
            None,
            owner=owner,
            explicit_run_start=True,
        )
        second = build_artist_embedding_sum(state, ref_context)

        self.assertFalse(torch.equal(first, second))
        self.assertEqual(len(dm.seen_ids), 4)
        self.assertEqual(len(state["_artist_embedding_cache"]), 1)

    def test_short_adaptive_trajectory_is_not_marked_ready(self):
        state = {
            "anchor_cache_points": 8,
            "anchor_keyframe_mode": ANCHOR_KEYFRAME_ADAPTIVE_Q,
            "anchor_deep_layer_threshold": -1,
            "stabilizer_min_sigma": None,
            "anchor_log_name": "test",
        }
        trajectory = _new_anchor_trajectory(state, ("test",), 1.0)
        state["_anchor_trajectory"] = trajectory
        for sigma in (1.0, 0.5):
            state["_anchor_cache"] = {0: torch.full((1, 1, 2), sigma)}
            _record_trajectory_step(state, trajectory, sigma)

        _finalize_anchor_trajectory(state)
        self.assertIsNone(state["_anchor_trajectory"])


if __name__ == "__main__":
    unittest.main()
