import os
from pathlib import Path
import sys
from types import ModuleType
from types import SimpleNamespace
import unittest
from unittest import mock
import uuid

import torch

from anima_mixer import conditioning_diagnostics as diagnostics


def _fake_runtime():
    class FakeCLIP:
        def encode_from_tokens_scheduled(self, tokens, *args, **kwargs):
            return tokens

    class FakeCFGGuider:
        def inner_set_conds(self, conds, *args, **kwargs):
            self.original_conds = conds
            return "set"

        def sample(self, *args, **kwargs):
            return "sampled"

    class FakeAnima:
        def preprocess_text_embeds(self, text_embeds, text_ids, **kwargs):
            return text_embeds

    class FakeLLMAdapter:
        def forward(self, source_hidden_states, target_input_ids, **kwargs):
            return source_hidden_states

    def encode_model_conds(model_function, conds, noise, device, prompt_type, **kwargs):
        return conds

    def get_area_and_mult(conds, *args, **kwargs):
        return conds

    def cond_cat(c_list, device=None):
        return c_list

    def calc_cond_batch(model, conds, x_in, timestep, model_options):
        return conds

    def sampling_function(*args, **kwargs):
        return "cfg"

    comfy_sd = SimpleNamespace(CLIP=FakeCLIP)
    samplers = SimpleNamespace(
        CFGGuider=FakeCFGGuider,
        encode_model_conds=encode_model_conds,
        get_area_and_mult=get_area_and_mult,
        cond_cat=cond_cat,
        _calc_cond_batch=calc_cond_batch,
        sampling_function=sampling_function,
    )
    anima_model = SimpleNamespace(Anima=FakeAnima, LLMAdapter=FakeLLMAdapter)
    return comfy_sd, samplers, anima_model


def _runtime_callables(comfy_sd, samplers, anima_model):
    return (
        anima_model.Anima.preprocess_text_embeds,
        anima_model.LLMAdapter.forward,
        comfy_sd.CLIP.encode_from_tokens_scheduled,
        samplers.CFGGuider.inner_set_conds,
        samplers.CFGGuider.sample,
        samplers.encode_model_conds,
        samplers.get_area_and_mult,
        samplers.cond_cat,
        samplers._calc_cond_batch,
        samplers.sampling_function,
    )


def _snapshot(fingerprint, *, bad=0):
    return {
        "is_tensor": True,
        "shape": (1, 2, 4),
        "dtype": "torch.float32",
        "device": "cpu",
        "bad": bad,
        "nan": bad,
        "inf": 0,
        "finite_max_abs": 1.0,
        "fingerprint": fingerprint,
    }


class ConditioningDiagnosticTests(unittest.TestCase):
    def setUp(self):
        diagnostics._reset_for_tests()
        self.environment = mock.patch.dict(os.environ, {
            diagnostics.MASTER_ENV_NAME: "1",
            diagnostics.ENV_NAME: "1",
        })
        self.environment.start()

    def tearDown(self):
        diagnostics._reset_for_tests()
        self.environment.stop()

    def test_install_is_idempotent_and_uninstall_restores_every_callable(self):
        comfy_sd, samplers, anima_model = _fake_runtime()
        originals = _runtime_callables(comfy_sd, samplers, anima_model)

        self.assertTrue(diagnostics.install(comfy_sd, samplers, anima_model))
        installed = _runtime_callables(comfy_sd, samplers, anima_model)
        self.assertEqual(len(diagnostics.get_install_status()), 10)
        self.assertTrue(all(after is not before for before, after in zip(originals, installed)))

        self.assertTrue(diagnostics.install(comfy_sd, samplers, anima_model))
        self.assertEqual(_runtime_callables(comfy_sd, samplers, anima_model), installed)
        self.assertEqual(len(diagnostics.get_install_status()), 10)

        restored = diagnostics.uninstall()
        self.assertEqual(len(restored), 10)
        self.assertEqual(_runtime_callables(comfy_sd, samplers, anima_model), originals)
        self.assertEqual(diagnostics.uninstall(), ())

    def test_disabled_install_does_not_patch_any_callable(self):
        for variable in (diagnostics.ENV_NAME, diagnostics.MASTER_ENV_NAME):
            with self.subTest(variable=variable):
                diagnostics._reset_for_tests()
                comfy_sd, samplers, anima_model = _fake_runtime()
                originals = _runtime_callables(comfy_sd, samplers, anima_model)
                with mock.patch.dict(os.environ, {variable: "0"}):
                    self.assertFalse(
                        diagnostics.install(comfy_sd, samplers, anima_model)
                    )
                self.assertEqual(
                    _runtime_callables(comfy_sd, samplers, anima_model),
                    originals,
                )
                self.assertEqual(diagnostics.get_install_status(), ())

    def test_partial_install_failure_rolls_back_prior_patches(self):
        comfy_sd, samplers, anima_model = _fake_runtime()
        originals = _runtime_callables(comfy_sd, samplers, anima_model)
        broken_anima_model = SimpleNamespace(Anima=anima_model.Anima)

        with self.assertLogs(
            "anima_mixer.conditioning_diagnostics",
            level="WARNING",
        ) as logs:
            self.assertFalse(
                diagnostics.install(comfy_sd, samplers, broken_anima_model)
            )

        self.assertEqual(
            _runtime_callables(comfy_sd, samplers, anima_model),
            originals,
        )
        self.assertEqual(diagnostics.get_install_status(), ())
        self.assertIn("event=diagnostic_error hook=install", "\n".join(logs.output))

    def test_root_entrypoint_loads_when_diagnostic_module_is_deleted(self):
        package_name = "anima_mixer_missing_probe_test"
        package = ModuleType(package_name)
        package.__package__ = package_name
        package.__path__ = []
        mixer = ModuleType(f"{package_name}.anima_mixer")
        mixer.__package__ = package_name
        mixer.__path__ = []
        mixer.NODE_CLASS_MAPPINGS = {"node": object()}
        mixer.NODE_DISPLAY_NAME_MAPPINGS = {"node": "Node"}
        semantic = ModuleType(
            f"{package_name}.anima_mixer.semantic_diagnostics"
        )
        semantic.setup_comfy_file_logging = lambda: None
        modules = {
            package_name: package,
            f"{package_name}.anima_mixer": mixer,
            f"{package_name}.anima_mixer.semantic_diagnostics": semantic,
        }
        namespace = {
            "__name__": package_name,
            "__package__": package_name,
            "__path__": [],
        }
        entrypoint = Path(__file__).resolve().parents[1] / "__init__.py"

        with mock.patch.dict(sys.modules, modules, clear=False):
            exec(
                compile(
                    entrypoint.read_text(encoding="utf-8"),
                    str(entrypoint),
                    "exec",
                ),
                namespace,
            )

        self.assertEqual(namespace["NODE_CLASS_MAPPINGS"], mixer.NODE_CLASS_MAPPINGS)
        self.assertIsNone(namespace["_install_conditioning_diagnostics"])

    def test_observer_failure_does_not_change_text_encoder_result(self):
        expected = [(
            torch.ones((1, 2, 4)),
            {"t5xxl_ids": torch.tensor([1, 2], dtype=torch.int32)},
        )]

        class Clip:
            def encode(self, tokens):
                return expected

        wrapped = diagnostics._make_clip_encode_wrapper(Clip.encode)
        with mock.patch.object(
            diagnostics,
            "_observe_encode",
            side_effect=RuntimeError("simulated observer failure"),
        ), self.assertLogs(
            "anima_mixer.conditioning_diagnostics",
            level="WARNING",
        ) as logs:
            result = wrapped(Clip(), {"t5xxl": [1, 2]})

        self.assertIs(result, expected)
        self.assertIn("event=diagnostic_error hook=text_encode", "\n".join(logs.output))

    def test_original_exception_is_reraised_unchanged(self):
        expected_error = RuntimeError("original encoder failure")

        def original(_clip, _tokens):
            raise expected_error

        wrapped = diagnostics._make_clip_encode_wrapper(original)
        with self.assertRaises(RuntimeError) as raised:
            wrapped(object(), {"t5xxl": [1, 2]})
        self.assertIs(raised.exception, expected_error)

    def test_changed_tokens_with_same_encoding_is_flagged_as_stale(self):
        clip = SimpleNamespace(patcher=None)
        encoded = [(
            torch.arange(8, dtype=torch.float32).reshape(1, 2, 4),
            {"t5xxl_ids": torch.tensor([1, 2], dtype=torch.int32)},
        )]
        events = []
        caller = {
            "caller": "test.prompt_encoder",
            "text_fp": "prompt",
            "text_chars": 6,
        }
        with mock.patch.object(
            diagnostics,
            "_caller_details",
            return_value=caller,
        ), mock.patch.object(
            diagnostics,
            "_log_event",
            side_effect=lambda event, **fields: events.append((event, fields)),
        ):
            diagnostics._observe_encode(clip, {"t5xxl": [1]}, encoded, 1.0)
            diagnostics._observe_encode(clip, {"t5xxl": [2]}, encoded, 1.0)

        second = [fields for event, fields in events if event == "text_encode"][-1]
        self.assertEqual(second["transition"], "tokens_changed_output_same_suspect")
        self.assertEqual(second["suspect"], "stale_text_encode_output")

    def test_model_condition_transition_flags_changed_input_with_same_output(self):
        patch = diagnostics._patcher_identity(None)
        first_in = ({
            "raw": _snapshot("raw-one"),
            "ids": _snapshot("ids"),
            "weights": _snapshot("weights"),
        },)
        second_in = ({
            "raw": _snapshot("raw-two"),
            "ids": _snapshot("ids"),
            "weights": _snapshot("weights"),
        },)
        output = ({"model": _snapshot("same-output")},)

        diagnostics._model_cond_transition("model", "positive", first_in, output, patch)
        transition, changes, suspect = diagnostics._model_cond_transition(
            "model",
            "positive",
            second_in,
            output,
            patch,
        )

        self.assertEqual(transition, "inputs_changed_output_same_suspect")
        self.assertEqual(changes, ("raw",))
        self.assertEqual(suspect, "stale_preprocess_text_embeds_output")

    def test_cond_cat_recovers_marker_uuid_and_role_from_comfy_caller_frame(self):
        first_uuid = uuid.uuid4()
        second_uuid = uuid.uuid4()
        sample = {"sample_id": "sample-test", "seen_cat": set()}
        diagnostics._register_uuid(first_uuid, {
            "sample_id": "sample-test",
            "sample": sample,
            "role": "negative",
        })
        diagnostics._register_uuid(second_uuid, {
            "sample_id": "sample-test",
            "sample": sample,
            "role": "positive",
        })
        events = []

        def original(c_list, device=None):
            return {"c_crossattn": torch.cat([
                item["c_crossattn"] for item in c_list
            ])}

        wrapped = diagnostics._make_cond_cat_wrapper(original)

        def comfy_caller_frame():
            cond_or_uncond = [1, 0]
            uuids = [first_uuid, second_uuid]
            return wrapped([
                {"c_crossattn": torch.ones((1, 2, 4))},
                {"c_crossattn": torch.zeros((1, 2, 4))},
            ])

        with mock.patch.object(
            diagnostics,
            "_log_event",
            side_effect=lambda event, **fields: events.append((event, fields)),
        ):
            result = comfy_caller_frame()

        self.assertEqual(tuple(result["c_crossattn"].shape), (2, 2, 4))
        event = [fields for name, fields in events if name == "cond_cat"][-1]
        self.assertEqual(event["markers"], (1, 0))
        self.assertEqual(event["uuids"], (str(first_uuid), str(second_uuid)))
        self.assertIn("marker:1;role:negative", event["inputs"])
        self.assertIn("marker:0;role:positive", event["inputs"])

    def test_preprocess_probe_reports_exact_nonfinite_boundary_and_role(self):
        class Owner:
            llm_adapter = object()

        expected = torch.full((1, 2, 4), float("nan"))

        def original(_owner, _embeds, _ids, t5xxl_weights=None):
            return expected

        wrapped = diagnostics._make_preprocess_text_embeds_wrapper(original)
        events = []
        token = diagnostics._current_model_cond.set({
            "sample_id": "sample-negative",
            "role": "negative",
            "model": "model-id",
            "patch": diagnostics._patcher_identity(None),
        })
        try:
            with mock.patch.object(
                diagnostics,
                "_log_event",
                side_effect=lambda event, **fields: events.append((event, fields)),
            ):
                result = wrapped(
                    Owner(),
                    torch.ones((1, 2, 4)),
                    torch.tensor([[1, 2]], dtype=torch.int32),
                    t5xxl_weights=torch.ones((1, 2, 1)),
                )
        finally:
            diagnostics._current_model_cond.reset(token)

        self.assertIs(result, expected)
        event = [
            fields for name, fields in events
            if name == "preprocess_text_embeds"
        ][-1]
        self.assertEqual(event["sample"], "sample-negative")
        self.assertEqual(event["role"], "negative")
        self.assertEqual(
            event["provenance"],
            "became_nonfinite_in_preprocess_text_embeds",
        )

    def test_llm_adapter_probe_identifies_first_bad_top_level_stage(self):
        class BadBlock(torch.nn.Module):
            def forward(self, value):
                return value * torch.tensor(float("nan"))

        class Adapter(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(8, 4)
                self.in_proj = torch.nn.Identity()
                self.blocks = torch.nn.ModuleList([BadBlock()])
                self.out_proj = torch.nn.Identity()
                self.norm = torch.nn.Identity()

            def forward(self, source_hidden_states, target_input_ids):
                value = self.in_proj(self.embed(target_input_ids))
                for block in self.blocks:
                    value = block(value)
                return self.norm(self.out_proj(value))

        adapter = Adapter()
        wrapped = diagnostics._make_llm_adapter_wrapper(Adapter.forward)
        events = []
        with mock.patch.object(
            diagnostics,
            "_log_event",
            side_effect=lambda event, **fields: events.append((event, fields)),
        ):
            result = wrapped(
                adapter,
                torch.ones((1, 2, 4)),
                torch.tensor([[1, 2]], dtype=torch.int64),
            )

        self.assertTrue(torch.isnan(result).all())
        event = [fields for name, fields in events if name == "llm_adapter"][-1]
        self.assertEqual(event["first_bad_stage"], "block_0")
        self.assertEqual(event["provenance"], "became_nonfinite_at_block_0")
        self.assertIn("embed:{", event["stages"])
        self.assertIn("block_0:{", event["stages"])


if __name__ == "__main__":
    unittest.main()
