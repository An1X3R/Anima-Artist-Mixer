import logging
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from anima_mixer import semantic_diagnostics as diagnostics
from anima_mixer.patching import tensor_diagnostic_snapshot, tensor_value_signature


def _snapshot(fingerprint, shape=(1, 2, 2), dtype="torch.float32"):
    return {
        "is_tensor": True,
        "shape": shape,
        "dtype": dtype,
        "device": "cpu",
        "bad": 0,
        "nan": 0,
        "inf": 0,
        "fingerprint": fingerprint,
    }


def _initialize(
    state,
    *,
    prompt,
    base_fp="base",
    ids_fp="ids",
    label="artist",
    artist_fp="artist",
):
    diagnostics.initialize_state(
        state,
        pack_id=f"pack-{prompt}",
        base_prompt=prompt,
        labels=(label,),
        weights=(1.0,),
        normalize_weights=True,
        alignment_mode="base_anchored",
        strength=1.0,
        apply_to_uncond=False,
        uncond_strength=1.0,
        base_raw_snapshot=_snapshot(base_fp),
        artist_raw_snapshots=(_snapshot(artist_fp),),
        artist_ids_snapshots=(
            _snapshot(f"{artist_fp}-ids", shape=(4,), dtype="torch.int32"),
        ),
        artist_t5_weights_snapshots=(
            _snapshot(f"{artist_fp}-weights", shape=(4,)),
        ),
        encoded_raw_snapshots={
            "base": _snapshot(base_fp),
            "artists": (_snapshot(artist_fp),),
        },
        base_ids_snapshot=_snapshot(ids_fp, shape=(4,), dtype="torch.int32"),
        base_weights_snapshot=_snapshot("weights", shape=(4,)),
    )


def _inputs(*, base_fp="base", ids_fp="ids", artist_fp="artist"):
    return {
        "base_raw": _snapshot(base_fp),
        "artist_raws": (_snapshot(artist_fp),),
        "artist_ids": (
            _snapshot(f"{artist_fp}-ids", shape=(4,), dtype="torch.int32"),
        ),
        "artist_t5_weights": (
            _snapshot(f"{artist_fp}-weights", shape=(4,)),
        ),
        "base_ids": _snapshot(ids_fp, shape=(4,), dtype="torch.int32"),
        "base_weights": _snapshot("weights", shape=(4,)),
    }


class SemanticDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.enabled_environment = mock.patch.dict(
            os.environ,
            {diagnostics.ENV_NAME: "1"},
        )
        self.enabled_environment.start()
        diagnostics._reset_history_for_tests()

    def tearDown(self):
        diagnostics._reset_history_for_tests()
        self.enabled_environment.stop()

    def test_prompt_change_with_identical_context_is_flagged(self):
        model = object()
        first = {"_shared_model_ref": model}
        second = {"_shared_model_ref": model}
        _initialize(first, prompt="old prompt", base_fp="old", ids_fp="old_ids")
        _initialize(second, prompt="new prompt", base_fp="new", ids_fp="new_ids")

        inputs = _inputs(base_fp="old", ids_fp="old_ids")
        context = _snapshot("same-context", shape=(2, 8, 4), dtype="torch.float16")
        with mock.patch.dict(os.environ, {diagnostics.ENV_NAME: "1"}):
            diagnostics.begin_run(
                first,
                execution_index=1,
                patcher_id=1,
                patch_identity=("patch", "patch", 1),
                current_inputs=inputs,
            )
            diagnostics.record_context(
                first,
                snapshot=context,
                context_key="c_crossattn",
                cond_or_uncond=(1, 0),
                conditioning_uuids=("first",),
            )
            diagnostics.record_stage(first, "artist_sum", _snapshot("first-sum"))
            diagnostics.record_stage(first, "mixed_context", _snapshot("first-mix"))
            diagnostics.end_run(first, outcome="cleanup")

            diagnostics.begin_run(
                second,
                execution_index=1,
                patcher_id=2,
                patch_identity=("patch", "patch", 1),
                current_inputs=_inputs(base_fp="new", ids_fp="new_ids"),
            )
            diagnostics.record_context(
                second,
                snapshot=context,
                context_key="c_crossattn",
                cond_or_uncond=(1, 0),
                conditioning_uuids=("second",),
            )
            diagnostics.end_run(second, outcome="cleanup")

        observed = second["_semantic_diag"]["last_run"]
        self.assertEqual(
            observed["context"]["transition"],
            "base_prompt_changed_context_same_suspect",
        )
        self.assertIn(
            "base_context_same_after_base_prompt_change",
            observed["context"]["suspects"],
        )

    def test_artist_only_change_does_not_flag_unchanged_base_context(self):
        model = object()
        first = {"_shared_model_ref": model}
        second = {"_shared_model_ref": model}
        _initialize(
            first,
            prompt="same base",
            label="old artist",
            artist_fp="old-artist",
        )
        _initialize(
            second,
            prompt="same base",
            label="new artist",
            artist_fp="new-artist",
        )
        context = _snapshot("same-context", shape=(2, 8, 4), dtype="torch.float16")

        diagnostics.begin_run(
            first,
            execution_index=1,
            patcher_id=1,
            patch_identity=("patch", "patch", 1),
            current_inputs=_inputs(artist_fp="old-artist"),
        )
        diagnostics.record_context(
            first,
            snapshot=context,
            context_key="c_crossattn",
            cond_or_uncond=(1, 0),
            conditioning_uuids=("first",),
        )
        diagnostics.record_stage(first, "artist_sum", _snapshot("old-sum"))
        diagnostics.record_stage(first, "mixed_context", _snapshot("old-mix"))
        diagnostics.end_run(first, outcome="cleanup")

        diagnostics.begin_run(
            second,
            execution_index=1,
            patcher_id=2,
            patch_identity=("patch", "patch", 1),
            current_inputs=_inputs(artist_fp="new-artist"),
        )
        diagnostics.record_context(
            second,
            snapshot=context,
            context_key="c_crossattn",
            cond_or_uncond=(1, 0),
            conditioning_uuids=("second",),
        )

        observed = second["_semantic_diag"]["active_run"]["context"]
        self.assertEqual(
            observed["transition"],
            "base_prompt_unchanged_context_same",
        )
        self.assertEqual(observed["suspects"], ())

    def test_same_artist_sum_after_changed_encoded_inputs_is_flagged(self):
        model = object()
        first = {"_shared_model_ref": model}
        second = {"_shared_model_ref": model}
        _initialize(first, prompt="base", artist_fp="old-artist")
        _initialize(
            second,
            prompt="base",
            label="replacement",
            artist_fp="new-artist",
        )
        context = _snapshot("context", shape=(2, 8, 4), dtype="torch.float16")
        artist_sum = _snapshot("stale-sum", shape=(1, 8, 4), dtype="torch.float16")

        diagnostics.begin_run(
            first,
            execution_index=1,
            patcher_id=1,
            patch_identity=("patch", "patch", 1),
            current_inputs=_inputs(artist_fp="old-artist"),
        )
        diagnostics.record_context(
            first,
            snapshot=context,
            context_key="c_crossattn",
            cond_or_uncond=(1, 0),
            conditioning_uuids=("first",),
        )
        diagnostics.record_stage(first, "artist_sum", artist_sum)
        diagnostics.record_stage(first, "mixed_context", _snapshot("first-mix"))
        diagnostics.end_run(first, outcome="cleanup")

        diagnostics.begin_run(
            second,
            execution_index=1,
            patcher_id=2,
            patch_identity=("patch", "patch", 1),
            current_inputs=_inputs(artist_fp="new-artist"),
        )
        diagnostics.record_context(
            second,
            snapshot=context,
            context_key="c_crossattn",
            cond_or_uncond=(1, 0),
            conditioning_uuids=("second",),
        )
        with self.assertLogs(
            "anima_mixer.semantic_diagnostics",
            level="WARNING",
        ) as logs:
            diagnostics.record_stage(second, "artist_sum", artist_sum)

        joined = "\n".join(logs.output)
        self.assertIn("inputs_changed_output_same_suspect", joined)
        self.assertIn("artist_sum_same_after_input_change", joined)

    def test_same_mixed_context_after_changed_dependencies_is_flagged(self):
        model = object()
        first = {"_shared_model_ref": model}
        second = {"_shared_model_ref": model}
        _initialize(first, prompt="old", base_fp="old", artist_fp="old-artist")
        _initialize(second, prompt="new", base_fp="new", artist_fp="new-artist")

        def run_context(
            state,
            context_fp,
            artist_fp,
            *,
            input_base_fp,
            input_ids_fp,
            mixed_fp=None,
        ):
            diagnostics.begin_run(
                state,
                execution_index=1,
                patcher_id=1,
                patch_identity=("patch", "patch", 1),
                current_inputs=_inputs(
                    base_fp=input_base_fp,
                    ids_fp=input_ids_fp,
                    artist_fp=artist_fp,
                ),
            )
            diagnostics.record_context(
                state,
                snapshot=_snapshot(
                    context_fp,
                    shape=(2, 8, 4),
                    dtype="torch.float16",
                ),
                context_key="c_crossattn",
                cond_or_uncond=(1, 0),
                conditioning_uuids=(context_fp,),
            )
            diagnostics.record_stage(
                state,
                "artist_sum",
                _snapshot(
                    f"{artist_fp}-sum",
                    shape=(1, 8, 4),
                    dtype="torch.float16",
                ),
            )
            diagnostics.record_stage(
                state,
                "mixed_context",
                _snapshot(
                    mixed_fp or f"{context_fp}-mixed",
                    shape=(2, 8, 4),
                    dtype="torch.float16",
                ),
            )

        run_context(
            first,
            "old-context",
            "old-artist",
            input_base_fp="old",
            input_ids_fp="ids",
        )
        diagnostics.end_run(first, outcome="cleanup")

        diagnostics.begin_run(
            second,
            execution_index=1,
            patcher_id=1,
            patch_identity=("patch", "patch", 1),
            current_inputs=_inputs(
                base_fp="new",
                ids_fp="ids",
                artist_fp="new-artist",
            ),
        )
        diagnostics.record_context(
            second,
            snapshot=_snapshot("new-context", shape=(2, 8, 4), dtype="torch.float16"),
            context_key="c_crossattn",
            cond_or_uncond=(1, 0),
            conditioning_uuids=("new-context",),
        )
        diagnostics.record_stage(
            second,
            "artist_sum",
            _snapshot("new-artist-sum", shape=(1, 8, 4), dtype="torch.float16"),
        )
        with self.assertLogs(
            "anima_mixer.semantic_diagnostics",
            level="WARNING",
        ) as logs:
            diagnostics.record_stage(
                second,
                "mixed_context",
                _snapshot("old-context-mixed", shape=(2, 8, 4), dtype="torch.float16"),
            )

        joined = "\n".join(logs.output)
        self.assertIn("mixed_context_same_after_input_change", joined)

    def test_finite_input_mutation_after_state_is_reported(self):
        state = {"_shared_model_ref": object()}
        _initialize(state, prompt="prompt", base_fp="encoded", ids_fp="ids")
        with mock.patch.dict(os.environ, {diagnostics.ENV_NAME: "1"}):
            diagnostics.begin_run(
                state,
                execution_index=1,
                patcher_id=1,
                patch_identity=("patch", "patch", 1),
                current_inputs=_inputs(base_fp="changed"),
            )
        self.assertEqual(
            state["_semantic_diag"]["active_run"]["state_drift"],
            ("base_raw",),
        )
        self.assertEqual(
            state["_semantic_diag"]["active_run"]["encode_to_state_drift"],
            (),
        )

    def test_encode_to_state_drift_is_separate_from_late_mutation(self):
        state = {"_shared_model_ref": object()}
        diagnostics.initialize_state(
            state,
            pack_id="pack",
            base_prompt="prompt",
            labels=("artist",),
            weights=(1.0,),
            normalize_weights=True,
            alignment_mode="base_anchored",
            strength=1.0,
            apply_to_uncond=False,
            uncond_strength=1.0,
            base_raw_snapshot=_snapshot("state"),
            artist_raw_snapshots=(_snapshot("artist"),),
            artist_ids_snapshots=(
                _snapshot("artist-ids", shape=(4,), dtype="torch.int32"),
            ),
            artist_t5_weights_snapshots=(
                _snapshot("artist-weights", shape=(4,)),
            ),
            encoded_raw_snapshots={
                "base": _snapshot("encoded"),
                "artists": (_snapshot("artist"),),
            },
            base_ids_snapshot=_snapshot("ids", shape=(4,), dtype="torch.int32"),
            base_weights_snapshot=_snapshot("weights", shape=(4,)),
        )
        with mock.patch.dict(os.environ, {diagnostics.ENV_NAME: "1"}):
            diagnostics.begin_run(
                state,
                execution_index=1,
                patcher_id=1,
                patch_identity=("patch", "patch", 1),
                current_inputs=_inputs(base_fp="state"),
            )
        run = state["_semantic_diag"]["active_run"]
        self.assertEqual(run["state_drift"], ())
        self.assertEqual(run["encode_to_state_drift"], ("base_raw",))

    def test_exact_snapshot_catches_weak_signature_collision(self):
        first = torch.zeros(128, dtype=torch.float32)
        second = first.clone()
        sampled = set(
            torch.linspace(0, 127, 64).round().to(dtype=torch.long).tolist()
        )
        free = [index for index in range(128) if index not in sampled]
        self.assertGreaterEqual(len(free), 2)
        first[free[0]] = 1.0
        first[free[1]] = 2.0
        second[free[0]] = 2.0
        second[free[1]] = 1.0

        self.assertEqual(tensor_value_signature(first), tensor_value_signature(second))
        self.assertNotEqual(
            tensor_diagnostic_snapshot(first)["fingerprint"],
            tensor_diagnostic_snapshot(second)["fingerprint"],
        )

    def test_disabled_mode_does_not_start_or_record(self):
        state = {"_shared_model_ref": object()}
        _initialize(state, prompt="prompt")
        with mock.patch.dict(os.environ, {diagnostics.ENV_NAME: "0"}):
            self.assertFalse(diagnostics.needs_run_start(state, 1))
            diagnostics.begin_run(
                state,
                execution_index=1,
                patcher_id=1,
                patch_identity=("patch", "patch", 1),
                current_inputs={},
            )
            diagnostics.record_cache_lookup(state, "mixed_context", True)
        self.assertIsNone(state["_semantic_diag"]["active_run"])

    def test_disabled_mode_skips_state_creation(self):
        state = {"_shared_model_ref": object()}
        with mock.patch.dict(os.environ, {diagnostics.ENV_NAME: "0"}):
            diagnostics.initialize_state(
                state,
                pack_id="pack",
                base_prompt="prompt",
                labels=("artist",),
                weights=(1.0,),
                normalize_weights=True,
                alignment_mode="base_anchored",
                strength=1.0,
                apply_to_uncond=False,
                uncond_strength=1.0,
                base_raw_snapshot=_snapshot("base"),
                artist_raw_snapshots=(_snapshot("artist"),),
                artist_ids_snapshots=(),
                artist_t5_weights_snapshots=(),
                encoded_raw_snapshots={},
                base_ids_snapshot=_snapshot("ids"),
                base_weights_snapshot=_snapshot("weights"),
            )
        self.assertNotIn("_semantic_diag", state)

    def test_file_logging_writes_relevant_session_log_and_latest_pointer(self):
        isolated_logger = logging.Logger("semantic-file-log-test")
        isolated_logger.setLevel(logging.DEBUG)
        isolated_logger.propagate = False
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                with mock.patch.dict(os.environ, {
                    diagnostics.FILE_LOG_ENV_NAME: "1",
                    diagnostics.LOG_DIR_ENV_NAME: temp_dir,
                }):
                    log_path = diagnostics.setup_comfy_file_logging(
                        isolated_logger
                    )
                    duplicate_path = diagnostics.setup_comfy_file_logging(
                        isolated_logger
                    )
                self.assertEqual(log_path, duplicate_path)
                self.assertEqual(
                    sum(
                        1
                        for handler in isolated_logger.handlers
                        if getattr(
                            handler,
                            diagnostics._FILE_HANDLER_MARKER,
                            False,
                        )
                    ),
                    1,
                )

                isolated_logger.info("unrelated record")
                isolated_logger.info(
                    "[AnimaAdapterMixer] relevant diagnostic"
                )
                isolated_logger.error(
                    "!!! Exception during processing !!! "
                    "[AnimaAdapterMixer] relevant failure"
                )
                for handler in isolated_logger.handlers:
                    handler.flush()

                content = Path(log_path).read_text(encoding="utf-8")
                self.assertIn("event=file_log_ready", content)
                self.assertIn("relevant diagnostic", content)
                self.assertIn("relevant failure", content)
                self.assertNotIn("unrelated record", content)
                latest = Path(temp_dir, "AnimaMixer_ComfyUI_latest.txt")
                self.assertEqual(
                    latest.read_text(encoding="utf-8").strip(),
                    log_path,
                )
            finally:
                diagnostics._reset_file_logging_for_tests(isolated_logger)

    def test_file_logging_can_be_disabled_independently(self):
        isolated_logger = logging.Logger("semantic-file-log-disabled-test")
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.dict(os.environ, {
                diagnostics.FILE_LOG_ENV_NAME: "0",
                diagnostics.LOG_DIR_ENV_NAME: temp_dir,
            }):
                log_path = diagnostics.setup_comfy_file_logging(
                    isolated_logger
                )
            self.assertIsNone(log_path)
            self.assertEqual(isolated_logger.handlers, [])
            self.assertEqual(list(Path(temp_dir).iterdir()), [])

    def test_file_logging_failure_is_warning_only(self):
        isolated_logger = logging.Logger("semantic-file-log-failure-test")
        with mock.patch.dict(os.environ, {
            diagnostics.FILE_LOG_ENV_NAME: "1",
            diagnostics.LOG_DIR_ENV_NAME: r"E:\codex logs",
        }), mock.patch.object(
            diagnostics.Path,
            "mkdir",
            side_effect=OSError("simulated write failure"),
        ), self.assertLogs(
            "anima_mixer.semantic_diagnostics",
            level="WARNING",
        ) as logs:
            log_path = diagnostics.setup_comfy_file_logging(isolated_logger)

        self.assertIsNone(log_path)
        self.assertEqual(isolated_logger.handlers, [])
        self.assertIn("event=file_log_error", "\n".join(logs.output))


if __name__ == "__main__":
    unittest.main()
