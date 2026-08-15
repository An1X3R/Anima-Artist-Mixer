import types
import unittest

from anima_mixer.model_compat import (
    ANIMA_2B_TO_29B_BLOCKS,
    ANIMA_29B_INSERTED_BLOCKS,
    install_anima_loader_patch,
    resolve_anima_block_layout,
)


class ModelCompatTests(unittest.TestCase):
    @staticmethod
    def _detector(config=None):
        config = dict(config or {"image_model": "anima", "num_blocks": 28})

        def detect_unet_config(state_dict, key_prefix, metadata=None):
            return dict(config)

        module = types.SimpleNamespace(detect_unet_config=detect_unet_config)
        return module

    def test_expanded_checkpoint_uses_serialized_block_count(self):
        module = self._detector()
        self.assertTrue(install_anima_loader_patch(module))

        state_dict = {
            "diffusion_model.blocks.0.self_attn.q_proj.weight": object(),
            "diffusion_model.blocks.39.self_attn.q_proj.weight": object(),
        }
        config = module.detect_unet_config(state_dict, "diffusion_model.")

        self.assertEqual(config["num_blocks"], 40)

    def test_regular_28_block_checkpoint_stays_unchanged(self):
        module = self._detector()
        install_anima_loader_patch(module)

        state_dict = {
            "diffusion_model.blocks.0.self_attn.q_proj.weight": object(),
            "diffusion_model.blocks.27.self_attn.q_proj.weight": object(),
        }
        config = module.detect_unet_config(state_dict, "diffusion_model.")

        self.assertEqual(config["num_blocks"], 28)

    def test_non_anima_detector_result_is_not_changed(self):
        module = self._detector({"image_model": "other", "num_blocks": 12})
        install_anima_loader_patch(module)

        state_dict = {"diffusion_model.blocks.39.weight": object()}
        config = module.detect_unet_config(state_dict, "diffusion_model.")

        self.assertEqual(config, {"image_model": "other", "num_blocks": 12})

    def test_install_is_idempotent(self):
        module = self._detector()

        self.assertTrue(install_anima_loader_patch(module))
        patched = module.detect_unet_config
        self.assertFalse(install_anima_loader_patch(module))
        self.assertIs(module.detect_unet_config, patched)

    def test_keyword_detector_arguments_are_supported(self):
        module = self._detector()
        install_anima_loader_patch(module)

        state_dict = {"blocks.39.weight": object()}
        config = module.detect_unet_config(
            state_dict=state_dict,
            key_prefix="",
            metadata=None,
        )

        self.assertEqual(config["num_blocks"], 40)

    def test_auto_layout_uses_legacy_28_mapping_on_29b(self):
        layout = resolve_anima_block_layout(40, "auto")

        self.assertEqual(layout.resolved_mode, "legacy_28")
        self.assertEqual(layout.selector_block_count, 28)
        self.assertEqual(layout.physical_blocks, ANIMA_2B_TO_29B_BLOCKS)
        self.assertEqual(layout.skipped_physical_blocks, ANIMA_29B_INSERTED_BLOCKS)

    def test_native_40_layout_keeps_every_physical_block(self):
        layout = resolve_anima_block_layout(40, "native_40")

        self.assertEqual(layout.resolved_mode, "native_40")
        self.assertEqual(layout.selector_block_count, 40)
        self.assertEqual(layout.physical_blocks, tuple(range(40)))
        self.assertEqual(layout.logical_index_by_physical[39], 39)

    def test_regular_28_block_model_ignores_29b_compat_mode(self):
        layout = resolve_anima_block_layout(28, "auto")

        self.assertEqual(layout.resolved_mode, "native")
        self.assertEqual(layout.selector_block_count, 28)
        self.assertEqual(layout.physical_blocks, tuple(range(28)))

    def test_legacy_selection_maps_logical_range_to_physical_blocks(self):
        layout = resolve_anima_block_layout(40, "legacy_28")

        self.assertEqual(
            layout.map_selector_blocks(range(20, 28)),
            (29, 31, 32, 34, 35, 37, 38, 39),
        )
        self.assertEqual(layout.logical_index_by_physical[29], 20)


if __name__ == "__main__":
    unittest.main()
