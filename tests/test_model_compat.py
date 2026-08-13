import types
import unittest

from anima_mixer.model_compat import install_anima_loader_patch


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


if __name__ == "__main__":
    unittest.main()
