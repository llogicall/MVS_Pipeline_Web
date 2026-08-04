import unittest
from pathlib import Path

import torch

from backend import mvs_maemvs


class MAEMVSIntegrationTests(unittest.TestCase):
    def test_dependency_loader_resolves_maemvs_components(self):
        deps = mvs_maemvs._get_maemvs_deps()

        self.assertEqual(deps.find_dataset_def("general_eval").__name__, "MVSDataset")
        self.assertEqual(deps.MultiTaskMVSNet.__name__, "MultiTaskMVSNet")
        self.assertTrue(callable(deps.save_pfm))
        self.assertTrue(callable(deps.fuse_depth_maps))

    def test_tocuda_accepts_explicit_cpu_device(self):
        sample = {
            "imgs": torch.ones(1),
            "proj_matrices": {"stage3": torch.zeros(1)},
            "out_name": ["00000000"],
        }

        moved = mvs_maemvs.tocuda(sample, torch.device("cpu"))

        self.assertEqual(moved["imgs"].device.type, "cpu")
        self.assertEqual(moved["proj_matrices"]["stage3"].device.type, "cpu")
        self.assertEqual(moved["out_name"], sample["out_name"])

    def test_checkpoint_keys_enable_mae_and_damw(self):
        use_mae, use_damw = mvs_maemvs._detect_model_options_from_keys(
            [
                "feature.conv0.conv.weight",
                "mae_decoder.decoder.fusion.0.weight",
                "multi_task_loss.raw_weights",
            ],
            use_mae=False,
            use_damw=False,
        )

        self.assertTrue(use_mae)
        self.assertTrue(use_damw)

    def test_pipeline_core_routes_maemvs(self):
        source = (
            Path(__file__).resolve().parents[1] / "backend" / "pipeline_core.py"
        ).read_text(encoding="utf-8")

        self.assertIn("maemvs", source.lower())
        self.assertIn("mvs_maemvs", source)


class MAEMVSProjectLayoutTests(unittest.TestCase):
    def test_maemvs_eval_layout_exists(self):
        root = Path(__file__).resolve().parents[1] / "MVS" / "MAEMVS_eval"

        self.assertTrue((root / "checkpoints").is_dir())
        self.assertTrue((root / "datasets" / "__init__.py").is_file())
        self.assertTrue((root / "datasets" / "general_eval.py").is_file())
        self.assertTrue((root / "models" / "__init__.py").is_file())


if __name__ == "__main__":
    unittest.main()
