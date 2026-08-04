import unittest
import importlib
from pathlib import Path
from types import SimpleNamespace

import torch

from backend import mvs_mvstta


class MVSTTAIntegrationTests(unittest.TestCase):
    def test_dependency_loader_resolves_mvstta_components(self):
        deps = mvs_mvstta._get_mvstta_deps()

        self.assertEqual(deps.find_dataset_def("general_eval").__name__, "MVSDataset")
        self.assertEqual(deps.MVSNet.__name__, "MVSNet")
        self.assertEqual(deps.TextureAdaptiveNorm.__name__, "TextureAdaptiveNorm")
        self.assertTrue(callable(deps.save_pfm))

    def test_default_args_use_texture_norm_and_reprojection_v5(self):
        args = mvs_mvstta._build_default_args()

        self.assertTrue(args.use_texture_norm)
        self.assertEqual(args.ds_mode, "reprojectionv5")
        self.assertGreater(args.tta_steps, 0)

    def test_built_model_exposes_texture_norm_flag(self):
        deps = mvs_mvstta._get_mvstta_deps()
        args = mvs_mvstta._build_default_args()

        model = mvs_mvstta._build_model(deps, args)

        self.assertTrue(model.use_texture_norm)

    def test_tocuda_accepts_explicit_cpu_device(self):
        sample = {
            "imgs": torch.ones(1),
            "proj_matrices": {"stage3": torch.zeros(1)},
            "filename": ["scene/{}/00000000{}"],
        }

        moved = mvs_mvstta.tocuda(sample, torch.device("cpu"))

        self.assertEqual(moved["imgs"].device.type, "cpu")
        self.assertEqual(moved["proj_matrices"]["stage3"].device.type, "cpu")
        self.assertEqual(moved["filename"], sample["filename"])

    def test_pipeline_core_routes_mvstta(self):
        source = (
            Path(__file__).resolve().parents[1] / "backend" / "pipeline_core.py"
        ).read_text(encoding="utf-8")

        self.assertIn("mvstta", source.lower())
        self.assertIn("mvs_mvstta", source)


class MVSTTAProjectLayoutTests(unittest.TestCase):
    def test_mvstta_eval_layout_exists(self):
        root = Path(__file__).resolve().parents[1] / "MVS" / "MVSTTA_eval"

        self.assertTrue((root / "checkpoints").is_dir())
        self.assertTrue((root / "datasets" / "__init__.py").is_file())
        self.assertTrue((root / "datasets" / "general_eval.py").is_file())
        self.assertTrue((root / "networks" / "mvsnet.py").is_file())
        self.assertTrue((root / "filter" / "pcd.py").is_file())

    def test_reprojection_v5_is_available(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "MVS"
            / "MVSTTA_eval"
            / "networks"
            / "module.py"
        ).read_text(encoding="utf-8")

        self.assertIn("reprojectionv5", source)
        self.assertIn("get_reprojection_depth_range_samples_v5", source)

    def test_reprojection_v5_preserves_requested_even_depth_count(self):
        mvs_mvstta._get_mvstta_deps()
        depth_sample = importlib.import_module("networks.depth_sample")
        cur_depth = torch.full((1, 4, 5), 10.0)
        reprojection_volume = torch.zeros(1, 4, 5, 1, 4)

        for ndepth in (32, 8):
            samples = depth_sample.get_reprojection_depth_range_samples_v5(
                cur_depth,
                reprojection_volume,
                ndepth,
                max_depth=20.0,
                min_depth=1.0,
            )

            self.assertEqual(tuple(samples.shape), (1, ndepth, 4, 5))


if __name__ == "__main__":
    unittest.main()
