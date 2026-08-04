import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from backend import mvs_depth, mvs_mvster, mvs_unimvsnet
from backend.services import task_manager


class DeviceTransferTests(unittest.TestCase):
    def test_tocuda_helpers_accept_explicit_cpu_device(self):
        sample = {
            "imgs": torch.ones(1),
            "nested": [torch.zeros(1), {"depth": torch.arange(2)}],
            "filename": ["scene/{}/00000000{}"],
        }
        device = torch.device("cpu")

        for module in (mvs_depth, mvs_mvster, mvs_unimvsnet):
            with self.subTest(module=module.__name__):
                moved = module.tocuda(sample, device)
                self.assertEqual(moved["imgs"].device.type, "cpu")
                self.assertEqual(moved["nested"][0].device.type, "cpu")
                self.assertEqual(moved["nested"][1]["depth"].device.type, "cpu")
                self.assertEqual(moved["filename"], sample["filename"])


class DependencyImportTests(unittest.TestCase):
    def test_mvs_dependency_loaders_resolve_model_classes(self):
        cases = (
            (mvs_depth._get_casmvsnet_deps, 3, "CascadeMVSNet"),
            (mvs_mvster._get_mvster_deps, 3, "MVS4net"),
            (mvs_unimvsnet._get_unimvsnet_deps, 3, "MVSNet"),
        )

        for loader, model_index, expected_class_name in cases:
            with self.subTest(loader=loader.__name__):
                deps = loader()
                self.assertEqual(deps[model_index].__name__, expected_class_name)

    def test_casmvsnet_filters_unsupported_extra_checkpoint_branches(self):
        state = {
            "feature.weight": torch.ones(1),
            "model_up.sr_decoder.conv1.weight": torch.ones(1),
        }

        filtered = mvs_depth._filter_casmvsnet_model_state(state)

        self.assertIn("feature.weight", filtered)
        self.assertNotIn("model_up.sr_decoder.conv1.weight", filtered)

    def test_unimvsnet_squeezes_batched_stage_maps_before_pfm_save(self):
        batched = np.zeros((1, 4, 5), dtype=np.float32)

        squeezed = mvs_unimvsnet._prepare_pfm_map(batched)

        self.assertEqual(squeezed.shape, (4, 5))
        self.assertEqual(squeezed.dtype, np.float32)


class MVSTERInputShapeTests(unittest.TestCase):
    def test_mvster_images_are_converted_to_views_first_layout(self):
        imgs = torch.zeros(1, 3, 3, 32, 32)

        converted = mvs_mvster._mvster_views_first(imgs)

        self.assertEqual(tuple(converted.shape), (3, 1, 3, 32, 32))

    def test_mvster_depth_interval_stays_as_torch_tensor(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "MVS"
            / "MVSTER_eval"
            / "models"
            / "MVS4Net.py"
        ).read_text(encoding="utf-8")

        self.assertIn(
            "depth_interval = (depth_values[:, -1] - depth_values[:, 0]) / depth_values.size(1)",
            source,
        )

    def test_mvster_tensor_images_are_converted_to_numpy_for_saving(self):
        img = torch.ones(3, 4, 5)

        converted = mvs_mvster._tensor_image_to_numpy(img)

        self.assertEqual(converted.shape, (3, 4, 5))
        self.assertEqual(converted.__class__.__module__, "numpy")

    def test_mvster_uses_64_pixel_resize_base_for_reg2d_alignment(self):
        self.assertEqual(mvs_mvster.MVSTER_RESIZE_BASE, 64)


class SourceGuardTests(unittest.TestCase):
    def test_backends_do_not_force_model_cuda_after_device_selection(self):
        backend_dir = Path(__file__).resolve().parents[1] / "backend"

        for filename in ("mvs_depth.py", "mvs_mvster.py"):
            with self.subTest(filename=filename):
                source = (backend_dir / filename).read_text(encoding="utf-8")
                self.assertNotIn("model.cuda()", source)

    def test_backends_do_not_reference_undefined_mvs_eval_dir(self):
        backend_dir = Path(__file__).resolve().parents[1] / "backend"

        for filename in ("mvs_mvster.py", "mvs_unimvsnet.py"):
            with self.subTest(filename=filename):
                source = (backend_dir / filename).read_text(encoding="utf-8")
                self.assertNotIn("MVS_EVAL_DIR", source)

    def test_casmvsnet_does_not_move_dataset_object_to_device(self):
        source = (
            Path(__file__).resolve().parents[1] / "backend" / "mvs_depth.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn(").to(device)\n    TestImgLoader", source)


class TaskCleanupTests(unittest.TestCase):
    def test_task_cleanup_releases_cuda_cache_when_available(self):
        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.empty_cache"
        ) as empty_cache, patch("torch.cuda.ipc_collect") as ipc_collect:
            task_manager._release_cuda_memory()

        empty_cache.assert_called_once_with()
        ipc_collect.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
