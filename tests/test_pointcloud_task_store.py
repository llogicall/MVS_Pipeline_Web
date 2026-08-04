import json
import tempfile
import unittest
from pathlib import Path

from backend.services.pointcloud_task_store import (
    FUSION_METHOD_ROOTS,
    PointCloudTaskStoreError,
    SUPPORTED_FUSION_SOURCES,
    build_processed_output,
    create_task_record,
    list_depth_results,
    list_pointcloud_state,
    resolve_input_ply,
    update_task_record,
)


class PointCloudTaskStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name)
        self.casmvs = self._create_depth_result("CasMVSNet")
        self.mvster = self._create_depth_result("MVSTER")

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_depth_result(self, name: str) -> Path:
        result = self.workspace / "outputs" / name
        depth_dir = result / "scene" / "depth_est"
        depth_dir.mkdir(parents=True)
        (depth_dir / "00000000.pfm").write_bytes(b"pf")
        (depth_dir / "00000000_stage1.pfm").write_bytes(b"pf")
        return result

    def test_lists_only_valid_depth_results_with_stable_ids(self):
        empty = self.workspace / "outputs" / "empty"
        empty.mkdir(parents=True)

        results = list_depth_results(self.workspace)

        self.assertEqual([item["id"] for item in results], ["CasMVSNet", "MVSTER"])
        self.assertEqual(results[0]["depth_count"], 1)
        self.assertTrue(results[0]["supports_fusion"])

    def test_rejects_input_ply_from_another_depth_result(self):
        foreign = self.mvster / "scene" / "foreign.ply"
        foreign.write_bytes(b"ply")

        with self.assertRaises(PointCloudTaskStoreError):
            resolve_input_ply(self.workspace, "CasMVSNet", str(foreign))

    def test_lists_original_and_processed_point_clouds_per_source(self):
        original = self.casmvs / "scene" / "fused.ply"
        original.write_bytes(b"ply")
        processed = self.casmvs / "pointcloud_processed" / "task_voxel.ply"
        processed.parent.mkdir()
        processed.write_bytes(b"ply")
        (self.mvster / "scene" / "other.ply").write_bytes(b"ply")

        state = list_pointcloud_state(self.workspace, "CasMVSNet")

        self.assertEqual(len(state["pointclouds"]), 2)
        categories = {item["category"] for item in state["pointclouds"]}
        self.assertEqual(categories, {"fusion", "processed"})
        self.assertTrue(all("other.ply" not in item["path"] for item in state["pointclouds"]))

    def test_persists_and_updates_records_inside_selected_source(self):
        record = create_task_record(
            self.workspace,
            "CasMVSNet",
            method="voxel",
            params={"voxel_size": 0.01},
            input_ply="scene/fused.ply",
        )
        updated = update_task_record(
            self.workspace,
            "CasMVSNet",
            record["id"],
            status="completed",
            output_ply="pointcloud_processed/result.ply",
        )

        store_path = self.casmvs / "pointcloud_tasks.json"
        payload = json.loads(store_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["tasks"][0]["status"], "completed")
        self.assertEqual(updated["output_ply"], "pointcloud_processed/result.ply")
        self.assertFalse(store_path.with_suffix(".json.tmp").exists())

    def test_builds_unique_processed_output_under_selected_source(self):
        output = build_processed_output(
            self.workspace,
            "CasMVSNet",
            task_id="task123",
            method="radius",
        )

        self.assertEqual(output.parent, self.casmvs / "pointcloud_processed")
        self.assertEqual(output.name, "task123_radius.ply")

    def test_fusion_method_roots_are_consistent_with_supported_sources(self):
        """FUSION_METHOD_ROOTS 和 SUPPORTED_FUSION_SOURCES 必须同步。"""
        self.assertEqual(set(FUSION_METHOD_ROOTS.keys()), SUPPORTED_FUSION_SOURCES)

    def test_list_depth_results_supports_fusion_matches_method_roots(self):
        """list_depth_results 返回的 supports_fusion 必须与 FUSION_METHOD_ROOTS 一致。"""
        results = list_depth_results(self.workspace)
        for item in results:
            expected = item["id"] in FUSION_METHOD_ROOTS
            self.assertEqual(
                item["supports_fusion"],
                expected,
                f"supports_fusion 不一致: {item['id']}",
            )

    def test_relative_to_depth_result_rejects_output_outside_source(self):
        from backend.services.pointcloud_task_store import relative_to_depth_result

        # 合法的输出路径
        legit = self.casmvs / "scene.ply"
        with open(legit, "w") as f:
            f.write("ply")
        relative = relative_to_depth_result(self.workspace, "CasMVSNet", legit)
        self.assertEqual(relative, "scene.ply")

        # 非法输出路径（跨越来源）
        other = self.casmvs / ".." / "MVSTER" / "other.ply"
        normalized = other.resolve()
        with self.assertRaises(PointCloudTaskStoreError):
            relative_to_depth_result(self.workspace, "CasMVSNet", normalized)


if __name__ == "__main__":
    unittest.main()
