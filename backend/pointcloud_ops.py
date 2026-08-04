"""
Open3D 点云后处理工具。

支持：
  - 体素下采样 (voxel_down_sample)
  - 统计滤波 (remove_statistical_outlier)
  - 半径滤波 (remove_radius_outlier)
"""

from __future__ import annotations

import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Tuple

import open3d as o3d

from backend.pipeline_core import find_point_clouds


LogFn = Callable[[str], None]


class PointCloudError(RuntimeError):
    """点云处理异常。"""


def _default_log(msg: str) -> None:
    print(msg)


def resolve_input_ply(
    workspace_dir: Path,
    scene_name: str = "scene",
    input_ply: str = "",
) -> Path:
    """
    解析输入 PLY 路径。

    优先级：
      1. 显式传入的 input_ply
      2. workspace/outputs 下最新 PLY
      3. workspace 下任意最新 PLY

    注意：当不指定 input_ply 时，回退扫描不遵守按深度结果来源隔离。
          建议通过 API 层提交任务（自动传递已经校验的 input_ply）。
    """
    if input_ply:
        ply = Path(input_ply).expanduser().resolve()
        if not ply.is_file():
            raise PointCloudError(f"输入点云不存在: {ply}")
        return ply

    warnings.warn(
        "resolve_input_ply() 未指定 input_ply，将回退扫描工作目录。"
        "这不会遵守按深度结果来源隔离，建议通过 API 提交任务。",
        UserWarning,
        stacklevel=2,
    )

    workspace_dir = Path(workspace_dir).expanduser().resolve()
    candidates = []
    outputs = workspace_dir / "outputs"
    if outputs.is_dir():
        candidates = find_point_clouds(outputs)
    if not candidates:
        candidates = find_point_clouds(workspace_dir)
    if not candidates:
        # 再尝试 scene 子目录
        scene_root = workspace_dir / scene_name
        if scene_root.is_dir():
            candidates = find_point_clouds(scene_root)
    if not candidates:
        raise PointCloudError(f"工作目录中未找到 PLY 点云: {workspace_dir}")
    return candidates[0]


def _load_point_cloud(ply_path: Path, log: LogFn) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(str(ply_path))
    if pcd.is_empty():
        raise PointCloudError(f"点云为空或无法读取: {ply_path}")
    log(f"读取点云: {ply_path}  点数={len(pcd.points)}")
    return pcd


def _save_point_cloud(
    pcd: o3d.geometry.PointCloud,
    workspace_dir: Path,
    tag: str,
    log: LogFn,
    output_path: Optional[Path] = None,
) -> Path:
    if output_path is None:
        out_dir = Path(workspace_dir).expanduser().resolve() / "outputs" / "pointcloud_processed"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"{tag}_{stamp}.ply"
    else:
        out_path = Path(output_path).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = o3d.io.write_point_cloud(str(out_path), pcd, write_ascii=False)
    if not ok:
        raise PointCloudError(f"写入点云失败: {out_path}")
    log(f"写出点云: {out_path}  点数={len(pcd.points)}")
    return out_path


def voxel_downsample(
    workspace_dir: Path,
    scene_name: str = "scene",
    input_ply: str = "",
    voxel_size: float = 0.01,
    log: Optional[LogFn] = None,
    output_path: Optional[Path] = None,
) -> Tuple[str, int, int]:
    """体素下采样。返回 (输出路径, 输入点数, 输出点数)。"""
    log = log or _default_log
    if voxel_size <= 0:
        raise PointCloudError("voxel_size 必须 > 0")

    src = resolve_input_ply(workspace_dir, scene_name, input_ply)
    pcd = _load_point_cloud(src, log)
    n0 = len(pcd.points)
    log(f"体素下采样: voxel_size={voxel_size}")
    out = pcd.voxel_down_sample(voxel_size=float(voxel_size))
    n1 = len(out.points)
    out_path = _save_point_cloud(out, workspace_dir, "voxel", log, output_path)
    log(f"体素下采样完成: {n0} -> {n1}")
    return str(out_path), n0, n1


def statistical_filter(
    workspace_dir: Path,
    scene_name: str = "scene",
    input_ply: str = "",
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
    log: Optional[LogFn] = None,
    output_path: Optional[Path] = None,
) -> Tuple[str, int, int]:
    """统计滤波。返回 (输出路径, 输入点数, 输出点数)。"""
    log = log or _default_log
    if nb_neighbors < 1:
        raise PointCloudError("nb_neighbors 必须 >= 1")
    if std_ratio <= 0:
        raise PointCloudError("std_ratio 必须 > 0")

    src = resolve_input_ply(workspace_dir, scene_name, input_ply)
    pcd = _load_point_cloud(src, log)
    n0 = len(pcd.points)
    log(f"统计滤波: nb_neighbors={nb_neighbors}, std_ratio={std_ratio}")
    cl, ind = pcd.remove_statistical_outlier(
        nb_neighbors=int(nb_neighbors),
        std_ratio=float(std_ratio),
    )
    out = pcd.select_by_index(ind)
    n1 = len(out.points)
    out_path = _save_point_cloud(out, workspace_dir, "statistical", log, output_path)
    log(f"统计滤波完成: {n0} -> {n1} (移除 {n0 - n1})")
    return str(out_path), n0, n1


def radius_filter(
    workspace_dir: Path,
    scene_name: str = "scene",
    input_ply: str = "",
    nb_points: int = 16,
    radius: float = 0.05,
    log: Optional[LogFn] = None,
    output_path: Optional[Path] = None,
) -> Tuple[str, int, int]:
    """半径滤波。返回 (输出路径, 输入点数, 输出点数)。"""
    log = log or _default_log
    if nb_points < 1:
        raise PointCloudError("nb_points 必须 >= 1")
    if radius <= 0:
        raise PointCloudError("radius 必须 > 0")

    src = resolve_input_ply(workspace_dir, scene_name, input_ply)
    pcd = _load_point_cloud(src, log)
    n0 = len(pcd.points)
    log(f"半径滤波: nb_points={nb_points}, radius={radius}")
    cl, ind = pcd.remove_radius_outlier(
        nb_points=int(nb_points),
        radius=float(radius),
    )
    out = pcd.select_by_index(ind)
    n1 = len(out.points)
    out_path = _save_point_cloud(out, workspace_dir, "radius", log, output_path)
    log(f"半径滤波完成: {n0} -> {n1} (移除 {n0 - n1})")
    return str(out_path), n0, n1
