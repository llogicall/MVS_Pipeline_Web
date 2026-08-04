"""按深度估计结果隔离点云文件和处理任务。"""

import json
import os
import time
import uuid
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List


TASKS_FILENAME = "pointcloud_tasks.json"
PROCESSED_DIRNAME = "pointcloud_processed"
FUSION_METHOD_ROOTS = {
    "CasMVSNet": "CasMVSNet_eval",
    "MAEMVS": "MAEMVS_eval",
    "MVSTER": "MVSTER_eval",
    "MVSTTA": "MVSTTA_eval",
    "UniMVSNet": "UniMVSNet_eval",
}
SUPPORTED_FUSION_SOURCES = set(FUSION_METHOD_ROOTS.keys())

_STORE_LOCK = RLock()


class PointCloudTaskStoreError(RuntimeError):
    """点云任务存储或路径校验失败。"""


def _outputs_root(workspace_dir: Path) -> Path:
    return Path(workspace_dir).expanduser().resolve() / "outputs"


def _count_depth_maps(result_root: Path) -> int:
    return sum(
        1
        for path in result_root.rglob("*.pfm")
        if path.parent.name == "depth_est" and "_stage" not in path.stem
    )


def resolve_depth_result(workspace_dir: Path, depth_result_id: str) -> Path:
    """解析当前项目内的深度结果目录并阻止目录穿越。"""
    result_id = (depth_result_id or "").strip()
    if not result_id or Path(result_id).name != result_id or "/" in result_id or "\\" in result_id:
        raise PointCloudTaskStoreError("无效的深度估计结果标识")

    outputs_root = _outputs_root(workspace_dir)
    result_root = (outputs_root / result_id).resolve()
    try:
        result_root.relative_to(outputs_root)
    except ValueError as exc:
        raise PointCloudTaskStoreError("深度估计结果不属于当前项目") from exc

    if not result_root.is_dir():
        raise PointCloudTaskStoreError(f"深度估计结果不存在: {result_id}")
    if _count_depth_maps(result_root) == 0:
        raise PointCloudTaskStoreError(f"深度估计结果中没有有效深度图: {result_id}")
    return result_root


def list_depth_results(workspace_dir: Path) -> List[Dict[str, Any]]:
    """列出当前项目内包含有效深度图的结果目录。"""
    outputs_root = _outputs_root(workspace_dir)
    if not outputs_root.is_dir():
        return []

    results = []
    for result_root in sorted(outputs_root.iterdir(), key=lambda path: path.name.lower()):
        if not result_root.is_dir():
            continue
        depth_count = _count_depth_maps(result_root)
        if depth_count == 0:
            continue
        ply_files = list(result_root.rglob("*.ply"))
        results.append(
            {
                "id": result_root.name,
                "method": result_root.name,
                "depth_count": depth_count,
                "has_ply": bool(ply_files),
                "ply_count": len(ply_files),
                "supports_fusion": result_root.name in SUPPORTED_FUSION_SOURCES,
            }
        )
    return results


def _tasks_path(result_root: Path) -> Path:
    return result_root / TASKS_FILENAME


def _read_tasks(result_root: Path) -> List[Dict[str, Any]]:
    tasks_path = _tasks_path(result_root)
    if not tasks_path.is_file():
        return []
    try:
        payload = json.loads(tasks_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PointCloudTaskStoreError(f"无法读取点云任务记录: {tasks_path}") from exc
    tasks = payload.get("tasks", []) if isinstance(payload, dict) else []
    if not isinstance(tasks, list):
        raise PointCloudTaskStoreError(f"点云任务记录格式无效: {tasks_path}")
    return tasks


def _write_tasks(result_root: Path, tasks: List[Dict[str, Any]]) -> None:
    tasks_path = _tasks_path(result_root)
    temporary_path = tasks_path.with_suffix(".json.tmp")
    payload = {"version": 1, "tasks": tasks}
    try:
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(str(temporary_path), str(tasks_path))
    except OSError as exc:
        raise PointCloudTaskStoreError(f"无法保存点云任务记录: {tasks_path}") from exc


def list_pointcloud_state(workspace_dir: Path, depth_result_id: str) -> Dict[str, Any]:
    """返回指定深度结果下的可用点云和持久化任务。"""
    result_root = resolve_depth_result(workspace_dir, depth_result_id)
    pointclouds = []
    for ply_path in result_root.rglob("*.ply"):
        relative_path = ply_path.relative_to(result_root)
        pointclouds.append(
            {
                "path": relative_path.as_posix(),
                "name": ply_path.name,
                "category": (
                    "processed"
                    if PROCESSED_DIRNAME in relative_path.parts
                    else "fusion"
                ),
                "modified_at": ply_path.stat().st_mtime,
                "size": ply_path.stat().st_size,
            }
        )
    pointclouds.sort(key=lambda item: item["modified_at"], reverse=True)
    with _STORE_LOCK:
        tasks = _read_tasks(result_root)
    tasks.sort(key=lambda item: item.get("created_at", 0), reverse=True)
    return {
        "depth_result_id": depth_result_id,
        "pointclouds": pointclouds,
        "tasks": tasks,
    }


def resolve_input_ply(
    workspace_dir: Path,
    depth_result_id: str,
    input_ply: str,
) -> Path:
    """解析深度结果内的相对 PLY 路径。"""
    relative_path = Path((input_ply or "").strip())
    if not str(relative_path) or relative_path.is_absolute():
        raise PointCloudTaskStoreError("必须选择当前深度结果内的输入点云")

    result_root = resolve_depth_result(workspace_dir, depth_result_id)
    ply_path = (result_root / relative_path).resolve()
    try:
        ply_path.relative_to(result_root)
    except ValueError as exc:
        raise PointCloudTaskStoreError("输入点云不属于所选深度估计结果") from exc
    if ply_path.suffix.lower() != ".ply" or not ply_path.is_file():
        raise PointCloudTaskStoreError(f"输入点云不存在或格式无效: {input_ply}")
    return ply_path


def create_task_record(
    workspace_dir: Path,
    depth_result_id: str,
    method: str,
    params: Dict[str, Any],
    input_ply: str = "",
) -> Dict[str, Any]:
    """在所选深度结果下创建持久化点云任务。"""
    result_root = resolve_depth_result(workspace_dir, depth_result_id)
    record = {
        "id": uuid.uuid4().hex[:12],
        "runtime_task_id": None,
        "depth_result_id": depth_result_id,
        "method": method,
        "status": "pending",
        "input_ply": input_ply,
        "output_ply": "",
        "params": params,
        "error": None,
        "created_at": time.time(),
        "started_at": None,
        "finished_at": None,
    }
    with _STORE_LOCK:
        tasks = _read_tasks(result_root)
        tasks.append(record)
        _write_tasks(result_root, tasks)
    return dict(record)


def update_task_record(
    workspace_dir: Path,
    depth_result_id: str,
    record_id: str,
    **updates: Any,
) -> Dict[str, Any]:
    """更新指定来源内的点云任务记录。"""
    result_root = resolve_depth_result(workspace_dir, depth_result_id)
    with _STORE_LOCK:
        tasks = _read_tasks(result_root)
        for record in tasks:
            if record.get("id") != record_id:
                continue
            record.update(updates)
            status = record.get("status")
            if status == "running" and not record.get("started_at"):
                record["started_at"] = time.time()
            if status in {"completed", "failed"} and not record.get("finished_at"):
                record["finished_at"] = time.time()
            _write_tasks(result_root, tasks)
            return dict(record)
    raise PointCloudTaskStoreError(f"点云任务记录不存在: {record_id}")


def build_processed_output(
    workspace_dir: Path,
    depth_result_id: str,
    task_id: str,
    method: str,
) -> Path:
    """生成所选深度结果内唯一的后处理输出路径。"""
    result_root = resolve_depth_result(workspace_dir, depth_result_id)
    output_dir = result_root / PROCESSED_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_task_id = "".join(ch for ch in task_id if ch.isalnum() or ch in "_-")
    safe_method = "".join(ch for ch in method if ch.isalnum() or ch in "_-")
    if not safe_task_id or not safe_method:
        raise PointCloudTaskStoreError("无效的点云任务输出名称")
    return output_dir / f"{safe_task_id}_{safe_method}.ply"


def relative_to_depth_result(
    workspace_dir: Path,
    depth_result_id: str,
    path: Path,
) -> str:
    """把来源内的绝对路径转换为持久化相对路径。"""
    result_root = resolve_depth_result(workspace_dir, depth_result_id)
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(result_root).as_posix()
    except ValueError as exc:
        raise PointCloudTaskStoreError("点云输出不属于所选深度估计结果") from exc
