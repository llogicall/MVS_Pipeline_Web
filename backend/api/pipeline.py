"""
管道执行 API 路由

提供 MVS 管道的前端交互接口：
  - POST /api/pipeline/run      : 创建并启动管道任务
  - GET  /api/pipeline/tasks    : 列出所有历史任务
  - GET  /api/pipeline/status   : 查询单个任务状态
  - POST /api/pipeline/cancel   : 取消正在运行的任务
  - GET  /api/pipeline/logs     : 按偏移量拉取任务日志
  - WS   /api/pipeline/ws/logs  : WebSocket 实时推送日志

核心设计：
  请求到达后立即返回 task_id，实际工作在后台线程池中执行。
  前端通过 WebSocket 被动接收日志，断连时自动回退为轮询模式。
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from backend.auth import get_current_username
from backend.db import get_user_workspace
from backend.pipeline_core import PipelineAction, PipelineConfig, normalize_path
from backend.services.task_manager import (
    TaskStatus,
    cancel_task,
    create_callable_task,
    create_task,
    get_task,
    list_tasks,
    list_user_tasks,
)
from backend.services.pointcloud_task_store import (
    FUSION_METHOD_ROOTS,
    PointCloudTaskStoreError,
    build_processed_output,
    create_task_record,
    relative_to_depth_result,
    resolve_depth_result,
    resolve_input_ply,
    update_task_record,
)

# ---------------------------------------------------------------------------
# 路由定义  每一个文件单独定义一个路由
# ---------------------------------------------------------------------------
# prefix 统一为 /api/pipeline，所有端点以此为前缀
router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])

MVS_ROOT_BY_DEPTH_RESULT = FUSION_METHOD_ROOTS


def _submit_persistent_pointcloud_task(
    workspace_dir: Path,
    depth_result_id: str,
    method: str,
    params: Dict[str, Any],
    input_ply: str,
    username: str,
    runner: Callable[[Callable[[str], None], str], str],
):
    """提交任务并把完整生命周期写入深度结果专属记录。"""
    record = create_task_record(
        workspace_dir,
        depth_result_id,
        method=method,
        params=params,
        input_ply=input_ply,
    )

    def _tracked_runner(log):
        update_task_record(
            workspace_dir,
            depth_result_id,
            record["id"],
            status="running",
        )
        try:
            output_path = runner(log, record["id"])
            relative_output = relative_to_depth_result(
                workspace_dir,
                depth_result_id,
                Path(output_path),
            )
            update_task_record(
                workspace_dir,
                depth_result_id,
                record["id"],
                status="completed",
                output_ply=relative_output,
                error=None,
            )
            return output_path
        except Exception as exc:
            update_task_record(
                workspace_dir,
                depth_result_id,
                record["id"],
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise

    try:
        task = create_callable_task(
            runner=_tracked_runner,
            username=username,
            output_dir=str(resolve_depth_result(workspace_dir, depth_result_id)),
            action=f"pointcloud_{method}",
        )
        update_task_record(
            workspace_dir,
            depth_result_id,
            record["id"],
            runtime_task_id=task.task_id,
        )
    except Exception as exc:
        update_task_record(
            workspace_dir,
            depth_result_id,
            record["id"],
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    return task, record


# ---------------------------------------------------------------------------
# Pydantic 请求/响应模型
# ---------------------------------------------------------------------------
class RunRequest(BaseModel):
    """
    启动管道任务的请求体。

    字段说明：
      image_dir       : 服务器本机图像目录路径，与 image_files 二选一
      workspace_dir   : 工作目录，用于存放 COLMAP、MVS 输入、最终输出等
      scene_name      : 场景标识符，用于在 workspace 下组织子目录
      mvs_root        : MVS 方法目录，默认指向项目根下的 CasMVSNet/
      checkpoint      : 模型权重文件路径，深度估计必需
      action          : 管道动作，可选 full|pose|convert|depth|fusion
      interval_scale  : 深度采样间隔缩放系数
      num_view        : 每张参考视角选取的源视角数（含自身）
      max_h / max_w   : 深度估计时输入图像的最大高/宽
      conf            : 点云融合的置信度阈值
      thres_view      : 几何一致性检查的最小一致视角数
      num_worker      : 多进程融合的 worker 数量
      filter_method   : 融合方法 normal（深度滤波）或 gipuma（需外部 fusibile 二进制）
      camera_model    : pycolmap 特征提取使用的相机模型类型
      single_camera   : 是否强制所有图像使用同一相机内参
      use_gpu         : 是否启用 CUDA（若为 False 则设置 CUDA_VISIBLE_DEVICES=""）
      fusibile_exe_path : gipuma 融合所需的外部 fusibile 二进制路径
      sfm_*           : pycolmap SIFT 特征提取的保守参数，防止 Windows 下崩溃
    """
    image_dir: Optional[str] = None
    workspace_dir: str = str(Path("runs").resolve())
    scene_name: str = "scene"
    mvs_root: str = ""
    checkpoint: Optional[str] = None
    action: str = "full"
    interval_scale: float = 1.06
    num_view: int = 5
    max_h: int = 864
    max_w: int = 1152
    conf: float = 0.9
    thres_view: int = 5
    num_worker: int = 4
    filter_method: str = "normal"
    camera_model: str = "SIMPLE_RADIAL"
    single_camera: bool = True
    use_gpu: bool = True
    fusibile_exe_path: Optional[str] = None
    sfm_max_image_size: int = 800
    sfm_max_num_features: int = 4096
    sfm_num_threads: int = 1


class TaskInfo(BaseModel):
    """
    任务信息的响应模型，用于向客户端返回任务快照。

    字段说明：
      task_id     : 任务的唯一标识符（UUID 前 12 位）
      status      : 当前状态 pending / running / completed / failed / cancelled
      created_at  : 任务创建时间戳（UNIX 时间）
      finished_at : 任务完成/失败时间戳，未结束时为 None
      error       : 错误信息，状态为 failed 时有值
      result_ply  : 最新 PLY 点云文件路径
      output_dir  : 管道输出目录
      log_count   : 已产生的日志行数
      action      : 管道动作（pose/convert/depth/fusion/full）
    """
    task_id: str
    status: str
    created_at: float
    finished_at: Optional[float] = None
    error: Optional[str] = None
    result_ply: Optional[str] = None
    output_dir: Optional[str] = None
    log_count: int = 0
    action: str = "full"


# ---------------------------------------------------------------------------
# 启动管道任务
# ---------------------------------------------------------------------------
@router.post("/run")
def run_pipeline_endpoint(req: RunRequest, username: str = Depends(get_current_username)):
    """
    接收前端参数，构造 PipelineConfig，提交到后台线程池执行。

    返回示例：
      {"task_id": "a1b2c3d4e5f6", "status": "pending"}
    """
    from backend.pipeline_core import PipelineError
    try:
        # 解析 MVS 方法目录，用户未指定时默认指向 CasMVSNet/
        resolved_mvs = normalize_path(
            req.mvs_root or str(Path(__file__).resolve().parent.parent.parent / "CasMVSNet"),
            "MVS root",
        )
        # 解析 checkpoint 路径，非深度估计动作时使用占位路径
        resolved_checkpoint = (
            normalize_path(req.checkpoint, "Checkpoint")
            if req.checkpoint
            else Path("unused.ckpt")
        )
        # 解析图像目录路径
        resolved_image_dir = (
            normalize_path(req.image_dir, "Image directory")
            if req.image_dir
            else Path(req.workspace_dir) / "uploaded_images" / req.scene_name
        )

        # 组装 PipelineConfig，传递给后台任务
        config = PipelineConfig(
            image_dir=resolved_image_dir,
            workspace_dir=normalize_path(req.workspace_dir, "Workspace directory"),
            scene_name=req.scene_name,
            mvs_root=resolved_mvs,
            checkpoint=resolved_checkpoint,
            interval_scale=req.interval_scale,
            num_view=req.num_view,
            max_h=req.max_h,
            max_w=req.max_w,
            conf=req.conf,
            thres_view=req.thres_view,
            num_worker=req.num_worker,
            filter_method=req.filter_method,
            camera_model=req.camera_model,
            single_camera=req.single_camera,
            use_gpu=req.use_gpu,
            fusibile_exe_path=Path(req.fusibile_exe_path).resolve()
            if req.fusibile_exe_path
            else None,
            action=PipelineAction(req.action),
            sfm_max_image_size=req.sfm_max_image_size,
            sfm_max_num_features=req.sfm_max_num_features,
            sfm_num_threads=req.sfm_num_threads,
        )

        task = create_task(config, username=username)
        return {"task_id": task.task_id, "status": task.status.value}
    except PipelineError as e:
        raise HTTPException(400, str(e))
    except ValueError as e:
        raise HTTPException(400, f"Invalid parameter: {e}")
    except Exception as e:
        raise HTTPException(500, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# 深度融合
# ---------------------------------------------------------------------------
class DeepFusionRequest(BaseModel):
    workspace_dir: str = ""
    scene_name: str = "scene"
    depth_result_id: str
    conf: float = 0.9
    thres_view: int = 5
    num_worker: int = 4


@router.post("/deep-fusion")
def run_deep_fusion_endpoint(req: DeepFusionRequest, username: str = Depends(get_current_username)):
    """运行深度融合任务"""
    from backend.pipeline_core import (
        PipelineError,
        prepare_workspace,
        run_point_fusion,
    )

    try:
        workspace_dir = (
            normalize_path(req.workspace_dir, "Workspace directory")
            if req.workspace_dir
            else get_user_workspace(username)
        )
        resolve_depth_result(workspace_dir, req.depth_result_id)
        method_dir = MVS_ROOT_BY_DEPTH_RESULT.get(req.depth_result_id)
        if not method_dir:
            raise PointCloudTaskStoreError(
                f"深度估计结果不支持点云融合: {req.depth_result_id}"
            )

        # 验证 CasMVSNet 输入目录存在（融合所需）
        from backend.pipeline_core import prepare_workspace as _pw
        _ws = _pw(workspace_dir, req.scene_name)
        if not _ws.casmvsnet_scene.is_dir():
            raise PointCloudTaskStoreError(
                f"CasMVSNet 输入目录不存在，请先运行格式转换: {_ws.casmvsnet_scene}"
            )
        required_casmvsnet = [
            _ws.casmvsnet_scene / "cams",
            _ws.casmvsnet_scene / "pair.txt",
        ]
        missing = [str(p) for p in required_casmvsnet if not p.exists()]
        if missing:
            raise PointCloudTaskStoreError(
                f"CasMVSNet 输入不完整，请先运行格式转换。缺失: {', '.join(missing)}"
            )

        config = PipelineConfig(
            image_dir=workspace_dir / "dummy",
            workspace_dir=workspace_dir,
            mvs_root=Path(__file__).resolve().parents[2] / "MVS" / method_dir,
            checkpoint=Path("unused"),
            scene_name=req.scene_name,
            conf=req.conf,
            thres_view=req.thres_view,
            num_worker=req.num_worker,
            action=PipelineAction.FUSION,
        )

        def _runner(log, record_id):
            del record_id
            ply_path = run_point_fusion(
                config,
                prepare_workspace(workspace_dir, req.scene_name),
                log,
            )
            if not ply_path:
                raise PipelineError("点云融合未生成 PLY 文件")
            return str(ply_path)

        task, record = _submit_persistent_pointcloud_task(
            workspace_dir=workspace_dir,
            depth_result_id=req.depth_result_id,
            method="fusion",
            params={
                "conf": req.conf,
                "thres_view": req.thres_view,
                "num_worker": req.num_worker,
            },
            input_ply="",
            username=username,
            runner=_runner,
        )
        return {
            "task_id": task.task_id,
            "record_id": record["id"],
            "depth_result_id": req.depth_result_id,
            "status": task.status.value,
        }
    except PointCloudTaskStoreError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as e:
        raise HTTPException(500, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Open3D 点云后处理
# ---------------------------------------------------------------------------
class PointCloudProcessRequest(BaseModel):
    """点云后处理请求"""
    method: str = Field(..., description="voxel | statistical | radius")
    workspace_dir: str = ""
    scene_name: str = "scene"
    depth_result_id: str
    input_ply: str
    # voxel downsample
    voxel_size: float = 0.01
    # statistical filter
    nb_neighbors: int = 20
    std_ratio: float = 2.0
    # radius filter
    nb_points: int = 16
    radius: float = 0.05


@router.post("/pointcloud/process")
def run_pointcloud_process(req: PointCloudProcessRequest, username: str = Depends(get_current_username)):
    """
    启动 Open3D 点云后处理任务：
      - voxel: 体素下采样
      - statistical: 统计滤波
      - radius: 半径滤波
    """
    from backend.pipeline_core import normalize_path
    from backend.pointcloud_ops import (
        PointCloudError,
        radius_filter,
        statistical_filter,
        voxel_downsample,
    )
    method = (req.method or "").strip().lower()
    if method not in {"voxel", "statistical", "radius"}:
        raise HTTPException(400, f"不支持的点云处理方法: {req.method}")

    try:
        workspace_dir = (
            normalize_path(req.workspace_dir, "Workspace directory")
            if req.workspace_dir
            else get_user_workspace(username)
        )
    except Exception as e:
        raise HTTPException(400, str(e))

    try:
        input_path = resolve_input_ply(
            workspace_dir,
            req.depth_result_id,
            req.input_ply,
        )
    except PointCloudTaskStoreError as exc:
        raise HTTPException(400, str(exc)) from exc

    params = {}
    if method == "voxel":
        params = {"voxel_size": req.voxel_size}
    elif method == "statistical":
        params = {
            "nb_neighbors": req.nb_neighbors,
            "std_ratio": req.std_ratio,
        }
    else:
        params = {"nb_points": req.nb_points, "radius": req.radius}

    def _runner(log, record_id):
        try:
            output_path = build_processed_output(
                workspace_dir,
                req.depth_result_id,
                record_id,
                method,
            )
            if method == "voxel":
                out_path, n0, n1 = voxel_downsample(
                    workspace_dir=workspace_dir,
                    scene_name=req.scene_name,
                    input_ply=str(input_path),
                    voxel_size=req.voxel_size,
                    log=log,
                    output_path=output_path,
                )
            elif method == "statistical":
                out_path, n0, n1 = statistical_filter(
                    workspace_dir=workspace_dir,
                    scene_name=req.scene_name,
                    input_ply=str(input_path),
                    nb_neighbors=req.nb_neighbors,
                    std_ratio=req.std_ratio,
                    log=log,
                    output_path=output_path,
                )
            else:
                out_path, n0, n1 = radius_filter(
                    workspace_dir=workspace_dir,
                    scene_name=req.scene_name,
                    input_ply=str(input_path),
                    nb_points=req.nb_points,
                    radius=req.radius,
                    log=log,
                    output_path=output_path,
                )
            log(f"点数变化: {n0} -> {n1}")
            return out_path
        except PointCloudError as exc:
            from backend.pipeline_core import PipelineError
            raise PipelineError(str(exc)) from exc

    try:
        task, record = _submit_persistent_pointcloud_task(
            workspace_dir=workspace_dir,
            depth_result_id=req.depth_result_id,
            method=method,
            params=params,
            input_ply=req.input_ply,
            username=username,
            runner=_runner,
        )
    except PointCloudTaskStoreError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {
        "task_id": task.task_id,
        "record_id": record["id"],
        "depth_result_id": req.depth_result_id,
        "status": task.status.value,
        "method": method,
    }


# ---------------------------------------------------------------------------
# 列出所有历史任务（按创建时间倒序）
# ---------------------------------------------------------------------------
@router.get("/tasks", response_model=List[TaskInfo])
def list_pipeline_tasks(username: str = Depends(get_current_username)):
    """返回当前用户的所有任务摘要列表，用于前端任务历史面板。"""
    return [
        TaskInfo(
            task_id=t.task_id,
            status=t.status.value,
            created_at=t.created_at,
            finished_at=t.finished_at,
            error=t.error,
            result_ply=t.result_ply,
            output_dir=t.output_dir,
            log_count=len(t.logs),
        )
        for t in list_user_tasks(username)
    ]


# ---------------------------------------------------------------------------
# 查询单个任务状态
# ---------------------------------------------------------------------------
@router.get("/status/{task_id}")
def get_task_status(task_id: str, username: str = Depends(get_current_username)):
    """
    按 task_id 查询任务的最新状态。

    参数：
      task_id : 任务的唯一标识符

    返回：
      任务不存在时返回 404 + {"error": "Task not found"}
      存在时返回 TaskInfo 结构体
    """
    task = get_task(task_id)
    if not task:
        return {"error": "Task not found"}, 404
    if task.username and task.username != username:
        return {"error": "Task not found"}, 404
    action = "full"
    if task.config is not None:
        cfg_action = getattr(task.config, "action", "full")
        action = cfg_action.value if hasattr(cfg_action, "value") else str(cfg_action)
    return TaskInfo(
        task_id=task.task_id,
        status=task.status.value,
        created_at=task.created_at,
        finished_at=task.finished_at,
        error=task.error,
        result_ply=task.result_ply,
        output_dir=task.output_dir,
        log_count=len(task.logs),
        action=action,
    )


# ---------------------------------------------------------------------------
# 取消正在运行的任务
# ---------------------------------------------------------------------------
@router.post("/cancel/{task_id}")
def cancel_pipeline_task(task_id: str, username: str = Depends(get_current_username)):
    """
    向运行中的任务发送取消信号。

    取消通过线程安全的 Event 实现。后台线程在每次写入日志前检查该标记，
    若已取消则抛 PipelineError 并终止。

    仅对 PENDING 或 RUNNING 状态的任务有效。
    """
    task = get_task(task_id)
    if not task:
        return {"error": "Task not found"}, 404
    if task.username and task.username != username:
        return {"error": "Task not found"}, 404
    if cancel_task(task_id):
        return {"status": "cancelled"}
    return {"error": "Task not found or already finished"}, 404


# ---------------------------------------------------------------------------
# 拉取任务日志（REST 轮询方式）
# ---------------------------------------------------------------------------
@router.get("/logs/{task_id}")
def get_task_logs(task_id: str, offset: int = 0, username: str = Depends(get_current_username)):
    """
    按偏移量拉取任务日志，用于前端 WebSocket 断连时的兜底轮询。

    参数：
      task_id : 任务唯一标识符
      offset  : 已收到的日志行数，服务端只返回 offset 之后的新行

    返回：
      {"logs": [...], "total": N}
      其中 logs 为新增日志行列表，total 为总行数
    """
    task = get_task(task_id)
    if not task:
        return {"error": "Task not found"}, 404
    if task.username and task.username != username:
        return {"error": "Task not found"}, 404
    return {"logs": task.logs[offset:], "total": len(task.logs)}


# ---------------------------------------------------------------------------
# WebSocket 实时日志流
# ---------------------------------------------------------------------------
@router.websocket("/ws/logs/{task_id}")
async def websocket_logs(websocket: WebSocket, task_id: str):
    """
    通过 WebSocket 向前端实时推送管道执行日志。

    工作流程：
      1. 接受 WebSocket 连接
      2. 轮询 task.logs 列表，发现新行立即推送
      3. 任务进入终态（completed / failed / cancelled）后发送最终状态并关闭
      4. 客户端断开时静默退出

    推送数据格式：
      - 增量日志：{"logs": ["行1", "行2"], "total": N}
      - 结束信号：{"logs": [], "total": N, "status": "...", "result_ply": "...", "error": "..."}

    设计考虑：
      - 轮询间隔 50ms，兼顾实时性和 CPU 开销
      - 使用 last_index 记录已推送位置，避免重复发送
      - 连接断开时不抛异常，保证服务端稳定
    """
    await websocket.accept()
    task = get_task(task_id)
    if not task:
        await websocket.send_json({"error": "Task not found"})
        await websocket.close()
        return

    # last_index 记录已经推送给客户端的日志条数
    last_index = 0
    try:
        # 任务运行期间持续推送新增日志
        while task.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
            if len(task.logs) > last_index:
                new_logs = task.logs[last_index:]
                last_index = len(task.logs)
                await websocket.send_json({"logs": new_logs, "total": len(task.logs)})
            await asyncio.sleep(0.05)

        # 推送最后一次未发送的日志
        if len(task.logs) > last_index:
            new_logs = task.logs[last_index:]
            await websocket.send_json({"logs": new_logs, "total": len(task.logs)})

        # 发送任务结束信号（含最终状态和结果）
        await websocket.send_json({
            "logs": [],
            "total": len(task.logs),
            "status": task.status.value,
            "result_ply": task.result_ply,
            "error": task.error,
        })
    except WebSocketDisconnect:
        # 前端断开 WebSocket 时静默退出，不视为错误
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
