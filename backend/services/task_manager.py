"""
后台任务管理器

负责 MVS 管道任务的异步执行、状态追踪和取消。

核心机制：
  1. ThreadPoolExecutor（最大 2 个并发 worker）执行阻塞式管道调用
  2. 每个任务由 PipelineTask 数据类追踪状态、日志和结果
  3. 取消通过 threading.Event 实现，线程在每次日志写入前检查标记
  4. 任务 ID 为 UUID 前 12 位，所有任务保存在进程内存字典中

使用示例：
    config = PipelineConfig(...)
    task = create_task(config)
    print(task.task_id, task.status)  # pending
    # ... 在另一个线程或 WebSocket 中检查状态
    status = get_task(task.task_id).status  # running / completed / failed
"""

import gc
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import Event
from typing import Callable, Optional

from backend.pipeline_core import (
    PipelineAction,
    PipelineConfig,
    PipelineError,
    build_action_plan,
    ensure_mvs_interface,
    find_point_clouds,
    list_images,
    prepare_workspace,
    run_pipeline,
    validate_existing_step_inputs,
    stage_uploaded_images,
)


# ---------------------------------------------------------------------------
# 任务状态枚举
# ---------------------------------------------------------------------------
class TaskStatus(str, Enum):
    """
    管道任务的生命周期状态。

    状态转移图：
      PENDING ──> RUNNING ──> COMPLETED
                        │──> FAILED
                        │──> CANCELLED
    """
    PENDING = "pending"      # 任务已创建，等待线程池调度
    RUNNING = "running"      # 正在执行管道流程
    COMPLETED = "completed"  # 正常执行完成
    FAILED = "failed"        # 执行过程中抛出异常
    CANCELLED = "cancelled"  # 收到取消信号后终止


# ---------------------------------------------------------------------------
# 任务数据类
# ---------------------------------------------------------------------------
@dataclass
class PipelineTask:
    """
    单个管道任务的数据容器，包含完整的状态、日志和执行上下文。

    属性说明：
      task_id      : UUID 前 12 位，用于外部识别和追踪
      username     : 所属用户名，用于多用户隔离
      status       : 当前生命周期状态
      created_at   : 任务创建时的 UNIX 时间戳
      finished_at  : 任务进入终态时的时间戳，运行中为 None
      logs         : 按时间顺序存储的日志行列表
      error        : 失败时的异常信息
      config       : 创建时使用的 PipelineConfig 副本
      result_ply   : 最新 PLY 点云文件的绝对路径
      output_dir   : 管道输出目录的绝对路径
      _cancel_event: 线程安全的取消信号 Event
    """
    task_id: str
    username: str = ""
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    logs: list = field(default_factory=list)
    error: Optional[str] = None
    config: Optional[PipelineConfig] = None
    result_ply: Optional[str] = None
    output_dir: Optional[str] = None
    # 私有字段：取消事件，不参与 dataclass 生成的 __init__
    _cancel_event: Event = field(default_factory=Event)

    def cancel(self):
        """
        触发取消信号。线程池中的工作线程在下次日志写入时检测到此标记并退出。
        此方法可在任意线程安全调用。
        """
        self._cancel_event.set()

    @property
    def is_cancelled(self) -> bool:
        """检查取消信号是否已被触发。"""
        return self._cancel_event.is_set()


# ---------------------------------------------------------------------------
# 全局线程池和任务字典
# ---------------------------------------------------------------------------
# max_workers=2 限制并发任务数，避免 GPU 显存竞争和 CPU 过载
_executor = ThreadPoolExecutor(max_workers=2)

# 进程内任务字典：task_id -> PipelineTask
# 服务器重启后丢失，适合开发和演示场景
# 生产环境可替换为数据库持久化
_tasks = {}


# ---------------------------------------------------------------------------
# 内部辅助函数
# ---------------------------------------------------------------------------
def _log_writer(task: PipelineTask) -> Callable[[str], None]:
    """
    构造一个绑定到特定任务的日志写入闭包。

    每次写入前检查取消标记，若已取消则抛 PipelineError 通知上层终止。
    这种设计确保取消信号最多延迟一条日志的时间即可生效。

    参数：
      task : 要绑定到的 PipelineTask 实例

    返回：
      接受 str 参数的日志写入函数
    """
    def write(message: str):
        if task.is_cancelled:
            raise PipelineError("Task was cancelled.")
        task.logs.append(message)
    return write


def _release_cuda_memory() -> None:
    """释放任务结束后不再被引用的 CUDA 缓存，且不影响原始异常。"""
    gc.collect()
    try:
        import torch

        if not torch.cuda.is_available():
            return
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        # 任务清理不能覆盖正在上报的原始管道异常。
        pass


def _run_task(task: PipelineTask):
    """
    后台线程的工作函数：执行完整的 MVS 管道并更新任务状态。

    执行流程：
      1. 解析所有路径为绝对路径
      2. 根据 action 构建执行步骤列表
      3. 校验步骤前置条件
      4. 调用 run_pipeline() 执行实际流程
      5. 查找输出 PLY 文件
      6. 更新任务状态为 COMPLETED 或 FAILED

    异常处理：
      - PipelineError：预期内的管道错误（如路径无效、命令失败）
      - Exception：非预期的运行时错误（如导入失败）
    """
    try:
        task.status = TaskStatus.RUNNING
        log = _log_writer(task)

        # 将所有路径解析为绝对路径，消除相对路径的歧义
        config = task.config
        config.image_dir = Path(config.image_dir).resolve()
        config.workspace_dir = Path(config.workspace_dir).resolve()
        config.mvs_root = Path(config.mvs_root).resolve()
        config.checkpoint = Path(config.checkpoint).resolve()
        if config.fusibile_exe_path:
            config.fusibile_exe_path = Path(config.fusibile_exe_path).resolve()

        # 构建步骤计划并校验前置条件
        steps = build_action_plan(config.action)
        images = list_images(config.image_dir) if "pose" in steps else []
        if "depth" in steps or "fusion" in steps:
            ensure_mvs_interface(config.mvs_root)

        workspace = prepare_workspace(config.workspace_dir, config.scene_name)
        workspace.root.mkdir(parents=True, exist_ok=True)
        validate_existing_step_inputs(config, workspace)

        if "pose" in steps:
            log(f"Found {len(images)} images in {config.image_dir}.")
        log(f"Workspace: {workspace.root}")
        log(f"Action plan: {' -> '.join(steps)}")

        # 执行核心管道
        run_pipeline(config, log)

        # 收集输出 PLY 点云
        clouds = find_point_clouds(workspace.output_root)
        if clouds:
            task.result_ply = str(clouds[0])
            log(f"Latest point cloud: {clouds[0]}")
        else:
            log("No PLY point cloud was found in the output directory.")

        task.output_dir = str(workspace.output_root.resolve())
        task.status = TaskStatus.COMPLETED
    except PipelineError as exc:
        task.error = str(exc)
        task.logs.append(f"ERROR: {exc}")
        task.status = TaskStatus.FAILED
    except Exception as exc:
        task.error = f"{type(exc).__name__}: {exc}"
        task.logs.append(f"UNEXPECTED ERROR: {task.error}")
        task.status = TaskStatus.FAILED
    finally:
        _release_cuda_memory()
        task.finished_at = time.time()


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------
def create_task(config: PipelineConfig, username: str = "") -> PipelineTask:
    """
    创建新任务并提交到线程池执行。

    参数：
      config   : 完整的 PipelineConfig 配置
      username : 所属用户名，用于多用户隔离

    返回：
      已创建的 PipelineTask 实例（status 为 PENDING，尚未开始执行）

    注意：
      本函数立即返回，任务在后台线程中异步执行。
      调用方应通过 task_id 轮询或 WebSocket 日志流获取执行结果。
    """
    task_id = uuid.uuid4().hex[:12]
    task = PipelineTask(task_id=task_id, config=config, username=username)
    _tasks[task_id] = task
    _executor.submit(_run_task, task)
    return task


def create_callable_task(
    runner: Callable[[Callable[[str], None]], Optional[str]],
    username: str = "",
    output_dir: str = "",
    action: str = "pointcloud",
) -> PipelineTask:
    """
    创建通用后台任务（不依赖完整 PipelineConfig）。

    runner 接收 log 回调，返回可选的结果 PLY 路径。
    """
    task_id = uuid.uuid4().hex[:12]
    task = PipelineTask(task_id=task_id, username=username)
    # 兼容前端/状态接口对 config.action 的读取
    task.config = type("Obj", (), {"action": action})()
    task.output_dir = output_dir or None
    _tasks[task_id] = task

    def _run_callable(t: PipelineTask):
        try:
            t.status = TaskStatus.RUNNING
            log = _log_writer(t)
            log(f"启动任务: {action}")
            result_ply = runner(log)
            if result_ply:
                t.result_ply = str(result_ply)
                log(f"结果点云: {t.result_ply}")
            t.status = TaskStatus.COMPLETED
            log("任务完成")
        except PipelineError as exc:
            t.error = str(exc)
            t.logs.append(f"ERROR: {exc}")
            t.status = TaskStatus.FAILED
        except Exception as exc:
            t.error = f"{type(exc).__name__}: {exc}"
            t.logs.append(f"UNEXPECTED ERROR: {t.error}")
            t.status = TaskStatus.FAILED
        finally:
            _release_cuda_memory()
            t.finished_at = time.time()

    _executor.submit(_run_callable, task)
    return task


def get_task(task_id: str) -> Optional[PipelineTask]:
    """
    按 task_id 查询任务对象。

    参数：
      task_id : UUID 前 12 位的任务标识符

    返回：
      找到时返回 PipelineTask 实例，否则返回 None
    """
    return _tasks.get(task_id)


def list_tasks() -> list:
    """
    返回所有已创建的任务列表，按创建时间倒序排列。

    返回：
      PipelineTask 列表，最新创建的任务排在最前面
    """
    return sorted(_tasks.values(), key=lambda t: t.created_at, reverse=True)


def list_user_tasks(username: str) -> list:
    """
    返回指定用户的所有任务列表，按创建时间倒序排列。

    参数：
      username : 用户名

    返回：
      该用户的 PipelineTask 列表
    """
    return sorted(
        (t for t in _tasks.values() if t.username == username),
        key=lambda t: t.created_at,
        reverse=True,
    )


def cancel_task(task_id: str) -> bool:
    """
    向指定任务发送取消信号。

    参数：
      task_id : 要取消的任务标识符

    返回：
      True  - 取消信号已发送
      False - 任务不存在或已处于终态，无法取消

    说明：
      取消是协作式的——工作线程在下次日志写入时检测到 Event 标记后才退出。
      已进入终态（COMPLETED / FAILED / CANCELLED）的任务不能再次取消。
    """
    task = _tasks.get(task_id)
    if task and task.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
        task.cancel()
        return True
    return False
