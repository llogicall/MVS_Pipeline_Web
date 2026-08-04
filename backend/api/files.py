"""
文件管理 API 路由

提供图像上传、工作目录浏览、PLY 点云文件服务和通用文件下载功能。

端点列表：
  POST /api/files/upload/images  : 上传图像文件到服务器临时目录
  GET  /api/files/list           : 浏览指定目录下的文件/子目录
  GET  /api/files/download/{path}: 下载任意文件
  GET  /api/files/ply            : 获取最新的 PLY 点云文件
  GET  /api/files/ply/info       : 查询 PLY 文件是否存在及路径
  GET  /api/files/cameras        : 读取 COLMAP sparse 模型，返回相机位姿 JSON
"""

import secrets
import shutil
from datetime import datetime
import zipfile
from pathlib import Path
from typing import List, Optional

import numpy as np

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.auth import get_current_username, get_current_username_flexible
from backend.db import get_user_uploads, get_user_workspace
from backend.services.pointcloud_task_store import (
    PointCloudTaskStoreError,
    list_depth_results as list_stored_depth_results,
    list_pointcloud_state,
    resolve_input_ply,
)

# ---------------------------------------------------------------------------
# 路由定义
# ---------------------------------------------------------------------------
router = APIRouter(prefix="/api/files", tags=["files"])


# ---------------------------------------------------------------------------
# 图像文件上传
# ---------------------------------------------------------------------------
@router.post("/upload/images")
async def upload_images(
    files: List[UploadFile],
    upload_dir: Optional[str] = None,
    username: str = Depends(get_current_username),
):
    """
    接收前端上传的图像文件，支持分批上传到同一目录。

    参数：
      files      : 前端通过 FormData 上传的文件列表（multipart/form-data）
      upload_dir : 可选，目标目录路径。首次上传不传此参数会在用户目录下创建缓存目录，
                   后续批次传入首次返回的 path 值即可续传至同一目录。

    校验规则：
      - 文件列表不能为空
      - 只接受常见图像扩展名：.jpg/.jpeg/.png/.bmp/.tif/.tiff/.webp
      - 非图像文件直接拒绝上传

    返回：
      {
        "path":    "data/<username>/uploads/...",  # 用户独立缓存目录
        "count":   12,
        "message": "Uploaded 12 images."
      }
    """
    if not files:
        raise HTTPException(400, "No files provided")

    # 检查文件扩展名是否均为图像格式
    image_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    non_images = [
        f.filename for f in files
        if f.filename and Path(f.filename).suffix.lower() not in image_ext
    ]
    if non_images:
        raise HTTPException(
            400,
            f"Non-image files not allowed: {', '.join(non_images[:5])}",
        )

    # 使用指定目录（续传）或创建用户独立缓存目录
    if upload_dir:
        dest_dir = Path(upload_dir).resolve()
        dest_dir.mkdir(parents=True, exist_ok=True)
    else:
        user_uploads = get_user_uploads(username)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        rand = secrets.token_hex(4)
        dest_dir = user_uploads / f"batch_{ts}_{rand}"
        dest_dir.mkdir(parents=True)

    for file in files:
        # 某些浏览器通过 webkitdirectory 上传时会携带子目录路径（如 images_post/00000000.jpg），
        # 使用 Path.name 只取文件名部分，确保写入扁平目录
        safe_name = Path(file.filename or "unknown").name
        dest = dest_dir / safe_name
        # IO密集型异步等待，在网络请求，数据库查询，文件读写等场景使用
        content = await file.read()
        dest.write_bytes(content)

    return {
        "path": str(dest_dir.resolve()),
        "count": len(files),
        "message": f"Uploaded {len(files)} images.",
    }


# ---------------------------------------------------------------------------
# 视频文件上传与帧提取
# ---------------------------------------------------------------------------
@router.post("/upload/video")
async def upload_video(
    file: UploadFile,
    upload_dir: Optional[str] = None,
    username: str = Depends(get_current_username),
):
    """
    接收前端上传的视频文件并进行帧提取。

    参数：
      file       : 前端通过 FormData 上传的视频文件
      upload_dir : 可选，目标目录路径

    返回：
      {
        "video_path": "...",       # 视频文件路径
        "frames_dir": "...",       # 提取的帧目录
        "frame_count": 120,        # 提取的帧数
        "fps": 30,                 # 视频帧率
        "duration": 4.0,           # 视频时长（秒）
        "message": "..."
      }
    """
    if not file:
        raise HTTPException(400, "No video file provided")

    # 校验视频格式
    video_ext = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
    if file.filename and Path(file.filename).suffix.lower() not in video_ext:
        raise HTTPException(
            400,
            f"Unsupported video format. Allowed: {', '.join(video_ext)}",
        )

    # 创建视频存储目录
    if upload_dir:
        dest_dir = Path(upload_dir).resolve()
        dest_dir.mkdir(parents=True, exist_ok=True)
    else:
        user_uploads = get_user_uploads(username)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        rand = secrets.token_hex(4)
        dest_dir = user_uploads / f"video_{ts}_{rand}"
        dest_dir.mkdir(parents=True)

    # 保存视频文件
    safe_name = Path(file.filename or "video.mp4").name
    video_path = dest_dir / safe_name
    content = await file.read()
    video_path.write_bytes(content)

    return {
        "video_path": str(video_path.resolve()),
        "upload_dir": str(dest_dir.resolve()),
        "message": f"Video uploaded: {safe_name}",
    }


class ExtractFramesRequest(BaseModel):
    """视频帧提取请求"""
    video_path: str
    output_dir: str = ""
    fps: float = 1.0  # 每秒提取帧数
    max_frames: int = 200
    image_format: str = "jpg"
    resize: Optional[int] = None  # 可选的图像缩放尺寸


@router.post("/upload/video/extract_frames")
async def extract_frames(
    req: ExtractFramesRequest,
    username: str = Depends(get_current_username),
):
    """
    从已上传的视频中提取帧。

    参数：
      video_path    : 视频文件路径
      output_dir    : 输出目录（可选，默认在视频同目录下创建 frames/ 子目录）
      fps           : 每秒提取帧数（默认 1.0）
      max_frames    : 最大提取帧数（默认 200）
      image_format  : 输出图像格式（默认 jpg）
      resize        : 可选的图像缩放尺寸

    返回：
      {
        "frames_dir": "...",
        "frame_count": 120,
        "fps": 30,
        "duration": 4.0,
        "message": "..."
      }
    """
    import cv2
    import gc

    video_path = Path(req.video_path).resolve()
    if not video_path.is_file():
        raise HTTPException(404, f"Video file not found: {video_path}")

    # 确定输出目录
    if req.output_dir:
        frames_dir = Path(req.output_dir).resolve()
    else:
        frames_dir = video_path.parent / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise HTTPException(500, f"Cannot open video: {video_path}")

        # 获取视频信息
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps

        # 计算采样间隔
        frame_interval = max(1, int(video_fps / req.fps))

        frame_count = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 按间隔采样
            if frame_count % frame_interval == 0:
                # 可选缩放
                if req.resize and req.resize > 0:
                    h, w = frame.shape[:2]
                    scale = req.resize / max(h, w)
                    if scale < 1:
                        new_w = int(w * scale)
                        new_h = int(h * scale)
                        frame = cv2.resize(frame, (new_w, new_h))

                # 保存帧
                frame_filename = f"{saved_count:08d}.{req.image_format}"
                frame_path = frames_dir / frame_filename
                cv2.imwrite(str(frame_path), frame)
                saved_count += 1

                # 达到最大帧数限制
                if saved_count >= req.max_frames:
                    break

            frame_count += 1

        cap.release()
        gc.collect()

        return {
            "frames_dir": str(frames_dir.resolve()),
            "frame_count": saved_count,
            "fps": video_fps,
            "duration": round(duration, 2),
            "message": f"Extracted {saved_count} frames from video.",
        }

    except cv2.error as e:
        raise HTTPException(500, f"OpenCV error: {str(e)}")
    except Exception as e:
        raise HTTPException(500, f"Frame extraction failed: {str(e)}")


@router.get("/upload/images/list")
def list_uploaded_images(
    upload_dir: str = "",
    username: str = Depends(get_current_username),
):
    """
    列出上传目录中的图像文件。

    参数：
      upload_dir : 上传目录路径，留空则返回用户默认上传目录

    返回：
      {
        "images": [
          {"name": "00000000.jpg", "url": "/api/files/upload/images/file?path=..."},
          ...
        ],
        "count": 5
      }
    """
    img_dir = Path(upload_dir) if upload_dir else get_user_uploads(username)
    if not img_dir.is_dir():
        return {"images": [], "count": 0}

    image_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    images = []
    for f in sorted(img_dir.iterdir()):
        if f.is_file() and f.suffix.lower() in image_ext:
            images.append({
                "name": f.name,
                "url": f"/api/files/upload/images/file?path={f.as_posix()}",
            })

    return {"images": images, "count": len(images)}


@router.get("/upload/images/file")
def serve_uploaded_image(
    path: str,
    token: Optional[str] = None,
    _username: str = Depends(get_current_username_flexible),
):
    """提供单个上传图像文件，支持通过 token 查询参数认证（用于 <img> 标签）"""
    # 支持正斜杠和反斜杠路径
    img_path = Path(path.replace("\\", "/")).resolve()
    if not img_path.is_file():
        raise HTTPException(404, f"Image not found: {img_path}")
    return FileResponse(img_path)


# ---------------------------------------------------------------------------
# 深度估计结果列表
# ---------------------------------------------------------------------------
@router.get("/depth_results")
def list_depth_results(
    workspace_dir: str = "",
    username: str = Depends(get_current_username),
):
    """
    列出工作目录下各 MVS 方法的深度估计结果。

    返回：
      {
        "results": [
          {
            "method": "CasMVSNet",
            "path": "data/<user>/workspace/<prj>/outputs/CasMVSNet",
            "depth_count": 10,
            "has_ply": true,
            "ply_path": "data/.../outputs/CasMVSNet/xxx.ply"
          },
          ...
        ]
      }
    """
    ws_base = Path(workspace_dir).resolve() if workspace_dir else get_user_workspace(username)
    return {"results": list_stored_depth_results(ws_base)}


@router.get("/depth_results/{depth_result_id}/pointclouds")
def get_depth_result_pointclouds(
    depth_result_id: str,
    workspace_dir: str = "",
    username: str = Depends(get_current_username),
):
    """返回指定深度结果下的点云文件和持久化处理任务。"""
    ws_base = Path(workspace_dir).resolve() if workspace_dir else get_user_workspace(username)
    try:
        return list_pointcloud_state(ws_base, depth_result_id)
    except PointCloudTaskStoreError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/depth_results/{depth_result_id}/pointcloud")
def get_depth_result_pointcloud(
    depth_result_id: str,
    path: str,
    workspace_dir: str = "",
    username: str = Depends(get_current_username_flexible),
):
    """读取指定深度结果内经过路径校验的 PLY 文件。"""
    ws_base = Path(workspace_dir).resolve() if workspace_dir else get_user_workspace(username)
    try:
        ply_path = resolve_input_ply(ws_base, depth_result_id, path)
    except PointCloudTaskStoreError as exc:
        raise HTTPException(400, str(exc)) from exc
    return FileResponse(ply_path, media_type="application/octet-stream", filename=ply_path.name)


# ---------------------------------------------------------------------------
# 目录文件列表浏览
# ---------------------------------------------------------------------------
@router.get("/list")
def list_files(path: str = "", username: str = Depends(get_current_username)):
    """
    浏览指定目录下的文件和子目录。

    参数：
      path : 目标目录路径（可选）。为空时返回用户数据目录的快捷入口。

    返回结构：
      {
        "entries": [
          {"name": "uploads", "type": "dir",  "path": "D:\\...\\data\\<user>\\uploads", "size": null},
          {"name": "workspace", "type": "dir", "path": "D:\\...\\data\\<user>\\workspace", "size": null},
        ],
        "path": "D:\\..."
      }

    安全限制：
      - path 不存在时返回 404
      - path 不是目录时返回 400
    """
    base = Path(path).resolve() if path else None
    if base and not base.exists():
        raise HTTPException(404, f"Path not found: {path}")
    if base and not base.is_dir():
        raise HTTPException(400, "Path must be a directory")

    # path 为空时返回用户数据目录的快捷入口
    if base is None:
        from backend.db import DATA_DIR
        user_dir = DATA_DIR / username
        if user_dir.is_dir():
            entries = [
                {"name": "uploads", "type": "dir", "path": str((user_dir / "uploads").resolve())},
                {"name": "workspace", "type": "dir", "path": str((user_dir / "workspace").resolve())},
            ]
        else:
            entries = []
        return {"entries": entries, "path": str(DATA_DIR)}

    # 遍历目标目录，收集所有条目
    entries = []
    for item in sorted(base.iterdir()):
        entries.append({
            "name": item.name,
            "type": "dir" if item.is_dir() else "file",
            "path": str(item.resolve()),
            "size": item.stat().st_size if item.is_file() else None,
        })
    return {"entries": entries, "path": str(base)}


# ---------------------------------------------------------------------------
# MVS 方法目录和 checkpoint 列表
# ---------------------------------------------------------------------------
@router.get("/mvs/methods")
def list_mvs_methods(username: str = Depends(get_current_username)):
    """列出 MVS 目录下的方法文件夹（排除 shared 和隐藏目录）"""
    mvs_dir = Path(__file__).resolve().parent.parent.parent / "MVS"
    if not mvs_dir.is_dir():
        return {"methods": []}

    methods = []
    for item in sorted(mvs_dir.iterdir()):
        if item.is_dir() and not item.name.startswith(('.', '_')) and item.name != 'shared':
            methods.append({
                "name": item.name,
                "path": str(item.resolve()),
            })
    return {"methods": methods}


@router.get("/mvs/checkpoints")
def list_mvs_checkpoints(method: str = "", username: str = Depends(get_current_username)):
    """列出指定 MVS 方法目录下的 checkpoint 文件"""
    if not method:
        return {"checkpoints": []}

    mvs_dir = Path(__file__).resolve().parent.parent.parent / "MVS"
    checkpoint_dir = mvs_dir / method / "checkpoints"
    if not checkpoint_dir.is_dir():
        return {"checkpoints": []}

    checkpoints = []
    for item in sorted(checkpoint_dir.iterdir()):
        if item.is_file() and item.suffix in ('.ckpt', '.pth', '.pt', '.bin'):
            checkpoints.append({
                "name": item.name,
                "path": str(item.resolve()),
            })
    return {"checkpoints": checkpoints}


# ---------------------------------------------------------------------------
# 文件下载
# FastAPI/Starlette 路径转换器（path converter） 语法。
# {参数名：类型}的意思是将URL片段解析为制定类型的路径参数

# ---------------------------------------------------------------------------
@router.get("/download/{path:path}")
def download_file(path: str):
    """
    下载指定路径的文件。path 参数使用 :path 捕获器以支持多层子目录。

    参数：
      path : 文件的绝对路径或相对于项目根目录的路径

    返回：
      - 文件存在时返回 FileResponse（流式下载）
      - 文件不存在时返回 404

    安全注意事项：
      当前未限制可下载的目录范围，生产环境应添加路径白名单检查。
    """
    target = Path(path).resolve()
    if not target.is_file():
        raise HTTPException(404, f"File not found: {path}")
    return FileResponse(str(target))


# ---------------------------------------------------------------------------
# 获取最新 PLY 点云文件
# ---------------------------------------------------------------------------
@router.get("/ply")
def get_latest_ply(
    workspace_dir: str = "",
    username: str = Depends(get_current_username),
):
    """
    返回工作目录下最新的 .ply 点云文件（按修改时间排序）。

    参数：
      workspace_dir : 工作目录路径，留空则使用该用户的默认 workspace/

    返回：
      - 找到 PLY 文件时直接以二进制流返回（content-type: application/octet-stream）
      - 未找到时返回 404

    搜索规则：
      在 workspace_dir/outputs/ 下递归搜索 *.ply 文件，
      按文件 mtime 降序排列，取第一个作为最新结果。
    """
    from backend.pipeline_core import find_point_clouds

    if workspace_dir:
        base = Path(workspace_dir)
    else:
        base = get_user_workspace(username)

    clouds = find_point_clouds(base / "outputs")

    if not clouds:
        raise HTTPException(404, "No PLY point cloud found")
    return FileResponse(str(clouds[0]))


# ---------------------------------------------------------------------------
# 查询 PLY 文件元信息
# ---------------------------------------------------------------------------
@router.get("/ply/info")
def get_ply_info(
    workspace_dir: str = "",
    username: str = Depends(get_current_username),
):
    """
    查询工作目录下是否存在 PLY 点云文件，供前端提前更新界面状态。

    参数：
      workspace_dir : 工作目录路径，留空则使用该用户的默认 workspace/

    返回：
      {
        "found": true,
        "path": "D:\\...\\data\\<username>\\workspace\\outputs\\scene.ply",
        "name": "scene.ply"
      }
      found 为 false 时 path 和 name 均为 null。
    """
    from backend.pipeline_core import find_point_clouds

    if workspace_dir:
        base = Path(workspace_dir)
    else:
        base = get_user_workspace(username)

    clouds = find_point_clouds(base / "outputs")

    return {
        "found": len(clouds) > 0,
        "path": str(clouds[0]) if clouds else None,
        "name": clouds[0].name if clouds else None,
    }


# ---------------------------------------------------------------------------
# 管道进度状态检查
# ---------------------------------------------------------------------------
@router.get("/pipeline/status")
def get_pipeline_status(
    workspace_dir: str = "",
    scene_name: str = "scene",
    username: str = Depends(get_current_username),
):
    """
    检查当前工作目录下的管道进度状态。

    返回各步骤是否已完成（基于文件存在性）：
    {
        "feature_extract": true,
        "feature_match": true,
        "incremental_mapping": true,
        "convert": false,
        "depth": false,
        "fusion": false
    }
    """
    base = (Path(workspace_dir) if workspace_dir else get_user_workspace(username)) / scene_name

    # 检查数据库文件（特征提取需要）
    db_exists = (base / "colmap" / "database.db").is_file()

    # 检查稀疏模型
    sparse_dir = base / "colmap" / "sparse" / "0"
    if not sparse_dir.is_dir():
        sparse_dir = base / "colmap" / "dense" / "sparse"

    cameras_exist = (sparse_dir / "cameras.bin").is_file() or (sparse_dir / "cameras.txt").is_file()
    images_exist = (sparse_dir / "images.bin").is_file() or (sparse_dir / "images.txt").is_file()
    points_exist = (sparse_dir / "points3D.bin").is_file() or (sparse_dir / "points3D.txt").is_file()

    # 检查 MVSNet 转换结果
    casmvsnet_scene = base / "casmvsnet_input" / scene_name
    convert_done = (casmvsnet_scene / "cams").is_dir() and (casmvsnet_scene / "pair.txt").is_file()

    # 检查点云输出
    from backend.pipeline_core import find_point_clouds
    clouds = find_point_clouds(base / "outputs")
    fusion_done = len(clouds) > 0

    return {
        "feature_extract": db_exists and cameras_exist,
        "feature_match": db_exists and images_exist,
        "incremental_mapping": cameras_exist and images_exist and points_exist,
        "convert": convert_done,
        "depth": False,  # 深度估计结果难以直接检测
        "fusion": fusion_done,
    }


# ---------------------------------------------------------------------------
# 读取 COLMAP 相机位姿
# ---------------------------------------------------------------------------
@router.get("/cameras")
def get_camera_poses(
    workspace_dir: str = "",
    scene_name: str = "scene",
    token: Optional[str] = None,
    username: str = Depends(get_current_username_flexible),
):
    """
    读取 COLMAP sparse 模型中的相机位姿，返回 JSON 供前端渲染。

    参数：
      workspace_dir : 工作目录，留空则使用该用户的默认 workspace/
      scene_name    : 场景名称，默认 scene

    搜索顺序：
      1. <workspace>/<scene>/dense/sparse/    （去畸变后的模型，优先）
      2. <workspace>/<scene>/colmap/sparse/0/ （原始稀疏重建模型）
      3. <workspace>/<scene>/sparse/0/        （扁平结构）

    返回：
      {
        "cameras": [
          {
            "image_id": 1,
            "image_name": "00000000.jpg",
            "position": [x, y, z],          # 相机中心世界坐标
            "rotation": [9-entry row-major], # 3x3 旋转矩阵（世界→相机）
            "focal_length": 1234.5,
            "width": 1920,
            "height": 1080
          }
        ]
      }
    """
    # 延迟导入，避免启动时加载过多依赖
    from backend.mvsnet_converter import (
        read_cameras_binary,
        read_cameras_text,
        read_images_binary,
        read_images_text,
        qvec2rotmat,
    )

    # 构建基础路径：先尝试 workspace_dir/scene_name，若不存在则用 workspace_dir
    ws_base = Path(workspace_dir) if workspace_dir else get_user_workspace(username)
    base = ws_base / scene_name
    if not base.is_dir():
        base = ws_base  # 场景子目录不存在，直接使用工作目录

    # 按优先级查找 sparse 模型目录
    sparse_dir = None
    # 1. 去畸变后的模型: <base>/dense/sparse/
    dense_sparse = base / "dense" / "sparse"
    if dense_sparse.is_dir():
        sparse_dir = dense_sparse
    else:
        # 2. COLMAP 原始稀疏重建: <base>/colmap/sparse/0/
        colmap_sparse = base / "colmap" / "sparse" / "0"
        if colmap_sparse.is_dir():
            sparse_dir = colmap_sparse
        else:
            # 3. 扁平结构: <base>/sparse/0/
            flat_sparse = base / "sparse" / "0"
            if flat_sparse.is_dir():
                sparse_dir = flat_sparse

    if not sparse_dir:
        raise HTTPException(404, f"No COLMAP sparse model found in {base}")

    # 自动检测格式（.bin 或 .txt）
    bin_mode = (sparse_dir / "cameras.bin").is_file() and (sparse_dir / "images.bin").is_file()
    txt_mode = (sparse_dir / "cameras.txt").is_file() and (sparse_dir / "images.txt").is_file()

    if not bin_mode and not txt_mode:
        raise HTTPException(404, f"No complete COLMAP model (cameras + images) in {sparse_dir}")

    try:
        if bin_mode:
            cameras = read_cameras_binary(sparse_dir / "cameras.bin")
            images = read_images_binary(sparse_dir / "images.bin")
        else:
            cameras = read_cameras_text(sparse_dir / "cameras.txt")
            images = read_images_text(sparse_dir / "images.txt")
    except Exception as e:
        raise HTTPException(500, f"Failed to read COLMAP model: {e}")

    result = []
    for img_id, img in images.items():
        # 相机中心: C = -R^T * t
        R = qvec2rotmat(img.qvec)
        center = -R.T @ img.tvec

        # 获取相机内参（用于计算视锥体宽高比）
        cam = cameras.get(img.camera_id)
        focal = float(cam.params[0]) if cam else 0.0
        w = int(cam.width) if cam else 0
        h = int(cam.height) if cam else 0

        result.append({
            "image_id": img_id,
            "image_name": img.name,
            "position": center.tolist(),
            "rotation": R.flatten().tolist(),  # 3x3 行主序
            "focal_length": focal,
            "width": w,
            "height": h,
        })

    return {"cameras": result}


# ---------------------------------------------------------------------------
# 稀疏点云（points3D.bin → PLY）
# ---------------------------------------------------------------------------
@router.get("/sparse_ply")
def get_sparse_ply(
    workspace_dir: str = "",
    scene_name: str = "scene",
    token: Optional[str] = None,
    username: str = Depends(get_current_username_flexible),
):
    """
    读取 COLMAP sparse 模型中的 points3D，转换为 PLY 格式返回。

    搜索顺序与 /cameras 端点一致。
    """
    from backend.mvsnet_converter import (
        read_points3d_binary,
        read_points3d_text,
        detect_model_ext,
    )

    # 构建基础路径：先尝试 workspace_dir/scene_name，若不存在则用 workspace_dir
    ws_base = Path(workspace_dir) if workspace_dir else get_user_workspace(username)
    base = ws_base / scene_name
    if not base.is_dir():
        base = ws_base

    # 查找 sparse 模型目录
    sparse_dir = None
    dense_sparse = base / "dense" / "sparse"
    if dense_sparse.is_dir():
        sparse_dir = dense_sparse
    else:
        colmap_sparse = base / "colmap" / "sparse" / "0"
        if colmap_sparse.is_dir():
            sparse_dir = colmap_sparse
        else:
            flat_sparse = base / "sparse" / "0"
            if flat_sparse.is_dir():
                sparse_dir = flat_sparse

    if not sparse_dir:
        raise HTTPException(404, f"No COLMAP sparse model found in {base}")

    try:
        ext = detect_model_ext(sparse_dir)
        if ext == ".bin":
            points3d = read_points3d_binary(sparse_dir / "points3D.bin")
        else:
            points3d = read_points3d_text(sparse_dir / "points3D.txt")
    except Exception as e:
        raise HTTPException(500, f"Failed to read points3D: {e}")

    if not points3d:
        raise HTTPException(404, "No points3D found in sparse model")

    # 生成 PLY 文件
    coords = np.array([p.xyz for p in points3d.values()])
    colors = np.array([p.rgb for p in points3d.values()])

    ply_path = base / "sparse_points.ply"
    _write_ply(ply_path, coords, colors)

    return FileResponse(str(ply_path), media_type="application/octet-stream")


def _write_ply(path: Path, coords: np.ndarray, colors: np.ndarray):
    """将点云坐标和颜色写入 ASCII PLY 文件"""
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(coords)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(len(coords)):
            x, y, z = coords[i]
            r, g, b = int(colors[i][0]), int(colors[i][1]), int(colors[i][2])
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
