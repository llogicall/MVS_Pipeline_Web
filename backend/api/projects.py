"""
项目管理 API 路由

每个项目保存一次完整的运行状态，包括配置参数、运行日志、结果路径。
项目数据存储在 data/<username>/projects/<id>.json 中。

端点列表：
  POST   /api/projects/create  : 创建新项目
  GET    /api/projects         : 列出当前用户所有项目
  GET    /api/projects/{id}    : 获取项目详情
  PUT    /api/projects/{id}    : 更新项目（保存状态）
  DELETE /api/projects/{id}    : 删除项目
"""
import json
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.auth import get_current_username
from backend.db import DATA_DIR, get_user_workspace

router = APIRouter(prefix="/api/projects", tags=["projects"])


def projects_dir(username: str) -> Path:
    """返回用户的项目数据目录"""
    d = DATA_DIR / username / "projects"
    d.mkdir(parents=True, exist_ok=True)
    return d


def project_path(username: str, project_id: str) -> Path:
    return projects_dir(username) / f"{project_id}.json"


def load_project(username: str, project_id: str) -> dict:
    path = project_path(username, project_id)
    if not path.is_file():
        raise HTTPException(404, "项目不存在")
    return json.loads(path.read_text(encoding="utf-8"))


def save_project(username: str, project_id: str, data: dict) -> None:
    path = project_path(username, project_id)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def list_projects(username: str) -> list:
    """列出用户所有项目，按更新时间倒序"""
    pdir = projects_dir(username)
    projects = []
    for f in pdir.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            projects.append({
                "id": data.get("id", f.stem),
                "name": data.get("name", ""),
                "created_at": data.get("created_at", ""),
                "updated_at": data.get("updated_at", ""),
            })
        except (json.JSONDecodeError, OSError):
            continue
    projects.sort(key=lambda p: p["updated_at"], reverse=True)
    return projects


# ---------------------------------------------------------------------------
# 请求/响应模型
# ---------------------------------------------------------------------------

class CreateProjectRequest(BaseModel):
    name: str = ""


class UpdateProjectRequest(BaseModel):
    name: Optional[str] = None
    config: Optional[dict] = None
    logs: Optional[List[str]] = None
    result_ply: Optional[str] = None
    image_upload_dir: Optional[str] = None
    completed_actions: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# 端点
# ---------------------------------------------------------------------------

@router.post("/create")
def create_project(req: CreateProjectRequest, username: str = Depends(get_current_username)):
    """创建新项目，自动设定工作目录到用户空间"""
    project_id = f"prj_{secrets.token_hex(8)}"
    now = datetime.now(timezone.utc).isoformat()
    # 默认工作目录：data/<username>/workspace/<project_id>/
    default_workspace = str((get_user_workspace(username) / project_id).resolve())
    data = {
        "id": project_id,
        "name": req.name.strip() or f"项目_{now[:10]}",
        "created_at": now,
        "updated_at": now,
        "config": {
            "workspace_dir": default_workspace,
            "scene_name": req.name.strip() or "scene",
        },
        "logs": [],
        "result_ply": "",
        "image_upload_dir": "",
        "completed_actions": [],
    }
    save_project(username, project_id, data)
    return {"id": project_id, **data}


@router.get("")
def get_projects(username: str = Depends(get_current_username)):
    """列出当前用户所有项目"""
    return {"projects": list_projects(username)}


@router.get("/{project_id}")
def get_project(project_id: str, username: str = Depends(get_current_username)):
    """获取项目完整数据"""
    return load_project(username, project_id)


@router.put("/{project_id}")
def update_project(
    project_id: str,
    req: UpdateProjectRequest,
    username: str = Depends(get_current_username),
):
    """更新项目状态（合并保存）"""
    data = load_project(username, project_id)
    now = datetime.now(timezone.utc).isoformat()
    data["updated_at"] = now

    if req.name is not None:
        data["name"] = req.name.strip()
    if req.config is not None:
        data["config"] = req.config
    if req.logs is not None:
        data["logs"] = req.logs
    if req.result_ply is not None:
        data["result_ply"] = req.result_ply
    if req.image_upload_dir is not None:
        data["image_upload_dir"] = req.image_upload_dir
    if req.completed_actions is not None:
        data["completed_actions"] = req.completed_actions

    save_project(username, project_id, data)
    return data


@router.delete("/{project_id}")
def delete_project(
    project_id: str,
    delete_files: bool = False,
    username: str = Depends(get_current_username),
):
    """删除项目，可选删除本地工作目录文件"""
    path = project_path(username, project_id)
    if not path.is_file():
        raise HTTPException(404, "项目不存在")

    # 读取项目数据以获取工作目录
    data = json.loads(path.read_text(encoding="utf-8"))
    workspace_dir = None
    if data.get("config", {}).get("workspace_dir"):
        workspace_dir = Path(data["config"]["workspace_dir"])

    # 删除项目配置文件
    path.unlink()

    # 删除本地工作目录文件
    deleted_files = False
    if delete_files and workspace_dir and workspace_dir.is_dir():
        import shutil
        shutil.rmtree(workspace_dir)
        deleted_files = True

    return {
        "message": "项目已删除",
        "files_deleted": deleted_files,
    }