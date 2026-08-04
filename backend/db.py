"""
用户数据库 — 简单的 JSON 文件存储

用户数据保存在 data/users.json，每个用户独立目录在 data/<username>/
"""
import json
from pathlib import Path
from datetime import datetime, timezone

from backend.auth import hash_password

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
USERS_FILE = DATA_DIR / "users.json"


def load_users() -> dict:
    """加载所有用户，返回 {username: {password, created_at}}"""
    if not USERS_FILE.is_file():
        return {}
    return json.loads(USERS_FILE.read_text(encoding="utf-8"))


def save_users(users: dict) -> None:
    """保存用户数据到 JSON 文件"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    USERS_FILE.write_text(
        json.dumps(users, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def ensure_user_dir(username: str) -> Path:
    """确保该用户的 data 目录存在，返回路径"""
    user_dir = DATA_DIR / username
    user_dir.mkdir(parents=True, exist_ok=True)
    (user_dir / "uploads").mkdir(parents=True, exist_ok=True)
    (user_dir / "workspace").mkdir(parents=True, exist_ok=True)
    return user_dir


def get_user_workspace(username: str) -> Path:
    """返回该用户的 workspace 目录"""
    return DATA_DIR / username / "workspace"


def get_user_uploads(username: str) -> Path:
    """返回该用户的上传缓存目录"""
    return DATA_DIR / username / "uploads"


# 默认开发账号（开发环境自动创建，无需手动注册）
DEV_USERNAME = "dev"
DEV_PASSWORD = "dev"


def ensure_dev_account() -> None:
    """确保默认开发账号存在且密码正确，首次启动时自动创建"""
    users = load_users()
    need_save = False

    if DEV_USERNAME not in users:
        # 用户不存在，创建
        users[DEV_USERNAME] = {
            "password": hash_password(DEV_PASSWORD),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        need_save = True
        print(f"[auth] 默认开发账号已创建: {DEV_USERNAME}/{DEV_PASSWORD}")
    else:
        # 用户存在，验证密码是否正确
        from backend.auth import verify_password
        stored_hash = users[DEV_USERNAME].get("password", "")
        if not verify_password(DEV_PASSWORD, stored_hash):
            # 密码哈希不匹配，重置密码
            users[DEV_USERNAME]["password"] = hash_password(DEV_PASSWORD)
            need_save = True
            print(f"[auth] 开发账号密码已重置: {DEV_USERNAME}/{DEV_PASSWORD}")

    if need_save:
        save_users(users)

    ensure_user_dir(DEV_USERNAME)