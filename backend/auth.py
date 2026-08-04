"""
认证工具 — 密码哈希、JWT 签发/验证、当前用户依赖注入
"""
import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import jwt
from fastapi import Header, HTTPException, Query

# 密钥文件路径（项目根目录下的 .secret_key）
SECRET_KEY_FILE = Path(__file__).resolve().parent.parent / ".secret_key"
ALGORITHM = "HS256"
TOKEN_EXPIRE_DAYS = 7

_secret_key: Optional[str] = None


def _get_secret_key() -> str:
    global _secret_key
    if _secret_key:
        return _secret_key
    if SECRET_KEY_FILE.is_file():
        _secret_key = SECRET_KEY_FILE.read_text().strip()
    else:
        _secret_key = secrets.token_hex(32)
        SECRET_KEY_FILE.write_text(_secret_key)
    return _secret_key


def hash_password(password: str) -> str:
    """返回 salt$hash 格式的密码哈希"""
    salt = secrets.token_hex(16)
    pwd_hash = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), salt.encode(), 100000
    ).hex()
    return f"{salt}${pwd_hash}"


def verify_password(password: str, stored: str) -> bool:
    """验证密码是否匹配存储的哈希"""
    try:
        salt, pwd_hash = stored.split("$")
    except ValueError:
        return False
    return (
        hashlib.pbkdf2_hmac(
            "sha256", password.encode(), salt.encode(), 100000
        ).hex()
        == pwd_hash
    )


def create_token(username: str) -> str:
    """签发 JWT token"""
    payload = {
        "sub": username,
        "exp": datetime.now(timezone.utc) + timedelta(days=TOKEN_EXPIRE_DAYS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, _get_secret_key(), algorithm=ALGORITHM)


def verify_token(token: str) -> Optional[str]:
    """验证 JWT token，返回用户名或 None"""
    try:
        payload = jwt.decode(token, _get_secret_key(), algorithms=[ALGORITHM])
        return payload["sub"]
    except jwt.PyJWTError:
        return None


async def get_current_username(authorization: Optional[str] = Header(None)) -> str:
    """
    FastAPI 依赖注入 — 从 Authorization header 提取当前用户名。

    用法：
        @router.get("/some-endpoint")
        def some_endpoint(username: str = Depends(get_current_username)):
            ...
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Not authenticated")
    parts = authorization.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    username = verify_token(parts[1])
    if not username:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return username


async def get_current_username_flexible(
    authorization: Optional[str] = Header(None),
    token: Optional[str] = Query(None),
) -> str:
    """
    FastAPI 依赖注入 — 支持 Authorization header 或 token 查询参数认证。
    用于 <img> 标签等无法携带 header 的场景。
    """
    # 优先使用 Authorization header
    if authorization:
        parts = authorization.split(None, 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            username = verify_token(parts[1])
            if username:
                return username

    # 回退到 token 查询参数
    if token:
        username = verify_token(token)
        if username:
            return username

    raise HTTPException(status_code=401, detail="Not authenticated")