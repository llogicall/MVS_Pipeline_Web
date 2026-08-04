"""
认证 API 路由

端点列表：
  POST /api/auth/register  : 注册新用户
  POST /api/auth/login     : 登录，返回 JWT token
  GET  /api/auth/me        : 获取当前用户信息
"""
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.auth import (
    create_token,
    get_current_username,
    hash_password,
    verify_password,
)
from backend.db import ensure_user_dir, load_users, save_users

router = APIRouter(prefix="/api/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


@router.post("/register")
def register(req: RegisterRequest):
    """注册新用户，自动创建用户数据目录，返回 JWT"""
    username = req.username.strip()
    if len(username) < 2:
        raise HTTPException(400, "用户名至少2个字符")
    if len(req.password) < 4:
        raise HTTPException(400, "密码至少4个字符")

    users = load_users()
    if username in users:
        raise HTTPException(409, "用户名已存在")

    users[username] = {
        "password": hash_password(req.password),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    save_users(users)

    # 创建用户数据目录
    ensure_user_dir(username)

    token = create_token(username)
    return {"token": token, "username": username, "message": "注册成功"}


@router.post("/login")
def login(req: LoginRequest):
    """登录，验证用户名密码，返回 JWT"""
    username = req.username.strip()
    users = load_users()
    user = users.get(username)
    if not user or not verify_password(req.password, user["password"]):
        raise HTTPException(401, "用户名或密码错误")

    token = create_token(username)
    return {"token": token, "username": username, "message": "登录成功"}


@router.get("/me")
def get_me(username: str = Depends(get_current_username)):
    """返回当前登录用户信息"""
    return {"username": username}