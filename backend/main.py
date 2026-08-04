"""
FastAPI 应用入口
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api.pipeline import router as pipeline_router
from backend.api.files import router as files_router
from backend.api.auth import router as auth_router
from backend.api.projects import router as projects_router
from backend.db import ensure_dev_account

app = FastAPI(
    title="MVS Pipeline Web Service",
    description="Multi-View Stereo Pipeline Web Service",
    version="1.0.0",
    max_form_memory_size=200 * 1024 * 1024,
    on_startup=[ensure_dev_account],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pipeline_router)
app.include_router(files_router)
app.include_router(auth_router)
app.include_router(projects_router)


@app.get("/api/health")
def health():
    return {"status": "ok"}


frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")




