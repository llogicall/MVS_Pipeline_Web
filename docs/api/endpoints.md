---
layout: default
title: API Reference
---

# API Reference

## Authentication

### Register
```
POST /api/auth/register
```
```json
{"username": "user", "password": "pass"}
```

### Login
```
POST /api/auth/login
```
```json
{"username": "user", "password": "pass"}
```
Returns JWT token.

### Current User
```
GET /api/auth/me
```

## Projects

### Create Project
```
POST /api/projects/create
```
```json
{"name": "my project"}
```

### List Projects
```
GET /api/projects
```

### Get Project
```
GET /api/projects/{id}
```

### Update Project
```
PUT /api/projects/{id}
```

### Delete Project
```
DELETE /api/projects/{id}
```

## Pipeline

### Run Pipeline
```
POST /api/pipeline/run
```
```json
{
  "workspace_dir": "data/dev/workspace",
  "action": "pose",
  "image_dir": "data/dev/uploads/images"
}
```

### Get Task Status
```
GET /api/pipeline/status/{task_id}
```

### Cancel Task
```
POST /api/pipeline/cancel/{task_id}
```

## Files

### Upload Images
```
POST /api/files/upload/images
```
Multipart form data with image files.

### Get Camera Poses
```
GET /api/files/cameras?workspace_dir=...&scene_name=...
```

### Get Sparse Point Cloud
```
GET /api/files/sparse_ply?workspace_dir=...&scene_name=...
```

### Get Dense Point Cloud
```
GET /api/files/ply?workspace_dir=...
```

### List MVS Methods
```
GET /api/files/mvs/methods
```

### List Checkpoints
```
GET /api/files/mvs/checkpoints?method=MethodName
```

## Health

### Health Check
```
GET /api/health
```
Returns `{"status": "ok"}`.