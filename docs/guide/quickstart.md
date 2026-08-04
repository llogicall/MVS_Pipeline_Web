---
layout: default
title: Quick Start Guide
---

# Quick Start Guide

## 1. Start the Server

```bash
cd /path/to/vis-new
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8005 --reload
```

## 2. Open the Web Interface

Navigate to `http://127.0.0.1:8005` in your browser.

The server will automatically log in with the default development account.

## 3. Create a Project

1. Click **+ New** to create a new project
2. Enter a project name

## 4. Upload Images

1. Select the **COLMAP** tab
2. Upload images or a video
3. Images will appear in the preview gallery

## 5. Run Pose Estimation

1. Click **Run Pose Estimation**
2. Monitor progress in the log panel
3. View camera poses and sparse point cloud after completion

## 6. Run MVS Depth Estimation

1. Switch to the **MVS** tab
2. Select an MVS method and checkpoint
3. Click **Convert** → **Depth Estimation** → **Point Cloud Fusion**

## 7. Process Point Cloud

1. Switch to the **Point Cloud** tab
2. Select a depth estimation source
3. Run fusion or filtering operations