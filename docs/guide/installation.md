---
layout: default
title: Installation Guide
---

# Installation Guide

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for MVS methods)
- Git

## Dependencies

```bash
# Install Python dependencies
pip install -r backend/requirements.txt

# Additional dependencies for MVS methods
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
pip install open3d
```

## Configuration

The server uses JWT for authentication. A secret key is automatically generated on first startup.

Default development account:
- Username: `dev`
- Password: `dev`

## Running

```bash
# Start the server
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8005 --reload
```

## MVS Checkpoints

Download checkpoints for each MVS method and place them in:

- `MVS/CasMVSNet_eval/checkpoints/`
- `MVS/UniMVSNet_eval/checkpoints/`
- `MVS/MVSTER_eval/checkpoints/`