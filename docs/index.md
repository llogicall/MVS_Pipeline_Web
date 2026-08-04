---
layout: default
title: MVS Pipeline Web Service
---

# MVS Pipeline Web Service

A web-based multi-view stereo (MVS) pipeline service.

## Getting Started

- [Installation Guide](guide/installation.md)
- [Quick Start](guide/quickstart.md)
- [API Reference](api/endpoints.md)

## Features

- **Multi-user system** with isolated data storage
- **Project management** with automatic configuration persistence
- **COLMAP integration** for pose estimation
- **Multiple MVS methods**: CasMVSNet, UniMVSNet, MVSTER
- **3D visualization** of camera poses and point clouds
- **Point cloud processing** with Open3D

## Results

<div align="center">
  <img src="assets/images/6121546667f8a91ae825670be3b5d585.png" width="45%" alt="Camera Pose Visualization" />
  <img src="assets/images/e7a163dc437fa623f04551beb0d90c36.png" width="45%" alt="Dense Point Cloud" />
  <p><em>Left: Camera pose visualization. Right: Dense point cloud result.</em></p>
</div>

## Project Structure

```
vis-new/
├── backend/          # FastAPI backend
├── frontend/         # Web frontend (HTML/CSS/JS)
├── MVS/              # MVS method directories
└── docs/             # Documentation
```

## License

MIT