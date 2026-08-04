import re
import shutil
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, List, Optional

from backend.mvsnet_converter import ConversionError, convert_colmap_dense_to_mvsnet


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
# 兼容不同 MVS 方法的目录结构：models/ 或 networks/
REQUIRED_MVS_ENTRIES_ALT = [
    {"models", "datasets", "test.py"},
    {"networks", "datasets", "test.py"},
]


class PipelineError(RuntimeError):
    pass


class PipelineAction(str, Enum):
    POSE = "pose"
    CONVERT = "convert"
    DEPTH = "depth"
    FUSION = "fusion"
    FULL = "full"
    FEATURE_EXTRACT = "feature_extract"
    FEATURE_MATCH = "feature_match"
    INCREMENTAL_MAPPING = "incremental_mapping"


@dataclass
class PipelineConfig:
    image_dir: Path
    workspace_dir: Path
    mvs_root: Path
    checkpoint: Path
    interval_scale: float = 1.06
    num_view: int = 5
    max_h: int = 864
    max_w: int = 1152
    conf: float = 0.9
    thres_view: int = 5
    num_worker: int = 4
    filter_method: str = "normal"
    scene_name: str = "scene"
    skip_sfm: bool = False
    skip_conversion: bool = False
    skip_mvs: bool = False
    camera_model: str = "SIMPLE_RADIAL"
    single_camera: bool = True
    use_gpu: bool = True
    fusibile_exe_path: Optional[Path] = None
    action: PipelineAction = PipelineAction.FULL
    sfm_max_image_size: int = 800
    sfm_max_num_features: int = 4096
    sfm_num_threads: int = 1
    ndepths: str = "48,32,8"


@dataclass
class Workspace:
    root: Path
    scene_name: str
    colmap_root: Path
    database_path: Path
    sparse_root: Path
    sparse_model: Path
    dense_folder: Path
    casmvsnet_root: Path
    casmvsnet_scene: Path
    output_root: Path


def normalize_path(value: str, field_name: str) -> Path:
    if not value or not str(value).strip():
        raise PipelineError(f"{field_name} is required.")
    return Path(str(value).strip()).expanduser().resolve()


def sanitize_scene_name(value: str) -> str:
    value = (value or "scene").strip()
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    value = value.strip("._-")
    return value or "scene"


def list_images(image_dir: Path) -> List[Path]:
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise PipelineError(f"Image directory does not exist: {image_dir}")

    files = [p for p in image_dir.iterdir() if p.is_file()]
    if not files:
        raise PipelineError(f"Image directory is empty: {image_dir}")

    non_images = [p.name for p in files if p.suffix.lower() not in IMAGE_EXTENSIONS]
    if non_images:
        preview = ", ".join(non_images[:5])
        raise PipelineError(
            f"Image directory must contain only image files. Non-image files: {preview}"
        )

    return sorted(files, key=lambda p: p.name.lower())


def stage_uploaded_images(uploaded_files: Iterable[object], workspace_dir: Path, scene_name: str) -> Path:
    files = list(uploaded_files or [])
    if not files:
        raise PipelineError("No uploaded image files were provided.")

    target_dir = Path(workspace_dir) / "uploaded_images" / sanitize_scene_name(scene_name)
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    used_names = set()
    for uploaded in files:
        source = Path(getattr(uploaded, "name", uploaded))
        if not source.is_file():
            raise PipelineError(f"Uploaded item is not a file: {source}")
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise PipelineError(f"Uploaded directory must contain only image files. Non-image file: {source.name}")

        target_name = _unique_name(source.name, used_names)
        shutil.copy2(source, target_dir / target_name)

    return target_dir


def _unique_name(name: str, used_names: set) -> str:
    path = Path(name)
    candidate = path.name
    index = 1
    while candidate.lower() in used_names:
        candidate = f"{path.stem}_{index}{path.suffix}"
        index += 1
    used_names.add(candidate.lower())
    return candidate


def pycolmap_camera_mode(single_camera: bool):
    try:
        import pycolmap
    except ImportError as exc:
        raise PipelineError("pycolmap is not installed in the current Python environment.") from exc
    return pycolmap.CameraMode.SINGLE if single_camera else pycolmap.CameraMode.AUTO


def pycolmap_undistort_kwargs(config: PipelineConfig, workspace: Workspace) -> dict:
    return {
        "output_path": str(workspace.dense_folder),
        "input_path": str(workspace.sparse_model),
        "image_path": str(config.image_dir),
        "output_type": "COLMAP",
    }


def ensure_mvs_interface(mvs_root: Path) -> None:
    mvs_root = Path(mvs_root)
    if not mvs_root.is_dir():
        raise PipelineError(f"MVS root does not exist: {mvs_root}")

    # 检查是否满足任一有效的目录结构（models/ 或 networks/）
    for required_set in REQUIRED_MVS_ENTRIES_ALT:
        missing = [name for name in required_set if not (mvs_root / name).exists()]
        if not missing:
            return  # 匹配到有效的目录结构，通过校验

    # 都不满足，报错并提示所有可能的结构
    expected = "models/ or networks/, datasets/, test.py"
    raise PipelineError(
        f"MVS root is missing required interface files. "
        f"Expected: {expected}"
    )


def prepare_workspace(workspace_dir: Path, scene_name: str) -> Workspace:
    root = Path(workspace_dir)
    safe_scene = sanitize_scene_name(scene_name)
    colmap_root = root / "colmap"
    sparse_root = colmap_root / "sparse"
    dense_folder = colmap_root / "dense"
    casmvsnet_root = root / "casmvsnet_input"
    output_root = root / "outputs"

    return Workspace(
        root=root,
        scene_name=safe_scene,
        colmap_root=colmap_root,
        database_path=colmap_root / "database.db",
        sparse_root=sparse_root,
        sparse_model=sparse_root / "0",
        dense_folder=dense_folder,
        casmvsnet_root=casmvsnet_root,
        casmvsnet_scene=casmvsnet_root / safe_scene,
        output_root=output_root,
    )


def validate_existing_step_inputs(config: PipelineConfig, workspace: Workspace) -> None:
    steps = build_action_plan(config.action)

    if "convert" in steps and "pose" not in steps:
        required = [
            workspace.dense_folder / "images",
            workspace.dense_folder / "sparse",
        ]
        _ensure_paths_exist(required, "This step requires an existing COLMAP dense folder")

    if "depth" in steps or "fusion" in steps:
        required = [
            workspace.casmvsnet_scene / "cams",
            workspace.casmvsnet_scene / "pair.txt",
        ]
        if not (
            (workspace.casmvsnet_scene / "images_post").is_dir()
            or (workspace.casmvsnet_scene / "images").is_dir()
        ):
            required.append(workspace.casmvsnet_scene / "images_post")
        _ensure_paths_exist(required, "This step requires existing CasMVSNet input")

    if "depth" in steps and not Path(config.checkpoint).is_file():
        raise PipelineError(f"Checkpoint file does not exist: {config.checkpoint}")


def _ensure_paths_exist(paths: List[Path], prefix: str) -> None:
    missing = [path.as_posix() for path in paths if not path.exists()]
    if missing:
        raise PipelineError(f"{prefix}. Missing: {', '.join(missing)}")


def build_action_plan(action: PipelineAction) -> List[str]:
    action = PipelineAction(action)
    if action == PipelineAction.FULL:
        return ["pose", "convert", "depth", "fusion"]
    return [action.value]


def run_pycolmap_feature_extraction(config: PipelineConfig, workspace: Workspace, log: Callable[[str], None]) -> None:
    """仅执行特征提取"""
    try:
        import pycolmap
    except ImportError as exc:
        raise PipelineError("pycolmap is not installed in the current Python environment.") from exc

    workspace.colmap_root.mkdir(parents=True, exist_ok=True)
    workspace.sparse_root.mkdir(parents=True, exist_ok=True)
    workspace.dense_folder.mkdir(parents=True, exist_ok=True)

    log("Running pycolmap feature extraction.")
    sift_options = pycolmap.SiftExtractionOptions()
    sift_options.max_image_size = int(config.sfm_max_image_size)
    sift_options.max_num_features = int(config.sfm_max_num_features)
    sift_options.num_threads = int(config.sfm_num_threads)
    log(
        "pycolmap SIFT options: "
        f"max_image_size={sift_options.max_image_size}, "
        f"max_num_features={sift_options.max_num_features}, "
        f"num_threads={sift_options.num_threads}"
    )
    pycolmap.extract_features(
        database_path=str(workspace.database_path),
        image_path=str(config.image_dir),
        camera_mode=pycolmap_camera_mode(config.single_camera),
        camera_model=config.camera_model,
        sift_options=sift_options,
    )
    log("Feature extraction completed.")


def run_pycolmap_feature_matching(config: PipelineConfig, workspace: Workspace, log: Callable[[str], None]) -> None:
    """仅执行图像匹配"""
    try:
        import pycolmap
    except ImportError as exc:
        raise PipelineError("pycolmap is not installed in the current Python environment.") from exc

    if not workspace.database_path.is_file():
        raise PipelineError("Database not found. Run feature extraction first.")

    log("Running pycolmap exhaustive matching.")
    pycolmap.match_exhaustive(database_path=str(workspace.database_path))
    log("Feature matching completed.")


def run_pycolmap_incremental_mapping(config: PipelineConfig, workspace: Workspace, log: Callable[[str], None]) -> None:
    """仅执行增量重建（位姿估计）"""
    try:
        import pycolmap
    except ImportError as exc:
        raise PipelineError("pycolmap is not installed in the current Python environment.") from exc

    if not workspace.database_path.is_file():
        raise PipelineError("Database not found. Run feature extraction and matching first.")

    log("Running pycolmap incremental mapping.")
    maps = pycolmap.incremental_mapping(
        database_path=str(workspace.database_path),
        image_path=str(config.image_dir),
        output_path=str(workspace.sparse_root),
    )
    if not maps:
        raise PipelineError("pycolmap did not produce a sparse reconstruction.")

    if not workspace.sparse_model.exists():
        first_model = sorted(p for p in workspace.sparse_root.iterdir() if p.is_dir())[0]
        workspace.sparse_model = first_model

    log("Incremental mapping completed.")


def run_pycolmap_sfm(config: PipelineConfig, workspace: Workspace, log: Callable[[str], None]) -> None:
    """执行完整的 SfM 流程：特征提取 + 图像匹配 + 增量重建 + 去畸变"""
    run_pycolmap_feature_extraction(config, workspace, log)
    run_pycolmap_feature_matching(config, workspace, log)
    run_pycolmap_incremental_mapping(config, workspace, log)

    log("Undistorting images into COLMAP dense layout.")
    try:
        import pycolmap
    except ImportError as exc:
        raise PipelineError("pycolmap is not installed in the current Python environment.") from exc
    pycolmap.undistort_images(**pycolmap_undistort_kwargs(config, workspace))

    sparse_dir = workspace.dense_folder / "sparse"
    if (sparse_dir / "0").is_dir():
        _flatten_sparse_model(sparse_dir / "0", sparse_dir)

    if not (workspace.dense_folder / "images").is_dir() or not sparse_dir.is_dir():
        raise PipelineError("pycolmap undistortion did not create dense/images and dense/sparse.")


def _flatten_sparse_model(source: Path, target: Path) -> None:
    for item in source.iterdir():
        destination = target / item.name
        if destination.exists():
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        shutil.move(str(item), str(destination))
    source.rmdir()


def convert_colmap_to_mvsnet(
    config: PipelineConfig,
    workspace: Workspace,
    log: Callable[[str], None],
) -> None:
    if workspace.casmvsnet_scene.exists():
        shutil.rmtree(workspace.casmvsnet_scene)
    workspace.casmvsnet_scene.mkdir(parents=True, exist_ok=True)
    log("Converting COLMAP dense output with the project-level MVSNet converter.")
    log(f"Dense input: {workspace.dense_folder}")
    log(f"Converted scene output: {workspace.casmvsnet_scene}")
    try:
        convert_colmap_dense_to_mvsnet(
            dense_folder=workspace.dense_folder,
            save_folder=workspace.casmvsnet_scene,
            interval_scale=config.interval_scale,
        )
    except ConversionError as exc:
        raise PipelineError(str(exc)) from exc


def run_depth_estimation(config: PipelineConfig, workspace: Workspace, log: Callable[[str], None]) -> None:
    """运行深度估计，根据 mvs_root 自动选择 CasMVSNet / UniMVSNet / MVSTER / MAEMVS"""
    workspace.output_root.mkdir(parents=True, exist_ok=True)

    mvs_root_str = str(config.mvs_root).lower()

    # 根据 MVS 方法名创建独立的输出目录
    if "mvstta" in mvs_root_str:
        from backend.mvs_mvstta import run_depth_estimation as _run_depth
        method_name = "MVSTTA"
    elif "maemvs" in mvs_root_str:
        from backend.mvs_maemvs import run_depth_estimation as _run_depth
        method_name = "MAEMVS"
    elif "mvster" in mvs_root_str:
        from backend.mvs_mvster import run_depth_estimation as _run_depth
        method_name = "MVSTER"
    elif "unimvsnet" in mvs_root_str:
        from backend.mvs_unimvsnet import run_depth_estimation as _run_depth
        method_name = "UniMVSNet"
    else:
        from backend.mvs_depth import run_depth_estimation as _run_depth
        method_name = "CasMVSNet"

    # 为每个方法创建独立的输出目录
    method_output_root = workspace.output_root / method_name
    method_output_root.mkdir(parents=True, exist_ok=True)
    log(f"深度估计方法: {method_name}")
    log(f"输出目录: {method_output_root}")

    _run_depth(
        testpath=str(workspace.casmvsnet_root),
        outdir=str(method_output_root),
        checkpoint=str(config.checkpoint),
        log=log,
        num_view=config.num_view,
        interval_scale=config.interval_scale,
        max_h=config.max_h,
        max_w=config.max_w,
        use_gpu=config.use_gpu,
        num_worker=config.num_worker,
    )


def run_point_fusion(config: PipelineConfig, workspace: Workspace, log: Callable[[str], None]) -> Optional[str]:
    """运行点云融合，根据 mvs_root 自动选择 CasMVSNet / UniMVSNet / MVSTER / MAEMVS"""
    workspace.output_root.mkdir(parents=True, exist_ok=True)

    mvs_root_str = str(config.mvs_root).lower()

    # 根据 MVS 方法名确定输出目录
    if "mvstta" in mvs_root_str:
        from backend.mvs_mvstta import run_point_fusion as _run_fusion
        method_name = "MVSTTA"
    elif "maemvs" in mvs_root_str:
        from backend.mvs_maemvs import run_point_fusion as _run_fusion
        method_name = "MAEMVS"
    elif "mvster" in mvs_root_str:
        from backend.mvs_mvster import run_point_fusion as _run_fusion
        method_name = "MVSTER"
    elif "unimvsnet" in mvs_root_str:
        from backend.mvs_unimvsnet import run_point_fusion as _run_fusion
        method_name = "UniMVSNet"
    else:
        from backend.mvs_depth import run_point_fusion as _run_fusion
        method_name = "CasMVSNet"

    # 使用方法专属的输出目录
    method_output_root = workspace.output_root / method_name
    method_output_root.mkdir(parents=True, exist_ok=True)
    log(f"点云融合方法: {method_name}")
    log(f"输出目录: {method_output_root}")

    if "mvstta" in mvs_root_str:
        ply_path = _run_fusion(
            testpath=str(workspace.casmvsnet_root),
            outdir=str(method_output_root),
            conf=[0.1, 0.15, config.conf],
            thres_view=config.thres_view,
            num_worker=config.num_worker,
            log=log,
        )
    elif "maemvs" in mvs_root_str:
        ply_path = _run_fusion(
            testpath=str(workspace.casmvsnet_root),
            outdir=str(method_output_root),
            conf=config.conf,
            thres_view=config.thres_view,
            log=log,
        )
    elif "mvster" in mvs_root_str:
        ply_path = _run_fusion(
            testpath=str(workspace.casmvsnet_root),
            outdir=str(method_output_root),
            conf=config.conf,
            thres_view=config.thres_view,
            log=log,
        )
    elif "unimvsnet" in mvs_root_str:
        ply_path = _run_fusion(
            testpath=str(workspace.casmvsnet_root),
            outdir=str(method_output_root),
            conf=[0.1, 0.15, config.conf],
            thres_view=config.thres_view,
            num_worker=config.num_worker,
            log=log,
        )
    else:
        testpath = str(workspace.casmvsnet_root)
        num_stage = len([int(nd) for nd in config.ndepths.split(",") if nd])
        ply_path = _run_fusion(
            testpath=testpath,
            outdir=str(method_output_root),
            conf=config.conf,
            thres_view=config.thres_view,
            num_stage=num_stage,
            log=log,
        )

    if ply_path:
        log(f"Final point cloud: {ply_path}")
    return ply_path or None


def run_mvs(config: PipelineConfig, workspace: Workspace, log: Callable[[str], None]) -> None:
    """运行完整的 MVS 流程：深度估计 + 点云融合"""
    run_depth_estimation(config, workspace, log)
    run_point_fusion(config, workspace, log)


def find_point_clouds(output_root: Path) -> List[Path]:
    if not Path(output_root).exists():
        return []
    return sorted(Path(output_root).rglob("*.ply"), key=lambda p: p.stat().st_mtime, reverse=True)


def run_pipeline(config: PipelineConfig, log: Callable[[str], None]) -> Workspace:
    started = time.time()
    config.image_dir = Path(config.image_dir).resolve()
    config.workspace_dir = Path(config.workspace_dir).resolve()
    config.mvs_root = Path(config.mvs_root).resolve()
    config.checkpoint = Path(config.checkpoint).resolve()
    if config.fusibile_exe_path:
        config.fusibile_exe_path = Path(config.fusibile_exe_path).resolve()

    steps = build_action_plan(config.action)
    images = list_images(config.image_dir) if any(s in steps for s in ["pose", "feature_extract", "feature_match", "incremental_mapping"]) else []
    if "depth" in steps or "fusion" in steps:
        ensure_mvs_interface(config.mvs_root)

    workspace = prepare_workspace(config.workspace_dir, config.scene_name)
    workspace.root.mkdir(parents=True, exist_ok=True)
    validate_existing_step_inputs(config, workspace)

    if images:
        log(f"Found {len(images)} images in {config.image_dir}.")
    log(f"Workspace: {workspace.root}")

    log(f"Action plan: {' -> '.join(steps)}")

    # 执行各个步骤
    if "feature_extract" in steps:
        run_pycolmap_feature_extraction(config, workspace, log)

    if "feature_match" in steps:
        run_pycolmap_feature_matching(config, workspace, log)

    if "incremental_mapping" in steps:
        run_pycolmap_incremental_mapping(config, workspace, log)

    if "pose" in steps:
        run_pycolmap_sfm(config, workspace, log)

    if "convert" in steps:
        convert_colmap_to_mvsnet(config, workspace, log)

    # 深度估计前清理 GPU 显存
    if "depth" in steps or "fusion" in steps:
        try:
            import torch
            import gc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                free_mem = torch.cuda.mem_get_info()[0] / 1024**3
                log(f"GPU 可用显存: {free_mem:.1f} GB")
        except ImportError:
            pass

    if "depth" in steps and "fusion" in steps:
        run_mvs(config, workspace, log)
    elif "depth" in steps:
        run_depth_estimation(config, workspace, log)
    elif "fusion" in steps:
        run_point_fusion(config, workspace, log)

    clouds = find_point_clouds(workspace.output_root)
    if clouds:
        log(f"Latest point cloud: {clouds[0]}")
    else:
        log("No PLY point cloud was found in the output directory.")
    log(f"Pipeline finished in {time.time() - started:.1f}s.")
    return workspace


def iter_log_tail(lines: Iterable[str], limit: int = 400) -> str:
    collected = list(lines)
    return "\n".join(collected[-limit:])
