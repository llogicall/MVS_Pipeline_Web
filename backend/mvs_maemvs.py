"""
MAEMVS 深度估计集成模块。

MAEMVS 是多任务学习框架，数据读取和融合输出与其他 MVS 方法不同。
本模块复用 MAEMVS 的模型、数据集和融合函数，但使用项目统一的后端函数入口。
"""

import gc
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, Iterable, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader


class DepthEstimationError(RuntimeError):
    pass


@dataclass
class MAEMVSDeps:
    find_dataset_def: Callable
    read_pfm: Callable
    save_pfm: Callable
    MultiTaskMVSNet: type
    fuse_depth_maps: Callable


def _activate_mvs_eval_dir(eval_dir: Path) -> None:
    eval_dir_str = str(eval_dir)
    sys.path[:] = [p for p in sys.path if p != eval_dir_str]
    sys.path.insert(0, eval_dir_str)
    for name in list(sys.modules):
        if (
            name == "datasets" or name.startswith("datasets.")
            or name == "models" or name.startswith("models.")
            or name == "networks" or name.startswith("networks.")
            or name == "save_depth_fusion"
            or name == "utils"
        ):
            del sys.modules[name]


def _get_maemvs_deps() -> MAEMVSDeps:
    """延迟导入 MAEMVS 依赖，避免不同 MVS 方法之间的模块名冲突。"""
    import importlib

    maemvs_dir = Path(__file__).resolve().parent.parent / "MVS" / "MAEMVS_eval"
    _activate_mvs_eval_dir(maemvs_dir)

    datasets_mod = importlib.import_module("datasets")
    io_mod = importlib.import_module("datasets.data_io")
    models_mod = importlib.import_module("models")
    fusion_mod = importlib.import_module("save_depth_fusion")

    return MAEMVSDeps(
        find_dataset_def=datasets_mod.find_dataset_def,
        read_pfm=io_mod.read_pfm,
        save_pfm=io_mod.save_pfm,
        MultiTaskMVSNet=models_mod.MultiTaskMVSNet,
        fuse_depth_maps=fusion_mod.fuse_depth_maps,
    )


def tocuda(sample, device: Optional[torch.device] = None):
    """递归移动 tensor 到指定设备，保留字符串和列表等元数据。"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(sample, torch.Tensor):
        return sample.to(device)
    if isinstance(sample, dict):
        return {key: tocuda(val, device) for key, val in sample.items()}
    if isinstance(sample, list):
        return [tocuda(val, device) for val in sample]
    if isinstance(sample, tuple):
        return tuple(tocuda(val, device) for val in sample)
    return sample


def tensor2numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().copy()
    if isinstance(value, dict):
        return {key: tensor2numpy(val) for key, val in value.items()}
    if isinstance(value, list):
        return [tensor2numpy(val) for val in value]
    if isinstance(value, tuple):
        return tuple(tensor2numpy(val) for val in value)
    return value


def _parse_int_list(value: str) -> list:
    return [int(item) for item in str(value).split(",") if item]


def _parse_float_list(value: str) -> list:
    return [float(item) for item in str(value).split(",") if item]


def _build_model(deps: MAEMVSDeps, ndepths: str, depth_inter_r: str, cr_base_chs: str,
                 share_cr: bool, grad_method: str, use_mae: bool, mask_ratio: float,
                 use_damw: bool, use_sparse_conv: bool):
    return deps.MultiTaskMVSNet(
        refine=False,
        ndepths=_parse_int_list(ndepths),
        depth_interals_ratio=_parse_float_list(depth_inter_r),
        share_cr=share_cr,
        grad_method=grad_method,
        arch_mode="fpn",
        cr_base_chs=_parse_int_list(cr_base_chs),
        use_mae=use_mae,
        mask_ratio=mask_ratio,
        use_damw=use_damw,
        use_sparse_conv=use_sparse_conv,
    )


def _detect_model_options_from_keys(keys: Iterable[str], use_mae: bool, use_damw: bool) -> Tuple[bool, bool]:
    """根据 checkpoint 参数名推断 MAEMVS 多任务分支配置。"""
    key_list = list(keys)
    if any(key.startswith("mae_decoder.") or key.startswith("mask_generator.") for key in key_list):
        use_mae = True
    if "multi_task_loss.raw_weights" in key_list:
        use_damw = True
    elif "multi_task_loss.fixed_weights" in key_list:
        use_damw = False
    return use_mae, use_damw


def _detect_checkpoint_model_options(
    checkpoint: str,
    use_mae: bool,
    use_damw: bool,
    log: Callable[[str], None],
) -> Tuple[bool, bool]:
    """读取 checkpoint 头部权重名，自动选择可 strict 加载的 MAEMVS 配置。"""
    state_dict = torch.load(checkpoint, map_location="cpu")
    model_state = state_dict.get("model", state_dict) if isinstance(state_dict, dict) else state_dict
    detected_use_mae, detected_use_damw = _detect_model_options_from_keys(
        model_state.keys(), use_mae, use_damw
    )
    if detected_use_mae != use_mae:
        log(f"Detected MAEMVS use_mae={detected_use_mae} from checkpoint")
    if detected_use_damw != use_damw:
        log(f"Detected MAEMVS use_damw={detected_use_damw} from checkpoint")
    del state_dict, model_state
    gc.collect()
    return detected_use_mae, detected_use_damw


def _load_checkpoint(model, checkpoint: str, log: Callable[[str], None]) -> None:
    state_dict = torch.load(checkpoint, map_location="cpu")
    model_state = state_dict.get("model", state_dict) if isinstance(state_dict, dict) else state_dict
    cleaned = {
        key[7:] if key.startswith("module.") else key: value
        for key, value in model_state.items()
    }
    try:
        model.load_state_dict(cleaned, strict=True)
    except RuntimeError as exc:
        if "CUDA out of memory" in str(exc):
            raise DepthEstimationError(
                f"GPU 显存不足，无法加载 MAEMVS。请降低分辨率 (max_h/max_w) 或使用 CPU (use_gpu=False)。详情: {exc}"
            ) from exc
        raise
    finally:
        del state_dict, model_state, cleaned
        gc.collect()
    log("MAEMVS checkpoint loaded with strict=True")


def _ensure_scene_dirs(scene_outdir: Path) -> None:
    for name in ("depth_est", "confidence", "cams", "images"):
        (scene_outdir / name).mkdir(parents=True, exist_ok=True)


def _to_numpy_image(img_chw: np.ndarray) -> np.ndarray:
    img = np.transpose(img_chw, (1, 2, 0)).astype(np.float32)
    if img.max() <= 2.0 and img.min() >= -1.0:
        img = np.clip(img, 0.0, 1.0) * 255.0
    else:
        img = img - img.min()
        img = img / max(float(img.max()), 1e-8) * 255.0
    return img.astype(np.uint8)


def _sample_base_name(sample: Dict, item_idx: int) -> str:
    if "out_name" in sample:
        name = sample["out_name"][item_idx]
        view = sample.get("out_view", ["view"])[item_idx]
        return f"{view}_{name}"

    filename = sample["filename"][item_idx]
    if isinstance(filename, (list, tuple)):
        filename = filename[0]
    return Path(filename.format("depth_est", ".pfm")).stem


def _save_camera_npz(filename: Path, proj_matrices: np.ndarray) -> None:
    np.savez_compressed(
        filename,
        intrinsics=proj_matrices[:, 1, :3, :3].astype(np.float32),
        extrinsics=proj_matrices[:, 0, :4, :4].astype(np.float32),
    )


def _write_cam_txt(filename: Path, ref_proj: np.ndarray, depth_values: np.ndarray) -> None:
    depth_min = float(depth_values[0]) if depth_values.size else 0.0
    if depth_values.size > 1:
        depth_interval = float(depth_values[1] - depth_values[0])
    else:
        depth_interval = 1.0

    with open(filename, "w", encoding="utf-8") as f:
        f.write("extrinsic\n")
        for row in ref_proj[0, :4, :4]:
            f.write(" ".join(str(float(v)) for v in row) + "\n")
        f.write("\nintrinsic\n")
        for row in ref_proj[1, :3, :3]:
            f.write(" ".join(str(float(v)) for v in row) + "\n")
        f.write(f"\n{depth_min} {depth_interval} {int(depth_values.size)}\n")


def _save_stage_maps(deps: MAEMVSDeps, scene_outdir: Path, name: str, outputs: Dict) -> None:
    for stage_name, stage_output in outputs.items():
        if not stage_name.startswith("stage") or not isinstance(stage_output, dict):
            continue
        suffix = stage_name.replace("stage", "_stage")
        if "depth" in stage_output:
            deps.save_pfm(str(scene_outdir / "depth_est" / f"{name}{suffix}.pfm"), stage_output["depth"])
        if "photometric_confidence" in stage_output:
            deps.save_pfm(
                str(scene_outdir / "confidence" / f"{name}{suffix}.pfm"),
                stage_output["photometric_confidence"],
            )


def run_depth_estimation(
    testpath: str,
    outdir: str,
    checkpoint: str,
    log: Callable[[str], None],
    dataset: str = "general_eval",
    num_view: int = 5,
    numdepth: int = 192,
    interval_scale: float = 1.06,
    max_h: int = 512,
    max_w: int = 640,
    batch_size: int = 1,
    ndepths: str = "48,32,8",
    depth_inter_r: str = "4,2,1",
    cr_base_chs: str = "8,8,8",
    share_cr: bool = False,
    grad_method: str = "detach",
    mask_ratio: float = 0.1,
    use_mae: bool = False,
    use_damw: bool = False,
    use_sparse_conv: bool = True,
    num_worker: int = 0,
    use_gpu: bool = True,
) -> None:
    """运行 MAEMVS 深度估计，保存深度图、置信度、相机和融合元数据。"""
    if not Path(checkpoint).is_file():
        raise DepthEstimationError(f"Checkpoint not found: {checkpoint}")

    test_root = Path(testpath)
    testlist = [item.name for item in test_root.iterdir() if item.is_dir()]
    if not testlist:
        raise DepthEstimationError(f"No scenes found in {testpath}")

    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    device = torch.device("cpu" if not use_gpu or not torch.cuda.is_available() else "cuda")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
        free_mem = torch.cuda.mem_get_info()[0] / 1024**3
        log(f"GPU 可用显存: {free_mem:.1f} GB")

    import platform
    if platform.system() == "Windows":
        num_worker = 0

    deps = _get_maemvs_deps()
    use_mae, use_damw = _detect_checkpoint_model_options(checkpoint, use_mae, use_damw, log)
    model = _build_model(
        deps, ndepths, depth_inter_r, cr_base_chs,
        share_cr, grad_method, use_mae, mask_ratio, use_damw, use_sparse_conv,
    ).to(device)
    log(f"Loading MAEMVS checkpoint: {checkpoint}")
    _load_checkpoint(model, checkpoint, log)
    model.eval()

    try:
        for scene in testlist:
            _save_scene_depth(
                deps=deps,
                scene=scene,
                testpath=str(test_root),
                outdir=Path(outdir),
                model=model,
                device=device,
                dataset=dataset,
                num_view=num_view,
                numdepth=numdepth,
                interval_scale=interval_scale,
                max_h=max_h,
                max_w=max_w,
                batch_size=batch_size,
                num_worker=num_worker,
                log=log,
            )
    except RuntimeError as exc:
        if "CUDA out of memory" in str(exc):
            if device.type == "cuda":
                torch.cuda.empty_cache()
            raise DepthEstimationError(
                f"GPU 显存不足。请降低分辨率 (max_h/max_w) 或使用 CPU (use_gpu=False)。详情: {exc}"
            ) from exc
        raise
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    log("MAEMVS depth estimation completed.")


@torch.no_grad()
def _save_scene_depth(
    deps: MAEMVSDeps,
    scene: str,
    testpath: str,
    outdir: Path,
    model,
    device: torch.device,
    dataset: str,
    num_view: int,
    numdepth: int,
    interval_scale: float,
    max_h: int,
    max_w: int,
    batch_size: int,
    num_worker: int,
    log: Callable[[str], None],
) -> None:
    """对单个场景运行 MAEMVS 深度估计。"""
    scene_outdir = outdir / scene
    _ensure_scene_dirs(scene_outdir)

    MVSDataset = deps.find_dataset_def(dataset)
    test_dataset = MVSDataset(
        testpath, [scene], "test", num_view,
        ndepths=numdepth,
        interval_scale=interval_scale,
        num_stage=3,
        max_h=max_h,
        max_w=max_w,
        fix_res=False,
    )
    loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_worker,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    log(f"MAEMVS processing scene: {scene}")
    metadata_path = scene_outdir / "metadata.jsonl"
    with open(metadata_path, "w", encoding="utf-8") as metadata:
        for batch_idx, sample in enumerate(loader):
            start_time = time.time()
            sample_cuda = tocuda(sample, device)
            outputs = model(
                sample_cuda["imgs"],
                sample_cuda["proj_matrices"],
                sample_cuda["depth_values"],
                return_mae=False,
            )
            outputs_np = tensor2numpy(outputs)
            confidence = outputs_np.get("photometric_confidence")
            if confidence is None:
                confidence = np.ones_like(outputs_np["depth"], dtype=np.float32)

            imgs = tensor2numpy(sample["imgs"])
            proj_stage3 = tensor2numpy(sample["proj_matrices"]["stage3"])
            depth_values = tensor2numpy(sample["depth_values"])

            for item_idx in range(outputs_np["depth"].shape[0]):
                name = _sample_base_name(sample, item_idx)
                depth_file = scene_outdir / "depth_est" / f"{name}.pfm"
                conf_file = scene_outdir / "confidence" / f"{name}.pfm"
                cam_file = scene_outdir / "cams" / f"{name}.npz"
                cam_txt_file = scene_outdir / "cams" / f"{name}_cam.txt"
                image_file = scene_outdir / "images" / f"{name}.jpg"

                deps.save_pfm(str(depth_file), outputs_np["depth"][item_idx].astype(np.float32))
                deps.save_pfm(str(conf_file), confidence[item_idx].astype(np.float32))
                _save_camera_npz(cam_file, proj_stage3[item_idx])
                _write_cam_txt(cam_txt_file, proj_stage3[item_idx][0], depth_values[item_idx])

                img = _to_numpy_image(imgs[item_idx, 0])
                cv2.imwrite(str(image_file), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

                stage_outputs = {}
                for key, value in outputs_np.items():
                    if key.startswith("stage") and isinstance(value, dict):
                        stage_outputs[key] = {
                            inner_key: inner_value[item_idx]
                            for inner_key, inner_value in value.items()
                            if inner_key in ("depth", "photometric_confidence")
                        }
                _save_stage_maps(deps, scene_outdir, name, stage_outputs)

                metadata.write(json.dumps({
                    "name": name,
                    "depth": os.path.relpath(depth_file, scene_outdir).replace("\\", "/"),
                    "confidence": os.path.relpath(conf_file, scene_outdir).replace("\\", "/"),
                    "camera": os.path.relpath(cam_file, scene_outdir).replace("\\", "/"),
                    "image": os.path.relpath(image_file, scene_outdir).replace("\\", "/"),
                    "src_names": [],
                }) + "\n")

            del sample_cuda, outputs, outputs_np
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            log(f"MAEMVS iter {batch_idx + 1}/{len(loader)}, Time: {time.time() - start_time:.2f}s")

    log(f"MAEMVS metadata saved: {metadata_path}")


def run_point_fusion(
    testpath: str,
    outdir: str,
    conf: float = 0.8,
    thres_view: int = 5,
    max_src_views: int = 4,
    max_points_per_view: int = 0,
    log: Optional[Callable[[str], None]] = None,
) -> str:
    """运行 MAEMVS 自带的 metadata/npz 相机点云融合。"""
    deps = _get_maemvs_deps()

    test_root = Path(testpath)
    testlist = [item.name for item in test_root.iterdir() if item.is_dir()]
    if not testlist:
        raise DepthEstimationError(f"No scenes found in {testpath}")

    result_ply = ""
    for scene in testlist:
        scene_outdir = Path(outdir) / scene
        metadata_path = scene_outdir / "metadata.jsonl"
        if not metadata_path.is_file():
            raise DepthEstimationError(f"MAEMVS metadata not found: {metadata_path}")

        ply_path = Path(outdir) / f"{scene}.ply"
        args = SimpleNamespace(
            depth_dir=str(scene_outdir),
            metadata=str(metadata_path),
            out_ply=str(ply_path),
            conf_thresh=float(conf),
            geo_pixel_thresh=1.0,
            geo_depth_thresh=0.01,
            min_consistent=max(1, min(int(thres_view), int(max_src_views))),
            max_src_views=int(max_src_views),
            max_points_per_view=int(max_points_per_view),
        )
        if log:
            log(f"MAEMVS fusing point cloud for scene: {scene}")
        deps.fuse_depth_maps(args)
        result_ply = str(ply_path)
        if log:
            log(f"Saved MAEMVS point cloud: {ply_path}")

    return result_ply
