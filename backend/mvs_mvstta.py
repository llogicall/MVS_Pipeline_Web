"""
MVSTTA 深度估计集成模块。

MVSTTA 是测试时自适应方法：测试阶段通过纹理引导的可学习特征归一化参数进行自适应，
并默认使用 reprojectionv5 进行重投影引导的假设深度重采样。
"""

import gc
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader


class DepthEstimationError(RuntimeError):
    pass


@dataclass
class MVSTTADeps:
    find_dataset_def: Callable
    read_pfm: Callable
    save_pfm: Callable
    MVSNet: type
    TextureAdaptiveNorm: type
    filter_depth: Callable


def _activate_mvs_eval_dir(eval_dir: Path) -> None:
    eval_dir_str = str(eval_dir)
    sys.path[:] = [p for p in sys.path if p != eval_dir_str]
    sys.path.insert(0, eval_dir_str)
    for name in list(sys.modules):
        if (
            name == "datasets" or name.startswith("datasets.")
            or name == "networks" or name.startswith("networks.")
            or name == "filter" or name.startswith("filter.")
            or name == "tools"
        ):
            del sys.modules[name]


def _get_mvstta_deps() -> MVSTTADeps:
    """延迟导入 MVSTTA 依赖，避免不同 MVS 方法之间的模块名冲突。"""
    import importlib

    mvstta_dir = Path(__file__).resolve().parent.parent / "MVS" / "MVSTTA_eval"
    _activate_mvs_eval_dir(mvstta_dir)

    datasets_mod = importlib.import_module("datasets")
    io_mod = importlib.import_module("datasets.data_io")
    mvsnet_mod = importlib.import_module("networks.mvsnet")
    texture_mod = importlib.import_module("networks.texture_norm")
    pcd_mod = importlib.import_module("filter.pcd")

    return MVSTTADeps(
        find_dataset_def=datasets_mod.find_dataset_def,
        read_pfm=io_mod.read_pfm,
        save_pfm=io_mod.save_pfm,
        MVSNet=mvsnet_mod.MVSNet,
        TextureAdaptiveNorm=texture_mod.TextureAdaptiveNorm,
        filter_depth=pcd_mod.filter_depth,
    )


def _build_default_args(**overrides) -> SimpleNamespace:
    """构造 MVSTTA 所需参数；默认启用 texture norm 和 reprojectionv5。"""
    args = SimpleNamespace(
        fea_mode="fpn",
        agg_mode="variance",
        depth_mode="unification",
        ndepths=[48, 32, 8],
        interval_ratio=[4.0, 2.0, 1.0],
        cr_base_chs=[8, 8, 8],
        ds_mode="reprojectionv5",
        use_texture_norm=True,
        tta_lr=5e-5,
        tta_steps=1,
        wd=0.0,
        dataset_name="general_eval",
        numdepth=192,
        interval_scale=1.06,
        inverse_depth=False,
        max_h=512,
        max_w=640,
        fix_res=False,
        batch_size=1,
        num_view=5,
        nviews=5,
        num_worker=0,
        conf=[0.1, 0.15, 0.9],
        thres_view=5,
        display=False,
        distributed=False,
        no_cuda=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


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


def write_cam(filename: str, cam: np.ndarray) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        f.write("extrinsic\n")
        for row in cam[0, :4, :4]:
            f.write(" ".join(str(float(v)) for v in row) + "\n")
        f.write("\nintrinsic\n")
        for row in cam[1, :3, :3]:
            f.write(" ".join(str(float(v)) for v in row) + "\n")
        f.write("\n" + " ".join(str(float(v)) for v in cam[1, 3, :4]) + "\n")


def _build_model(deps: MVSTTADeps, args: SimpleNamespace):
    return deps.MVSNet(
        ndepths=args.ndepths,
        depth_interval_ratio=args.interval_ratio,
        cr_base_chs=args.cr_base_chs,
        fea_mode=args.fea_mode,
        agg_mode=args.agg_mode,
        depth_mode=args.depth_mode,
        args=args,
    )


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
                f"GPU 显存不足，无法加载 MVSTTA。请降低分辨率 (max_h/max_w) 或使用 CPU (use_gpu=False)。详情: {exc}"
            ) from exc
        raise
    finally:
        del state_dict, model_state, cleaned
        gc.collect()
    log("MVSTTA checkpoint loaded with strict=True")


def _texture_norm_parameters(model) -> List[torch.nn.Parameter]:
    return [
        param for name, param in model.named_parameters()
        if "texture_norm" in name and param.requires_grad
    ]


def _run_tta_steps(model, sample_cuda: Dict, optimizer, steps: int, log: Callable[[str], None]) -> None:
    if steps <= 0:
        return
    model.eval()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
        loss = outputs.get("warp_loss")
        if loss is None:
            raise DepthEstimationError("MVSTTA forward did not return warp_loss for test-time adaptation")
        loss.backward()
        optimizer.step()
        log(f"MVSTTA TTA step {step + 1}/{steps}, warp_loss={float(loss.detach().cpu()):.6f}")
        del outputs, loss


def _save_stage_maps(deps: MVSTTADeps, outdir: str, filename_fmt: str, outputs: Dict, final_conf: np.ndarray) -> None:
    final_h, final_w = final_conf.shape
    for stage in ("stage1", "stage2"):
        stage_output = outputs.get(stage)
        if not isinstance(stage_output, dict):
            continue
        suffix = f"_{stage}.pfm"
        if "depth" in stage_output:
            deps.save_pfm(os.path.join(outdir, filename_fmt.format("depth_est", suffix)), stage_output["depth"])
        if "photometric_confidence" in stage_output:
            conf = stage_output["photometric_confidence"]
            conf = cv2.resize(conf, (final_w, final_h), interpolation=cv2.INTER_NEAREST)
            deps.save_pfm(os.path.join(outdir, filename_fmt.format("confidence", suffix)), conf)


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
    ndepths: Optional[List[int]] = None,
    interval_ratio: Optional[List[float]] = None,
    cr_base_chs: Optional[List[int]] = None,
    tta_lr: float = 5e-5,
    tta_steps: int = 1,
    ds_mode: str = "reprojectionv5",
    use_texture_norm: bool = True,
    num_worker: int = 0,
    use_gpu: bool = True,
) -> None:
    """运行 MVSTTA 深度估计，并在测试时自适应 texture norm 参数。"""
    if not Path(checkpoint).is_file():
        raise DepthEstimationError(f"Checkpoint not found: {checkpoint}")

    test_root = Path(testpath)
    testlist = [item.name for item in test_root.iterdir() if item.is_dir()]
    if not testlist:
        raise DepthEstimationError(f"No scenes found in {testpath}")

    import platform
    if platform.system() == "Windows":
        num_worker = 0

    if ndepths is None:
        ndepths = [48, 32, 8]
    if interval_ratio is None:
        interval_ratio = [4.0, 2.0, 1.0]
    if cr_base_chs is None:
        cr_base_chs = [8, 8, 8]

    args = _build_default_args(
        dataset_name=dataset,
        num_view=num_view,
        nviews=num_view,
        numdepth=numdepth,
        interval_scale=interval_scale,
        max_h=max_h,
        max_w=max_w,
        batch_size=batch_size,
        num_worker=num_worker,
        ndepths=ndepths,
        interval_ratio=interval_ratio,
        cr_base_chs=cr_base_chs,
        tta_lr=tta_lr,
        tta_steps=tta_steps,
        ds_mode=ds_mode,
        use_texture_norm=use_texture_norm,
        no_cuda=(not use_gpu),
    )
    if args.ds_mode != "reprojectionv5":
        log(f"MVSTTA ds_mode override: {args.ds_mode}")

    device = torch.device("cpu" if not use_gpu or not torch.cuda.is_available() else "cuda")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
        free_mem = torch.cuda.mem_get_info()[0] / 1024**3
        log(f"GPU 可用显存: {free_mem:.1f} GB")

    deps = _get_mvstta_deps()
    model = _build_model(deps, args).to(device)
    log(f"Loading MVSTTA checkpoint: {checkpoint}")
    _load_checkpoint(model, checkpoint, log)

    tta_params = _texture_norm_parameters(model)
    if args.use_texture_norm and not tta_params:
        raise DepthEstimationError("未找到 texture_norm 参数，无法执行 MVSTTA 测试时自适应")
    optimizer_tta = torch.optim.Adam(tta_params, lr=args.tta_lr, betas=(0.9, 0.999), weight_decay=args.wd)

    try:
        for scene in testlist:
            _save_scene_depth(
                deps=deps,
                args=args,
                scene=scene,
                testpath=str(test_root),
                outdir=outdir,
                model=model,
                optimizer_tta=optimizer_tta,
                device=device,
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

    log("MVSTTA depth estimation completed.")


def _save_scene_depth(
    deps: MVSTTADeps,
    args: SimpleNamespace,
    scene: str,
    testpath: str,
    outdir: str,
    model,
    optimizer_tta,
    device: torch.device,
    log: Callable[[str], None],
) -> None:
    """对单个场景运行 MVSTTA 深度估计。"""
    MVSDataset = deps.find_dataset_def(args.dataset_name)
    dataset = MVSDataset(
        testpath,
        [scene],
        "test",
        args.num_view,
        ndepths=args.numdepth,
        interval_scale=args.interval_scale,
        inverse_depth=args.inverse_depth,
        max_h=args.max_h,
        max_w=args.max_w,
        fix_res=args.fix_res,
    )
    loader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.num_worker, drop_last=False)

    log(f"MVSTTA processing scene: {scene}, ds_mode={args.ds_mode}, texture_norm={args.use_texture_norm}")
    for batch_idx, sample in enumerate(loader):
        sample_cuda = tocuda(sample, device)
        start_time = time.time()

        _run_tta_steps(model, sample_cuda, optimizer_tta, int(args.tta_steps), log)

        with torch.no_grad():
            outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
        elapsed = time.time() - start_time
        outputs_np = tensor2numpy(outputs)

        filenames = sample["filename"]
        cams = sample["proj_matrices"][f"stage{len(args.ndepths)}"].numpy()
        imgs = sample["imgs"].numpy()
        log(f"MVSTTA iter {batch_idx + 1}/{len(loader)}, Time: {elapsed:.2f}s")

        for filename, cam, img, depth_est, photometric_confidence in zip(
            filenames, cams, imgs, outputs_np["depth"], outputs_np["photometric_confidence"]
        ):
            img = img[0]
            cam = cam[0]
            depth_filename = os.path.join(outdir, filename.format("depth_est", ".pfm"))
            confidence_filename = os.path.join(outdir, filename.format("confidence", ".pfm"))
            cam_filename = os.path.join(outdir, filename.format("cams", "_cam.txt"))
            img_filename = os.path.join(outdir, filename.format("images", ".jpg"))

            for path in (depth_filename, confidence_filename, cam_filename, img_filename):
                os.makedirs(os.path.dirname(path), exist_ok=True)

            deps.save_pfm(depth_filename, depth_est.astype(np.float32))
            deps.save_pfm(confidence_filename, photometric_confidence.astype(np.float32))
            write_cam(cam_filename, cam)
            img = np.clip(np.transpose(img, (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
            cv2.imwrite(img_filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            _save_stage_maps(deps, outdir, filename, outputs_np, photometric_confidence)

        del sample_cuda, outputs, outputs_np
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()


def run_point_fusion(
    testpath: str,
    outdir: str,
    conf: Optional[List[float]] = None,
    thres_view: int = 5,
    num_worker: int = 0,
    log: Optional[Callable[[str], None]] = None,
) -> str:
    """运行 MVSTTA 的 PCD 几何一致性点云融合。"""
    deps = _get_mvstta_deps()
    if conf is None or isinstance(conf, (float, int)):
        final_conf = 0.9 if conf is None else float(conf)
        conf = [0.1, 0.15, final_conf]

    args = _build_default_args(conf=conf, thres_view=thres_view, num_worker=num_worker)
    test_root = Path(testpath)
    testlist = [item.name for item in test_root.iterdir() if item.is_dir()]
    if not testlist:
        raise DepthEstimationError(f"No scenes found in {testpath}")

    result_ply = ""
    for scene in testlist:
        pair_folder = os.path.join(testpath, scene)
        scan_folder = os.path.join(outdir, scene)
        out_folder = os.path.join(outdir, scene)
        ply_path = os.path.join(outdir, f"{scene}.ply")
        if log:
            log(f"MVSTTA fusing point cloud for scene: {scene}")
        deps.filter_depth(args, pair_folder, scan_folder, out_folder, ply_path)
        result_ply = ply_path
        if log:
            log(f"Saved MVSTTA point cloud: {ply_path}")

    return result_ply
