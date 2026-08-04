"""
UniMVSNet 深度估计集成模块

将 MVS/UniMVSNet_eval/test.py 的深度估计和点云融合逻辑封装为可直接调用的函数，
避免使用 subprocess 调用外部脚本。
"""

import gc
import os
import sys
import time
import signal
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from PIL import Image
from plyfile import PlyData, PlyElement

cudnn.benchmark = True


class DepthEstimationError(RuntimeError):
    pass


def _activate_mvs_eval_dir(eval_dir: Path) -> None:
    eval_dir_str = str(eval_dir)
    sys.path[:] = [p for p in sys.path if p != eval_dir_str]
    sys.path.insert(0, eval_dir_str)
    for name in list(sys.modules):
        if (
            name == "datasets" or name.startswith("datasets.")
            or name == "models" or name.startswith("models.")
            or name == "networks" or name.startswith("networks.")
        ):
            del sys.modules[name]


def _get_unimvsnet_deps():
    """延迟导入 UniMVSNet 依赖，避免与其他 MVS 模块的 sys.path 冲突"""
    import importlib
    from pathlib import Path

    unet_dir = Path(__file__).resolve().parent.parent / "MVS" / "UniMVSNet_eval"
    _activate_mvs_eval_dir(unet_dir)

    datasets_mod = importlib.import_module("datasets")
    find_dataset_def = datasets_mod.find_dataset_def

    io_mod = importlib.import_module("datasets.data_io")
    read_pfm = io_mod.read_pfm
    save_pfm = io_mod.save_pfm

    model_mod = importlib.import_module("networks.mvsnet")
    MVSNet = model_mod.MVSNet

    return find_dataset_def, read_pfm, save_pfm, MVSNet


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def write_cam(file, cam):
    with open(file, "w") as f:
        f.write('extrinsic\n')
        for i in range(4):
            for j in range(4):
                f.write(str(cam[0][i][j]) + ' ')
            f.write('\n')
        f.write('\n')
        f.write('intrinsic\n')
        for i in range(3):
            for j in range(3):
                f.write(str(cam[1][i][j]) + ' ')
            f.write('\n')
        f.write('\n' + str(cam[1][3][0]) + ' ' + str(cam[1][3][1]) + ' ' + str(cam[1][3][2]) + ' ' + str(cam[1][3][3]) + '\n')


def make_recursive_func(func):
    def wrapper(vars):
        if isinstance(vars, list):
            return [wrapper(x) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v) for k, v in vars.items()}
        else:
            return func(vars)
    return wrapper


@make_recursive_func
def tensor2numpy(vars):
    if isinstance(vars, np.ndarray):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.detach().cpu().numpy().copy()
    else:
        raise NotImplementedError("invalid input type {}".format(type(vars)))


def tocuda(vars, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(vars, torch.Tensor):
        return vars.to(device)
    elif isinstance(vars, dict):
        return {key: tocuda(val, device) for key, val in vars.items()}
    elif isinstance(vars, list):
        return [tocuda(val, device) for val in vars]
    elif isinstance(vars, tuple):
        return tuple(tocuda(val, device) for val in vars)
    else:
        return vars


def _prepare_pfm_map(value):
    """将模型输出整理成 save_pfm 接受的 HxW/HxWx1/HxWx3 数组。"""
    array = np.asarray(value)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.dtype != np.float32:
        array = array.astype(np.float32)
    return array


# ---------------------------------------------------------------------------
# 深度估计核心函数
# ---------------------------------------------------------------------------

def run_depth_estimation(
    testpath: str,
    outdir: str,
    checkpoint: str,
    log: Callable[[str], None],
    dataset: str = "general_eval",
    num_view: int = 5,
    numdepth: int = 192,
    interval_scale: float = 1.06,
    max_h: int = 864,
    max_w: int = 1152,
    batch_size: int = 1,
    ndepths: List[int] = None,
    interval_ratio: List[float] = None,
    fea_mode: str = "fpn",
    agg_mode: str = "variance",
    depth_mode: str = "regression",
    num_worker: int = 0,
    use_gpu: bool = True,
) -> None:
    """
    运行 UniMVSNet 深度估计。

    参数：
      testpath: 包含场景目录的数据路径
      outdir: 输出目录
      checkpoint: 模型权重文件路径
      log: 日志回调函数
    """
    if ndepths is None:
        ndepths = [48, 32, 8]
    if interval_ratio is None:
        interval_ratio = [4, 2, 1]

    if not Path(checkpoint).is_file():
        raise DepthEstimationError(f"Checkpoint not found: {checkpoint}")

    # 确定场景列表
    testlist = [e for e in os.listdir(testpath) if os.path.isdir(os.path.join(testpath, e))]
    if not testlist:
        raise DepthEstimationError(f"No scenes found in {testpath}")

    num_stage = len(ndepths)

    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    log(f"UniMVSNet: {len(testlist)} scene(s) in {testpath}")
    log(f"Checkpoint: {checkpoint}")
    log(f"Output: {outdir}")
    log(f"ndepths: {ndepths}, interval_ratio: {interval_ratio}")

    # 延迟导入 UniMVSNet 依赖
    find_dataset_def, read_pfm, save_pfm, MVSNet = _get_unimvsnet_deps()

    # 加载 checkpoint 并自动检测参数
    log(f"Loading model: {checkpoint}")
    state_dict = torch.load(checkpoint, map_location="cpu")
    model_state = state_dict.get("model", state_dict)

    # 自动检测 agg_mode：如果 checkpoint 包含 weight_net，则为 adaptive
    has_weight_net = any("weight_net" in k for k in model_state.keys())
    if has_weight_net:
        agg_mode = "adaptive"
        log("Detected adaptive aggregation mode from checkpoint")

    # 构建模型
    device = torch.device("cpu" if not use_gpu or not torch.cuda.is_available() else "cuda")

    # GPU 显存检查
    if device.type == "cuda":
        free_mem = torch.cuda.mem_get_info()[0] / 1024**3
        log(f"GPU 可用显存: {free_mem:.1f} GB")
        if free_mem < 2.0:
            log("警告: GPU 可用显存不足，可能导致 OOM 错误")

    model = MVSNet(
        ndepths=ndepths,
        depth_interval_ratio=interval_ratio,
        fea_mode=fea_mode,
        agg_mode=agg_mode,
        depth_mode=depth_mode,
    ).to(device)

    try:
        model.load_state_dict(model_state, strict=True)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            raise DepthEstimationError(
                f"GPU 显存不足，无法加载模型。请降低分辨率 (max_h/max_w) 或使用 CPU (use_gpu=False)。"
                f"详情: {e}"
            ) from e
        raise
    del state_dict, model_state
    model.eval()

    # Windows 下强制 num_worker=0，避免 DataLoader worker 崩溃
    import platform
    if platform.system() == "Windows":
        num_worker = 0

    try:
        with torch.no_grad():
            for scene in testlist:
                log(f"Processing scene: {scene}")
                _save_scene_depth(
                    scene=scene,
                    testpath=testpath,
                    outdir=outdir,
                    model=model,
                    device=device,
                    dataset=dataset,
                    num_view=num_view,
                    numdepth=numdepth,
                    interval_scale=interval_scale,
                    max_h=max_h,
                    max_w=max_w,
                    batch_size=batch_size,
                    num_stage=num_stage,
                    num_worker=num_worker,
                    log=log,
                )
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            torch.cuda.empty_cache()
            raise DepthEstimationError(
                f"GPU 显存不足。请降低分辨率 (max_h/max_w) 或使用 CPU (use_gpu=False)。详情: {e}"
            ) from e
        raise
    finally:
        del model
        torch.cuda.empty_cache()
        gc.collect()

    log("UniMVSNet depth estimation completed.")


def _save_scene_depth(
    scene: str,
    testpath: str,
    outdir: str,
    model,
    device,
    dataset: str,
    num_view: int,
    numdepth: int,
    interval_scale: float,
    max_h: int,
    max_w: int,
    batch_size: int,
    num_stage: int,
    num_worker: int,
    log: Callable[[str], None],
) -> None:
    """对单个场景运行深度估计"""
    find_dataset_def, read_pfm, save_pfm, _ = _get_unimvsnet_deps()

    MVSDataset = find_dataset_def(dataset)
    test_dataset = MVSDataset(
        testpath, [scene], "test", num_view,
        ndepths=numdepth,
        interval_scale=interval_scale,
        inverse_depth=False,
        num_stage=3,
        max_h=max_h, max_w=max_w, fix_res=False,
    )
    TestImgLoader = DataLoader(
        test_dataset, batch_size, shuffle=False, num_workers=num_worker, drop_last=False
    )

    for batch_idx, sample in enumerate(TestImgLoader):
        sample_cuda = tocuda(sample, device)
        start_time = time.time()
        outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
        end_time = time.time()
        outputs = tensor2numpy(outputs)
        del sample_cuda
        torch.cuda.empty_cache()

        filenames = sample["filename"]
        cams = sample["proj_matrices"]["stage{}".format(num_stage)].numpy()
        imgs = sample["imgs"].numpy()
        log(f'Iter {batch_idx+1}/{len(TestImgLoader)}, Time:{end_time - start_time:.2f}s')

        for filename, cam, img, depth_est, photometric_confidence in zip(
            filenames, cams, imgs, outputs["depth"], outputs["photometric_confidence"]
        ):
            depth_est = _prepare_pfm_map(depth_est)
            photometric_confidence = _prepare_pfm_map(photometric_confidence)
            img = img[0]
            cam = cam[0]
            depth_filename = os.path.join(outdir, filename.format('depth_est', '.pfm'))
            confidence_filename = os.path.join(outdir, filename.format('confidence', '.pfm'))
            cam_filename = os.path.join(outdir, filename.format('cams', '_cam.txt'))
            img_filename = os.path.join(outdir, filename.format('images', '.jpg'))

            for d in [depth_filename, confidence_filename, cam_filename, img_filename]:
                os.makedirs(os.path.dirname(d), exist_ok=True)

            save_pfm(depth_filename, depth_est)
            save_pfm(confidence_filename, photometric_confidence)
            write_cam(cam_filename, cam)
            img = np.clip(np.transpose(img, (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(img_filename, img_bgr)

            # 保存中间 stage 的深度图（如果有）
            if "stage2" in outputs and "depth" in outputs["stage2"]:
                depth2 = outputs["stage2"]["depth"][0] if isinstance(outputs["stage2"]["depth"], list) else outputs["stage2"]["depth"]
                save_pfm(os.path.join(outdir, filename.format('depth_est', '_stage2.pfm')), _prepare_pfm_map(depth2))
                pc2 = outputs["stage2"]["photometric_confidence"][0] if isinstance(outputs["stage2"]["photometric_confidence"], list) else outputs["stage2"]["photometric_confidence"]
                h, w = photometric_confidence.shape
                pc2_resized = cv2.resize(_prepare_pfm_map(pc2), (w, h), interpolation=cv2.INTER_NEAREST)
                save_pfm(os.path.join(outdir, filename.format('confidence', '_stage2.pfm')), pc2_resized)

            if "stage1" in outputs and "depth" in outputs["stage1"]:
                depth1 = outputs["stage1"]["depth"][0] if isinstance(outputs["stage1"]["depth"], list) else outputs["stage1"]["depth"]
                save_pfm(os.path.join(outdir, filename.format('depth_est', '_stage1.pfm')), _prepare_pfm_map(depth1))
                pc1 = outputs["stage1"]["photometric_confidence"][0] if isinstance(outputs["stage1"]["photometric_confidence"], list) else outputs["stage1"]["photometric_confidence"]
                h, w = photometric_confidence.shape
                pc1_resized = cv2.resize(_prepare_pfm_map(pc1), (w, h), interpolation=cv2.INTER_NEAREST)
                save_pfm(os.path.join(outdir, filename.format('confidence', '_stage1.pfm')), pc1_resized)

        del sample, outputs, filenames, cams, imgs
        torch.cuda.empty_cache()
        gc.collect()


# ---------------------------------------------------------------------------
# 点云融合核心函数（复用 filter/pcd.py）
# ---------------------------------------------------------------------------

def run_point_fusion(
    testpath: str,
    outdir: str,
    conf: List[float] = None,
    thres_view: int = 5,
    num_stage: int = 3,
    num_worker: int = 0,
    log: Optional[Callable[[str], None]] = None,
) -> str:
    """
    运行点云融合：过滤深度图并生成最终点云。

    返回生成的 PLY 文件路径。
    """
    if conf is None:
        conf = [0.1, 0.15, 0.9]

    # 动态导入 filter 模块（避免循环导入）
    mvs_eval_dir = Path(__file__).resolve().parent.parent / "MVS" / "UniMVSNet_eval"
    sys.path.insert(0, str(mvs_eval_dir))
    from filter.pcd import filter_depth

    testlist = [e for e in os.listdir(testpath) if os.path.isdir(os.path.join(testpath, e))]
    if not testlist:
        raise DepthEstimationError(f"No scenes found in {testpath}")

    # 创建一个简单的 args 对象来传递参数
    class Args:
        pass
    args = Args()
    args.ndepths = [48, 32, 8]  # 默认值
    args.conf = conf
    args.thres_view = thres_view
    args.num_view = 5
    args.display = False

    result_ply = None
    for scene in testlist:
        pair_folder = os.path.join(testpath, scene)
        scan_folder = os.path.join(outdir, scene)
        out_folder = os.path.join(outdir, scene)
        ply_path = os.path.join(outdir, f"{scene}.ply")

        if log:
            log(f"Fusing point cloud for scene: {scene}")

        filter_depth(args, pair_folder, scan_folder, out_folder, ply_path)
        result_ply = ply_path

        if log:
            log(f"Saved point cloud: {ply_path}")

    return result_ply or ""
