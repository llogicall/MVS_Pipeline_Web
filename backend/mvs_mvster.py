"""
MVSTER 深度估计集成模块

将 MVS/MVSTER_eval 的深度估计和点云融合逻辑封装为可直接调用的函数。
"""

import gc
import os
import sys
import time
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

MVSTER_RESIZE_BASE = 64


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


def _get_mvster_deps():
    """延迟导入 MVSTER 依赖，避免与其他 MVS 模块的 sys.path 冲突"""
    import importlib
    from pathlib import Path

    mvster_dir = Path(__file__).resolve().parent.parent / "MVS" / "MVSTER_eval"
    _activate_mvs_eval_dir(mvster_dir)

    datasets_mod = importlib.import_module("datasets")
    find_dataset_def = datasets_mod.find_dataset_def

    io_mod = importlib.import_module("datasets.data_io")
    read_pfm = io_mod.read_pfm
    save_pfm = io_mod.save_pfm

    model_mod = importlib.import_module("models.MVS4Net")
    MVS4net = model_mod.MVS4net

    return find_dataset_def, read_pfm, save_pfm, MVS4net


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


def _mvster_views_first(imgs: torch.Tensor) -> torch.Tensor:
    if imgs.ndim != 5:
        return imgs
    return imgs.transpose(0, 1).contiguous()


def _tensor_image_to_numpy(img):
    if isinstance(img, torch.Tensor):
        return img.detach().cpu().numpy()
    return img


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
    interval_scale: float = 1.06,
    max_h: int = 864,
    max_w: int = 1152,
    batch_size: int = 1,
    num_worker: int = 0,
    use_gpu: bool = True,
    ndepths: str = "8,8,4,4",
    depth_inter_r: str = "0.5,0.5,0.5,1",
    fpn_base_channel: int = 8,
    reg_channel: int = 8,
    reg_mode: str = "reg2d",
    group_cor: bool = False,
    group_cor_dim: str = "8,8,4,4",
    inverse_depth: bool = False,
    agg_type: str = "ConvBnReLU3D",
    dcn: bool = False,
    arch_mode: str = "fpn",
    mono: bool = False,
    asff: bool = False,
    attn_temp: float = 2.0,
) -> None:
    """
    运行 MVSTER 深度估计。
    """
    if not Path(checkpoint).is_file():
        raise DepthEstimationError(f"Checkpoint not found: {checkpoint}")

    testlist = [e for e in os.listdir(testpath) if os.path.isdir(os.path.join(testpath, e))]
    if not testlist:
        raise DepthEstimationError(f"No scenes found in {testpath}")

    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    log(f"MVSTER: {len(testlist)} scene(s) in {testpath}")
    log(f"Checkpoint: {checkpoint}")
    log(f"Output: {outdir}")
    log(f"ndepths: {ndepths}, depth_inter_r: {depth_inter_r}")

    # 延迟导入 MVSTER 依赖
    find_dataset_def, read_pfm, save_pfm, MVS4net = _get_mvster_deps()

    device = torch.device("cpu" if not use_gpu or not torch.cuda.is_available() else "cuda")

    # 加载 checkpoint 并自动检测参数
    log(f"Loading model: {checkpoint}")
    state_dict = torch.load(checkpoint, map_location="cpu")
    model_state = state_dict.get("model", state_dict)

    # 从 checkpoint 推断 stage_splits 和 fpn_base_channel
    stage_splits_list = [int(n) for n in ndepths.split(",") if n]
    num_stage = len(stage_splits_list)

    # 检测 fpn_base_channel：从 feature.conv0.0.conv.weight shape 推断
    # shape: [fpn_base_channel, 3, 3, 3]
    fpn_key = "feature.conv0.0.conv.weight"
    if fpn_key in model_state:
        detected_fpn_base = model_state[fpn_key].shape[0]
        if detected_fpn_base > 0:
            fpn_base_channel = detected_fpn_base
            log(f"Detected from checkpoint: fpn_base_channel={fpn_base_channel}")

    # 检测 reg_channel 和 group_cor 模式
    # reg.0.conv0.conv.weight shape: [reg_channel, input_channels, 1, 3, 3]
    # - group_cor=False: input_channels = fpn_base_channel * stage_splits[i]
    # - group_cor=True:  input_channels = group_cor_dim[i]
    reg_key = "reg.0.conv0.conv.weight"
    if reg_key in model_state:
        reg_weight = model_state[reg_key]
        detected_reg_channel = reg_weight.shape[0]
        detected_input_channels = reg_weight.shape[1]
        if detected_reg_channel > 0:
            reg_channel = detected_reg_channel

        # 检测是否为 group_cor 模式
        # 如果 input_channels == stage_splits[0]，则是 group_cor 模式
        if detected_input_channels == stage_splits_list[0]:
            group_cor = True
            # 从各 stage 的 reg 层推断 group_cor_dim
            detected_group_cor_dim = []
            for i in range(num_stage):
                rk = f"reg.{i}.conv0.conv.weight"
                if rk in model_state:
                    detected_group_cor_dim.append(model_state[rk].shape[1])
                else:
                    detected_group_cor_dim.append(stage_splits_list[i])
            group_cor_dim = ",".join(str(d) for d in detected_group_cor_dim)
            log(f"Detected group_cor=True, group_cor_dim={group_cor_dim}, reg_channel={reg_channel}")
        else:
            log(f"Detected from checkpoint: reg_channel={reg_channel}")

    model = MVS4net(
        arch_mode=arch_mode,
        reg_net=reg_mode,
        num_stage=4,
        fpn_base_channel=fpn_base_channel,
        reg_channel=reg_channel,
        stage_splits=stage_splits_list,
        depth_interals_ratio=[float(ir) for ir in depth_inter_r.split(",")],
        group_cor=group_cor,
        group_cor_dim=[int(n) for n in group_cor_dim.split(",")],
        inverse_depth=inverse_depth,
        agg_type=agg_type,
        dcn=dcn,
        mono=mono,
        asff=asff,
        attn_temp=attn_temp,
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
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.eval()

    # 清理 GPU 缓存
    if device.type == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

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
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    if device.type == "cuda":
        free_mem = torch.cuda.mem_get_info()[0] / 1024**3
        log(f"MVSTER 深度估计完成，GPU 剩余显存: {free_mem:.1f} GB")


def _save_scene_depth(
    scene: str,
    testpath: str,
    outdir: str,
    model,
    device,
    dataset: str,
    num_view: int,
    interval_scale: float,
    max_h: int,
    max_w: int,
    batch_size: int,
    num_stage: int,
    num_worker: int,
    log: Callable[[str], None],
) -> None:
    """对单个场景运行深度估计"""
    find_dataset_def, read_pfm, save_pfm, _ = _get_mvster_deps()

    MVSDataset = find_dataset_def(dataset)
    test_dataset = MVSDataset(
        testpath, [scene], "test", num_view,
        ndepths=192,
        interval_scale=interval_scale,
        num_stage=4,
        fill_to_nviews=True,
        max_h=max_h, max_w=max_w, fix_res=False,
        resize_base=MVSTER_RESIZE_BASE,
    )
    TestImgLoader = DataLoader(
        test_dataset, batch_size, shuffle=False, num_workers=0, drop_last=False
    )

    for batch_idx, sample in enumerate(TestImgLoader):
        sample_cuda = tocuda(sample, device)
        start_time = time.time()
        outputs = model(
            _mvster_views_first(sample_cuda["imgs"]),
            sample_cuda["proj_matrices"],
            sample_cuda["depth_values"],
            sample["filename"],
        )
        end_time = time.time()
        outputs = tensor2numpy(outputs)
        del sample_cuda
        torch.cuda.empty_cache()

        filenames = sample["filename"]
        cams = sample["proj_matrices"]["stage{}".format(num_stage)].numpy()
        imgs = sample["imgs"]
        log(f'Iter {batch_idx+1}/{len(TestImgLoader)}, Time:{end_time - start_time:.2f}s')

        for filename, cam, img, depth_est, photometric_confidence in zip(
            filenames, cams, imgs, outputs["depth"], outputs["photometric_confidence"]
        ):
            img = _tensor_image_to_numpy(img[0])
            cam = cam[0]
            depth_filename = os.path.join(outdir, filename.format('depth_est', '.pfm'))
            confidence_filename = os.path.join(outdir, filename.format('confidence', '.pfm'))
            cam_filename = os.path.join(outdir, filename.format('cams', '_cam.txt'))
            img_filename = os.path.join(outdir, filename.format('images', '.jpg'))

            for d in [depth_filename, confidence_filename, cam_filename, img_filename]:
                os.makedirs(os.path.dirname(d), exist_ok=True)

            save_pfm(depth_filename, depth_est)

            # 保存多阶段置信度
            try:
                if 'stage1' in outputs and 'photometric_confidence' in outputs['stage1']:
                    confidence_list = []
                    for i in range(1, 5):
                        conf_val = outputs['stage{}'.format(i)]['photometric_confidence']
                        # 兼容 float / numpy scalar / numpy array
                        if isinstance(conf_val, (int, float)):
                            conf_arr = np.full_like(depth_est, float(conf_val))
                        elif isinstance(conf_val, np.ndarray):
                            conf_arr = conf_val.squeeze(0) if conf_val.ndim > 2 else conf_val
                        else:
                            conf_arr = np.array(conf_val)
                        confidence_list.append(conf_arr)
                    save_pfm(confidence_filename, confidence_list[-1])
                    for stage_idx in range(4):
                        save_pfm(
                            os.path.join(outdir, filename.format('confidence', '_stage{}.pfm'.format(stage_idx + 1))),
                            confidence_list[stage_idx]
                        )
                else:
                    # 兼容 photometric_confidence 为 float 的情况
                    if isinstance(photometric_confidence, (int, float)):
                        photometric_confidence = np.full_like(depth_est, float(photometric_confidence))
                    save_pfm(confidence_filename, photometric_confidence)
            except Exception as conf_err:
                log(f"警告: 保存置信度失败: {conf_err}")
                # 写入全 1 置信度作为兜底
                save_pfm(confidence_filename, np.ones_like(depth_est))
                save_pfm(confidence_filename, photometric_confidence)

            write_cam(cam_filename, cam)
            img = np.clip(np.transpose(img, (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(img_filename, img_bgr)

        del sample, outputs, filenames, cams, imgs
        torch.cuda.empty_cache()
        gc.collect()


# ---------------------------------------------------------------------------
# 点云融合
# ---------------------------------------------------------------------------

def run_point_fusion(
    testpath: str,
    outdir: str,
    conf: float = 0.9,
    thres_view: int = 5,
    log: Optional[Callable[[str], None]] = None,
) -> str:
    """
    运行点云融合：过滤深度图并生成最终点云。

    返回生成的 PLY 文件路径。
    """
    mvs_eval_dir = Path(__file__).resolve().parent.parent / "MVS" / "MVSTER_eval"
    sys.path.insert(0, str(mvs_eval_dir))
    from filter.pcd import filter_depth

    testlist = [e for e in os.listdir(testpath) if os.path.isdir(os.path.join(testpath, e))]
    if not testlist:
        raise DepthEstimationError(f"No scenes found in {testpath}")

    result_ply = None
    for scene in testlist:
        pair_folder = os.path.join(testpath, scene)
        scan_folder = os.path.join(outdir, scene)
        out_folder = os.path.join(outdir, scene)
        ply_path = os.path.join(outdir, f"{scene}.ply")

        if log:
            log(f"Fusing point cloud for scene: {scene}")

        filter_depth(pair_folder, scan_folder, out_folder, ply_path, conf, thres_view)
        result_ply = ply_path

        if log:
            log(f"Saved point cloud: {ply_path}")

    return result_ply or ""
