"""
CasMVSNet 深度估计集成模块

将 MVS/CasMVSNet_eval/test.py 的深度估计和点云融合逻辑封装为可直接调用的函数，
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
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from PIL import Image
from plyfile import PlyData, PlyElement

cudnn.benchmark = False


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


def _get_casmvsnet_deps():
    """延迟导入 CasMVSNet 依赖，避免与其他 MVS 模块的 sys.path 冲突"""
    import importlib
    from pathlib import Path

    casmvs_dir = Path(__file__).resolve().parent.parent / "MVS" / "CasMVSNet_eval"
    _activate_mvs_eval_dir(casmvs_dir)

    datasets_mod = importlib.import_module("datasets")
    find_dataset_def = datasets_mod.find_dataset_def

    io_mod = importlib.import_module("datasets.data_io")
    read_pfm = io_mod.read_pfm
    save_pfm = io_mod.save_pfm

    models_mod = importlib.import_module("models")
    CascadeMVSNet = models_mod.CascadeMVSNet

    gipuma_mod = importlib.import_module("gipuma")
    gipuma_filter = gipuma_mod.gipuma_filter

    return find_dataset_def, read_pfm, save_pfm, CascadeMVSNet, gipuma_filter


def _filter_casmvsnet_model_state(
    model_state: Dict[str, torch.Tensor],
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, torch.Tensor]:
    """过滤普通 CasMVSNet 推理模型不使用的 checkpoint 附加分支。"""
    unsupported_prefixes = ("model_up.",)
    filtered = {
        key: value
        for key, value in model_state.items()
        if not key.startswith(unsupported_prefixes)
    }
    skipped = len(model_state) - len(filtered)
    if skipped and log:
        log(f"Ignored {skipped} unsupported CasMVSNet checkpoint keys: {unsupported_prefixes}")
    return filtered

cudnn.benchmark = False


class DepthEstimationError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# 工具函数（从 test.py 提取）
# ---------------------------------------------------------------------------

def read_camera_parameters(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    return intrinsics, extrinsics


def read_img(filename):
    img = Image.open(filename)
    np_img = np.array(img, dtype=np.float32) / 255.
    return np_img


def read_mask(filename):
    return read_img(filename) > 0.5


def save_mask(filename, mask):
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(filename)


def read_pair_file(filename):
    data = []
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            if len(src_views) > 0:
                data.append((ref_view, src_views))
    return data


def write_cam(file, cam):
    f = open(file, "w")
    f.write('extrinsic\n')
    for i in range(0, 4):
        for j in range(0, 4):
            f.write(str(cam[0][i][j]) + ' ')
        f.write('\n')
    f.write('\n')
    f.write('intrinsic\n')
    for i in range(0, 3):
        for j in range(0, 3):
            f.write(str(cam[1][i][j]) + ' ')
        f.write('\n')
    f.write('\n' + str(cam[1][3][0]) + ' ' + str(cam[1][3][1]) + ' ' + str(cam[1][3][2]) + ' ' + str(cam[1][3][3]) + '\n')
    f.close()


def generate_pointcloud(img, depth, ply_filename, intrinsic):
    height, width = depth.shape[:2]
    x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    valid_points = depth > 0
    x, y, depth_valid = x[valid_points], y[valid_points], depth[valid_points]
    color = img[valid_points]
    xyz_ref = np.matmul(np.linalg.inv(intrinsic),
                        np.vstack((x, y, np.ones_like(x))) * depth_valid)
    vertexs = xyz_ref.transpose((1, 0))
    vertex_colors = (color * 255).astype(np.uint8)
    vertexs = np.array([tuple(v) for v in vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]
    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(ply_filename)


def tocuda(sample, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(sample, torch.Tensor):
        return sample.to(device)
    elif isinstance(sample, dict):
        return {key: tocuda(val, device) for key, val in sample.items()}
    elif isinstance(sample, list):
        return [tocuda(val, device) for val in sample]
    elif isinstance(sample, tuple):
        return tuple(tocuda(val, device) for val in sample)
    else:
        return sample


def tensor2numpy(inputs):
    if isinstance(inputs, torch.Tensor):
        return inputs.cpu().numpy()
    elif isinstance(inputs, dict):
        return {k: tensor2numpy(v) for k, v in inputs.items()}
    elif isinstance(inputs, list):
        return [tensor2numpy(v) for v in inputs]
    else:
        return inputs


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
    max_h: int = 512,
    max_w: int = 640,
    batch_size: int = 1,
    ndepths: str = "48,32,8",
    depth_inter_r: str = "4,2,1",
    cr_base_chs: str = "8,8,8",
    share_cr: bool = False,
    grad_method: str = "detach",
    num_worker: int = 0,
    save_freq: int = 20,
    use_gpu: bool = True,
) -> None:
    """
    运行 CasMVSNet 深度估计。

    参数：
      testpath: 包含场景目录的数据路径（如 casmvsnet_input/）
      outdir: 输出目录
      checkpoint: 模型权重文件路径
      log: 日志回调函数
      其他参数同 test.py
    """
    if not Path(checkpoint).is_file():
        raise DepthEstimationError(f"Checkpoint not found: {checkpoint}")

    # 确定场景列表
    testlist = [e for e in os.listdir(testpath) if os.path.isdir(os.path.join(testpath, e))]
    if not testlist:
        raise DepthEstimationError(f"No scenes found in {testpath}")

    num_stage = len([int(nd) for nd in ndepths.split(",") if nd])

    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    device = torch.device("cpu" if not use_gpu or not torch.cuda.is_available() else "cuda")

    log(f"CasMVSNet: {len(testlist)} scene(s) in {testpath}")
    log(f"Checkpoint: {checkpoint}")
    log(f"Output: {outdir}")

    for scene in testlist:
        log(f"Processing scene: {scene}")
        _save_scene_depth(
            scene=scene,
            testpath=testpath,
            outdir=outdir,
            checkpoint=checkpoint,
            dataset=dataset,
            num_view=num_view,
            numdepth=numdepth,
            interval_scale=interval_scale,
            max_h=max_h,
            max_w=max_w,
            batch_size=batch_size,
            ndepths=ndepths,
            depth_inter_r=depth_inter_r,
            cr_base_chs=cr_base_chs,
            share_cr=share_cr,
            grad_method=grad_method,
            num_stage=num_stage,
            num_worker=num_worker,
            save_freq=save_freq,
            device=device,
            log=log,
        )

    log("Depth estimation completed.")


def _save_scene_depth(
    scene: str,
    testpath: str,
    outdir: str,
    checkpoint: str,
    dataset: str,
    num_view: int,
    numdepth: int,
    interval_scale: float,
    max_h: int,
    max_w: int,
    batch_size: int,
    ndepths: str,
    depth_inter_r: str,
    cr_base_chs: str,
    share_cr: bool,
    grad_method: str,
    num_stage: int,
    num_worker: int,
    save_freq: int,
    device: torch.device,
    log: Callable[[str], None],
) -> None:
    """对单个场景运行深度估计"""
    import gc

    # 清理 GPU 缓存，释放之前操作占用的显存
    if device.type == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
        free_mem = torch.cuda.mem_get_info()[0] / 1024**3
        log(f"GPU 可用显存: {free_mem:.1f} GB")

    find_dataset_def, read_pfm, save_pfm, CascadeMVSNet, gipuma_filter = _get_casmvsnet_deps()

    MVSDataset = find_dataset_def(dataset)
    test_dataset = MVSDataset(
        testpath, [scene], "test", num_view,
        ndepths=numdepth,
        interval_scale=interval_scale,
        num_stage=3,
        max_h=max_h, max_w=max_w, fix_res=False,
    )
    TestImgLoader = DataLoader(
        test_dataset, batch_size, shuffle=False, num_workers=num_worker, drop_last=False
    )

    model = CascadeMVSNet(
        refine=False,
        ndepths=[int(nd) for nd in ndepths.split(",") if nd],
        depth_interals_ratio=[float(d_i) for d_i in depth_inter_r.split(",") if d_i],
        share_cr=share_cr,
        cr_base_chs=[int(ch) for ch in cr_base_chs.split(",") if ch],
        grad_method=grad_method,
    )

    log(f"Loading model: {checkpoint}")
    state_dict = torch.load(checkpoint, map_location=torch.device("cpu"))
    model_state = _filter_casmvsnet_model_state(state_dict['model'], log)
    try:
        model.load_state_dict(model_state, strict=True)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            raise DepthEstimationError(
                f"GPU 显存不足，无法加载模型。请降低分辨率 (max_h/max_w) 或使用 CPU (use_gpu=False)。"
                f"详情: {e}"
            ) from e
        raise
    del state_dict, model_state  # 释放 checkpoint 内存
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.eval()

    # Windows 下强制 num_worker=0，避免 DataLoader worker 崩溃
    import platform
    if platform.system() == "Windows":
        num_worker = 0

    try:
        with torch.no_grad():
            for batch_idx, sample in enumerate(TestImgLoader):
                sample_cuda = tocuda(sample, device)
                start_time = time.time()
                outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
                end_time = time.time()
                outputs = tensor2numpy(outputs)
                del sample_cuda
                if device.type == "cuda":
                    torch.cuda.empty_cache()  # 每批次清理 GPU 缓存
                filenames = sample["filename"]
                cams = sample["proj_matrices"]["stage{}".format(num_stage)].numpy()
            imgs = sample["imgs"].numpy()
            log(f'Iter {batch_idx+1}/{len(TestImgLoader)}, Time:{end_time - start_time:.2f}s')

            for filename, cam, img, depth_est, photometric_confidence in zip(
                filenames, cams, imgs, outputs["depth"], outputs["photometric_confidence"]
            ):
                img = img[0]
                cam = cam[0]
                depth_filename = os.path.join(outdir, filename.format('depth_est', '.pfm'))
                confidence_filename = os.path.join(outdir, filename.format('confidence', '.pfm'))
                cam_filename = os.path.join(outdir, filename.format('cams', '_cam.txt'))
                img_filename = os.path.join(outdir, filename.format('images', '.jpg'))
                ply_filename = os.path.join(outdir, filename.format('ply_local', '.ply'))

                for d in [depth_filename, confidence_filename, cam_filename, img_filename, ply_filename]:
                    os.makedirs(os.path.dirname(d), exist_ok=True)

                save_pfm(depth_filename, depth_est)
                save_pfm(confidence_filename, photometric_confidence)
                write_cam(cam_filename, cam)
                img = np.clip(np.transpose(img, (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(img_filename, img_bgr)

                if num_stage == 1:
                    downsample_img = cv2.resize(img, (int(img.shape[1] * 0.25), int(img.shape[0] * 0.25)))
                elif num_stage == 2:
                    downsample_img = cv2.resize(img, (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5)))
                else:
                    downsample_img = img

                if batch_idx % save_freq == 0:
                    generate_pointcloud(downsample_img, depth_est, ply_filename, cam[1, :3, :3])

            del sample, outputs, filenames, cams, imgs
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            if device.type == "cuda":
                torch.cuda.empty_cache()
            raise DepthEstimationError(
                f"GPU 显存不足。请降低分辨率 (max_h/max_w) 或使用 CPU (use_gpu=False)。详情: {e}"
            ) from e
        raise
    finally:
        # 释放模型显存
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        if device.type == "cuda":
            free_mem = torch.cuda.mem_get_info()[0] / 1024**3
            log(f"深度估计完成，GPU 剩余显存: {free_mem:.1f} GB")


# ---------------------------------------------------------------------------
# 点云融合核心函数
# ---------------------------------------------------------------------------

def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    xyz_ref = np.matmul(np.linalg.inv(intrinsics_ref),
                        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]))
    xyz_src = np.matmul(np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)),
                        np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)
    xyz_src = np.matmul(np.linalg.inv(intrinsics_src),
                        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
    xyz_reprojected = np.matmul(np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
                                np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)
    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(
        depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src
    )
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)
    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref
    mask = np.logical_and(dist < 1, relative_depth_diff < 0.01)
    depth_reprojected[~mask] = 0
    return mask, depth_reprojected, x2d_src, y2d_src


def filter_depth(pair_folder, scan_folder, out_folder, plyfilename, conf=0.9, thres_view=5, num_stage=3):
    """过滤深度图并生成点云"""
    _, read_pfm, _, _, _ = _get_casmvsnet_deps()

    pair_file = os.path.join(pair_folder, "pair.txt")
    vertexs = []
    vertex_colors = []

    pair_data = read_pair_file(pair_file)

    for ref_view, src_views in pair_data:
        ref_intrinsics, ref_extrinsics = read_camera_parameters(
            os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(ref_view)))
        ref_img = read_img(os.path.join(scan_folder, 'images/{:0>8}.jpg'.format(ref_view)))
        ref_depth_est = read_pfm(os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(ref_view)))[0]
        confidence = read_pfm(os.path.join(out_folder, 'confidence/{:0>8}.pfm'.format(ref_view)))[0]
        photo_mask = confidence > conf

        all_srcview_depth_ests = []
        all_srcview_x = []
        all_srcview_y = []
        all_srcview_geomask = []

        geo_mask_sum = 0
        for src_view in src_views:
            src_intrinsics, src_extrinsics = read_camera_parameters(
                os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(src_view)))
            src_depth_est = read_pfm(os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(src_view)))[0]

            geo_mask, depth_reprojected, x2d_src, y2d_src = check_geometric_consistency(
                ref_depth_est, ref_intrinsics, ref_extrinsics,
                src_depth_est, src_intrinsics, src_extrinsics
            )
            geo_mask_sum += geo_mask.astype(np.int32)
            all_srcview_depth_ests.append(depth_reprojected)
            all_srcview_x.append(x2d_src)
            all_srcview_y.append(y2d_src)
            all_srcview_geomask.append(geo_mask)

        depth_est_averaged = (sum(all_srcview_depth_ests) + ref_depth_est) / (geo_mask_sum + 1)
        geo_mask = geo_mask_sum >= thres_view
        final_mask = np.logical_and(photo_mask, geo_mask)

        valid_points = final_mask
        x, y = np.meshgrid(np.arange(0, ref_depth_est.shape[1]), np.arange(0, ref_depth_est.shape[0]))
        x, y, depth = x[valid_points], y[valid_points], depth_est_averaged[valid_points]

        if num_stage == 1:
            color = ref_img[1::4, 1::4, :][valid_points]
        elif num_stage == 2:
            color = ref_img[1::2, 1::2, :][valid_points]
        else:
            color = ref_img[valid_points]

        xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics),
                            np.vstack((x, y, np.ones_like(x))) * depth)
        xyz_world = np.matmul(np.linalg.inv(ref_extrinsics),
                              np.vstack((xyz_ref, np.ones_like(x))))[:3]
        vertexs.append(xyz_world.transpose((1, 0)))
        vertex_colors.append((color * 255).astype(np.uint8))

    if not vertexs:
        raise DepthEstimationError("No valid points after filtering")

    vertexs = np.concatenate(vertexs, axis=0)
    vertex_colors = np.concatenate(vertex_colors, axis=0)
    vertexs = np.array([tuple(v) for v in vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(plyfilename)
    return plyfilename


def run_point_fusion(
    testpath: str,
    outdir: str,
    conf: float = 0.9,
    thres_view: int = 5,
    num_stage: int = 3,
    log: Optional[Callable[[str], None]] = None,
) -> str:
    """
    运行点云融合：过滤深度图并生成最终点云。

    返回生成的 PLY 文件路径。
    """
    _, read_pfm, save_pfm, _, _ = _get_casmvsnet_deps()

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

        result_ply = filter_depth(pair_folder, scan_folder, out_folder, ply_path, conf, thres_view, num_stage)

        if log:
            log(f"Saved point cloud: {ply_path}")

    return result_ply or ""
