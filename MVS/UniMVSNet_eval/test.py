"""
UniMVSNet 推理脚本

与 CasMVSNet_eval/test.py 接口一致，支持：
  - 深度估计（保存 depth_est/*.pfm + confidence/*.pfm）
  - 点云融合（pcd filter）

用法：
  python test.py --testpath_single_scene <scene_dir> --outdir <output_dir> --loadckpt <checkpoint>
"""

import argparse
import os
import sys
import time
import gc
import platform
import signal
from functools import partial
from multiprocessing import Pool

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from PIL import Image
from plyfile import PlyData, PlyElement

from datasets import find_dataset_def
from datasets.data_io import read_pfm, save_pfm
from networks.mvsnet import MVSNet
from filter.pcd import pcd_filter
from filter.tank_test_config import tank_cfg

cudnn.benchmark = True

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def print_args(args):
    print("################################  args  ################################")
    for k, v in args.__dict__.items():
        print("{0: <10}\t{1: <30}\t{2: <20}".format(k, str(v), str(type(v))))
    print("########################################################################")


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


@make_recursive_func
def tensor2numpy(vars):
    if isinstance(vars, np.ndarray):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.detach().cpu().numpy().copy()
    else:
        raise NotImplementedError("invalid input type {}".format(type(vars)))


def tocuda(vars):
    if isinstance(vars, torch.Tensor):
        return vars.cuda()
    elif isinstance(vars, dict):
        return {key: tocuda(val) for key, val in vars.items()}
    elif isinstance(vars, list):
        return [tocuda(val) for val in vars]
    else:
        return vars


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


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='UniMVSNet: Predict depth, filter, and fuse')

parser.add_argument('--dataset', default='general_eval', help='select dataset')
parser.add_argument('--testpath', help='testing data dir for some scenes')
parser.add_argument('--testpath_single_scene', default='', help='testing data path for single scene')
parser.add_argument('--testlist', default='all', help='testing scene list')

parser.add_argument('--batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')
parser.add_argument('--ndepths', type=int, nargs='+', default=[48, 32, 8])
parser.add_argument('--interval_ratio', type=float, nargs='+', default=[4, 2, 1])
parser.add_argument('--fea_mode', type=str, default='fpn', choices=['fpn', 'unet'])
parser.add_argument('--agg_mode', type=str, default='variance', choices=['variance', 'adaptive'])
parser.add_argument('--depth_mode', type=str, default='regression', choices=['regression', 'classification', 'unification'])

parser.add_argument('--loadckpt', default='checkpoints/unimvsnet.ckpt', help='load a specific checkpoint')
parser.add_argument('--outdir', default='./output', help='output dir')
parser.add_argument('--display', action='store_true', help='display depth images and masks')

parser.add_argument('--interval_scale', type=float, default=1.06, help='the depth interval scale')
parser.add_argument('--num_view', type=int, default=5, help='num of view')
parser.add_argument('--max_h', type=int, default=864, help='testing max h')
parser.add_argument('--max_w', type=int, default=1152, help='testing max w')
parser.add_argument('--fix_res', action='store_true', help='scene all using same res')

parser.add_argument('--num_worker', type=int, default=0 if platform.system() == "Windows" else 4,
                    help='worker count for depth filtering and test dataloader')
parser.add_argument('--save_freq', type=int, default=20, help='save freq of local pcd')

parser.add_argument('--filter_method', type=str, default='pcd', choices=["gipuma", "pcd", "dypcd"], help="filter method")

# filter
parser.add_argument('--conf', type=float, nargs='+', default=[0.1, 0.15, 0.9], help='prob confidence for pcd/dypcd')
parser.add_argument('--thres_view', type=int, default=5, help='threshold of num view, only for pcd')

# dypcd
parser.add_argument('--dist_base', type=float, default=1/4)
parser.add_argument('--rel_diff_base', type=float, default=1/1300)

# gipuma
parser.add_argument('--fusibile_exe_path', type=str, default='../fusibile/fusibile')
parser.add_argument('--prob_threshold', type=float, default='0.3')
parser.add_argument('--disp_threshold', type=float, default='0.25')
parser.add_argument('--num_consistent', type=float, default='3')

# device
parser.add_argument('--no_cuda', action='store_true')
parser.add_argument('--local_rank', type=int, default=0)

args = parser.parse_args()
print("argv:", sys.argv[1:])
print_args(args)

if args.testpath_single_scene:
    args.testpath = os.path.dirname(args.testpath_single_scene)

num_stage = len(args.ndepths)
args.distributed = False


# ---------------------------------------------------------------------------
# 深度估计
# ---------------------------------------------------------------------------

def save_depth(testlist):
    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")

    # 构建模型
    model = MVSNet(
        ndepths=args.ndepths,
        depth_interval_ratio=args.interval_ratio,
        fea_mode=args.fea_mode,
        agg_mode=args.agg_mode,
        depth_mode=args.depth_mode,
    ).to(device)

    # 加载 checkpoint
    print(f"Loading model {args.loadckpt}")
    state_dict = torch.load(args.loadckpt, map_location="cpu")
    if "model" in state_dict:
        model.load_state_dict(state_dict["model"], strict=True)
    else:
        model.load_state_dict(state_dict, strict=True)
    del state_dict
    model.eval()

    with torch.no_grad():
        for scene in testlist:
            if scene in tank_cfg.scenes:
                scene_cfg = getattr(tank_cfg, scene)
                args.max_h = scene_cfg.max_h
                args.max_w = scene_cfg.max_w

            # 获取 dataset
            MVSDataset = find_dataset_def(args.dataset)
            test_dataset = MVSDataset(
                args.testpath, [scene], "test", args.num_view, args.numdepth, args.interval_scale,
                max_h=args.max_h, max_w=args.max_w, fix_res=args.fix_res,
            )
            TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=args.num_worker, drop_last=False)

            for batch_idx, sample in enumerate(TestImgLoader):
                sample_cuda = tocuda(sample)
                start_time = time.time()
                outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
                end_time = time.time()
                outputs = tensor2numpy(outputs)
                del sample_cuda
                filenames = sample["filename"]
                cams = sample["proj_matrices"]["stage{}".format(num_stage)].numpy()
                imgs = sample["imgs"].numpy()
                print('Iter {}/{}, Time:{:.2f}s Res:{}'.format(batch_idx, len(TestImgLoader), end_time - start_time, imgs[0].shape))

                for filename, cam, img, depth_est, depth2, depth1, photometric_confidence, pc2, pc1 \
                        in zip(filenames, cams, imgs, outputs["depth"],
                               outputs.get("stage2", {}).get("depth", [None]),
                               outputs.get("stage1", {}).get("depth", [None]),
                               outputs["photometric_confidence"],
                               outputs.get("stage2", {}).get("photometric_confidence", [None]),
                               outputs.get("stage1", {}).get("photometric_confidence", [None])):

                    img = img[0]
                    cam = cam[0]
                    depth_filename = os.path.join(args.outdir, filename.format('depth_est', '.pfm'))
                    confidence_filename = os.path.join(args.outdir, filename.format('confidence', '.pfm'))
                    cam_filename = os.path.join(args.outdir, filename.format('cams', '_cam.txt'))
                    img_filename = os.path.join(args.outdir, filename.format('images', '.jpg'))

                    for d in [depth_filename, confidence_filename, cam_filename, img_filename]:
                        os.makedirs(os.path.dirname(d), exist_ok=True)

                    save_pfm(depth_filename, depth_est)
                    save_pfm(confidence_filename, photometric_confidence)
                    write_cam(cam_filename, cam)
                    img = np.clip(np.transpose(img, (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(img_filename, img_bgr)

                    # 保存中间 stage 的深度图（如果有）
                    if depth2 is not None:
                        save_pfm(os.path.join(args.outdir, filename.format('depth_est', '_stage2.pfm')), depth2)
                        h, w = photometric_confidence.shape
                        pc2_resized = cv2.resize(pc2, (w, h), interpolation=cv2.INTER_NEAREST)
                        save_pfm(os.path.join(args.outdir, filename.format('confidence', '_stage2.pfm')), pc2_resized)
                    if depth1 is not None:
                        save_pfm(os.path.join(args.outdir, filename.format('depth_est', '_stage1.pfm')), depth1)
                        h, w = photometric_confidence.shape
                        pc1_resized = cv2.resize(pc1, (w, h), interpolation=cv2.INTER_NEAREST)
                        save_pfm(os.path.join(args.outdir, filename.format('confidence', '_stage1.pfm')), pc1_resized)

                del sample, outputs, filenames, cams, imgs
                torch.cuda.empty_cache()
                gc.collect()

    del model
    torch.cuda.empty_cache()
    gc.collect()


# ---------------------------------------------------------------------------
# 点云融合
# ---------------------------------------------------------------------------

def filter_depth(args, pair_folder, scan_folder, out_folder, plyfilename):
    from filter.pcd import filter_depth as _filter_depth
    _filter_depth(args, pair_folder, scan_folder, out_folder, plyfilename)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # 确定场景列表
    if args.testlist != "all":
        with open(args.testlist) as f:
            testlist = [line.rstrip() for line in f.readlines()]
    else:
        testlist = [e for e in os.listdir(args.testpath) if os.path.isdir(os.path.join(args.testpath, e))] \
            if not args.testpath_single_scene else [os.path.basename(args.testpath_single_scene)]

    print(f"Scenes: {testlist}")

    # step1. 深度估计
    save_depth(testlist)

    # step2. 点云融合
    if args.filter_method == "pcd":
        from functools import partial as _partial
        from filter.pcd import pcd_filter_worker
        for scan in testlist:
            pcd_filter_worker(args, scan)
    elif args.filter_method == "dypcd":
        from filter.pcd import dypcd_filter
        dypcd_filter(args, testlist, args.num_worker)
    else:
        from filter.pcd import gipuma_filter
        gipuma_filter(testlist, args.outdir, args.prob_threshold, args.disp_threshold, args.num_consistent,
                      args.fusibile_exe_path)

    print("Done.")
