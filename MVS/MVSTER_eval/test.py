"""
MVSTER 推理脚本

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
from models.MVS4Net import MVS4net
from filter.pcd import filter_depth

cudnn.benchmark = True


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


def tocuda(vars):
    if isinstance(vars, torch.Tensor):
        return vars.cuda()
    elif isinstance(vars, dict):
        return {key: tocuda(val) for key, val in vars.items()}
    elif isinstance(vars, list):
        return [tocuda(val) for val in vars]
    else:
        return vars


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='MVSTER: Predict depth, filter, and fuse')

parser.add_argument('--dataset', default='general_eval', help='select dataset')
parser.add_argument('--testpath', help='testing data dir for some scenes')
parser.add_argument('--testpath_single_scene', default='', help='testing data path for single scene')
parser.add_argument('--testlist', default='all', help='testing scene list')

parser.add_argument('--batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--loadckpt', default='checkpoints/mvster.ckpt', help='load a specific checkpoint')
parser.add_argument('--outdir', default='./output', help='output dir')

parser.add_argument('--ndepths', type=str, default="8,8,4,4", help='ndepths')
parser.add_argument('--depth_inter_r', type=str, default="0.5,0.5,0.5,1", help='depth_intervals_ratio')

parser.add_argument('--interval_scale', type=float, default=1.06, help='the depth interval scale')
parser.add_argument('--num_view', type=int, default=5, help='num of view')
parser.add_argument('--max_h', type=int, default=864, help='testing max h')
parser.add_argument('--max_w', type=int, default=1152, help='testing max w')
parser.add_argument('--fix_res', action='store_true', help='scene all using same res')

parser.add_argument('--num_worker', type=int, default=0 if platform.system() == "Windows" else 4,
                    help='worker count for depth filtering')
parser.add_argument('--save_freq', type=int, default=20, help='save freq of local pcd')

parser.add_argument('--filter_method', type=str, default='normal', choices=["gipuma", "normal"], help="filter method")
parser.add_argument('--conf', type=float, default=0.9, help='prob confidence')
parser.add_argument('--thres_view', type=int, default=5, help='threshold of num view')

# MVSTER specific
parser.add_argument("--fpn_base_channel", type=int, default=8)
parser.add_argument("--reg_channel", type=int, default=8)
parser.add_argument('--reg_mode', type=str, default="reg2d")
parser.add_argument('--group_cor', action='store_true', help='group correlation')
parser.add_argument('--group_cor_dim', type=str, default="8,8,4,4", help='group correlation dim')
parser.add_argument('--inverse_depth', action='store_true', help='inverse depth')
parser.add_argument('--agg_type', type=str, default="ConvBnReLU3D", help='cost regularization type')
parser.add_argument('--dcn', action='store_true', help='dcn')
parser.add_argument('--arch_mode', type=str, default="fpn")
parser.add_argument('--mono', action='store_true', help='mono depth prediction')
parser.add_argument('--ASFF', action='store_true')
parser.add_argument('--attn_temp', type=float, default=2)

# device
parser.add_argument('--no_cuda', action='store_true')

args = parser.parse_args()
print("argv:", sys.argv[1:])
print_args(args)

if args.testpath_single_scene:
    args.testpath = os.path.dirname(args.testpath_single_scene)

num_stage = len([int(nd) for nd in args.ndepths.split(",") if nd])


# ---------------------------------------------------------------------------
# 深度估计
# ---------------------------------------------------------------------------

def save_scene_depth(testlist):
    MVSDataset = find_dataset_def(args.dataset)
    test_dataset = MVSDataset(args.testpath, testlist, "test", args.num_view, args.interval_scale,
                              max_h=args.max_h, max_w=args.max_w, fix_res=args.fix_res)
    TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=args.num_worker, drop_last=False)

    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    model = MVS4net(
        arch_mode=args.arch_mode,
        reg_net=args.reg_mode,
        num_stage=4,
        fpn_base_channel=args.fpn_base_channel,
        reg_channel=args.reg_channel,
        stage_splits=[int(n) for n in args.ndepths.split(",")],
        depth_interals_ratio=[float(ir) for ir in args.depth_inter_r.split(",")],
        group_cor=args.group_cor,
        group_cor_dim=[int(n) for n in args.group_cor_dim.split(",")],
        inverse_depth=args.inverse_depth,
        agg_type=args.agg_type,
        dcn=args.dcn,
        mono=args.mono,
        asff=args.ASFF,
        attn_temp=args.attn_temp,
    ).to(device)

    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict['model'], strict=True)
    del state_dict
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.cuda()
    model.eval()

    with torch.no_grad():
        for batch_idx, sample in enumerate(TestImgLoader):
            sample_cuda = tocuda(sample)
            start_time = time.time()
            outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"], sample["filename"])
            end_time = time.time()
            outputs = tensor2numpy(outputs)
            del sample_cuda
            torch.cuda.empty_cache()

            filenames = sample["filename"]
            cams = sample["proj_matrices"]["stage{}".format(num_stage)].numpy()
            imgs = sample["imgs"]
            print('Iter {}/{}, Time:{:.2f}s Res:{}'.format(batch_idx, len(TestImgLoader), end_time - start_time, imgs[0].shape))

            for filename, cam, img, depth_est, photometric_confidence in zip(
                    filenames, cams, imgs, outputs["depth"], outputs["photometric_confidence"]):
                img = img[0].numpy()
                cam = cam[0]
                depth_filename = os.path.join(args.outdir, filename.format('depth_est', '.pfm'))
                confidence_filename = os.path.join(args.outdir, filename.format('confidence', '.pfm'))
                cam_filename = os.path.join(args.outdir, filename.format('cams', '_cam.txt'))
                img_filename = os.path.join(args.outdir, filename.format('images', '.jpg'))

                for d in [depth_filename, confidence_filename, cam_filename, img_filename]:
                    os.makedirs(os.path.dirname(d), exist_ok=True)

                save_pfm(depth_filename, depth_est)

                # 保存多阶段置信度（MVSTER 特有）
                confidence_list = [outputs['stage{}'.format(i)]['photometric_confidence'].squeeze(0) for i in range(1, 5)]
                photometric_confidence_final = confidence_list[-1]
                save_pfm(confidence_filename, photometric_confidence_final)

                # 保存各阶段置信度（用于深度融合）
                for stage_idx in range(4):
                    stage_conf = confidence_list[stage_idx]
                    stage_conf_filename = os.path.join(args.outdir, filename.format('confidence', '_stage{}.pfm'.format(stage_idx + 1)))
                    save_pfm(stage_conf_filename, stage_conf)

                write_cam(cam_filename, cam)
                img = np.clip(np.transpose(img, (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(img_filename, img_bgr)

            del sample, outputs, filenames, cams, imgs
            torch.cuda.empty_cache()
            gc.collect()

    del model
    torch.cuda.empty_cache()
    gc.collect()


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    if args.testlist != "all":
        with open(args.testlist) as f:
            testlist = [line.rstrip() for line in f.readlines()]
    else:
        testlist = [e for e in os.listdir(args.testpath) if os.path.isdir(os.path.join(args.testpath, e))] \
            if not args.testpath_single_scene else [os.path.basename(args.testpath_single_scene)]

    print(f"Scenes: {testlist}")

    # step1. 深度估计
    save_scene_depth(testlist)

    # step2. 点云融合
    for scan in testlist:
        pair_folder = os.path.join(args.testpath, scan)
        scan_folder = os.path.join(args.outdir, scan)
        out_folder = os.path.join(args.outdir, scan)
        ply_path = os.path.join(args.outdir, f"{scan}.ply")
        filter_depth(pair_folder, scan_folder, out_folder, ply_path, args.conf, args.thres_view)

    print("Done.")
