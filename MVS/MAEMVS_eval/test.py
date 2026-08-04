"""Run depth-map inference and save results for point-cloud fusion.

This follows the common MVSNet/PatchmatchNet split: one script estimates and
saves per-view depth maps, while a second script filters/fuses them to PLY.
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from datasets import find_dataset_def
from datasets.data_io import save_pfm
from models import MultiTaskMVSNet
from utils import tocuda, tensor2numpy


def build_parser():
    parser = argparse.ArgumentParser(description="MultiTask MVSNet depth-map testing")
    parser.add_argument("--dataset", default="general_eval", choices=["general_eval", "aerial_mvs", "dtu_yao"])
    parser.add_argument("--testpath", required=True)
    parser.add_argument("--testpath_single_scene", default=None)
    parser.add_argument("--testlist", default=None)
    parser.add_argument("--loadckpt", required=True)
    parser.add_argument("--outdir", default="./result/depths")
    parser.add_argument("--nviews", "--num_view", dest="nviews", type=int, default=5)
    parser.add_argument("--numdepth", type=int, default=192)
    parser.add_argument("--interval_scale", type=float, default=1.06)
    parser.add_argument("--max_h", type=int, default=512)
    parser.add_argument("--max_w", type=int, default=640)
    parser.add_argument("--ndepths", type=str, default="48,32,8")
    parser.add_argument("--depth_inter_r", type=str, default="4,2,1")
    parser.add_argument("--cr_base_chs", type=str, default="8,8,8")
    parser.add_argument("--share_cr", action="store_true")
    parser.add_argument("--grad_method", default="detach", choices=["detach", "undetach"])
    parser.add_argument("--mask_ratio", type=float, default=0.1)
    parser.add_argument("--use_mae", action="store_true", default=False)
    parser.add_argument("--use_damw", action="store_true", default=False)
    parser.add_argument("--use_sparse_conv", action="store_true", default=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", "--num_worker", dest="num_workers", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.8)
    parser.add_argument("--thres_view", type=int, default=5)
    parser.add_argument("--no_gpu", dest="use_gpu", action="store_false")
    parser.set_defaults(use_gpu=True)
    return parser


def ensure_dirs(outdir):
    for name in ["depth_est", "confidence", "cams", "images"]:
        os.makedirs(os.path.join(outdir, name), exist_ok=True)


def build_dataset(args):
    dataset_cls = find_dataset_def(args.dataset)
    if args.testpath_single_scene:
        scene_path = os.path.abspath(args.testpath_single_scene)
        testpath = os.path.dirname(scene_path)
        testlist = [os.path.basename(scene_path)]
    else:
        testpath = args.testpath
        if args.testlist and args.testlist != "all":
            with open(args.testlist, "r", encoding="utf-8") as f:
                testlist = [line.strip() for line in f if line.strip()]
        else:
            testlist = [name for name in sorted(os.listdir(testpath))
                        if os.path.isdir(os.path.join(testpath, name))]

    if args.dataset == "aerial_mvs":
        return dataset_cls(
            testpath,
            mode="test",
            nviews=args.nviews,
            img_size=[args.max_h, args.max_w],
            ndepths=args.numdepth,
            interval_scale=args.interval_scale,
        )
    return dataset_cls(
        testpath,
        testlist,
        mode="test",
        nviews=args.nviews,
        ndepths=args.numdepth,
        interval_scale=args.interval_scale,
        max_h=args.max_h,
        max_w=args.max_w,
        fix_res=False,
    )


def build_model(args):
    ndepths = [int(v) for v in args.ndepths.split(",") if v]
    depth_interals_ratio = [float(v) for v in args.depth_inter_r.split(",") if v]
    cr_base_chs = [int(v) for v in args.cr_base_chs.split(",") if v]
    return MultiTaskMVSNet(
        refine=False,
        ndepths=ndepths,
        depth_interals_ratio=depth_interals_ratio,
        share_cr=args.share_cr,
        grad_method=args.grad_method,
        arch_mode="fpn",
        cr_base_chs=cr_base_chs,
        use_mae=args.use_mae,
        mask_ratio=args.mask_ratio,
        use_damw=args.use_damw,
        use_sparse_conv=args.use_sparse_conv,
    )


def load_checkpoint(model, filename):
    checkpoint = torch.load(filename, map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    cleaned = {}
    for key, value in state_dict.items():
        cleaned[key[7:] if key.startswith("module.") else key] = value
    model.load_state_dict(cleaned, strict=True)


def to_numpy_image(img_chw):
    img = np.transpose(img_chw, (1, 2, 0))
    img = np.asarray(img, dtype=np.float32)
    if img.max() <= 2.0 and img.min() >= -1.0:
        img = np.clip(img, 0.0, 1.0) * 255.0
    else:
        img = img - img.min()
        img = img / max(float(img.max()), 1e-8) * 255.0
    return img.astype(np.uint8)


def sample_base_name(sample, batch_idx):
    if "out_name" in sample:
        name = sample["out_name"][0] if isinstance(sample["out_name"], (list, tuple)) else sample["out_name"]
        view = sample.get("out_view", ["view"])
        view = view[0] if isinstance(view, (list, tuple)) else view
        return "{}_{}".format(view, name)
    return "{:08d}".format(batch_idx)


def save_camera_npz(filename, proj_matrices):
    proj = np.asarray(proj_matrices, dtype=np.float32)
    np.savez_compressed(
        filename,
        intrinsics=proj[:, 1, :3, :3],
        extrinsics=proj[:, 0, :4, :4],
    )


def to_device(sample, device):
    if isinstance(sample, torch.Tensor):
        return sample.to(device)
    if isinstance(sample, dict):
        return {key: to_device(value, device) for key, value in sample.items()}
    if isinstance(sample, list):
        return [to_device(value, device) for value in sample]
    if isinstance(sample, tuple):
        return tuple(to_device(value, device) for value in sample)
    return sample


@torch.no_grad()
def run_inference(args):
    ensure_dirs(args.outdir)
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    dataset = build_dataset(args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = build_model(args)
    load_checkpoint(model, args.loadckpt)
    model.to(device)
    model.eval()

    metadata_path = os.path.join(args.outdir, "metadata.jsonl")
    with open(metadata_path, "w", encoding="utf-8") as metadata:
        for batch_idx, sample in enumerate(loader):
            sample_cuda = to_device(sample, device)
            outputs = model(
                sample_cuda["imgs"],
                sample_cuda["proj_matrices"],
                sample_cuda["depth_values"],
                return_mae=False,
            )

            depth = tensor2numpy(outputs["depth"])
            confidence = tensor2numpy(outputs.get("photometric_confidence", torch.ones_like(outputs["depth"])))
            imgs = tensor2numpy(sample["imgs"])
            proj_stage3 = tensor2numpy(sample["proj_matrices"]["stage3"])

            for item_idx in range(depth.shape[0]):
                name = sample_base_name(sample, batch_idx * args.batch_size + item_idx)
                depth_file = os.path.join(args.outdir, "depth_est", name + ".pfm")
                conf_file = os.path.join(args.outdir, "confidence", name + ".pfm")
                cam_file = os.path.join(args.outdir, "cams", name + ".npz")
                image_file = os.path.join(args.outdir, "images", name + ".png")

                save_pfm(depth_file, depth[item_idx].astype(np.float32))
                save_pfm(conf_file, confidence[item_idx].astype(np.float32))
                save_camera_npz(cam_file, proj_stage3[item_idx])
                cv2.imwrite(image_file, cv2.cvtColor(to_numpy_image(imgs[item_idx, 0]), cv2.COLOR_RGB2BGR))

                src_names = []
                if args.batch_size == 1:
                    # The current datasets provide source images but not stable source output names.
                    # Fusion will use all other saved views by default when src_names is empty.
                    src_names = []
                metadata.write(json.dumps({
                    "name": name,
                    "depth": os.path.relpath(depth_file, args.outdir).replace("\\", "/"),
                    "confidence": os.path.relpath(conf_file, args.outdir).replace("\\", "/"),
                    "camera": os.path.relpath(cam_file, args.outdir).replace("\\", "/"),
                    "image": os.path.relpath(image_file, args.outdir).replace("\\", "/"),
                    "src_names": src_names,
                }) + "\n")

            print("Saved {}/{} batches".format(batch_idx + 1, len(loader)))

    print("Depth maps saved to {}".format(args.outdir))
    print("Metadata saved to {}".format(metadata_path))


def main():
    args = build_parser().parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
