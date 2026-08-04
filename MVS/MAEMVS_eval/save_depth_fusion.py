"""Fuse saved MVS depth maps into a colored point cloud.

The filtering strategy follows common MVSNet/PatchmatchNet practice:
photometric confidence thresholding plus geometric consistency from source
views, then back-project accepted reference pixels and write a PLY file.
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from datasets.data_io import read_pfm


def build_parser():
    parser = argparse.ArgumentParser(description="Fuse saved depth maps to a PLY point cloud")
    parser.add_argument("--depth_dir", default="./result/depths")
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--out_ply", default="./result/fused.ply")
    parser.add_argument("--conf_thresh", type=float, default=0.8)
    parser.add_argument("--geo_pixel_thresh", type=float, default=1.0)
    parser.add_argument("--geo_depth_thresh", type=float, default=0.01)
    parser.add_argument("--min_consistent", type=int, default=2)
    parser.add_argument("--max_src_views", type=int, default=4)
    parser.add_argument("--max_points_per_view", type=int, default=0)
    return parser


def save_camera_npz(filename, intrinsics, extrinsics):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.savez_compressed(
        filename,
        intrinsics=np.asarray(intrinsics, dtype=np.float32),
        extrinsics=np.asarray(extrinsics, dtype=np.float32),
    )


def read_camera_npz(filename, view_idx=0):
    data = np.load(filename)
    intrinsics = data["intrinsics"]
    extrinsics = data["extrinsics"]
    if intrinsics.ndim == 3:
        intrinsics = intrinsics[view_idx]
    if extrinsics.ndim == 3:
        extrinsics = extrinsics[view_idx]
    return intrinsics.astype(np.float32), extrinsics.astype(np.float32)


def read_image(filename):
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(filename)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_depth(filename):
    return np.asarray(read_pfm(filename)[0], dtype=np.float32)


def load_metadata(depth_dir, metadata_file=None):
    metadata_file = metadata_file or os.path.join(depth_dir, "metadata.jsonl")
    records = []
    with open(metadata_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                for key in ["depth", "confidence", "camera", "image"]:
                    item[key] = os.path.join(depth_dir, item[key])
                records.append(item)
    return records


def build_default_src_names(records, ref_idx, max_src_views):
    indices = [idx for idx in range(len(records)) if idx != ref_idx]
    indices.sort(key=lambda idx: abs(idx - ref_idx))
    return [records[idx]["name"] for idx in indices[:max_src_views]]


def pixel_grid(height, width):
    xs, ys = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    return xs, ys


def backproject(depth, intrinsics, extrinsics):
    height, width = depth.shape
    xs, ys = pixel_grid(height, width)
    pixels = np.stack([xs.reshape(-1), ys.reshape(-1), np.ones(height * width, dtype=np.float32)], axis=0)
    cam_points = np.linalg.inv(intrinsics).dot(pixels) * depth.reshape(1, -1)
    world_points = np.linalg.inv(extrinsics).dot(
        np.vstack([cam_points, np.ones((1, cam_points.shape[1]), dtype=np.float32)])
    )[:3]
    return world_points, xs, ys


def project(world_points, intrinsics, extrinsics):
    cam_points = extrinsics.dot(
        np.vstack([world_points, np.ones((1, world_points.shape[1]), dtype=np.float32)])
    )[:3]
    pixels = intrinsics.dot(cam_points)
    zs = pixels[2:3].copy()
    pixels[:2] /= np.maximum(zs, 1e-8)
    return pixels[0], pixels[1], cam_points[2]


def sample_depth(depth, xs, ys):
    return cv2.remap(
        depth.astype(np.float32),
        xs.astype(np.float32),
        ys.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref,
                         depth_src, intrinsics_src, extrinsics_src):
    height, width = depth_ref.shape
    world_ref, _, _ = backproject(depth_ref, intrinsics_ref, extrinsics_ref)
    x_src, y_src, _ = project(world_ref, intrinsics_src, extrinsics_src)
    sampled_depth_src = sample_depth(depth_src, x_src.reshape(height, width), y_src.reshape(height, width))

    world_src, _, _ = backproject(sampled_depth_src, intrinsics_src, extrinsics_src)
    x_reprojected, y_reprojected, depth_reprojected = project(world_src, intrinsics_ref, extrinsics_ref)
    return (
        depth_reprojected.reshape(height, width).astype(np.float32),
        x_reprojected.reshape(height, width).astype(np.float32),
        y_reprojected.reshape(height, width).astype(np.float32),
    )


def geometric_consistency_mask(depth_ref, intrinsics_ref, extrinsics_ref,
                              depth_src, intrinsics_src, extrinsics_src,
                              pixel_thresh, depth_thresh):
    height, width = depth_ref.shape
    xs_ref, ys_ref = pixel_grid(height, width)
    depth_reprojected, x_reprojected, y_reprojected = reproject_with_depth(
        depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src
    )
    pixel_error = np.sqrt((x_reprojected - xs_ref) ** 2 + (y_reprojected - ys_ref) ** 2)
    relative_depth_error = np.abs(depth_reprojected - depth_ref) / np.maximum(depth_ref, 1e-8)
    mask = (
        (pixel_error < pixel_thresh)
        & (relative_depth_error < depth_thresh)
        & (depth_ref > 0)
        & (depth_reprojected > 0)
    )
    return mask, depth_reprojected


def write_ply_ascii(filename, vertices, colors):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    with open(filename, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex {}\n".format(vertices.shape[0]))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for xyz, rgb in zip(vertices, colors):
            f.write("{:.6f} {:.6f} {:.6f} {} {} {}\n".format(
                float(xyz[0]), float(xyz[1]), float(xyz[2]),
                int(rgb[0]), int(rgb[1]), int(rgb[2])
            ))


def fuse_depth_maps(args):
    records = load_metadata(args.depth_dir, args.metadata)
    record_by_name = {record["name"]: record for record in records}
    vertices_all = []
    colors_all = []

    for ref_idx, ref in enumerate(records):
        depth_ref = read_depth(ref["depth"])
        conf_ref = read_depth(ref["confidence"])
        image_ref = read_image(ref["image"])
        intr_ref, ext_ref = read_camera_npz(ref["camera"], view_idx=0)

        if image_ref.shape[:2] != depth_ref.shape:
            image_ref = cv2.resize(image_ref, (depth_ref.shape[1], depth_ref.shape[0]), interpolation=cv2.INTER_LINEAR)

        photo_mask = conf_ref >= args.conf_thresh
        src_names = ref.get("src_names") or build_default_src_names(records, ref_idx, args.max_src_views)
        src_names = src_names[:args.max_src_views]

        geo_sum = np.zeros_like(photo_mask, dtype=np.int32)
        depth_sum = depth_ref.copy()

        for src_name in src_names:
            src = record_by_name.get(src_name)
            if src is None:
                continue
            depth_src = read_depth(src["depth"])
            intr_src, ext_src = read_camera_npz(src["camera"], view_idx=0)
            geo_mask, depth_reprojected = geometric_consistency_mask(
                depth_ref, intr_ref, ext_ref,
                depth_src, intr_src, ext_src,
                args.geo_pixel_thresh, args.geo_depth_thresh,
            )
            geo_sum += geo_mask.astype(np.int32)
            depth_sum += depth_reprojected * geo_mask.astype(np.float32)

        final_mask = photo_mask & (geo_sum >= args.min_consistent)
        refined_depth = depth_sum / np.maximum(geo_sum + 1, 1)
        refined_depth = np.where(final_mask, refined_depth, 0.0).astype(np.float32)

        world_points, _, _ = backproject(refined_depth, intr_ref, ext_ref)
        mask_flat = final_mask.reshape(-1)
        points = world_points.T[mask_flat]
        colors = image_ref.reshape(-1, 3)[mask_flat]

        if args.max_points_per_view > 0 and points.shape[0] > args.max_points_per_view:
            selected = np.linspace(0, points.shape[0] - 1, args.max_points_per_view).astype(np.int64)
            points = points[selected]
            colors = colors[selected]

        if points.size:
            vertices_all.append(points.astype(np.float32))
            colors_all.append(colors.astype(np.uint8))
        print("{}: kept {} points".format(ref["name"], points.shape[0]))

    vertices = np.concatenate(vertices_all, axis=0) if vertices_all else np.zeros((0, 3), dtype=np.float32)
    colors = np.concatenate(colors_all, axis=0) if colors_all else np.zeros((0, 3), dtype=np.uint8)
    write_ply_ascii(args.out_ply, vertices, colors)
    print("Saved {} points to {}".format(vertices.shape[0], args.out_ply))


def main():
    args = build_parser().parse_args()
    fuse_depth_maps(args)


if __name__ == "__main__":
    main()
