"""
统一 MVS 数据集模块

所有 MVS 方法共享的数据集加载逻辑，以 CasMVSNet_eval 的 general_eval.py 为基准。
不同方法的差异通过参数控制，避免代码重复。

参数差异：
  - CasMVSNet / UniMVSNet: ndepths 参数, 3 stages, 缩放 1/2/4
  - MVSTER: 无 ndepths 参数(硬编码192), 4 stages, 缩放 0.5/1/2/4
  - UniMVSNet: 额外 inverse_depth 参数
"""

from torch.utils.data import Dataset
import numpy as np
import os
import cv2
from PIL import Image
from datasets.data_io import read_pfm

_s_h, _s_w = 0, 0


class MVSDataset(Dataset):
    """
    通用 MVS 数据集，兼容所有方法。

    参数：
      datapath:       数据根目录（含 scene 子目录）
      listfile:       场景名称列表
      mode:           必须为 "test"
      nviews:         每次取的视角数
      ndepths:        深度采样数（默认 192）
      interval_scale: 深度间隔缩放系数（默认 1.06）
      inverse_depth:  是否使用逆深度采样（UniMVSNet 专用，默认 False）
      num_stage:      模型阶段数（3 或 4，影响投影矩阵缩放，默认 3）
      fill_to_nviews: src_views 填充目标（True=nviews, False=nviews-1，默认 False）
      **kwargs:
        max_h, max_w: 最大分辨率
        fix_res:      是否固定分辨率
    """

    def __init__(self, datapath, listfile, mode, nviews,
                 ndepths=192, interval_scale=1.06,
                 inverse_depth=False, num_stage=3,
                 fill_to_nviews=False, **kwargs):
        super().__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale
        self.inverse_depth = inverse_depth
        self.num_stage = num_stage
        self.fill_to_nviews = fill_to_nviews
        self.max_h = kwargs["max_h"]
        self.max_w = kwargs["max_w"]
        self.fix_res = kwargs.get("fix_res", False)
        self.fix_wh = False

        assert self.mode == "test"
        self.metas = self.build_list()

    # -----------------------------------------------------------------
    # 构建配对列表
    # -----------------------------------------------------------------
    def build_list(self):
        metas = []
        interval_scale_dict = {}

        for scan in self.listfile:
            if isinstance(self.interval_scale, float):
                interval_scale_dict[scan] = self.interval_scale
            else:
                interval_scale_dict[scan] = self.interval_scale[scan]

            pair_file = os.path.join(self.datapath, "{}/pair.txt".format(scan))
            with open(pair_file) as f:
                num_viewpoint = int(f.readline())
                for _ in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    if len(src_views) > 0:
                        # 填充源视角
                        fill_target = self.nviews if self.fill_to_nviews else self.nviews - 1
                        if len(src_views) < fill_target:
                            src_views += [src_views[0]] * (fill_target - len(src_views))
                        metas.append((scan, ref_view, src_views, scan))

        self.interval_scale = interval_scale_dict
        return metas

    def __len__(self):
        return len(self.metas)

    # -----------------------------------------------------------------
    # 读取工具
    # -----------------------------------------------------------------
    def read_cam_file(self, filename, interval_scale):
        with open(filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        intrinsics[:2, :] /= 4.0

        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1])
        if len(lines[11].split()) >= 3:
            num_depth = lines[11].split()[2]
            depth_max = depth_min + int(float(num_depth)) * depth_interval
            depth_interval = (depth_max - depth_min) / self.ndepths

        depth_interval *= interval_scale
        return intrinsics, extrinsics, depth_min, depth_interval

    def read_img(self, filename):
        img = Image.open(filename)
        return np.array(img, dtype=np.float32) / 255.0

    def scale_mvs_input(self, img, intrinsics, max_w, max_h, base=32):
        h, w = img.shape[:2]
        if h > max_h or w > max_w:
            scale = 1.0 * max_h / h
            if scale * w > max_w:
                scale = 1.0 * max_w / w
            new_w, new_h = scale * w // base * base, scale * h // base * base
        else:
            new_w, new_h = 1.0 * w // base * base, 1.0 * h // base * base

        scale_w = 1.0 * new_w / w
        scale_h = 1.0 * new_h / h
        intrinsics[0, :] *= scale_w
        intrinsics[1, :] *= scale_h
        img = cv2.resize(img, (int(new_w), int(new_h)))
        return img, intrinsics

    # -----------------------------------------------------------------
    # 核心：构建单样本
    # -----------------------------------------------------------------
    def __getitem__(self, idx):
        global _s_h, _s_w
        scan, ref_view, src_views, scene_name = self.metas[idx]
        view_ids = [ref_view] + src_views[:self.nviews - 1]

        imgs = []
        proj_matrices = []
        depth_values = None

        for i, vid in enumerate(view_ids):
            # 图像路径：优先 images_post/，回退 images/
            img_filename = os.path.join(self.datapath, '{}/images_post/{:0>8}.jpg'.format(scan, vid))
            if not os.path.exists(img_filename):
                img_filename = os.path.join(self.datapath, '{}/images/{:0>8}.jpg'.format(scan, vid))

            cam_filename = os.path.join(self.datapath, '{}/cams/{:0>8}_cam.txt'.format(scan, vid))

            img = self.read_img(img_filename)
            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(
                cam_filename, interval_scale=self.interval_scale[scene_name])

            img, intrinsics = self.scale_mvs_input(img, intrinsics, self.max_w, self.max_h)

            if self.fix_res:
                _s_h, _s_w = img.shape[:2]
                self.fix_res = False
                self.fix_wh = True

            if i == 0 and not self.fix_wh:
                _s_h, _s_w = img.shape[:2]

            # 统一分辨率
            c_h, c_w = img.shape[:2]
            if (c_h != _s_h) or (c_w != _s_w):
                scale_h = 1.0 * _s_h / c_h
                scale_w = 1.0 * _s_w / c_w
                img = cv2.resize(img, (_s_w, _s_h))
                intrinsics[0, :] *= scale_w
                intrinsics[1, :] *= scale_h

            imgs.append(img)

            # 投影矩阵: [extrinsic(4x4), intrinsic(3x3)]
            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics
            proj_matrices.append(proj_mat)

            # 深度值：仅参考视角计算
            if i == 0:
                if self.inverse_depth:
                    depth_end = depth_interval * self.ndepths + depth_min
                    depth_values = np.linspace(1.0 / depth_min, 1.0 / depth_end,
                                               self.ndepths, endpoint=False)
                    depth_values = (1.0 / depth_values).astype(np.float32)
                else:
                    depth_values = np.arange(
                        depth_min,
                        depth_interval * (self.ndepths - 0.5) + depth_min,
                        depth_interval,
                        dtype=np.float32,
                    )

        # 堆叠: (N, H, W, C) -> (N, C, H, W)
        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        proj_matrices = np.stack(proj_matrices)

        # 多阶段投影矩阵（内参缩放）
        # CasMVSNet/UniMVSNet (3 stages): scale = 1, 2, 4
        # MVSTER (4 stages): scale = 0.5, 1, 2, 4
        proj_matrices_ms = {}
        for stage in range(1, self.num_stage + 1):
            if self.num_stage == 4:
                # MVSTER: stage1=0.5x, stage2=1x, stage3=2x, stage4=4x
                scale_factor = 2.0 ** (stage - 2)
            else:
                # CasMVSNet/UniMVSNet: stage1=1x, stage2=2x, stage3=4x
                scale_factor = 2.0 ** (stage - 1)

            scaled = proj_matrices.copy()
            if scale_factor != 1.0:
                scaled[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * scale_factor
            proj_matrices_ms["stage{}".format(stage)] = scaled

        return {
            "imgs": imgs,
            "proj_matrices": proj_matrices_ms,
            "depth_values": depth_values,
            "filename": scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}",
        }
