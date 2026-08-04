import sys
import os
current_file = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file))
sys.path.append(parent_dir)

import random
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import os, cv2, time, math
from PIL import Image
from datasets.data_io import *
from imageio import imread, imsave, imwrite
from PIL import Image, ImageEnhance, ImageOps, ImageFile


# data augment
def image_augment(image):

    image = randomColor(image)
    #image = randomGaussian(image, mean=0.2, sigma=0.3)

    return image
    

def randomColor(image):

    random_factor = np.random.randint(1, 301) / 100.
    color_image = ImageEnhance.Color(image).enhance(random_factor)  # Image Color
    random_factor = np.random.randint(10, 201) / 100.
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # Image Brightness
    random_factor = np.random.randint(10, 201) / 100.
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # Image Contrast
    random_factor = np.random.randint(0, 301) / 100.
    sharpness_image = ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # Image Sharpness

    return sharpness_image

# the DTU dataset preprocessed by Yao Yao (only for training)
class MVSDataset(Dataset):
    def __init__(self, datapath, mode, nviews, img_size=None, ndepths=192, interval_scale=1.06, **kwargs):
        super(MVSDataset, self).__init__()
        self.img_size = img_size if img_size is not None else [512, 640]
        # assert self.img_size[0] % 32 == 0 and self.img_size[1] % 32 == 0, 'img_wh must both be multiples of 32!'

        self.datapath = datapath
        # self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale
        self.kwargs = kwargs
        print("mvsdataset kwargs", self.kwargs)

        assert self.mode in ["train", "val", "test"]
        self.metas = self.build_list()

        # # less data
        # np.random.shuffle(self.metas)
        # self.metas = self.metas[:int(0.5 * len(self.metas))]

    def build_list(self):
        gt_fext = '.png'
        sample_list = []

        # image index
        train_cluster_path = self.datapath + '/index.txt'
        data_cluster = open(train_cluster_path).read().split()

        # pair
        view_pair_path = self.datapath + '/pair.txt'
        ref_indexs = []
        src_indexs = []
        with open(view_pair_path) as f:
            cluster_num = int(f.readline().rstrip())
            for idx in range(cluster_num):
                ref_index = int(f.readline().rstrip())
                src_index = [int(x) for x in f.readline().rstrip().split()][1:]
                ref_indexs.append(ref_index)
                src_indexs.append(src_index)

        # for each data scene
        for i in data_cluster:
            image_folder = os.path.join(self.datapath, ('Images/%s' % i)).replace("\\", "/")
            cam_folder = os.path.join(self.datapath, ('Cams/%s' % i)).replace("\\", "/")
            depth_folder = os.path.join(self.datapath, ('Depths/%s' % i)).replace("\\", "/")

            # for each view
            for ref_ind, view_inds in zip(ref_indexs, src_indexs):  # 0-4
                image_folder2 = os.path.join(image_folder, ('%d' % ref_ind)).replace("\\", "/")
                image_files = sorted(os.listdir(image_folder2))

                view_cnts = min(self.nviews, len(view_inds) + 1)

                for j in range(0, int(np.size(image_files))):
                    paths = []
                    portion = os.path.splitext(image_files[j])
                    newcamname = portion[0] + '.txt'
                    newdepthname = portion[0] + gt_fext

                    # ref image
                    ref_image_path = os.path.join(os.path.join(image_folder, ('%d' % ref_ind)),
                                                  image_files[j]).replace("\\", "/")
                    ref_cam_path = os.path.join(os.path.join(cam_folder, ('%d' % ref_ind)), newcamname).replace(
                        "\\", "/")

                    paths.append(ref_image_path)
                    paths.append(ref_cam_path)

                    # view images
                    for view in range(view_cnts - 1):
                        # print(image_folder)
                        view_ind = view_inds[view]  # selected view image
                        view_image_path = os.path.join(os.path.join(image_folder, ('%d' % view_ind)),
                                                       image_files[j]).replace("\\", "/")

                        view_cam_path = os.path.join(os.path.join(cam_folder, ('%d' % view_ind)),
                                                     newcamname).replace("\\", "/")
                        paths.append(view_image_path)
                        paths.append(view_cam_path)

                    # depth path
                    depth_image_path = os.path.join(os.path.join(depth_folder, ('%d' % ref_ind)),
                                                    newdepthname).replace("\\", "/")
                    paths.append(depth_image_path)
                    sample_list.append((view_cnts, paths))

        return sample_list

    def center_image(self, img, mode='mean'):
        # normalize image input
        if mode == 'standard':
            np_img = np.array(img, dtype=np.float32) / 255.

        elif mode == 'mean':
            img_array = np.array(img)
            img = img_array.astype(np.float32)
            var = np.var(img, axis=(0, 1), keepdims=True)
            mean = np.mean(img, axis=(0, 1), keepdims=True)
            np_img = (img - mean) / (np.sqrt(var) + 0.00000001)

        else:
            raise Exception("{}? Not implemented yet!".format(mode))

        return np_img


    def __len__(self):
        return len(self.metas)

    def read_cam_file_old(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1]) * self.interval_scale
        return intrinsics, extrinsics, depth_min, depth_interval
    
    def read_cam_file(self, file):
        with open(file) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]    

        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        
        # if cam ori is XrightYup
        O = np.eye(3, dtype=np.float32)
        O[1, 1] = -1
        O[2, 2] = -1
        R = extrinsics[0:3, 0:3]
        R2 = np.matmul(R, O)
        extrinsics[0:3, 0:3] = R2
        extrinsics = np.linalg.inv(extrinsics)  # Tcw   天呐，怎么会有这样的操作
        
        intrinsics = np.zeros((3, 3), dtype=np.float32)
        # print(lines[6])
        # # intrinsics_info = np.fromstring(lines[6])
        intrinsics_info = lines[6].split()
        intrinsics[0][0] = intrinsics_info[0]  # 焦距f
        intrinsics[1][1] = intrinsics_info[0]  # 焦距f
        intrinsics[0][2] = intrinsics_info[1]  # x0
        intrinsics[1][2] = intrinsics_info[2]  # y0
        intrinsics[2][2] = 1  # 齐次坐标补充

        # depth_range_info = np.fromstring(lines[8])
        depth_range_info = lines[8].split()
        depth_min = float(depth_range_info[0])
        depth_max = float(depth_range_info[1])
        depth_interval = float(depth_range_info[2])

        return intrinsics, extrinsics, depth_min, depth_max, depth_interval

    def read_img(self, filename):
        img = Image.open(filename)
        # # img = img.resize(self.img_size[::-1], Image.BILINEAR)
        # # scale 0~255 to 0~1
        # np_img = np.array(img, dtype=np.float32) / 255.0
        # return np_img
        return img

    def prepare_img(self, hr_img):
        # w1600-h1200-> 800-600 ; crop -> 640, 512; downsample 1/4 -> 160, 128

        # downsample
        h, w = hr_img.shape
        hr_img_ds = cv2.resize(hr_img, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)
        # crop
        h, w = hr_img_ds.shape
        target_h, target_w = 512, 640
        start_h, start_w = (h - target_h) // 2, (w - target_w) // 2
        hr_img_crop = hr_img_ds[start_h: start_h + target_h, start_w: start_w + target_w]

        # #downsample
        # lr_img = cv2.resize(hr_img_crop, (target_w//4, target_h//4), interpolation=cv2.INTER_NEAREST)

        return hr_img_crop

    def read_mask_hr(self, filename):
        img = Image.open(filename)
        np_img = np.array(img, dtype=np.float32)
        np_img = (np_img > 10).astype(np.float32)
        np_img = self.prepare_img(np_img)

        h, w = np_img.shape
        np_img_ms = {
            "stage1": cv2.resize(np_img, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(np_img, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
            "stage3": np_img,
        }
        return np_img_ms

    def read_depth(self, filename):
        # read pfm depth file
        return np.array(read_pfm(filename)[0], dtype=np.float32)

    def read_depth_hr(self, filename):
        # read pfm depth file
        # w1600-h1200-> 800-600 ; crop -> 640, 512; downsample 1/4 -> 160, 128
        depth_hr = np.array(read_pfm(filename)[0], dtype=np.float32)
        depth_lr = self.prepare_img(depth_hr)

        h, w = depth_lr.shape
        depth_lr_ms = {
            "stage1": cv2.resize(depth_lr, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(depth_lr, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
            "stage3": depth_lr,
        }
        return depth_lr_ms

    def __getitem__(self, idx):
        meta = self.metas[idx]
        # scan, light_idx, ref_view, src_views = meta
        view_ids, paths = meta    # 解析的元数据中为路径和样本视角数
        # # use only the reference view and first nviews-1 source views
        # view_ids = [ref_view] + src_views[:self.nviews - 1]

        imgs = []
        mask = None
        depth_ms = None
        depth_values = None
        proj_matrices = []

        # depth
        depth_dir = os.path.join(paths[2 * view_ids]) 
        depimg = imread(depth_dir) # read png depth file
        depth_image = np.array(np.float32(depimg) / 64.0)  # WHU-MVS dataset

        for i,vid in enumerate(range(view_ids)):
            # images
            if self.mode == "train":
                image = image_augment(self.read_img(paths[2 * vid]))
            else:
                image = self.read_img(paths[2 * vid])
            image = np.array(image)

            intrinsics, extrinsics, depth_min, depth_max, depth_interval = self.read_cam_file(paths[2 * vid + 1])

            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)  #
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics

            proj_matrices.append(proj_mat)


            if i == 0:  # reference view
                # # 由于浮点数的精度问题，有时候最后一个值可能会因为精度问题而无法达到，导致生成的数组只有self.ndepths-1个元素。
                # # depth_max = depth_interval * self.ndepths + depth_min
                depth_values = np.linspace(depth_min, depth_max, self.ndepths, endpoint=False, dtype=np.float32)    # 等间隔分布的深度样本
                ref_depth_interval = depth_interval
                # depth_values = np.array([depth_min, depth_max, depth_interval], dtype=np.float32)
                # # depth_values = np.arange(depth_min, depth_max, depth_interval, dtype=np.float32)
                mask = np.float32((depth_image >= depth_min) * 1.0) * np.float32((depth_image <= depth_max) * 1.0)

            imgs.append(self.center_image(image))

        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        # # ms proj_mats
        # proj_matrices = np.stack(proj_matrices)
        # stage2_pjmats = proj_matrices.copy()
        # stage2_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 2
        # stage3_pjmats = proj_matrices.copy()
        # stage3_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 4

        # proj_matrices_ms = {
        #     "stage1": proj_matrices,
        #     "stage2": stage2_pjmats,
        #     "stage3": stage3_pjmats
        # }

        proj_matrices = np.stack(proj_matrices)
        stage2_pjmats = proj_matrices.copy()
        stage2_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] / 2
        stage3_pjmats = proj_matrices.copy()
        stage3_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] / 4
        proj_matrices_ms = {
            "stage1": stage3_pjmats,
            "stage2": stage2_pjmats,
            "stage3": proj_matrices
        }

        h, w = depth_image.shape
        depth_ms = {
            "stage1": cv2.resize(depth_image, (w//4, h//4), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(depth_image, (w//2, h//2), interpolation=cv2.INTER_NEAREST),
            "stage3": depth_image,
        }
        mask_ms = {
            "stage1": cv2.resize(mask, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(mask, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
            "stage3": mask,
        }

        name = os.path.splitext(os.path.basename(paths[0]))[0]
        vid = os.path.dirname(paths[0]).split("/")[-2]

        return {"imgs": imgs,
                "proj_matrices": proj_matrices_ms,
                "depth": depth_ms,
                "depth_values": depth_values,
                "out_name": name,
                "depth_interval": ref_depth_interval,
                "out_view": vid,
                "mask": mask_ms}


if __name__ == "__main__":

    dataset = MVSDataset('/root/data/datasets/WHU_MVS_dataset/WHU_MVS_dataset/train', 'train', 5)
    print(dataset[0])



