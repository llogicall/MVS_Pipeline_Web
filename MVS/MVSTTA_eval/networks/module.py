import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
from torch.autograd import Variable
from .texture_norm import *
from .depth_sample import *
from .fea_warping import *

sys.path.append("..")


def init_bn(module):
    if module.weight is not None:
        nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return


def init_uniform(module, init_method):
    if module.weight is not None:
        if init_method == "kaiming":
            nn.init.kaiming_uniform_(module.weight)
        elif init_method == "xavier":
            nn.init.xavier_uniform_(module.weight)
    return


class Conv2d(nn.Module):
    """Applies a 2D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Conv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        # self.bn = nn.GroupNorm(8, out_channels) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class Deconv2d(nn.Module):
    """Applies a 2D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv2d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        # self.bn = nn.GroupNorm(8, out_channels) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        y = self.conv(x)
        if self.stride == 2:
            h, w = list(x.size())[2:]
            y = y[:, :, :2 * h, :2 * w].contiguous()
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class Conv3d(nn.Module):
    """Applies a 3D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Conv3d, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        # self.bn = nn.GroupNorm(8, out_channels) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class Deconv3d(nn.Module):
    """Applies a 3D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv3d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        # self.bn = nn.GroupNorm(8, out_channels) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        y = self.conv(x)
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


def homo_warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth] o [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth,
                                                                                            -1)  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        proj_xyz[:, 2:3][proj_xyz[:, 2:3] == 0] += 0.00001  # NAN BUG, not on dtu, but on blendedmvs

        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy

    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                   padding_mode='zeros').type(torch.float32)
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

    return warped_src_fea


class DeConv2dFuse(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, relu=True, bn=True,
                 bn_momentum=0.1):
        super(DeConv2dFuse, self).__init__()

        self.deconv = Deconv2d(in_channels, out_channels, kernel_size, stride=2, padding=1, output_padding=1,
                               bn=True, relu=relu, bn_momentum=bn_momentum)

        self.conv = Conv2d(2 * out_channels, out_channels, kernel_size, stride=1, padding=1,
                           bn=bn, relu=relu, bn_momentum=bn_momentum)

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x_pre, x):
        x = self.deconv(x)
        x = torch.cat((x, x_pre), dim=1)
        x = self.conv(x)
        return x


class FeatureNet(nn.Module):
    def __init__(self, base_channels, num_stage=3, stride=4, mode="fpn", use_texture_norm=True):
        super(FeatureNet, self).__init__()
        assert mode in ["unet", "fpn"], print("mode must be in 'unet', 'fpn', but get:{}".format(mode))
        self.mode = mode
        self.stride = stride
        self.base_channels = base_channels
        self.num_stage = num_stage
        self.use_texture_norm = use_texture_norm  # 新增：是否使用纹理归一化

        self.conv0 = nn.Sequential(
            Conv2d(3, base_channels, 3, 1, padding=1),
            Conv2d(base_channels, base_channels, 3, 1, padding=1),
        )

        self.conv1 = nn.Sequential(
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )

        self.conv2 = nn.Sequential(
            Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
        )

        if self.mode == 'unet':
            self.out1 = nn.Conv2d(base_channels * 4, base_channels * 4, 1, bias=False)
            self.out_channels = [4 * base_channels]
            if num_stage == 3:
                self.deconv1 = DeConv2dFuse(base_channels * 4, base_channels * 2, 3)
                self.deconv2 = DeConv2dFuse(base_channels * 2, base_channels, 3)

                self.out2 = nn.Conv2d(base_channels * 2, base_channels * 2, 1, bias=False)
                self.out3 = nn.Conv2d(base_channels, base_channels, 1, bias=False)
                self.out_channels.append(2 * base_channels)
                self.out_channels.append(base_channels)

            elif num_stage == 2:
                self.deconv1 = DeConv2dFuse(base_channels * 4, base_channels * 2, 3)

                self.out2 = nn.Conv2d(base_channels * 2, base_channels * 2, 1, bias=False)
                self.out_channels.append(2 * base_channels)
        elif self.mode == "fpn":
            self.out1 = nn.Conv2d(base_channels * 4, base_channels * 4, 1, bias=False)
            self.out_channels = [4 * base_channels]
            final_chs = base_channels * 4
            if num_stage == 3:
                self.inner1 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)
                self.inner2 = nn.Conv2d(base_channels * 1, final_chs, 1, bias=True)

                self.out2 = nn.Conv2d(final_chs, base_channels * 2, 3, padding=1, bias=False)
                self.out3 = nn.Conv2d(final_chs, base_channels, 3, padding=1, bias=False)
                self.out_channels.append(base_channels * 2)
                self.out_channels.append(base_channels)

            elif num_stage == 2:
                self.inner1 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)

                self.out2 = nn.Conv2d(final_chs, base_channels, 3, padding=1, bias=False)
                self.out_channels.append(base_channels)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

        # 新增：为每个输出层添加纹理自适应归一化
        if self.use_texture_norm:
            self.texture_norm1 = TextureAdaptiveNorm(base_channels * 4)
            if num_stage >= 2:
                self.texture_norm2 = TextureAdaptiveNorm(base_channels * 2)
            if num_stage >= 3:
                self.texture_norm3 = TextureAdaptiveNorm(base_channels)


    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)

        intra_feat = conv2
        outputs = {}
        if self.mode == "unet":
            out = self.out1(intra_feat)
            if self.use_texture_norm:
                out = self.texture_norm1(out, img=x)
            outputs["stage1"] = out
            if self.num_stage == 3:
                intra_feat = self.deconv1(conv1, intra_feat)
                out = self.out2(intra_feat)
                # 添加归一化
                if self.use_texture_norm:
                    out = self.texture_norm2(out, img=x)                
                outputs["stage2"] = out

                intra_feat = self.deconv2(conv0, intra_feat)
                out = self.out3(intra_feat)
                # 添加归一化
                if self.use_texture_norm:
                    out = self.texture_norm3(out, img=x)                
                outputs["stage3"] = out

            elif self.num_stage == 2:
                intra_feat = self.deconv1(conv1, intra_feat)
                out = self.out2(intra_feat)
                # 添加归一化
                if self.use_texture_norm:
                    out = self.texture_norm2(out, img=x)
                outputs["stage2"] = out

        elif self.mode == "fpn":
            out = self.out1(intra_feat)
            if self.use_texture_norm:
                out = self.texture_norm1(out, img=x)
            outputs["stage1"] = out
            if self.num_stage == 3:
                intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner1(conv1)
                out = self.out2(intra_feat)
                # 添加归一化
                if self.use_texture_norm:
                    out = self.texture_norm2(out, img=x)                
                outputs["stage2"] = out

                intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner2(conv0)
                out = self.out3(intra_feat)
                # 添加归一化
                if self.use_texture_norm:
                    out = self.texture_norm3(out, img=x)                
                outputs["stage3"] = out

            elif self.num_stage == 2:
                intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner1(conv1)
                out = self.out2(intra_feat)
                # 添加归一化
                if self.use_texture_norm:
                    out = self.texture_norm2(out, img=x)                
                outputs["stage2"] = out

        return outputs


class CostRegNet(nn.Module):
    def __init__(self, in_channels, base_channels):
        super(CostRegNet, self).__init__()
        self.conv0 = Conv3d(in_channels, base_channels, padding=1)

        self.conv1 = Conv3d(base_channels, base_channels * 2, stride=2, padding=1)
        self.conv2 = Conv3d(base_channels * 2, base_channels * 2, padding=1)

        self.conv3 = Conv3d(base_channels * 2, base_channels * 4, stride=2, padding=1)
        self.conv4 = Conv3d(base_channels * 4, base_channels * 4, padding=1)

        self.conv5 = Conv3d(base_channels * 4, base_channels * 8, stride=2, padding=1)
        self.conv6 = Conv3d(base_channels * 8, base_channels * 8, padding=1)

        self.conv7 = Deconv3d(base_channels * 8, base_channels * 4, stride=2, padding=1, output_padding=1)

        self.conv9 = Deconv3d(base_channels * 4, base_channels * 2, stride=2, padding=1, output_padding=1)

        self.conv11 = Deconv3d(base_channels * 2, base_channels * 1, stride=2, padding=1, output_padding=1)

        self.prob = nn.Conv3d(base_channels, 1, 3, stride=1, padding=1, bias=False)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.prob(x)
        return x


class AggWeightNetVolume(nn.Module):
    def __init__(self, in_channels=32):
        super(AggWeightNetVolume, self).__init__()
        self.w_net = nn.Sequential(
            Conv3d(in_channels, 1, kernel_size=1, stride=1, padding=0),
            Conv3d(1, 1, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        """
        :param x: (b, c, d, h, w)
        :return: (b, 1, d, h, w)
        """
        w = self.w_net(x)
        return w


def depth_regression(p, depth_values):
    if depth_values.dim() <= 2:
        # print("regression dim <= 2")
        depth_values = depth_values.view(*depth_values.shape, 1, 1)
    depth = torch.sum(p * depth_values, 1)

    return depth


def winner_take_all(prob_volume, depth_values):
    """
    :param prob_volume: (b, d, h, w)
    :param depth_values: (b, d, h, w)
    :return: (b, h, w)
    """
    _, idx = torch.max(prob_volume, dim=1, keepdim=True)
    depth = torch.gather(depth_values, 1, idx).squeeze(1)
    return depth


def unity_regression(prob_volume, depth_values, interval):
    """
    :param interval: (b, )
    :param prob_volume: (b, d, h, w)
    :param depth_values: (b, d, h, w)
    :return: (b, h, w)
    """
    val, idx = torch.max(prob_volume, dim=1, keepdim=True)

    wta_depth = torch.gather(depth_values, 1, idx)
    offset = (1 - val) * interval

    depth = wta_depth + offset
    depth = depth.squeeze(1)

    return depth


def get_cur_depth_range_samples(last_depth, ndepth, depth_inteval_pixel):
    # cur_depth: (B, H, W)
    # return depth_range_values: (B, D, H, W)
    last_depth_min = (last_depth - ndepth / 2 * depth_inteval_pixel)  # (B, H, W)
    last_depth_max = (last_depth + ndepth / 2 * depth_inteval_pixel)
    # cur_depth_min = (cur_depth - ndepth / 2 * depth_inteval_pixel).clamp(min=0.0)   #(B, H, W)
    # cur_depth_max = (cur_depth_min + (ndepth - 1) * depth_inteval_pixel).clamp(max=max_depth)

    new_interval = (last_depth_max - last_depth_min) / (ndepth - 1)  # (B, H, W)

    depth_range_samples = last_depth_min.unsqueeze(1) + (torch.arange(0, ndepth, device=last_depth.device,
                                                                      dtype=last_depth.dtype,
                                                                      requires_grad=False).reshape(1, -1, 1,
                                                                                                   1) * new_interval.unsqueeze(1))

    return depth_range_samples, (ndepth * depth_inteval_pixel) / (ndepth - 1)


def get_depth_range_samples(last_depth, ndepth, depth_inteval_pixel, shape=None, prob_volume=None, reprojection_volume=None, prev_depth_hypotheses=None, args=None):
    # cur_depth: (B, H, W) or (B, D)
    # return depth_range_samples: (B, D, H, W)

    if last_depth.dim() == 2:  # 第一阶段
        last_depth_min = last_depth[:, 0]  # (B,)
        last_depth_max = last_depth[:, -1]
        new_interval = (last_depth_max - last_depth_min) / (ndepth - 1)  # (B, )
        stage_interval = new_interval[0]

        depth_range_samples = last_depth_min.unsqueeze(1) + (torch.arange(0, ndepth, device=last_depth.device, dtype=last_depth.dtype,
                                                                          requires_grad=False).reshape(1, -1) * new_interval.unsqueeze(
            1))  # (B, D)

        # (B, D, H, W)
        depth_range_samples = depth_range_samples.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, shape[0], shape[1])

    else:   # 使用深度来调整采样

        # depth_range_samples, stage_interval = get_cur_depth_range_samples(last_depth, ndepth, depth_inteval_pixel)
        # depth_range_samples = get_deformable_depth_range_samples(last_depth, prob_volume, prev_depth_hypotheses, ndepth, eta=1.0, 
        #                                max_depth=max_depth, min_depth=min_depth)

        max_depth = torch.max(prev_depth_hypotheses).item()
        min_depth = torch.min(prev_depth_hypotheses).item()
        stage_interval = (ndepth * depth_inteval_pixel) / (ndepth - 1)
        cur_depth = last_depth

        if args.ds_mode == 'uniform':
            depth_range_samples = get_cur_depth_range_samples(cur_depth, ndepth, depth_inteval_pixel, shape, max_depth, min_depth)
        elif args.ds_mode == 'deformable':
            depth_range_samples = get_deformable_depth_range_samples(cur_depth, prob_volume, prev_depth_hypotheses, ndepth, eta=1.0, 
                                        max_depth=max_depth, min_depth=min_depth)
        elif args.ds_mode == 'reprojectionv1':
            depth_range_samples,_ = get_reprojection_depth_range_samples(cur_depth, prob_volume, prev_depth_hypotheses, reprojection_volume, ndepth, eta=1.0, 
                                        max_depth=max_depth, min_depth=min_depth)
        elif args.ds_mode == 'reprojectionv2':
            depth_range_samples,_ = get_reprojection_depth_range_samples_v2(cur_depth, prob_volume, prev_depth_hypotheses, reprojection_volume, ndepth, eta=1.0, 
                                        max_depth=max_depth, min_depth=min_depth)        
        elif args.ds_mode == 'reprojectionv3':
            depth_range_samples = get_reprojection_depth_range_samples_v3(cur_depth, reprojection_volume, ndepth, max_depth=max_depth, min_depth=min_depth)
        elif args.ds_mode == 'reprojectionv4':
            depth_range_samples = get_reprojection_depth_range_samples_v4(cur_depth, reprojection_volume, ndepth, max_depth=max_depth, min_depth=min_depth)
        elif args.ds_mode == 'reprojectionv5':
            depth_range_samples = get_reprojection_depth_range_samples_v5(cur_depth, reprojection_volume, ndepth, max_depth=max_depth, min_depth=min_depth)
        else:
            depth_range_samples = None


        

    return depth_range_samples, stage_interval




def get_deformable_depth_range_samples(cur_depth, prob_map, prev_depth_hypotheses, ndepth, eta=1.0, 
                                       max_depth=192.0, min_depth=0.0, shape=None):
    """
    基于深度假设可变形离散化(DHD)的深度范围采样
    根据论文公式(10)-(13)实现
    
    Args:
        cur_depth: (B, H, W) 当前阶段预测的深度图
        prob_map: (B, D_prev, H, W) 前一阶段的深度概率图
        prev_depth_hypotheses: (B, D_prev, H, W) 前一阶段的深度假设平面
        ndepth: 当前阶段深度假设数量
        eta: 范围扩展系数，控制深度范围的自适应调整程度
        max_depth: 最大深度值
        min_depth: 最小深度值
        shape: (B, H, W) 输入形状
    
    Returns:
        depth_range_samples: (B, D, H, W) 新生成的深度假设平面
    """
    B, H, W = cur_depth.shape
    new_shape = [B, prev_depth_hypotheses.shape[1], H, W]
    
    # 步骤1: 计算深度假设的方差（公式10）
    # cur_depth_expanded: (B, 1, H, W) 扩展维度用于广播
    cur_depth_expanded = cur_depth.unsqueeze(1)
    
    # 计算每个像素深度假设的方差 S_k(x)
    prev_depth_hypotheses = F.interpolate(prev_depth_hypotheses, size=(H, W), mode='bilinear')
    diff = prev_depth_hypotheses - cur_depth_expanded  # (B, D_prev, H, W)
    variance = torch.sum(prob_map * diff.pow(2), dim=1)  # (B, H, W)
    
    # 步骤2: 计算标准差 σ_k(x) = sqrt(S_k(x))
    std_dev = torch.sqrt(variance + 1e-8)  # 添加小量防止数值不稳定
    
    # 步骤3: 构建自适应深度范围（公式11）
    # I_{k+1}(x) = [D_k(x) - ησ_k(x), D_k(x) + ησ_k(x)]
    depth_min = cur_depth - eta * std_dev
    depth_max = cur_depth + eta * std_dev
    
    # 确保深度范围在有效区间内
    depth_min = torch.clamp(depth_min, min=min_depth)
    depth_max = torch.clamp(depth_max, max=max_depth)
    
    # 步骤4: 中心线性递增离散化(Centered Linear Increasing Discretization, CLID)
    # 根据论文公式(12)-(13)实现
    # 注意：这里假设ndepth为偶数，如果为奇数需要适当调整
    
    # 准备深度索引，假设ndepth为偶数
    half_ndepth = ndepth // 2
    
    # 生成索引di，用于公式(12)-(13)
    # 对于上半部分(di=1,2,...,half_ndepth)
    di_up = torch.arange(1, half_ndepth + 1, device=cur_depth.device, 
                         dtype=cur_depth.dtype).view(1, -1, 1, 1)  # (1, half_ndepth, 1, 1)
    
    # 对于下半部分(di=1,2,...,half_ndepth)
    di_down = torch.arange(1, half_ndepth + 1, device=cur_depth.device, 
                           dtype=cur_depth.dtype).view(1, -1, 1, 1)  # (1, half_ndepth, 1, 1)
    
    # 计算分母项: (ndepth/2)*(ndepth/2+1)
    denom = (half_ndepth * (half_ndepth + 1))
    denom = torch.tensor(denom).to(cur_depth.dtype)
    
    # 扩展cur_depth和std_dev用于广播
    # cur_depth_expanded = cur_depth.unsqueeze(1)  # (B, 1, H, W)
    eta_std = (eta * std_dev).unsqueeze(1)  # (B, 1, H, W)
    
    # 生成上半部分深度假设（公式12）
    # R_UP_{k,di} = D_k(x) + (ησ_k(x)/(ndepth/2*(ndepth/2+1))) * di(di+1)
    depth_up = cur_depth_expanded + (eta_std / denom) * di_up * (di_up + 1)
    
    # 生成下半部分深度假设（公式13）
    # R_DOWN_{k,di} = D_k(x) - (ησ_k(x)/(ndepth/2*(ndepth/2+1))) * di(di+1)
    depth_down = cur_depth_expanded - (eta_std / denom) * di_down * (di_down + 1)
    
    # 合并上下两部分，并添加中心点
    # 注意：如果ndepth为奇数，需要单独处理中心点
    if ndepth % 2 == 1:  # 奇数个深度假设
        depth_range_samples = torch.cat([depth_down.flip(1), cur_depth_expanded, depth_up], dim=1)
    else:  # 偶数个深度假设
        depth_range_samples = torch.cat([depth_down.flip(1), depth_up], dim=1)
    
    # 确保深度值在有效范围内
    depth_range_samples = torch.clamp(depth_range_samples, min=min_depth, max=max_depth)
    
    # 对深度假设进行排序（确保每个像素的深度假设是递增的）
    depth_range_samples, _ = torch.sort(depth_range_samples, dim=1)
    
    return depth_range_samples





def get_depth_range_samples_sdlmvs(last_depth, ndepth, depth_inteval_pixel,
                            shape=None, eta=2.5, deform=True):
    """
    级联 MVS 的 DHD 模块实现（SDL-MVS）
    last_depth: (B,H,W) 或 (B,D)
    return  depth_range_samples: (B,ndepth,H,W)
            stage_interval: 平均间隔（仅作日志）
    """
    if last_depth.dim() == 2:                       # 1D 序列 -> 均匀插值
        last_depth_min = last_depth[:, 0]           # (B,)
        last_depth_max = last_depth[:, -1]
        new_interval = (last_depth_max - last_depth_min) / (ndepth - 1)
        stage_interval = new_interval[0]
        depth_range_samples = (
            last_depth_min.unsqueeze(1)
            + torch.arange(ndepth, device=last_depth.device, dtype=last_depth.dtype
                          ).unsqueeze(0) * new_interval.unsqueeze(1)
        )                                           # (B,ndepth)
        depth_range_samples = depth_range_samples.unsqueeze(-1).unsqueeze(-1
                              ).expand(-1, -1, shape[0], shape[1])

    else:                                           # ========= DHD 入口 ========
        assert last_depth.dim() == 3
        depth_range_samples, stage_interval = dhd_module(
                last_depth, ndepth, eta, deform)
    return depth_range_samples, stage_interval

def dhd_module(depth_map, ndepth, eta=2.5, deform=True):
    """
    depth_map: (B,H,W)  上一阶段预测深度
    return   : (B,ndepth,H,W) + scalar(avg_interval)
    """
    B, H, W = depth_map.shape
    device = depth_map.device

    # 1) 计算像素级不确定性 σ(x)
    #    这里用 3×3 邻域标准差近似，速度快且无需概率体
    padding = 1
    unfolded = F.unfold(depth_map.unsqueeze(1), 3, padding=padding)  # (B,9,HW)
    mean = unfolded.mean(dim=1, keepdim=True)                      # (B,1,HW)
    std  = unfolded.std(dim=1, keepdim=True)                       # (B,1,HW)
    sigma = std.view(B, H, W)                                      # (B,H,W)

    # 2) 可变形区间 [μ-ησ, μ+ησ]
    center = depth_map
    d_min  = center - eta * sigma                                  # (B,H,W)
    d_max  = center + eta * sigma                                  # (B,H,W)

    # 3) 线性递增离散化（上下各 ndepth//2 层）
    nh = ndepth // 2
    # 上半区间指数坐标
    indices = torch.arange(1, nh + 1, device=device, dtype=depth_map.dtype)
    S = (nh * (nh + 1)) / 2                                        # 归一化因子
    up_delta   = (eta * sigma).unsqueeze(1) * (indices * (indices + 1)) / S
    up_depths  = center.unsqueeze(1) + up_delta.unsqueeze(2)       # (B,nh,H,W)

    # 下半区间镜像
    low_delta  = (eta * sigma).unsqueeze(1) * (indices * (indices + 1)) / S
    low_depths = center.unsqueeze(1) - low_delta.unsqueeze(2)      # (B,nh,H,W)

    # 合并并排序
    depth_hypo = torch.cat([low_depths.flip(1), center.unsqueeze(1), up_depths], 1)
    depth_hypo = depth_hypo.clamp(min=1e-2)                        # 防止负深度

    # 4) 平均间隔（仅日志）
    avg_interval = (d_max.mean() - d_min.mean()) / (ndepth - 1)

    return depth_hypo, avg_interval




