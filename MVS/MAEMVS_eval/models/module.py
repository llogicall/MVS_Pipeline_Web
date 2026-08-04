import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
sys.path.append("..")
from utils import local_pcd

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



class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ConvBn3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = ConvBnReLU(in_channels, out_channels, kernel_size=3, stride=stride, pad=1)
        self.conv2 = ConvBn(out_channels, out_channels, kernel_size=3, stride=1, pad=1)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        return out


class Hourglass3d(nn.Module):
    def __init__(self, channels):
        super(Hourglass3d, self).__init__()

        self.conv1a = ConvBnReLU3D(channels, channels * 2, kernel_size=3, stride=2, pad=1)
        self.conv1b = ConvBnReLU3D(channels * 2, channels * 2, kernel_size=3, stride=1, pad=1)

        self.conv2a = ConvBnReLU3D(channels * 2, channels * 4, kernel_size=3, stride=2, pad=1)
        self.conv2b = ConvBnReLU3D(channels * 4, channels * 4, kernel_size=3, stride=1, pad=1)

        self.dconv2 = nn.Sequential(
            nn.ConvTranspose3d(channels * 4, channels * 2, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(channels * 2))

        self.dconv1 = nn.Sequential(
            nn.ConvTranspose3d(channels * 2, channels, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(channels))

        self.redir1 = ConvBn3D(channels, channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = ConvBn3D(channels * 2, channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1b(self.conv1a(x))
        conv2 = self.conv2b(self.conv2a(conv1))
        dconv2 = F.relu(self.dconv2(conv2) + self.redir2(conv1), inplace=True)
        dconv1 = F.relu(self.dconv1(dconv2) + self.redir1(x), inplace=True)
        return dconv1


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
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy

    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                   padding_mode='zeros')
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

    return warped_src_fea

class DeConv2dFuse(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, relu=True, bn=True,
                 bn_momentum=0.1):
        super(DeConv2dFuse, self).__init__()

        self.deconv = Deconv2d(in_channels, out_channels, kernel_size, stride=2, padding=1, output_padding=1,
                               bn=True, relu=relu, bn_momentum=bn_momentum)

        self.conv = Conv2d(2*out_channels, out_channels, kernel_size, stride=1, padding=1,
                           bn=bn, relu=relu, bn_momentum=bn_momentum)

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x_pre, x):
        x = self.deconv(x)
        x = torch.cat((x, x_pre), dim=1)
        x = self.conv(x)
        return x


class FeatureNet(nn.Module):
    def __init__(self, base_channels, num_stage=3, stride=4, arch_mode="unet", use_texture_norm=True):
        super(FeatureNet, self).__init__()
        assert arch_mode in ["unet", "fpn"], print("mode must be in 'unet' or 'fpn', but get:{}".format(arch_mode))
        print("*************feature extraction arch mode:{}****************".format(arch_mode))
        self.arch_mode = arch_mode
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

        self.out1 = nn.Conv2d(base_channels * 4, base_channels * 4, 1, bias=False)
        self.out_channels = [4 * base_channels]

        if self.arch_mode == 'unet':
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
        elif self.arch_mode == "fpn":
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
        out = self.out1(intra_feat)

        # 新增：在输出前应用纹理归一化
        if self.use_texture_norm:
            out = self.texture_norm1(out, img=x)

        outputs["stage1"] = out
        if self.arch_mode == "unet":
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

        elif self.arch_mode == "fpn":
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
                    out = self.texture_norm3(out, img=x)
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

class RefineNet(nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()
        self.conv1 = ConvBnReLU(4, 32)
        self.conv2 = ConvBnReLU(32, 32)
        self.conv3 = ConvBnReLU(32, 32)
        self.res = ConvBnReLU(32, 1)

    def forward(self, img, depth_init):
        concat = F.cat((img, depth_init), dim=1)
        depth_residual = self.res(self.conv3(self.conv2(self.conv1(concat))))
        depth_refined = depth_init + depth_residual
        return depth_refined


def depth_regression(p, depth_values):
    if depth_values.dim() <= 2:
        # print("regression dim <= 2")
        depth_values = depth_values.view(*depth_values.shape, 1, 1)
    depth = torch.sum(p * depth_values, 1)

    return depth

def depth_to_invdepth_torch(depth, eps=1e-6):
    return 1.0 / (depth + eps)

def invdepth_to_depth_torch(invdepth, eps=1e-6):
    return 1.0 / (invdepth + eps)

def depth_to_logdepth(depth, eps=1e-6):
    return torch.log(depth + eps)

def logdepth_to_depth(logdepth):
    return torch.exp(logdepth)

def cas_mvsnet_loss(inputs, depth_gt_ms, mask_ms, **kwargs):
    depth_loss_weights = kwargs.get("dlossw", None)

    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)

    for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
        depth_est = stage_inputs["depth"]
        depth_gt = depth_gt_ms[stage_key]
        mask = mask_ms[stage_key]
        mask = mask > 0.5

        depth_loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction='mean')

        if depth_loss_weights is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            total_loss += depth_loss_weights[stage_idx] * depth_loss
        else:
            total_loss += 1.0 * depth_loss

    return total_loss, depth_loss


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



def get_cur_depth_range_samples(cur_depth, ndepth, depth_inteval_pixel, shape, max_depth=192.0, min_depth=0.0):
    #shape, (B, H, W)
    #cur_depth: (B, H, W)
    #return depth_range_values: (B, D, H, W)
    cur_depth_min = (cur_depth - ndepth / 2 * depth_inteval_pixel)  # (B, H, W)
    cur_depth_max = (cur_depth + ndepth / 2 * depth_inteval_pixel)
    # cur_depth_min = (cur_depth - ndepth / 2 * depth_inteval_pixel).clamp(min=0.0)   #(B, H, W)
    # cur_depth_max = (cur_depth_min + (ndepth - 1) * depth_inteval_pixel).clamp(max=max_depth)

    # assert cur_depth.shape == torch.Size(shape), "cur_depth:{}, input shape:{}".format(cur_depth.shape, shape)
    new_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1)  # (B, H, W)

    depth_range_samples = cur_depth_min.unsqueeze(1) + (torch.arange(0, ndepth, device=cur_depth.device,
                                                                  dtype=cur_depth.dtype,
                                                                  requires_grad=False).reshape(1, -1, 1,
                                                                                               1) * new_interval.unsqueeze(1))
    return depth_range_samples


def get_depth_range_samples(cur_depth, ndepth, depth_inteval_pixel, device, dtype, shape,
                           max_depth=192.0, min_depth=0.0, prob_volume=None, prev_depth_hypotheses=None):
    #shape: (B, H, W)
    #cur_depth: (B, H, W) or (B, D)
    #return depth_range_samples: (B, D, H, W)
    # stage0_shape:(B, H/4, W/4)
    if cur_depth.dim() == 2:
        cur_depth_min = cur_depth[:, 0]  # (B,)
        cur_depth_max = cur_depth[:, -1]
        new_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1)  # (B, )
        stage0_shape = shape
        stage0_shape[1] = shape[1]//4
        stage0_shape[2] = shape[2]//4

        depth_range_samples = cur_depth_min.unsqueeze(1) + (torch.arange(0, ndepth, device=device, dtype=dtype,
                                                                       requires_grad=False).reshape(1, -1) * new_interval.unsqueeze(1)) #(B, D)

        depth_range_samples = depth_range_samples.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, stage0_shape[1], stage0_shape[2]) #(B, D, H, W)   -> (B,D,H/4,W/4)

    else:

        # depth_range_samples = get_cur_depth_range_samples(cur_depth, ndepth, depth_inteval_pixel, shape, max_depth, min_depth)

        depth_range_samples = get_deformable_depth_range_samples(cur_depth, prob_volume, prev_depth_hypotheses, ndepth, eta=1.0, 
                                       max_depth=max_depth, min_depth=min_depth)



    return depth_range_samples


class TextureAdaptiveNorm(nn.Module):
    """
    基于纹理的自适应归一化层（非学习方式）
    根据图像的局部纹理强度动态调整特征的分布
    """
    def __init__(self, num_features, window_size=7):
        super(TextureAdaptiveNorm, self).__init__()
        self.num_features = num_features
        self.window_size = window_size
        # Sobel算子用于计算梯度
        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1],
                                                     [-2, 0, 2],
                                                     [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([[-1, -2, -1],
                                                     [ 0,  0,  0],
                                                     [ 1,  2,  1]], dtype=torch.float32).view(1, 1, 3, 3))
        
    def extract_texture_map(self, img):
        """
        提取纹理强度图（抽象版本）
        
        Args:
            img: [B, 3, H, W] 输入图像
        Returns:
            texture_strength: [B, 1, H, W] 纹理强度图
        """
        # 转换为灰度图
        gray = 0.299 * img[:, 0:1, :, :] + 0.587 * img[:, 1:2, :, :] + 0.114 * img[:, 2:3, :, :]
        
        # 计算梯度
        grad_x = F.conv2d(gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(gray, self.sobel_y, padding=1)
        
        # 梯度幅值作为纹理强度
        texture_strength = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        
        # 局部平均平滑纹理强度图（减少噪声影响）
        kernel_size = self.window_size
        padding = kernel_size // 2
        texture_strength = F.avg_pool2d(
            F.pad(texture_strength, (padding, padding, padding, padding), mode='reflect'),
            kernel_size=kernel_size, stride=1
        )
        
        return texture_strength
    
    def forward(self, feature, img=None, texture_map=None):
        """
        Args:
            feature: [B, C, H, W] 待归一化的特征
            img: [B, 3, H, W] 原始图像（用于提取纹理，可选）
            texture_map: [B, 1, H, W] 预计算的纹理强度图（可选）
        
        Returns:
            normalized_feature: [B, C, H, W] 归一化后的特征
        """
        B, C, H, W = feature.shape
        
        # 如果没有提供纹理图，从图像提取
        if texture_map is None and img is not None:
            texture_map = self.extract_texture_map(img)
        elif texture_map is None:
            # 如果都没有提供，使用简单的方差作为纹理估计
            texture_map = torch.var(feature, dim=1, keepdim=True)
        
        # 纹理强度归一化到 [0, 1] 范围
        texture_map = (texture_map - texture_map.min()) / (texture_map.max() - texture_map.min() + 1e-8)
        
        # 计算特征的通道统计量   每个通道的所有元素计算统计量
        mean = feature.mean(dim=[2, 3], keepdim=True)  # [B, C, 1, 1]
        std = feature.std(dim=[2, 3], keepdim=True) + 1e-8  # [B, C, 1, 1]

        # 基础归一化   （最普通的归一化）
        normalized = (feature - mean) / std
        
        # 根据纹理强度动态调整   自适应调制
        # 纹理强的区域：保持更高的对比度（减少归一化程度）
        # 纹理弱的区域：更强的归一化（增强对比度）
        
        # 自适应缩放因子：纹理越强，scale越接近1（保持原特征）
        scale = 0.5 + 0.5 * texture_map  # [B, 1, H, W] 范围 [0.5, 1.0]
        
        # 自适应偏移：纹理越强，偏移越小
        offset = (1.0 - texture_map) * mean  # [B, C, H, W]
        
        # 根据特征图的尺寸将scale进行缩放的过程
        scale = F.interpolate(scale, [H,W], mode='bilinear')
        offset = F.interpolate(offset, [H,W], mode='bilinear')
        # 应用自适应调制
        normalized_feature = normalized * scale + offset
        
        return normalized_feature



if __name__ == "__main__":
    # some testing code, just IGNORE it
    import sys
    sys.path.append("../")
    from datasets import find_dataset_def
    from torch.utils.data import DataLoader
    import numpy as np
    import cv2
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    # MVSDataset = find_dataset_def("colmap")
    # dataset = MVSDataset("../data/results/ford/num10_1/", 3, 'test',
    #                      128, interval_scale=1.06, max_h=1250, max_w=1024)

    MVSDataset = find_dataset_def("dtu_yao")
    num_depth = 48
    dataset = MVSDataset("../data/DTU/mvs_training/dtu/", '../lists/dtu/train.txt', 'train',
                         3, num_depth, interval_scale=1.06 * 192 / num_depth)

    dataloader = DataLoader(dataset, batch_size=1)
    item = next(iter(dataloader))

    imgs = item["imgs"][:, :, :, ::4, ::4]  #(B, N, 3, H, W)
    # imgs = item["imgs"][:, :, :, :, :]
    proj_matrices = item["proj_matrices"]   #(B, N, 2, 4, 4) dim=N: N view; dim=2: index 0 for extr, 1 for intric
    proj_matrices[:, :, 1, :2, :] = proj_matrices[:, :, 1, :2, :]
    # proj_matrices[:, :, 1, :2, :] = proj_matrices[:, :, 1, :2, :] * 4
    depth_values = item["depth_values"]     #(B, D)

    imgs = torch.unbind(imgs, 1)
    proj_matrices = torch.unbind(proj_matrices, 1)
    ref_img, src_imgs = imgs[0], imgs[1:]
    ref_proj, src_proj = proj_matrices[0], proj_matrices[1:][0]  #only vis first view

    src_proj_new = src_proj[:, 0].clone()
    src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
    ref_proj_new = ref_proj[:, 0].clone()
    ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])

    warped_imgs = homo_warping(src_imgs[0], src_proj_new, ref_proj_new, depth_values)

    ref_img_np = ref_img.permute([0, 2, 3, 1])[0].detach().cpu().numpy()[:, :, ::-1] * 255
    cv2.imwrite('../tmp/ref.png', ref_img_np)
    cv2.imwrite('../tmp/src.png', src_imgs[0].permute([0, 2, 3, 1])[0].detach().cpu().numpy()[:, :, ::-1] * 255)

    for i in range(warped_imgs.shape[2]):
        warped_img = warped_imgs[:, :, i, :, :].permute([0, 2, 3, 1]).contiguous()
        img_np = warped_img[0].detach().cpu().numpy()
        img_np = img_np[:, :, ::-1] * 255

        alpha = 0.5
        beta = 1 - alpha
        gamma = 0
        img_add = cv2.addWeighted(ref_img_np, alpha, img_np, beta, gamma)
        cv2.imwrite('../tmp/tmp{}.png'.format(i), np.hstack([ref_img_np, img_np, img_add])) #* ratio + img_np*(1-ratio)]))