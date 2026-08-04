"""
多任务学习的FeatureNet - 支持稀疏CNN特征提取和MAE辅助任务
基于论文方法整合的模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .sparse_cnn import SparseConvBlockV2, PseudoSparseBatchNorm2d


class Conv2d(nn.Module):
    """Applies a 2D convolution (optionally with batch normalization and relu activation)"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Conv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x


class Deconv2d(nn.Module):
    """Applies a 2D deconvolution (optionally with batch normalization and relu activation)"""
    
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

    def forward(self, x):
        y = self.conv(x)
        if self.stride == 2:
            h, w = list(x.size())[2:]
            y = y[:, :, :2 * h, :2 * w].contiguous()
        if self.bn is not None:
            y = self.bn(y)
        if self.relu:
            y = F.relu(y, inplace=True)
        return y


class DeConv2dFuse(nn.Module):
    """上采样融合模块"""
    
    def __init__(self, in_channels, out_channels, kernel_size, relu=True, bn=True,
                 bn_momentum=0.1):
        super(DeConv2dFuse, self).__init__()

        self.deconv = Deconv2d(in_channels, out_channels, kernel_size, stride=2, padding=1, output_padding=1,
                               bn=True, relu=relu, bn_momentum=bn_momentum)

        self.conv = Conv2d(2*out_channels, out_channels, kernel_size, stride=1, padding=1,
                           bn=bn, relu=relu, bn_momentum=bn_momentum)

    def forward(self, x_pre, x):
        x = self.deconv(x)
        x = torch.cat((x, x_pre), dim=1)
        x = self.conv(x)
        return x


class SparseFeatureNet(nn.Module):
    """
    稀疏CNN特征提取网络
    
    支持两种模式：
    1. 标准Dense模式：用于MVS特征提取
    2. 稀疏Sparse模式：用于MAE掩码特征提取
    
    论文方法：
    - FPN结构作为共享特征提取器
    - 稀疏卷积保留掩码区域信息
    - 多尺度输出用于级联MVS和MAE解码
    """
    
    def __init__(self, base_channels=8, num_stage=3, stride=4, arch_mode="fpn", 
                 use_sparse_conv=False):
        super(SparseFeatureNet, self).__init__()
        
        assert arch_mode in ["unet", "fpn"], f"mode must be in 'unet' or 'fpn', got: {arch_mode}"
        
        self.arch_mode = arch_mode
        self.stride = stride
        self.base_channels = base_channels
        self.num_stage = num_stage
        self.use_sparse_conv = use_sparse_conv
        self.collect_visuals = False
        self.visual_cache = {}
        
        print(f"************* SparseFeatureNet *************")
        print(f"  - arch mode: {arch_mode}")
        print(f"  - use_sparse_conv: {use_sparse_conv}")
        print(f"  - base_channels: {base_channels}")
        print(f"*********************************************")
        
        # 特征提取层
        if use_sparse_conv:
            # 使用稀疏卷积
            self.conv0 = nn.Sequential(
                SparseConvBlockV2(3, base_channels, 3, 1, padding=1),
                SparseConvBlockV2(base_channels, base_channels, 3, 1, padding=1),
            )
            
            self.conv1 = nn.Sequential(
                SparseConvBlockV2(base_channels, base_channels * 2, 5, stride=2, padding=2),
                SparseConvBlockV2(base_channels * 2, base_channels * 2, 3, 1, padding=1),
                SparseConvBlockV2(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            )
            
            self.conv2 = nn.Sequential(
                SparseConvBlockV2(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
                SparseConvBlockV2(base_channels * 4, base_channels * 4, 3, 1, padding=1),
                SparseConvBlockV2(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            )
        else:
            # 标准卷积
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
        
        # FPN或UNet上采样结构
        if arch_mode == 'unet':
            if num_stage == 3:
                self.deconv1 = DeConv2dFuse(base_channels * 4, base_channels * 2, 3)
                self.deconv2 = DeConv2dFuse(base_channels * 2, base_channels, 3)

                self.out2 = nn.Conv2d(base_channels * 2, base_channels * 2, 1, bias=False)
                self.out3 = nn.Conv2d(base_channels, base_channels, 1, bias=False)
                self.out_channels.append(2 * base_channels)
                self.out_channels.append(base_channels)
                
        elif arch_mode == "fpn":
            final_chs = base_channels * 4
            if num_stage == 3:
                self.inner1 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)
                self.inner2 = nn.Conv2d(base_channels * 1, final_chs, 1, bias=True)

                self.out2 = nn.Conv2d(final_chs, base_channels * 2, 3, padding=1, bias=False)
                self.out3 = nn.Conv2d(final_chs, base_channels, 3, padding=1, bias=False)
                self.out_channels.append(base_channels * 2)
                self.out_channels.append(base_channels)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: 输入图像 [B, 3, H, W]
            mask: 二进制掩码 [B, 1, H, W]（可选，用于稀疏模式）
            
        Returns:
            outputs: 多尺度特征字典
        """
        if mask is None:
            mask = torch.ones_like(x[:, :1, :, :])
            
        current_mask = mask
        if self.collect_visuals:
            self.visual_cache = {}
        captured_sparse_layer = False
        
        # Stage 0
        for layer in self.conv0:
            if isinstance(layer, SparseConvBlockV2):
                if self.collect_visuals and not captured_sparse_layer:
                    self.visual_cache["sparse_input"] = x.detach()
                    self.visual_cache["sparse_input_mask"] = current_mask.detach()
                x, current_mask = layer(x, current_mask)
                if self.collect_visuals and not captured_sparse_layer:
                    self.visual_cache["sparse_output"] = x.detach()
                    if current_mask is not None:
                        self.visual_cache["sparse_output_mask"] = current_mask.detach()
                    captured_sparse_layer = True
            else:
                x = layer(x)
        conv0 = x
        
        # Stage 1
        for layer in self.conv1:
            if isinstance(layer, SparseConvBlockV2):
                x, current_mask = layer(x, current_mask)
            else:
                x = layer(x)
        conv1 = x
        
        # Stage 2 (最终特征)
        for layer in self.conv2:
            if isinstance(layer, SparseConvBlockV2):
                x, current_mask = layer(x, current_mask)
            else:
                x = layer(x)
        conv2 = x
        
        intra_feat = conv2
        outputs = {}
        
        # Stage 1 输出 (1/4分辨率)
        out = self.out1(intra_feat)
        outputs["stage1"] = out
        
        # Stage 2 输出 (1/2分辨率)
        if self.arch_mode == "unet":
            if self.num_stage >= 2:
                intra_feat = self.deconv1(conv1, intra_feat)
                out = self.out2(intra_feat)
                outputs["stage2"] = out
                
        elif self.arch_mode == "fpn":
            if self.num_stage >= 2:
                intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner1(conv1)
                out = self.out2(intra_feat)
                outputs["stage2"] = out
        
        # Stage 3 输出 (原始分辨率)
        if self.num_stage >= 3:
            if self.arch_mode == "unet":
                intra_feat = self.deconv2(conv0, intra_feat)
                out = self.out3(intra_feat)
                outputs["stage3"] = out
                
            elif self.arch_mode == "fpn":
                intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner2(conv0)
                out = self.out3(intra_feat)
                outputs["stage3"] = out
        
        return outputs


# 为了兼容性，保留原名
FeatureNet = SparseFeatureNet


if __name__ == "__main__":
    print("Testing SparseFeatureNet...")
    
    # 测试标准模式
    model_dense = SparseFeatureNet(base_channels=8, num_stage=3, arch_mode="fpn", use_sparse_conv=False)
    x = torch.randn(2, 3, 256, 320)
    outputs = model_dense(x)
    print(f"Dense mode:")
    for k, v in outputs.items():
        print(f"  {k}: {v.shape}")
    
    # 测试稀疏模式
    model_sparse = SparseFeatureNet(base_channels=8, num_stage=3, arch_mode="fpn", use_sparse_conv=True)
    mask = torch.ones(2, 1, 256, 320)
    mask[:, :, 64:192, 80:240] = 0  # 掩码中心区域
    outputs_sparse = model_sparse(x, mask)
    print(f"\nSparse mode (mask ratio: {(mask==0).float().mean():.2%}):")
    for k, v in outputs_sparse.items():
        print(f"  {k}: {v.shape}")
    
    print("\nAll SparseFeatureNet tests passed!")
