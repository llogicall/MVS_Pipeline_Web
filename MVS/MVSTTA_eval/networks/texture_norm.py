import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torchvision.utils import save_image
import sys
sys.path.append("..")
# from kornia.filters import gabor_kernel



class TextureAdaptiveNorm(nn.Module):
    """
    基于纹理的自适应归一化层（非学习方式）
    根据图像的局部纹理强度动态调整特征的分布
    """
    def __init__(self, num_features, window_size=9):
        super(TextureAdaptiveNorm, self).__init__()
        self.num_features = num_features
        self.window_size = window_size
        # 可学习的纹理权重和偏移
        self.texture_weight = nn.Parameter(torch.ones(1, num_features, 1, 1) * 0.5)
        self.texture_bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        # Sobel算子用于计算梯度
        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1],
                                                     [-2, 0, 2],
                                                     [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([[-1, -2, -1],
                                                     [ 0,  0,  0],
                                                     [ 1,  2,  1]], dtype=torch.float32).view(1, 1, 3, 3))
        
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
            # texture_map = self.extract_texture_map(img)
            # texture_map = self.extract_texture_map_laplacian(img)
            # texture_map = self.extract_texture_map_gabor(img)
            texture_map = self.extract_texture_map_lbp(img)
            # texture_map = self.extract_texture_map_fft(img)

        elif texture_map is None:
            # 如果都没有提供，使用简单的方差作为纹理估计
            texture_map = torch.var(feature, dim=1, keepdim=True)
        
        # 纹理强度归一化到 [0, 1] 范围
        texture_map = (texture_map - texture_map.min()) / (texture_map.max() - texture_map.min() + 1e-8)
        
        # 计算特征的通道统计量
        mean = feature.mean(dim=[2, 3], keepdim=True)  # [B, C, 1, 1]
        std = feature.std(dim=[2, 3], keepdim=True) + 1e-8  # [B, C, 1, 1]
        
        # 基础归一化
        normalized = (feature - mean) / std
        
        # # 根据纹理强度动态调整   自适应调制
        # # 纹理强的区域：保持更高的对比度（减少归一化程度）
        # # 纹理弱的区域：更强的归一化（增强对比度）
        
        # # 自适应缩放因子：纹理越强，scale越接近1（保持原特征）
        # scale = 0.5 + 0.5 * texture_map  # [B, 1, H, W] 范围 [0.5, 1.0]
        # # 自适应偏移：纹理越强，偏移越小
        # offset = (1.0 - texture_map) * mean  # [B, C, H, W]
        
        # 可学习的纹理调制
        # texture_map = texture_map.unsqueeze(1)  # [B,1,H,W]
        scale = torch.sigmoid(self.texture_weight * texture_map)  # [B,C,H,W]
        offset = self.texture_bias * (1.0 - texture_map)

        # 根据特征图的尺寸将scale进行缩放的过程
        scale = F.interpolate(scale, [H,W], mode='bilinear')
        offset = F.interpolate(offset, [H,W], mode='bilinear')
        # 应用自适应调制
        normalized_feature = normalized * scale + offset
        
        return normalized_feature
    
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


    def extract_texture_map_laplacian(self, img):
        """拉普拉斯算子方法（提取边缘和纹理）"""
        gray = 0.299 * img[:, 0:1, :, :] + 0.587 * img[:, 1:2, :, :] + 0.114 * img[:, 2:3, :, :]
        laplacian_kernel = torch.tensor([[0, 1, 0],
                                        [1, -4, 1],
                                        [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3).to(img.device)
        laplacian = F.conv2d(gray, laplacian_kernel, padding=1)
        texture_strength = torch.abs(laplacian)
        
        kernel_size = self.window_size
        padding = kernel_size // 2
        texture_strength = F.avg_pool2d(
            F.pad(texture_strength, (padding, padding, padding, padding), mode='reflect'),
            kernel_size=kernel_size, stride=1
        )
        return texture_strength

    def extract_texture_map_gabor(self, img):
        """Gabor滤波器方法（多尺度多方向纹理）"""
        gray = 0.299 * img[:, 0:1, :, :] + 0.587 * img[:, 1:2, :, :] + 0.114 * img[:, 2:3, :, :]
        
        # 定义多个方向和尺度的Gabor核
        orientations = [0, torch.pi/4, torch.pi/2, 3*torch.pi/4]
        scales = [2, 4, 8]
        
        texture_maps = []
        for theta in orientations:
            for sigma in scales:
                # kernel = gabor_kernel(theta, sigma=sigma, frequency=0.1).to(img.device)
                kernel = 1

                filtered = F.conv2d(gray, kernel, padding=sigma)
                texture_maps.append(torch.abs(filtered))
        
        # 融合多尺度多方向纹理
        texture_strength = torch.mean(torch.cat(texture_maps, dim=1), dim=1, keepdim=True)
        
        kernel_size = self.window_size
        padding = kernel_size // 2
        texture_strength = F.avg_pool2d(
            F.pad(texture_strength, (padding, padding, padding, padding), mode='reflect'),
            kernel_size=kernel_size, stride=1
        )
        return texture_strength

    def extract_texture_map_lbp(self, img, radius=1, n_points=8):
        """局部二值模式LBP方法（纹理统计特征）"""
        gray = 0.299 * img[:, 0:1, :, :] + 0.587 * img[:, 1:2, :, :] + 0.114 * img[:, 2:3, :, :]
        
        # 获取邻域像素
        h, w = gray.shape[2:]
        offsets = []
        for i in range(n_points):
            angle = 2.0 * torch.pi * i / n_points
            angle = torch.tensor(angle)
            dx = int(torch.round(radius * torch.cos(angle)))
            dy = int(torch.round(radius * torch.sin(angle)))
            offsets.append((dx, dy))
        
        # 计算LBP
        lbp = torch.zeros_like(gray)
        for dx, dy in offsets:
            if dx == 0 and dy == 0:
                continue
            shifted = F.pad(gray, (abs(dx), abs(dx), abs(dy), abs(dy)), mode='reflect')
            shifted = shifted[:, :, abs(dy):abs(dy)+h, abs(dx):abs(dx)+w]
            lbp += (shifted >= gray).float() * (2 ** offsets.index((dx, dy)))
        
        # 计算LBP直方图作为纹理强度
        texture_strength = F.avg_pool2d(lbp, kernel_size=3, stride=1, padding=1)
        
        kernel_size = self.window_size
        padding = kernel_size // 2
        texture_strength = F.avg_pool2d(
            F.pad(texture_strength, (padding, padding, padding, padding), mode='reflect'),
            kernel_size=kernel_size, stride=1
        )
        return texture_strength

    def extract_texture_map_fft(self, img):
        """傅里叶变换方法（全局纹理特征）"""
        gray = 0.299 * img[:, 0:1, :, :] + 0.587 * img[:, 1:2, :, :] + 0.114 * img[:, 2:3, :, :]
        
        # 计算傅里叶变换
        fft = torch.fft.fft2(gray)
        fft_shift = torch.fft.fftshift(fft)
        magnitude = torch.log(torch.abs(fft_shift) + 1e-8)
        
        # 提取高频分量作为纹理强度
        h, w = magnitude.shape[2:]
        center_h, center_w = h // 2, w // 2
        radius = min(h, w) // 8
        
        # 创建高频掩码
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        mask = ((y - center_h)**2 + (x - center_w)**2) > radius**2
        mask = mask.to(img.device).float()
        
        texture_strength = magnitude * mask
        
        kernel_size = self.window_size
        padding = kernel_size // 2
        texture_strength = F.avg_pool2d(
            F.pad(texture_strength, (padding, padding, padding, padding), mode='reflect'),
            kernel_size=kernel_size, stride=1
        )
        return texture_strength








