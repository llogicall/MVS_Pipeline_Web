"""
基于视差动态调整的深度采样策略模块
实现根据场景多视角图像之间的视差情况实时调整假设深度采样策略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def estimate_disparity(features, proj_matrices, min_depth, max_depth, num_samples=32):
    """
    使用特征快速估计视差图
    Args:
        features: list of features from different views [ref_feature, src_feature1, src_feature2, ...]
        proj_matrices: list of projection matrices [ref_proj, src_proj1, src_proj2, ...]
        min_depth: minimum depth value
        max_depth: maximum depth value
        num_samples: number of depth samples for disparity estimation
    Returns:
        disparity_map: (B, H, W) 视差图
        confidence_map: (B, H, W) 置信度图
    """
    from .module import homo_warping, depth_regression
    
    B, C, H, W = features[0].shape
    device = features[0].device
    dtype = features[0].dtype
    
    # 快速生成深度采样点用于视差估计
    depth_values = torch.linspace(min_depth, max_depth, num_samples, device=device, dtype=dtype)
    depth_values = depth_values.view(1, num_samples, 1, 1).repeat(B, 1, H, W)
    
    # 使用简单代价体计算视差
    ref_feature, src_features = features[0], features[1:]
    ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]
    
    # 构建初始代价值
    ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_samples, 1, 1)
    volume_sum = ref_volume
    volume_sq_sum = ref_volume ** 2
    del ref_volume
    
    # 简单特征匹配计算视差（仅使用第一个源视图以加速）
    src_fea, src_proj = src_features[0], src_projs[0]
    src_proj_new = src_proj[:, 0].clone()
    src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
    ref_proj_new = ref_proj[:, 0].clone()
    ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
    
    warped_volume = homo_warping(src_fea, src_proj_new, ref_proj_new, depth_values)
    
    volume_sum = volume_sum + warped_volume
    volume_sq_sum = volume_sq_sum + warped_volume ** 2
    del warped_volume
    
    # 使用方差作为代价值
    volume_variance = volume_sq_sum.div_(2).sub_(volume_sum.div_(2).pow_(2))
    
    # 简单代价正则化
    cost_reg = F.avg_pool3d(volume_variance, (3, 1, 1), stride=1, padding=(1, 0, 0))
    prob_volume = F.softmax(cost_reg.squeeze(1), dim=1)
    
    # 估计初始深度并转换为视差
    estimated_depth = depth_regression(prob_volume, depth_values)
    # 视差 = 基线距离 / 深度
    # 近似：假设基线约等于焦距，这里简化为深度的倒数乘以缩放因子
    baseline_factor = max_depth / 2
    disparity_map = baseline_factor / estimated_depth
    
    # 计算置信度：使用响应锐度
    with torch.no_grad():
        prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), 
                                           (4, 1, 1), stride=1, padding=0).squeeze(1)
        depth_index = depth_regression(prob_volume, 
                                     depth_values=torch.arange(num_samples, device=device, dtype=dtype)).long()
        depth_index = depth_index.clamp(min=0, max=num_samples-1)
        confidence_map = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)
    
    return disparity_map, confidence_map


def compute_adaptive_depth_interval(disparity_map, confidence_map, base_interval, cur_depth):
    """
    根据视差和置信度动态计算深度间隔
    
    核心策略：
    - 视差大的区域（近景）：深度间隔小，使用密集采样
    - 视差小的区域（远景）：深度间隔大，使用稀疏采样
    - 置信度高的区域：可以适当增大间隔，因为匹配已经准确
    - 置信度低的区域：需要减小间隔，提高采样密度
    
    Args:
        disparity_map: (B, H, W) 视差图
        confidence_map: (B, H, W) 置信度图
        base_interval: float 基础深度间隔
        cur_depth: (B, H, W) 当前深度图
    Returns:
        adaptive_interval: (B, H, W) 自适应深度间隔
    """
    B, H, W = disparity_map.shape
    device = disparity_map.device
    dtype = disparity_map.dtype
    
    # 策略1: 基于视差的间隔调整
    # 归一化视差到[0, 1]范围
    disparity_normalized = (disparity_map - disparity_map.min()) / (disparity_map.max() - disparity_map.min() + 1e-8)
    
    # 视差大的区域使用小间隔，视差小的区域使用大间隔
    # 使用指数函数实现非线性调整
    disparity_factor = torch.exp(-disparity_normalized * 2)  # 视差大→因子小
    
    # 策略2: 基于置信度的间隔调整
    # 归一化置信度到[0, 1]范围
    confidence_normalized = (confidence_map - confidence_map.min()) / (confidence_map.max() - confidence_map.min() + 1e-8)
    
    # 置信度高的区域可以增大间隔
    confidence_factor = 1.0 + 0.5 * confidence_normalized  # 最大增大1.5倍
    
    # 策略3: 基于深度的间隔调整（考虑透视投影）
    # 深度越远，深度间隔应该越大
    depth_normalized = (cur_depth - cur_depth.min()) / (cur_depth.max() - cur_depth.min() + 1e-8)
    depth_factor = 1.0 + depth_normalized
    
    # 综合计算自适应间隔
    adaptive_interval = base_interval * disparity_factor * confidence_factor * depth_factor
    
    # 限制调整范围，防止过大或过小
    adaptive_interval = torch.clamp(adaptive_interval, min=base_interval * 0.25, max=base_interval * 4.0)
    
    return adaptive_interval


def get_adaptive_depth_range_samples(cur_depth, ndepth, depth_inteval_pixel, shape, max_depth=192.0, min_depth=0.0, 
                                    disparity_map=None, confidence_map=None):
    """
    自适应深度范围采样，支持基于视差的不均匀分布
    
    实现多种不均匀分布策略：
    1. 线性非均匀：在深度范围两端密集采样，中间稀疏
    2. 指数分布：在近景密集，远景稀疏
    3. 置信度加权：在置信度高的区域密集采样
    
    Args:
        cur_depth: (B, H, W) 当前深度图
        ndepth: int 采样深度数量
        depth_inteval_pixel: float 基础深度间隔
        shape: (B, H, W) 输入形状
        max_depth: float 最大深度
        min_depth: float 最小深度
        disparity_map: (B, H, W) 可选，视差图
        confidence_map: (B, H, W) 可选，置信度图
    Returns:
        depth_range_samples: (B, D, H, W) 深度假设平面
    """
    B, H, W = cur_depth.shape
    device = cur_depth.device
    dtype = cur_depth.dtype
    
    # 计算自适应深度间隔
    if disparity_map is not None and confidence_map is not None:
        adaptive_intervals = compute_adaptive_depth_interval(disparity_map, confidence_map, 
                                                           depth_inteval_pixel, cur_depth)
    else:
        adaptive_intervals = torch.full_like(cur_depth, depth_inteval_pixel)
    
    # 计算自适应深度范围
    # 视差大的区域（近景）使用较小的深度范围
    if disparity_map is not None:
        disparity_normalized = (disparity_map - disparity_map.mean()) / (disparity_map.std() + 1e-8)
        range_factor = torch.exp(-disparity_normalized).clamp(min=0.5, max=2.0)
    else:
        range_factor = 1.0
    
    cur_depth_min = (cur_depth - ndepth / 2 * adaptive_intervals * range_factor).clamp(min=min_depth)
    cur_depth_max = (cur_depth + ndepth / 2 * adaptive_intervals * range_factor).clamp(max=max_depth)
    
    # 实现不均匀深度采样
    if confidence_map is not None:
        # 策略：在置信度高的区域密集采样
        confidence_normalized = (confidence_map - confidence_map.min()) / (confidence_map.max() - confidence_map.min() + 1e-8)
        
        # 使用非线性映射实现不均匀分布
        # 基本思想：在置信度高的区域，采样点更密集
        positions = torch.linspace(-0.5, 0.5, steps=ndepth, device=device, dtype=dtype)
        positions = positions.view(1, ndepth, 1, 1)
        
        # 根据置信度调整采样位置分布
        # 置信度高的区域→采样点向中心聚集（更精确）
        conf_adjustment = 1.0 - 0.3 * confidence_normalized.unsqueeze(1)  # 最大调整30%
        adjusted_positions = positions * conf_adjustment.unsqueeze(2).unsqueeze(3)
        
        # 根据调整后的位置计算深度
        depth_range_samples = cur_depth.unsqueeze(1) + adjusted_positions * (cur_depth_max - cur_depth_min).unsqueeze(1)
    else:
        # 使用不均匀线性分布（两端密集，中间稀疏）
        # 使用二次函数实现：x^2，使得采样点在两端更密集
        positions = torch.linspace(-1, 1, steps=ndepth, device=device, dtype=dtype)
        # 使用sigmoid函数进行平滑调整
        positions = torch.sign(positions) * torch.pow(torch.abs(positions), 0.7)
        positions = positions / positions.max() * 0.5
        positions = positions.view(1, ndepth, 1, 1)
        
        depth_range_samples = cur_depth.unsqueeze(1) + positions * (cur_depth_max - cur_depth_min).unsqueeze(1)
    
    # 对深度假设进行排序（确保每个像素的深度假设是递增的）
    depth_range_samples, _ = torch.sort(depth_range_samples, dim=1)
    
    return depth_range_samples


class AdaptiveDepthSampler(nn.Module):
    """
    自适应深度采样器类
    集成了视差估计、间隔计算和深度采样功能
    """
    def __init__(self, base_interval=1.0, num_disparity_samples=32):
        super(AdaptiveDepthSampler, self).__init__()
        self.base_interval = base_interval
        self.num_disparity_samples = num_disparity_samples
    
    def forward(self, features, proj_matrices, cur_depth, ndepth, depth_inteval_pixel, 
                shape, max_depth=192.0, min_depth=0.0):
        """
        执行自适应深度采样
        
        Args:
            features: list of features from different views
            proj_matrices: list of projection matrices
            cur_depth: (B, H, W) 当前深度图
            ndepth: int 采样深度数量
            depth_inteval_pixel: float 基础深度间隔
            shape: (B, H, W) 输入形状
            max_depth: float 最大深度
            min_depth: float 最小深度
        Returns:
            depth_range_samples: (B, D, H, W) 自适应深度假设平面
            disparity_map: (B, H, W) 视差图（用于调试和可视化）
            confidence_map: (B, H, W) 置信度图（用于调试和可视化）
        """
        # 估计视差和置信度
        disparity_map, confidence_map = estimate_disparity(
            features, proj_matrices, min_depth, max_depth, self.num_disparity_samples
        )
        
        # 调整视差图和置信度图到与cur_depth相同的尺寸
        if disparity_map.shape[1:] != cur_depth.shape[1:]:
            disparity_map = F.interpolate(disparity_map.unsqueeze(1), size=cur_depth.shape[1:], 
                                         mode='bilinear', align_corners=False).squeeze(1)
            confidence_map = F.interpolate(confidence_map.unsqueeze(1), size=cur_depth.shape[1:], 
                                         mode='bilinear', align_corners=False).squeeze(1)
        
        # 生成自适应深度采样
        depth_range_samples = get_adaptive_depth_range_samples(
            cur_depth, ndepth, depth_inteval_pixel, shape, max_depth, min_depth,
            disparity_map, confidence_map
        )
        
        return depth_range_samples, disparity_map, confidence_map


def integrate_adaptive_sampling(module_path='module.py'):
    """
    集成函数：展示如何将自适应深度采样集成到现有代码中
    """
    integration_guide = """
    # 在 cas_mvsnet.py 的 CascadeMVSNet 类中集成自适应深度采样
    
    1. 在 __init__ 中添加自适应采样器：
        from adaptive_depth_sampling import AdaptiveDepthSampler
        self.adaptive_sampler = AdaptiveDepthSampler(base_interval=depth_interval, num_disparity_samples=32)
    
    2. 在 forward 方法中修改深度采样部分：
        
        # 原代码：
        # depth_range_samples = get_depth_range_samples(...)
        
        # 修改为：
        if self.training:  # 仅在训练时使用自适应采样
            depth_range_samples, disparity_map, confidence_map = self.adaptive_sampler(
                features_stage, proj_matrices_stage, cur_depth, self.ndepths[stage_idx],
                self.depth_interals_ratio[stage_idx] * depth_interval,
                [img.shape[0], img.shape[2], img.shape[3]], depth_max, depth_min
            )
        else:
            # 推理时使用固定采样以加速
            depth_range_samples = get_depth_range_samples(...)
    
    3. 在损失函数中可以添加视差一致性损失（可选）：
        # 利用生成的 disparity_map 和 confidence_map 增强训练稳定性
    """
    
    return integration_guide


if __name__ == "__main__":
    # 测试代码
    print("自适应深度采样策略模块")
    print("=" * 50)
    print("\n核心特性：")
    print("1. 基于视差的动态间隔调整")
    print("2. 置信度加权的不均匀采样")
    print("3. 透视投影感知的深度范围自适应")
    print("4. 训练推理模式切换")
    
    print("\n集成指南：")
    print(integrate_adaptive_sampling())
    
    # 简单功能测试
    B, H, W, D = 2, 64, 64, 48
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试数据
    cur_depth = torch.linspace(10, 100, steps=H*W).view(B, H, W).to(device)
    disparity_map = torch.rand(B, H, W).to(device) * 5.0
    confidence_map = torch.rand(B, H, W).to(device)
    
    # 测试自适应采样
    depth_samples = get_adaptive_depth_range_samples(
        cur_depth, D, 1.0, [B, H, W], 192.0, 0.0, disparity_map, confidence_map
    )
    
    print(f"\n测试结果：")
    print(f"输入深度范围: [{cur_depth.min():.2f}, {cur_depth.max():.2f}]")
    print(f"输出深度采样形状: {depth_samples.shape}")
    print(f"采样范围: [{depth_samples.min():.2f}, {depth_samples.max():.2f}]")
    print(f"视差图形状: {disparity_map.shape}")
    print(f"置信度图形状: {confidence_map.shape}")
    
    print("\n✓ 自适应深度采样策略模块加载成功！")
