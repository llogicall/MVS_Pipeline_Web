import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torchvision.utils import save_image
import sys
sys.path.append("..")

def extract_uncertainty_from_reprojection(reprojection_volume, method='min_error', 
                                          valid_mask_threshold=1e4):
    """
    从重投影体积中提取每个像素的不确定性度量
    
    Args:
        reprojection_volume: [B, H, W, 1, N_src] 多视角重投影损失体积
        method: 不确定性提取方法
            - 'min_error': 最小重投影误差（对遮挡鲁棒）
            - 'variance': 重投影误差方差（更敏感）
            - 'topk_mean': top-k最小误差的均值
        valid_mask_threshold: 无效区域的阈值（你的实现中用1e4标记）
    
    Returns:
        uncertainty: [B, H, W] 不确定性图，值域[0,1]，越大表示越不确定
    """
    # squeeze掉维度3 [B, H, W, N_src]
    reproj_vol = reprojection_volume.squeeze(3)
    B, H, W, N_src = reproj_vol.shape
    
    # 创建有效区域掩码
    valid_mask = reproj_vol < (valid_mask_threshold * 0.9)  # [B, H, W, N_src]
    
    if method == 'min_error':
        # 方法1：最小重投影误差（对遮挡鲁棒）
        # 对于每个像素，取所有源视角中的最小重投影误差
        # 将无效区域设为大值，避免影响min计算
        reproj_vol_masked = torch.where(valid_mask, reproj_vol, 
                                        torch.full_like(reproj_vol, 1e6))
        min_error, _ = torch.min(reproj_vol_masked, dim=-1)  # [B, H, W]
        
        # 归一化到[0,1]范围
        # 使用sigmoid函数进行平滑归一化
        uncertainty = torch.sigmoid(min_error - 2.0)  # 中心点在2.0附近
        
    elif method == 'variance':
        # 方法2：重投影误差方差（更敏感于视角间的不一致）
        # 只计算有效区域的方差
        reproj_vol_masked = torch.where(valid_mask, reproj_vol, torch.zeros_like(reproj_vol))
        valid_count = valid_mask.float().sum(dim=-1).clamp(min=1)  # [B, H, W]
        
        mean_error = reproj_vol_masked.sum(dim=-1) / valid_count
        variance = ((reproj_vol_masked - mean_error.unsqueeze(-1)) ** 2 * valid_mask.float()).sum(dim=-1) / valid_count
        
        uncertainty = torch.sigmoid(variance - 1.0)
        
    elif method == 'topk_mean':
        # 方法3：top-k最小误差的均值（平衡鲁棒性和敏感性）
        k = min(2, N_src)
        reproj_vol_masked = torch.where(valid_mask, reproj_vol, 
                                        torch.full_like(reproj_vol, 1e6))
        topk_errors, _ = torch.topk(reproj_vol_masked, k=k, dim=-1, largest=False)
        topk_mean = topk_errors.mean(dim=-1)
        
        uncertainty = torch.sigmoid(topk_mean - 2.0)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return uncertainty


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


def get_reprojection_depth_range_samples(
    cur_depth,                    # [B, H, W] 当前阶段估计的深度
    prob_volume,                  # [B, D_prev, H, W] 上一阶段的概率体积
    prev_depth_hypotheses,        # [B, D_prev, H, W] 上一阶段的深度假设
    reprojection_volume,          # [B, H, W, 1, N_src] 重投影体积
    ndepth,                       # 当前阶段深度假设数量
    max_depth=192.0,
    min_depth=0.0,
    uncertainty_method='min_error',
    eta=1.0,                      # 基础范围扩展系数
    range_expansion_factor=1.5,   # 不确定性驱动的范围扩展系数
    density_control_factor=2.0,   # 密度控制系数（越大，边缘越稀疏）
    center_density_factor=0.5,    # 中心密度增强系数（越小，中心越密集）
):
    """
    自适应假设深度采样
    
    核心思想：
    1. 从重投影体积提取几何不确定性
    2. 根据不确定性自适应调整深度搜索范围
    3. 采用非均匀递增步长采样：中心密集、边缘稀疏
    
    Args:
        cur_depth: [B, H, W] 当前阶段估计的深度图
        prob_volume: [B, D_prev, H, W] 上一阶段的概率体积
        prev_depth_hypotheses: [B, D_prev, H, W] 上一阶段的深度假设
        reprojection_volume: [B, H, W, 1, N_src] 重投影体积
        ndepth: 当前阶段深度假设数量
        max_depth, min_depth: 深度范围限制
        uncertainty_method: 不确定性提取方法
        eta: 基础范围扩展系数
        range_expansion_factor: 不确定性驱动的范围扩展系数
        density_control_factor: 密度控制系数
        center_density_factor: 中心密度增强系数
    
    Returns:
        depth_range_samples: [B, D, H, W] 新生成的深度假设
    """
    B, H, W = cur_depth.shape
    device = cur_depth.device
    dtype = cur_depth.dtype
    
    # ============== 步骤1：提取几何不确定性 ==============
    uncertainty = extract_uncertainty_from_reprojection(
        reprojection_volume, 
        method=uncertainty_method
    )  # [B, H, W]
    
    # ============== 步骤2：估计深度分布的统计量 ==============
    # 从概率体积计算深度方差（作为基础范围估计）
    cur_depth_expanded = cur_depth.unsqueeze(1)  # [B, 1, H, W]
    
    # 确保prev_depth_hypotheses尺寸匹配
    if prev_depth_hypotheses.shape[2:] != cur_depth.shape[1:]:
        prev_depth_hypotheses = F.interpolate(
            prev_depth_hypotheses, 
            size=(H, W), 
            mode='bilinear',
            align_corners=True
        )
    
    # 计算深度假设的方差
    diff = prev_depth_hypotheses - cur_depth_expanded  # [B, D_prev, H, W]
    variance = torch.sum(prob_volume * diff.pow(2), dim=1)  # [B, H, W]
    std_dev = torch.sqrt(variance + 1e-8)  # [B, H, W]
    
    # ============== 步骤3：自适应深度范围估计 ==============
    # 基础范围：基于概率体积方差
    base_range = eta * std_dev
    
    # 不确定性自适应范围调制
    # 高不确定性 -> 扩大搜索范围
    # 低不确定性 -> 收缩搜索范围
    adaptive_range = base_range * (1.0 + uncertainty * range_expansion_factor)
    
    # 计算深度上下界
    depth_min = cur_depth - adaptive_range
    depth_max = cur_depth + adaptive_range
    
    # 确保在有效范围内
    depth_min = torch.clamp(depth_min, min=min_depth)
    depth_max = torch.clamp(depth_max, max=max_depth)
    
    # ============== 步骤4：非均匀递增步长采样 ==============
    # 核心创新：二次递增步长，实现中心密集、边缘稀疏
    
    # 准备索引
    # 假设ndepth为偶数，分为上下两部分
    half_ndepth = ndepth // 2
    
    # 生成相对索引：i = 0.5, 1.5, 2.5, ..., half_ndepth-0.5
    # 这样可以使采样点对称分布
    indices_up = torch.arange(0.5, half_ndepth, 1.0, device=device, dtype=dtype)  # [half_ndepth]
    indices_down = torch.arange(0.5, half_ndepth, 1.0, device=device, dtype=dtype)  # [half_ndepth]
    
    # 计算非均匀步长
    # 步长函数：delta_i = base_interval * (1 + alpha * i^2)
    # 其中 alpha 由不确定性控制
    
    # 基础步长（中心附近的步长）
    base_interval = (depth_max - depth_min) / (ndepth - 1)  # [B, H, W]
    
    # 不确定性越大，步长增长越快（边缘越稀疏）
    alpha = density_control_factor * uncertainty  # [B, H, W]
    
    # 中心密度增强：不确定性越小，中心越密集
    center_interval = base_interval * (center_density_factor + uncertainty * 0.5)  # [B, H, W]
    
    # 计算累积偏移量（非均匀采样）
    # 使用二次函数：offset_i = center_interval * i * (1 + alpha * i / half_ndepth)
    
    # 上半部分（深度增加方向）
    # 归一化索引
    norm_indices_up = indices_up / half_ndepth  # [half_ndepth], 范围[0, 1)
    
    # 计算每个点的偏移量
    # alpha扩展维度用于广播: [B, H, W] -> [B, 1, H, W]
    alpha_expanded = alpha.unsqueeze(1)  # [B, 1, H, W]
    center_interval_expanded = center_interval.unsqueeze(1)  # [B, 1, H, W]
    norm_indices_up_expanded = norm_indices_up.view(1, -1, 1, 1)  # [1, half_ndepth, 1, 1]
    
    # 二次递增步长
    offsets_up = center_interval_expanded * norm_indices_up_expanded * half_ndepth * \
                 (1.0 + alpha_expanded * norm_indices_up_expanded)  # [B, half_ndepth, H, W]
    
    # 下半部分（深度减小方向）
    norm_indices_down = indices_down / half_ndepth
    norm_indices_down_expanded = norm_indices_down.view(1, -1, 1, 1)
    offsets_down = center_interval_expanded * norm_indices_down_expanded * half_ndepth * \
                   (1.0 + alpha_expanded * norm_indices_down_expanded)  # [B, half_ndepth, H, W]
    
    # ============== 步骤5：生成深度假设 ==============
    # 中心深度
    cur_depth_expanded = cur_depth.unsqueeze(1)  # [B, 1, H, W]
    
    # 上半部分深度（大于中心）
    depth_up = cur_depth_expanded + offsets_up  # [B, half_ndepth, H, W]
    
    # 下半部分深度（小于中心）
    depth_down = cur_depth_expanded - offsets_down  # [B, half_ndepth, H, W]
    
    # 合并深度假设
    if ndepth % 2 == 1:  # 奇数个深度假设，包含中心点
        depth_range_samples = torch.cat([depth_down.flip(1), cur_depth_expanded, depth_up], dim=1)
    else:  # 偶数个深度假设，不包含中心点
        depth_range_samples = torch.cat([depth_down.flip(1), depth_up], dim=1)
    
    # ============== 步骤6：后处理 ==============
    # 确保深度值在有效范围内
    depth_range_samples = torch.clamp(depth_range_samples, min=min_depth, max=max_depth)
    
    # 对深度假设进行排序（确保递增）
    depth_range_samples, _ = torch.sort(depth_range_samples, dim=1)
    
    return depth_range_samples, uncertainty


def get_reprojection_depth_range_samples_v2(
    cur_depth,
    prob_volume,
    prev_depth_hypotheses,
    reprojection_volume,
    ndepth,
    max_depth=192.0,
    min_depth=0.0,
    uncertainty_method='min_error',
    eta=1.0,
    range_expansion_factor=1.5,    # 不确定性驱动的范围扩展系数
):
    """
    自适应假设深度采样 - 简化版本
    采用更直接的指数步长策略
    
    核心思想：
    从中心向两侧，步长按指数增长
    
    步长模式：delta, delta, delta*k, delta*k, delta*k^2, delta*k^2, ...
    其中 k > 1 是增长因子，由不确定性控制
    """
    B, H, W = cur_depth.shape
    device = cur_depth.device
    dtype = cur_depth.dtype
    
    # 步骤1：提取不确定性
    uncertainty = extract_uncertainty_from_reprojection(
        reprojection_volume
    )
    
    # 步骤2：估计基础范围
    cur_depth_expanded = cur_depth.unsqueeze(1)
    
    if prev_depth_hypotheses.shape[2:] != cur_depth.shape[1:]:
        prev_depth_hypotheses = F.interpolate(
            prev_depth_hypotheses,
            size=(H, W),
            mode='bilinear',
            align_corners=True
        )
    
    diff = prev_depth_hypotheses - cur_depth_expanded
    variance = torch.sum(prob_volume * diff.pow(2), dim=1)
    std_dev = torch.sqrt(variance + 1e-8)
    
    # 自适应范围
    adaptive_range = eta * std_dev * (1.0 + uncertainty * range_expansion_factor)
    depth_min = torch.clamp(cur_depth - adaptive_range, min=min_depth)
    depth_max = torch.clamp(cur_depth + adaptive_range, max=max_depth)
    
    # 步骤3：指数步长采样
    # 增长因子：不确定性越高，增长越快
    k = 1.0 + uncertainty * 0.5  # [B, H, W], 范围约[1.0, 1.5]
    
    # 生成累积步长
    half_ndepth = ndepth // 2
    
    # 步长序列：1, 1, k, k, k^2, k^2, k^3, k^3, ...
    # 对于第i个点（i从1开始），步长为 k^floor((i-1)/2)
    
    # 计算每个位置的增长指数
    exponents = torch.floor(torch.arange(0, half_ndepth, device=device, dtype=dtype) / 2)
    exponents = exponents.view(1, -1, 1, 1)  # [1, half_ndepth, 1, 1]
    
    # 扩展k用于广播
    k_expanded = k.unsqueeze(1)  # [B, 1, H, W]
    
    # 计算每个点的步长
    step_lengths = torch.pow(k_expanded, exponents)  # [B, half_ndepth, H, W]
    
    # 基础步长（最小步长）
    base_step = (depth_max - depth_min) / (ndepth * 0.7)  # 经验系数0.7
    base_step = base_step.unsqueeze(1)  # [B, 1, H, W]
    
    # 累积偏移量
    cumulative_offsets = torch.cumsum(step_lengths * base_step, dim=1)  # [B, half_ndepth, H, W]
    
    # 生成深度假设
    cur_depth_expanded = cur_depth.unsqueeze(1)  # [B, 1, H, W]
    
    # 第一个点就是中心
    depth_up = cur_depth_expanded + cumulative_offsets  # [B, half_ndepth, H, W]
    depth_down = cur_depth_expanded - cumulative_offsets  # [B, half_ndepth, H, W]
    
    # 合并
    if ndepth % 2 == 1:
        depth_range_samples = torch.cat([depth_down.flip(1), cur_depth_expanded, depth_up], dim=1)
    else:
        # 对于偶数情况，调整起始位置
        first_offset = step_lengths[:, 0:1, :, :] * base_step * 0.5
        # depth_up = cur_depth_expanded + first_offset + cumulative_offsets[:, :-1, :, :]   # 这里为啥会维度1会-1
        # depth_down = cur_depth_expanded - first_offset - cumulative_offsets[:, :-1, :, :]
        depth_up = cur_depth_expanded + first_offset + cumulative_offsets
        depth_down = cur_depth_expanded - first_offset - cumulative_offsets   
        depth_range_samples = torch.cat([depth_down.flip(1), depth_up], dim=1)
    
    # 后处理
    depth_range_samples = torch.clamp(depth_range_samples, min=min_depth, max=max_depth)
    depth_range_samples, _ = torch.sort(depth_range_samples, dim=1)
    
    return depth_range_samples, uncertainty


def get_reprojection_depth_range_samples_v3(
    cur_depth,              # [B, H, W] 当前阶段预测的深度图
    reprojection_volume,    # [B, H, W, 1, N_src] 重投影体积 (唯一的数据输入)
    ndepth,                 # int 当前阶段深度假设数量
    max_depth=192.0,        # float 场景最大深度
    min_depth=0.0,          # float 场景最小深度
    base_range_ratio=0.05,  # float 【新参数】基础搜索范围占总深度的比例 (如5%)
    range_expansion_factor=2.0, # float 不确定性驱动的范围拉伸系数
    uncertainty_method='min_error'
):
    """
    极简版：无概率体积依赖，纯靠重投影误差驱动的非均匀指数采样
    """
    B, H, W = cur_depth.shape
    device = cur_depth.device
    dtype = cur_depth.dtype

    # ============== 步骤1：提取几何不确定性 ==============
    uncertainty = extract_uncertainty_from_reprojection(
        reprojection_volume, method=uncertainty_method
    ) # [B, H, W], 值域 [0, 1]

    # ============== 步骤2：计算自适应深度边界 (极简逻辑) ==============
    # 摒弃了复杂的方差计算，直接使用全局深度跨度的一个固定比例作为基础半径
    depth_span = max_depth - min_depth
    base_radius = depth_span * base_range_ratio
    
    # 唯一的自适应性来源：不确定性拉伸
    # 不确定性为0时，搜索半径就是 base_radius；不确定性为1时，半径扩大 (1 + factor) 倍
    adaptive_range = base_radius * (1.0 + uncertainty * range_expansion_factor)
    
    depth_min = torch.clamp(cur_depth - adaptive_range, min=min_depth)
    depth_max = torch.clamp(cur_depth + adaptive_range, max=max_depth)

    # ============== 步骤3：阶梯状指数步长采样 (核心保持不变) ==============
    # 增长因子 k: [1.0, 1.5]
    k = 1.0 + uncertainty * 0.5 
    half_ndepth = ndepth // 2

    # 计算阶梯指数序列: [0, 0, 1, 1, 2, 2, ...]
    exponents = torch.floor(torch.arange(0, half_ndepth, device=device, dtype=dtype) / 2)
    exponents = exponents.view(1, -1, 1, 1) # [1, half_ndepth, 1, 1]
    k_expanded = k.unsqueeze(1)             # [B, 1, H, W]

    # 阶梯步长: [1, 1, k, k, k^2, k^2, ...]
    step_lengths = torch.pow(k_expanded, exponents) # [B, half_ndepth, H, W]

    # 基础步长（乘0.7是经验系数，补偿指数累加导致的总跨度缩小）
    base_step = (depth_max - depth_min) / (ndepth * 0.7)
    base_step = base_step.unsqueeze(1) # [B, 1, H, W]

    # 累积偏移量
    cumulative_offsets = torch.cumsum(step_lengths * base_step, dim=1) # [B, half_ndepth, H, W]
    cur_depth_expanded = cur_depth.unsqueeze(1) # [B, 1, H, W]

    # ============== 步骤4：组装深度假设 ==============
    if ndepth % 2 == 1:
        depth_up = cur_depth_expanded + cumulative_offsets
        depth_down = cur_depth_expanded - cumulative_offsets
        depth_range_samples = torch.cat([depth_down.flip(1), cur_depth_expanded, depth_up], dim=1)
    else:
        # 偶数个假设的完美对称平移
        first_offset = step_lengths[:, 0:1, :, :] * base_step * 0.5
        # depth_up = cur_depth_expanded + first_offset + cumulative_offsets[:, :-1, :, :]   # 这里为啥会维度1会-1
        # depth_down = cur_depth_expanded - first_offset - cumulative_offsets[:, :-1, :, :]
        depth_up = cur_depth_expanded + first_offset + cumulative_offsets
        depth_down = cur_depth_expanded - first_offset - cumulative_offsets   
        depth_range_samples = torch.cat([depth_down.flip(1), depth_up], dim=1)

    # 后处理：截断与严格排序
    depth_range_samples = torch.clamp(depth_range_samples, min=min_depth, max=max_depth)
    depth_range_samples, _ = torch.sort(depth_range_samples, dim=1)

    return depth_range_samples



def get_reprojection_depth_range_samples_v4(
    cur_depth,              
    reprojection_volume,    
    ndepth,                 
    max_depth=192.0,        
    min_depth=0.0,          
    # --- 解耦后的两组参数 ---
    base_range_ratio=0.05,        # 控制：绝对物理搜索范围
    range_expansion_factor=2.0,   # 控制：遇到遮挡时范围扩大多少
    density_skew_boost=4.0,       # 【核心新增】控制：不确定性对内部密度的扭曲程度
    uncertainty_method='min_error'
):
    B, H, W = cur_depth.shape
    device = cur_depth.device
    dtype = cur_depth.dtype

    # 1. 提取不确定性
    uncertainty = extract_uncertainty_from_reprojection(
        reprojection_volume, method=uncertainty_method
    ) # [B, H, W] in [0, 1]

    # ================= 第一层解耦：边界作用域 =================
    # 边界仅由场景尺度和不确定性决定，与内部形状无关
    base_radius = (max_depth - min_depth) * base_range_ratio
    adaptive_range = base_radius * (1.0 + uncertainty * range_expansion_factor)
    depth_min = torch.clamp(cur_depth - adaptive_range, min=min_depth)
    depth_max = torch.clamp(cur_depth + adaptive_range, max=max_depth)

    # ================= 第二层解耦：内部密度分布 =================
    half_ndepth = ndepth // 2
    
    # 【核心创新】：定义密度倾斜指数 gamma (γ)
    # γ = 1.0 代表绝对均匀分布
    # γ > 1.0 代表中心密集，边缘稀疏 (γ越大，中心越密，边缘越空旷)
    gamma = 1.0 + uncertainty * density_skew_boost  # 范围从 1.0 到 5.0

    # 生成归一化的等差序列索引 t (0到1之间)
    # 例如 half_ndepth=4 -> t = [0.125, 0.375, 0.625, 0.875]
    t = (torch.arange(0, half_ndepth, device=device, dtype=dtype) + 0.5) / half_ndepth
    t = t.view(1, -1, 1, 1) # [1, N_half, 1, 1]
    gamma_expanded = gamma.unsqueeze(1) # [B, 1, H, W]

    # 【数学核心】：使用幂律映射 f(t) = t^(1/γ) 重塑分布
    # 当 γ=1，f(t)=t (均匀)
    # 当 γ=5，f(t)=t^0.2 (这是一个极其平缓的曲线，前段极长，后段极短)
    mapped_t = torch.pow(t, gamma_expanded) # [B, N_half, H, W]

    # 计算相邻点的间距 (差分)
    # 为了保持对称，我们在前面补一个 0
    mapped_t_with_zero = torch.cat([torch.zeros_like(mapped_t[:, :1, :, :]), mapped_t], dim=1)
    step_weights = mapped_t_with_zero[:, 1:, :, :] - mapped_t_with_zero[:, :-1, :, :] # [B, N_half, H, W]

    # 将比例权重应用到真实的物理区间 [depth_min, depth_max] 上
    interval_length = (depth_max - depth_min).unsqueeze(1) # [B, 1, H, W]
    cumulative_offsets = torch.cumsum(step_weights * interval_length, dim=1) # [B, N_half, H, W]

    # ================= 组装与对称平移 =================
    cur_depth_expanded = cur_depth.unsqueeze(1) # [B, 1, H, W]

    if ndepth % 2 == 1:
        depth_up = cur_depth_expanded + cumulative_offsets
        depth_down = cur_depth_expanded - cumulative_offsets
        depth_range_samples = torch.cat([depth_down.flip(1), cur_depth_expanded, depth_up], dim=1)
    else:
        # 偶数完美对称平移（取第一段的半个步长作为起点）
        first_offset = step_weights[:, 0:1, :, :] * interval_length * 0.5
        preceding_offsets = torch.cat(
            [torch.zeros_like(cumulative_offsets[:, :1, :, :]), cumulative_offsets[:, :-1, :, :]],
            dim=1,
        )
        depth_up = cur_depth_expanded + first_offset + preceding_offsets
        depth_down = cur_depth_expanded - first_offset - preceding_offsets
        depth_range_samples = torch.cat([depth_down.flip(1), depth_up], dim=1)

    # 后处理
    depth_range_samples = torch.clamp(depth_range_samples, min=min_depth, max=max_depth)
    depth_range_samples, _ = torch.sort(depth_range_samples, dim=1)

    return depth_range_samples

def get_reprojection_depth_range_samples_v5(
    cur_depth,
    reprojection_volume,
    ndepth,
    max_depth=192.0,
    min_depth=0.0,
    base_range_ratio=0.04,
    range_expansion_factor=2.5,
    density_skew_boost=3.0,
    uncertainty_method='topk_mean'
):
    """
    v5: 默认用于 Web 集成的重投影引导假设深度重采样。

    在 v4 的“搜索边界”和“内部密度”解耦思想上，v5 使用 top-k
    重投影误差均值估计不确定性，降低单个遮挡源视角对采样范围的影响。
    """
    return get_reprojection_depth_range_samples_v4(
        cur_depth=cur_depth,
        reprojection_volume=reprojection_volume,
        ndepth=ndepth,
        max_depth=max_depth,
        min_depth=min_depth,
        base_range_ratio=base_range_ratio,
        range_expansion_factor=range_expansion_factor,
        density_skew_boost=density_skew_boost,
        uncertainty_method=uncertainty_method,
    )

def save_depth_samples_incremental(
    depth_range_samples: torch.Tensor, 
    h_indices: list, 
    w_indices: list, 
    file_path: str, 
    batch_idx: int = 0,
    precision: int = 4
):
    """
    从 [B, D, H, W] 的深度假设张量中提取指定像素的序列，增量式写入 txt。
    
    Args:
        depth_range_samples (torch.Tensor): 形状为 [B, D, H, W] 的深度假设值
        h_indices (list): 需要提取的像素行坐标列表，如 [10, 20]
        w_indices (list): 需要提取的像素列坐标列表，如 [15, 25]
        file_path (str): 保存的 txt 文件路径
        batch_idx (int): 从哪个 batch 提取数据，默认为 0
        precision (int): 保存浮点数的小数位数，默认为 4
    """
    if len(h_indices) != len(w_indices):
        raise ValueError("h_indices 和 w_indices 的长度必须一致")
    
    if depth_range_samples.dim() != 4:
        raise ValueError("输入张量形状必须是 [B, D, H, W]")
        
    # 1. 将指定 batch 的数据移至 CPU 并转为 numpy，此时形状为 [D, H, W]
    samples_np = depth_range_samples[batch_idx].detach().cpu().numpy()
    
    # 2. 以 'a' (append追加) 模式打开文件。
    # 如果文件不存在，Python 会自动创建；如果存在，光标会移到文件末尾，绝不会覆盖之前的内容。
    with open(file_path, 'a', encoding='utf-8') as f:
        for h, w in zip(h_indices, w_indices):
            # 3. 提取该像素位置完整的 D 维深度序列
            depth_seq = samples_np[:, h, w]
            
            # 4. 格式化字符串：保留指定小数位，用逗号分隔
            seq_str = ", ".join([f"{val:.{precision}f}" for val in depth_seq])
            
            # 5. 拼接坐标标签，并写入一行（加上 \n 换行符）
            line_content = f"[h={h:03d}, w={w:03d}] {seq_str}\n"
            f.write(line_content)
            
    print(f"成功追加 {len(h_indices)} 个像素的深度序列至: {file_path}")

    return depth_range_samples



def save_depth_gt_incremental(
    depth_gt: torch.Tensor, 
    h_indices: list, 
    w_indices: list, 
    file_path: str, 
    batch_idx: int = 0,
    precision: int = 4
):
    """
    从 [B, H, W] 的深度假设张量中提取指定像素的序列，增量式写入 txt。
    
    Args:
        depth_range_samples (torch.Tensor): 形状为 [B, D, H, W] 的深度假设值
        h_indices (list): 需要提取的像素行坐标列表，如 [10, 20]
        w_indices (list): 需要提取的像素列坐标列表，如 [15, 25]
        file_path (str): 保存的 txt 文件路径
        batch_idx (int): 从哪个 batch 提取数据，默认为 0
        precision (int): 保存浮点数的小数位数，默认为 4
    """
    if len(h_indices) != len(w_indices):
        raise ValueError("h_indices 和 w_indices 的长度必须一致")
    
    if depth_gt.dim() != 3:
        raise ValueError("输入张量形状必须是 [B, H, W]")
        
    # 1. 将指定 batch 的数据移至 CPU 并转为 numpy，此时形状为 [D, H, W]
    samples_np = depth_gt[batch_idx].detach().cpu().numpy()
    
    # 2. 以 'a' (append追加) 模式打开文件。
    # 如果文件不存在，Python 会自动创建；如果存在，光标会移到文件末尾，绝不会覆盖之前的内容。
    with open(file_path, 'a', encoding='utf-8') as f:
        for h, w in zip(h_indices, w_indices):
            # 3. 提取该像素位置完整的 D 维深度序列
            depth_gtv = samples_np[h, w]
            
            # 4. 格式化字符串：保留指定小数位，用逗号分隔
            seq_str = ", ".join(str(depth_gtv))
            
            # 5. 拼接坐标标签，并写入一行（加上 \n 换行符）
            line_content = f"[h={h:03d}, w={w:03d}] {seq_str}\n"
            f.write(line_content)
            
    print(f"成功追加 {len(h_indices)} 个像素的深度序列至: {file_path}")

    return depth_gt


