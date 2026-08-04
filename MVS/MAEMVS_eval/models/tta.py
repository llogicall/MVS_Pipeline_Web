import torch
import torch.nn as nn
import torch.nn.functional as F
from tools import *
from torchvision.utils import save_image



class EssentialFeatureUpdater(nn.Module):
    """
    本质特征驱动的参数更新器
    """
    def update_geometry_prototypes(self, current_geometry, memory_geometry, consistency):
        """
        更新几何原型
        """
        # 根据一致性调整更新速度
        adaptive_momentum = 0.9 * (1.0 - consistency * 0.5)

        # 距离度量：寻找最相似的几何原型
        similarities = torch.cdist(
            current_geometry.unsqueeze(0),
            self.geometry_prototypes
        )

        # Soft赋权：更新所有原型
        prototype_weights = F.softmax(-similarities, dim=1)

        # 加权更新
        for i in range(len(self.geometry_prototypes)):
            self.geometry_prototypes.data[i] = (
                adaptive_momentum * self.geometry_prototypes.data[i] +
                (1 - adaptive_momentum) * current_geometry * prototype_weights[0, i]
            )

    def update_texture_scales(self, current_texture, target_texture, uncertainty):
        """
        更新纹理缩放因子
        """
        # 纹理密度差异
        density_diff = current_texture[:, :64] - target_texture[:, :64]

        # 方向性差异
        orientation_diff = current_texture[:, 64:80] - target_texture[:, 64:80]

        # 综合差异
        texture_diff = torch.cat([density_diff, orientation_diff], dim=-1)

        # 根据不确定性加权
        weighted_diff = texture_diff * (1.0 - uncertainty)

        # 更新纹理统计向量
        self.frequency_stats.data = (
            0.95 * self.frequency_stats.data +
            0.05 * weighted_diff.mean(dim=0)
        )

    def update_semantic_modulation(self, semantics, depth_quality, geometry_complexity):
        """
        更新语义调制参数
        """
        # 遍历每个语义类别
        for class_id in range(self.semantic_modulation['disparity_scale'].shape[0]):
            # 该类别的掩码
            class_mask = (semantics == class_id)

            if class_mask.sum() > 0:
                # 该类别的深度质量
                class_quality = depth_quality[class_mask].mean()

                # 该类别的几何复杂度
                class_complexity = geometry_complexity[class_mask].mean()

                # 更新规则
                # 高质量 + 低复杂度 → 提高视差权重
                if class_quality > 0.8 and class_complexity < 0.5:
                    self.semantic_modulation['disparity_scale'].data[class_id] *= 1.01
                    self.semantic_modulation['uncertainty_weight'].data[class_id] *= 0.99

                # 低质量 + 高复杂度 → 降低视差权重，增加不确定性容忍
                elif class_quality < 0.5 or class_complexity > 0.8:
                    self.semantic_modulation['disparity_scale'].data[class_id] *= 0.99
                    self.semantic_modulation['uncertainty_weight'].data[class_id] *= 1.02




class SceneMemoryBank(nn.Module):
    """
    场景记忆库：存储历史场景的本质特征和对应的优化参数
    """
    def __init__(self, capacity=100, feature_dim=256):
        super().__init__()

        # 记忆容量
        self.capacity = capacity

        # 本质特征记忆
        self.essential_features_memory = torch.zeros(capacity, feature_dim)

        # 优化参数记忆
        self.optimized_params_memory = []

        # 记忆的年龄(用于淘汰)
        self.memory_ages = torch.zeros(capacity)

        # 记忆的使用频率
        self.memory_usage = torch.zeros(capacity)

        # 当前记忆数量
        self.current_size = 0

    def query(self, query_features, top_k=5):
        """
        查询最相似的k个历史场景
        """
        if self.current_size == 0:
            return None

        # 计算相似度
        similarities = F.cosine_similarity(
            query_features.unsqueeze(0),
            self.essential_features_memory[:self.current_size],
            dim=-1
        )

        # 找到top-k最相似的
        top_indices = similarities.topk(k=min(top_k, self.current_size))[1]

        # 返回对应的参数
        similar_params = [
            self.optimized_params_memory[idx.item()]
            for idx in top_indices
        ]

        # 更新使用频率
        self.memory_usage[top_indices] += 1

        return similar_params, similarities[top_indices]

    def store(self, essential_features, optimized_params):
        """
        存储新的场景
        """
        if self.current_size < self.capacity:
            # 直接存储
            idx = self.current_size
            self.essential_features_memory[idx] = essential_features
            self.optimized_params_memory.append(optimized_params)
            self.current_size += 1
        else:
            # 替换最老且最少使用的记忆
            # 计算替换分数 = age / (usage + 1)
            replace_scores = self.memory_ages / (self.memory_usage + 1)
            idx = replace_scores.argmax().item()

            self.essential_features_memory[idx] = essential_features
            self.optimized_params_memory[idx] = optimized_params
            self.memory_ages[idx] = 0
            self.memory_usage[idx] = 0

        # 更新年龄
        self.memory_ages[:self.current_size] += 1

