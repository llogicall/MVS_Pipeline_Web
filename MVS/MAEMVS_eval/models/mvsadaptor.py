import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class MVSTTAAdaptor(nn.Module):
    """
    MVS测试时自适应模块
    在训练时禁用，测试时启用
    """
    def __init__(self, feature_net, depth_net):
        super().__init__()
        self.feature_net = feature_net
        self.depth_net = depth_net
        
        # 标识是否处于TTA模式
        self.is_tta_mode = False
        
        # 冻结主干网络权重
        for param in self.feature_net.parameters():
            param.requires_grad = False
        for param in self.depth_net.parameters():
            param.requires_grad = False
        
        # 仅解冻归一化层的γ, β参数
        self.tentable_params = []
        for module in self.feature_net.modules():
            if hasattr(module, 'weight') and hasattr(module, 'bias'):
                # 这是归一化层（BatchNorm, InstanceNorm, etc.）
                module.weight.requires_grad = True   # γ
                module.bias.requires_grad = True    # β
                self.tentable_params.append(module.weight)
                self.tentable_params.append(module.bias)
    
    def set_tta_mode(self, enable=True):
        """
        设置TTA模式
        Args:
            enable: True-启用TTA, False-禁用TTA（训练模式）
        """
        self.is_tta_mode = enable
        
        if enable:
            # 测试模式：启用TTA优化
            for module in self.feature_net.modules():
                if hasattr(module, 'weight'):
                    module.weight.requires_grad = True
                    module.bias.requires_grad = True
        else:
            # 训练模式：使用标准训练流程
            for param in self.feature_net.parameters():
                param.requires_grad = True
            for param in self.depth_net.parameters():
                param.requires_grad = True
    
    def tent_optimize(self, imgs, proj_matrices, depth_values, num_steps=5):
        """
        测试时自适应优化
        Args:
            imgs: (B, N, 3, H, W) 多视图图像
            proj_matrices: 投影矩阵
            depth_values: 深度假设
            num_steps: TTA优化步数
        Returns:
            depth: 自适应后的深度图
        """
        # 创建优化器（仅优化归一化参数）
        optimizer = torch.optim.Adam(self.tentable_params, lr=0.001)
        
        for step in range(num_steps):
            optimizer.zero_grad()
            
            # 提取特征
            features = []
            for nview_idx in range(imgs.size(1)):
                img = imgs[:, nview_idx]
                feat = self.feature_net(img)
                features.append(feat)
            
            # 构建代价体积（参考视图和源视图）
            ref_feature = features[0]["stage1"]
            src_features = features[1:]
            
            # 扭曲源视图特征到参考视图
            warped_srcs = []
            for src_feat, src_proj in zip(src_features, proj_matrices["stage1"][1:]):
                warped = self.homo_warp_tta(
                    src_feat["stage1"], 
                    src_proj, 
                    proj_matrices["stage1"][0], 
                    depth_values
                )
                warped_srcs.append(warped)
            
            # 计算跨视图一致性损失（无监督）
            consistency_loss = self.compute_consistency_loss(
                ref_feature, 
                warped_srcs
            )
            
            # 反向传播
            consistency_loss.backward()
            optimizer.step()
            
            if step < num_steps - 1:
                print(f"TTA Step {step+1}/{num_steps}, Consistency Loss: {consistency_loss.item():.4f}")
        
        # 最终推理
        outputs = self.depth_net(
            features, 
            proj_matrices, 
            depth_values, 
            num_depth=depth_values.size(1),
            cost_regularization=self.cost_regularization
        )
        
        return outputs
    
    def compute_consistency_loss(self, ref_feature, warped_srcs):
        """
        计算跨视图一致性损失（TTA的监督信号）
        这是一个无监督损失，不需要真值深度图
        """
        consistency_loss = 0.0
        
        for warped_src in warped_srcs:
            # L2距离（特征层面的对齐）
            l2_loss = torch.mean((ref_feature - warped_src) ** 2)
            
            # 感知一致性损失（使用中间特征）
            perceptual_loss = torch.mean(torch.abs(ref_feature - warped_src))
            
            consistency_loss += 0.7 * l2_loss + 0.3 * perceptual_loss
        
        return consistency_loss / len(warped_srcs)
    
    def homo_warp_tta(self, src_fea, src_proj, ref_proj, depth_values):
        """
        单应性扭曲（支持梯度计算）
        """
        # ... 与homo_warping相同的实现 ...
        # 确保所有操作都支持梯度计算
        pass
    
    def forward(self, imgs, proj_matrices, depth_values):
        """
        前向传播（根据模式选择路径）
        """
        if self.is_tta_mode:
            # TTA模式：执行测试时自适应
            return self.tent_optimize(imgs, proj_matrices, depth_values)
        else:
            # 标准模式：正常推理
            features = []
            for nview_idx in range(imgs.size(1)):
                img = imgs[:, nview_idx]
                features.append(self.feature_net(img))
            
            outputs = self.depth_net(
                features, 
                proj_matrices, 
                depth_values, 
                num_depth=depth_values.size(1),
                cost_regularization=self.cost_regularization
            )
            
            return outputs
        

