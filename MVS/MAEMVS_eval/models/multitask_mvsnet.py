"""
多任务学习MVSNet - 整合稀疏CNN特征提取、MAE辅助任务和DAMW损失平衡
基于论文《Generalized deep aerial Multi-view Stereo network based on gradient balance masked representation learning》
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import (
    homo_warping, depth_regression, Conv3d, Deconv3d,
    CostRegNet, RefineNet, cas_mvsnet_loss
)
from .multitask_module import SparseFeatureNet
from .sparse_cnn import MaskGenerator
from .mae_decoder import MAEDecoderWithMask
from .damw import MultiTaskLoss, GradientTracker


Align_Corners_Range = False


class DepthNet(nn.Module):
    """深度估计网络"""
    
    def __init__(self):
        super(DepthNet, self).__init__()

    def forward(self, features, proj_matrices, depth_values, num_depth, cost_regularization, 
                prob_volume_init=None):
        """
        Args:
            features: 特征列表 [ref_feature, src_features...]
            proj_matrices: 投影矩阵
            depth_values: 深度假设值
            num_depth: 深度假设数量
            cost_regularization: 代价体正则化网络
            prob_volume_init: 初始概率体（用于级联）
        """
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(features) == len(proj_matrices), "Different number of images and projection matrices"

        num_views = len(features)

        # 参考视图特征和源视图特征
        ref_feature = features[0]
        ref_proj = proj_matrices[0]
        src_features = features[1:]
        src_projs = proj_matrices[1:]

        # 构建代价体（方差聚合）
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
        volume_sum = ref_volume
        volume_sq_sum = ref_volume ** 2
        del ref_volume
        
        for src_fea, src_proj in zip(src_features, src_projs):
            # 可微单应性变换
            src_proj_new = src_proj[:, 0].clone()
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
            ref_proj_new = ref_proj[:, 0].clone()
            ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
            warped_volume = homo_warping(src_fea, src_proj_new, ref_proj_new, depth_values)

            if self.training:
                volume_sum = volume_sum + warped_volume
                volume_sq_sum = volume_sq_sum + warped_volume ** 2
            else:
                volume_sum += warped_volume
                volume_sq_sum += warped_volume.pow_(2)
            del warped_volume
            
        # 方差聚合
        volume_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2))

        # 代价体正则化
        cost_reg = cost_regularization(volume_variance)
        prob_volume_pre = cost_reg.squeeze(1)

        if prob_volume_init is not None:
            prob_volume_pre += prob_volume_init

        prob_volume = F.softmax(prob_volume_pre, dim=1)
        depth = depth_regression(prob_volume, depth_values=depth_values)

        with torch.no_grad():
            # 光度置信度
            prob_volume_sum4 = 4 * F.avg_pool3d(
                F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), 
                (4, 1, 1), stride=1, padding=0
            ).squeeze(1)
            depth_index = depth_regression(
                prob_volume, 
                depth_values=torch.arange(num_depth, device=prob_volume.device, dtype=torch.float)
            ).long()
            depth_index = depth_index.clamp(min=0, max=num_depth-1)
            photometric_confidence = torch.gather(
                prob_volume_sum4, 1, depth_index.unsqueeze(1)
            ).squeeze(1)

        return {
            "depth": depth, 
            "photometric_confidence": photometric_confidence, 
            "prob_volume": prob_volume
        }


def get_cur_depth_range_samples(cur_depth, ndepth, depth_inteval_pixel, shape, max_depth=192.0, min_depth=0.0):
    #shape, (B, H, W)
    #cur_depth: (B, H, W)
    #return depth_range_values: (B, D, H, W)
    cur_depth_min = (cur_depth - ndepth / 2 * depth_inteval_pixel)  # (B, H, W)
    cur_depth_max = (cur_depth + ndepth / 2 * depth_inteval_pixel)

    # assert cur_depth.shape == torch.Size(shape), "cur_depth:{}, input shape:{}".format(cur_depth.shape, shape)
    new_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1)  # (B, H, W)

    depth_range_samples = cur_depth_min.unsqueeze(1) + (torch.arange(0, ndepth, device=cur_depth.device,
                                                                  dtype=cur_depth.dtype,
                                                                  requires_grad=False).reshape(1, -1, 1,
                                                                                               1) * new_interval.unsqueeze(1))

    return depth_range_samples


def get_depth_range_samples(cur_depth, ndepth, depth_inteval_pixel, device, dtype, shape, max_depth=192.0, min_depth=0.0):
    #shape: (B, H, W)
    #cur_depth: (B, H, W) or (B, D)
    #return depth_range_samples: (B, D, H, W)
    if cur_depth.dim() == 2:
        cur_depth_min = cur_depth[:, 0]  # (B,)
        cur_depth_max = cur_depth[:, -1]
        new_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1)  # (B, )

        depth_range_samples = cur_depth_min.unsqueeze(1) + (torch.arange(0, ndepth, device=device, dtype=dtype,
                                                                       requires_grad=False).reshape(1, -1) * new_interval.unsqueeze(1)) #(B, D)

        depth_range_samples = depth_range_samples.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, shape[1], shape[2]) #(B, D, H, W)

    else:
        depth_range_samples = get_cur_depth_range_samples(cur_depth, ndepth, depth_inteval_pixel, shape, max_depth, min_depth)

    return depth_range_samples


class MultiTaskMVSNet(nn.Module):
    """
    多任务学习MVSNet
    
    整合了：
    1. 稀疏CNN特征提取（共享Encoder）
    2. MAE辅助解码器（掩码图像重建）
    3. MVS深度估计解码器
    4. DAMW多任务损失平衡
    
    训练时同时输出MVS损失和MAE损失
    测试时只使用MVS分支
    """
    
    def __init__(self, refine=False, ndepths=[48, 32, 8], depth_interals_ratio=[4, 2, 1], 
                 share_cr=False, grad_method="detach", arch_mode="fpn", 
                 cr_base_chs=[8, 8, 8], use_mae=True, mask_ratio=0.1,
                 use_damw=True, use_sparse_conv=True):
        """
        Args:
            refine: 是否使用深度细化
            ndepths: 各阶段深度假设数量
            depth_interals_ratio: 深度间隔比例
            share_cr: 是否共享代价体正则化网络
            grad_method: 梯度方法
            arch_mode: 特征提取架构模式
            cr_base_chs: 代价正则化基础通道数
            use_mae: 是否使用MAE辅助任务
            mask_ratio: MAE掩码比例
            use_damw: 是否使用DAMW损失平衡
            use_sparse_conv: 是否使用稀疏卷积
        """
        super(MultiTaskMVSNet, self).__init__()
        
        self.refine = refine
        self.share_cr = share_cr
        self.ndepths = ndepths
        self.depth_interals_ratio = depth_interals_ratio
        self.grad_method = grad_method
        self.arch_mode = arch_mode
        self.cr_base_chs = cr_base_chs
        self.num_stage = len(ndepths)
        
        # MAE相关
        self.use_mae = use_mae
        self.mask_ratio = mask_ratio
        self.use_damw = use_damw
        self.use_sparse_conv = use_sparse_conv
        
        print("********** MultiTaskMVSNet **********")
        print(f"  ndepths: {ndepths}, depth_intervals_ratio: {depth_interals_ratio}")
        print(f"  grad_method: {grad_method}, arch_mode: {arch_mode}")
        print(f"  use_mae: {use_mae}, mask_ratio: {mask_ratio}")
        print(f"  use_damw: {use_damw}, use_sparse_conv: {use_sparse_conv}")
        print("**************************************")
        
        assert len(ndepths) == len(depth_interals_ratio)
        
        self.stage_infos = {
            "stage1": {"scale": 4.0},
            "stage2": {"scale": 2.0},
            "stage3": {"scale": 1.0}
        }
        
        # 共享特征提取网络（稀疏CNN）
        self.feature = SparseFeatureNet(
            base_channels=8, stride=4, num_stage=self.num_stage, 
            arch_mode=self.arch_mode, use_sparse_conv=use_sparse_conv
        )
        
        # 代价体正则化网络
        if self.share_cr:
            self.cost_regularization = CostRegNet(
                in_channels=self.feature.out_channels[0], base_channels=8
            )
        else:
            self.cost_regularization = nn.ModuleList([
                CostRegNet(in_channels=self.feature.out_channels[i],
                           base_channels=self.cr_base_chs[i])
                for i in range(self.num_stage)
            ])
            
        # 深度细化网络
        if self.refine:
            self.refine_network = RefineNet()
            
        # 深度估计网络
        self.DepthNet = DepthNet()
        
        # MAE解码器
        if self.use_mae:
            self.mask_generator = MaskGenerator(mask_ratio=mask_ratio)
            self.mae_decoder = MAEDecoderWithMask(
                in_channels_list=self.feature.out_channels,
                out_channels=3
            )
            
        # 多任务损失计算器
        if self.use_damw:
            self.multi_task_loss = MultiTaskLoss(use_damw=True)
        else:
            self.multi_task_loss = MultiTaskLoss(use_damw=False)
            
        # 梯度追踪器
        self.gradient_tracker = GradientTracker(num_stages=self.num_stage)
        
    def forward(self, imgs, proj_matrices, depth_values, return_mae=True, collect_visuals=False):
        """
        Args:
            imgs: 输入图像 [B, N, 3, H, W]
            proj_matrices: 投影矩阵
            depth_values: 深度值范围
            return_mae: 是否返回MAE结果
            
        Returns:
            outputs: 输出字典，包含MVS和MAE结果
        """
        # 解析深度范围
        depth_min = float(depth_values[0, 0].cpu().numpy())
        depth_max = float(depth_values[0, -1].cpu().numpy())
        depth_interval = (depth_max - depth_min) / depth_values.size(1) if depth_values.size(1) > 1 else 1.0
        
        # 参考图像
        ref_img = imgs[:, 0]  # [B, 3, H, W]
        
        # MAE处理
        mae_loss = None
        mae_output = None
        masks_dict = None
        imgs_for_feat = imgs
        
        if self.use_mae and return_mae and self.training:
            # 生成掩码
            masks_dict, masked_img = self.mask_generator(ref_img)
            # 将掩码应用到所有图像
            imgs_for_feat = imgs.clone()
            for i in range(imgs.size(1)):
                imgs_for_feat[:, i] = imgs[:, i] * masks_dict["stage3"]
        
        # 特征提取
        features = []
        for nview_idx in range(imgs.size(1)):
            img = imgs_for_feat[:, nview_idx]
            self.feature.collect_visuals = bool(collect_visuals and nview_idx == 0)
            if self.use_sparse_conv and masks_dict is not None:
                mask = masks_dict.get("stage3")
                feat = self.feature(img, mask)
            else:
                feat = self.feature(img, None)
            features.append(feat)
        self.feature.collect_visuals = False
            
        # MVS深度估计
        outputs = {}
        depth, cur_depth, prob_volume, prev_depth_hypotheses = None, None, None, None
        
        for stage_idx in range(self.num_stage):
            # 获取当前阶段的特征和投影矩阵
            features_stage = [feat[f"stage{stage_idx + 1}"] for feat in features]
            proj_matrices_stage = proj_matrices[f"stage{stage_idx + 1}"]
            stage_scale = self.stage_infos[f"stage{stage_idx + 1}"]["scale"]
            
            # 获取当前深度范围
            if depth is not None:
                if self.grad_method == "detach":
                    cur_depth = depth.detach()
                else:
                    cur_depth = depth
            else:
                cur_depth = depth_values
                
            # 计算深度范围采样
            depth_range_samples = get_depth_range_samples(cur_depth=cur_depth,
                                                        ndepth=self.ndepths[stage_idx],
                                                        depth_inteval_pixel=self.depth_interals_ratio[stage_idx] * depth_interval,
                                                        dtype=img[0].dtype,
                                                        device=img[0].device,
                                                        shape=[img.shape[0], img.shape[2], img.shape[3]],
                                                        max_depth=depth_max,
                                                        min_depth=depth_min)
            
            
            prev_depth_hypotheses = depth_range_samples
            
            # 深度估计
            # outputs_stage = self.DepthNet(
            #     features_stage, 
            #     proj_matrices_stage,
            #     depth_values=F.interpolate(
            #         depth_range_samples.unsqueeze(1),
            #         [self.ndepths[stage_idx], img.shape[2]//int(stage_scale), 
            #          img.shape[3]//int(stage_scale)], 
            #         mode='trilinear',
            #         align_corners=Align_Corners_Range
            #     ).squeeze(1),
            #     num_depth=self.ndepths[stage_idx],
            #     cost_regularization=self.cost_regularization if self.share_cr 
            #                        else self.cost_regularization[stage_idx]
            # )
            outputs_stage = self.DepthNet(features_stage, proj_matrices_stage,
                                          depth_values=F.interpolate(depth_range_samples.unsqueeze(1),
                                                                     [self.ndepths[stage_idx], img.shape[2]//int(stage_scale), img.shape[3]//int(stage_scale)], mode='trilinear',
                                                                     align_corners=Align_Corners_Range).squeeze(1),
                                          num_depth=self.ndepths[stage_idx],
                                          cost_regularization=self.cost_regularization if self.share_cr else self.cost_regularization[stage_idx])
            

            depth = outputs_stage['depth']
            prob_volume = outputs_stage['prob_volume']
            
            outputs[f"stage{stage_idx + 1}"] = outputs_stage
            outputs.update(outputs_stage)
            
        # 深度细化
        if self.refine:
            refined_depth = self.refine_network(torch.cat((ref_img, depth), 1))
            outputs["refined_depth"] = refined_depth
            
        # MAE解码
        if self.use_mae and return_mae and self.training:
            # 密集化特征
            dense_features = {}
            for stage_key in ["stage1", "stage2", "stage3"]:
                if stage_key in features[0]:
                    dense_features[stage_key] = features[0][stage_key]

            self.mae_decoder.collect_visuals = bool(collect_visuals)
            mae_output, mae_loss = self.mae_decoder(dense_features, ref_img, masks_dict)
            self.mae_decoder.collect_visuals = False
            outputs["mae_output"] = mae_output
            outputs["mae_loss"] = mae_loss
            outputs["masks"] = masks_dict

        # 保存特征图用于DAMW梯度追踪
        if self.training and len(features) > 0:
            self._saved_features = features[0]  # 参考视图的多尺度特征

        return outputs
    
    def compute_loss(self, outputs, depth_gt_ms, mask_ms, dlossw=None):
        """
        计算多任务损失
        
        Args:
            outputs: 模型输出
            depth_gt_ms: 多尺度深度真值
            mask_ms: 多尺度掩码
            dlossw: 各阶段深度损失权重
            
        Returns:
            total_loss: 总损失
            loss_dict: 损失字典
        """
        # MVS深度损失
        mvs_loss, depth_loss = cas_mvsnet_loss(
            outputs, depth_gt_ms, mask_ms, dlossw=dlossw
        )
        
        # MAE重建损失
        mae_loss = torch.tensor(0.0, device=mvs_loss.device)
        if self.use_mae and self.training and "mae_loss" in outputs:
            mae_loss = outputs["mae_loss"]
            
        # 多任务损失加权
        total_loss, loss_dict = self.multi_task_loss(
            mvs_loss, mae_loss, gradients_dict=None
        )
        
        loss_dict["depth_loss"] = depth_loss
        loss_dict["mvs_loss"] = mvs_loss
        loss_dict["mae_loss"] = mae_loss

        return total_loss, loss_dict

    def update_damw_weights(self, loss_mvs=None, loss_mae=None):
        """
        在反向传播后更新DAMW权重。

        使用保存的特征图梯度注册到梯度追踪器，
        然后调用DAMW优化器更新任务权重。

        必须在 loss.backward() 之后、optimizer.step() 之前调用。
        """
        if not (self.use_damw and self.training):
            return None, self.multi_task_loss.get_weights()

        if loss_mvs is not None and loss_mae is not None:
            if not hasattr(self, '_saved_features') or self._saved_features is None:
                return None, self.multi_task_loss.get_weights()
            return self.multi_task_loss.assign_gradnorm_gradients(loss_mvs, loss_mae, self._saved_features)

        if not hasattr(self, '_saved_features') or self._saved_features is None:
            return

        ref_features = self._saved_features  # dict: {"stage1": feat, "stage2": feat, "stage3": feat}

        # 对于中间特征张量，需要显式调用 retain_grad() 才能在 backward 后访问 .grad
        for _, feat in ref_features.items():
            if isinstance(feat, dict):
                # 字典结构 {"layer1": feat, "layer2": feat}
                for _, layer_feat in feat.items():
                    if layer_feat.requires_grad:
                        layer_feat.retain_grad()
            elif isinstance(feat, torch.Tensor) and feat.requires_grad:
                # 直接是 Tensor 结构
                feat.retain_grad()

        # 注册MVS任务的梯度
        self.gradient_tracker.register_gradients(ref_features, "mvs")

        # 注册MAE任务的梯度（使用相同的特征图）
        if self.use_mae:
            self.gradient_tracker.register_gradients(ref_features, "mae")

        # 获取合并的梯度字典
        gradients_dict = self.gradient_tracker.get_combined_gradients()

        # 更新DAMW权重
        grad_loss, weights = self.multi_task_loss.step(gradients_dict)

        # 清空缓存
        self._saved_features = None
        self.gradient_tracker.clear()

        return grad_loss, weights


# 别名
CascadeMVSNet = MultiTaskMVSNet


if __name__ == "__main__":
    print("Testing MultiTaskMVSNet...")
    
    # 创建模型
    model = MultiTaskMVSNet(
        ndepths=[48, 32, 8],
        depth_interals_ratio=[4, 2, 1],
        arch_mode="fpn",
        use_mae=True,
        mask_ratio=0.1,
        use_damw=True,
        use_sparse_conv=True
    )
    
    # 模拟输入
    B, N, C, H, W = 1, 5, 3, 256, 320
    imgs = torch.randn(B, N, C, H, W)
    
    # 投影矩阵
    proj_matrices = {
        "stage1": torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(B, N, 2, 4, 4),
        "stage2": torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(B, N, 2, 4, 4),
        "stage3": torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(B, N, 2, 4, 4),
    }
    
    depth_values = torch.linspace(100, 200, 48)
    
    # 前向传播
    model.train()
    outputs = model(imgs, proj_matrices, depth_values)
    
    print("Forward pass successful!")
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")
        elif isinstance(v, dict):
            print(f"  {k}: dict with keys {list(v.keys())}")
        else:
            print(f"  {k}: {type(v)}")
            
    print("\nAll MultiTaskMVSNet tests passed!")
