import torch
import torch.nn as nn
import torch.nn.functional as F

# 添加特征的跨视角一致性约束 - 完整实现
def fea_warp_loss(data, outputs, mask_ms, **kwargs):
    """
    三阶段跨视角特征Warping一致性损失
    
    Args:
        data: 输入数据字典
        outputs: 模型输出
        mask_ms: 掩码
        **kwargs: 其他参数，包括dlossw
    
    Returns:
        total_loss: 总warping损失
    """
    inputs = outputs
    imgs = data["imgs"]
    cams = data["proj_matrices"]
    depth_loss_weights = kwargs.get("dlossw", None)
    scalar_outputs = {}
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=imgs.device, requires_grad=True)

    for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
        # stage_idx = int(stage_key.replace("stage", "")) - 1
        stage_idx = int(stage_key.replace("stage", ""))
        depth_est = stage_inputs["depth"]
        #   输入当前阶段的特征图（要在模型类里添加），对应阶段的相机参数，
        # output:stage1,stage2,stage3          cams:stage1,stage2,stage3
        # warp_loss = fea_warp_stage(outputs["stage{}".format(stage_idx)], cams[stage_key], depth_est, stage_idx)
        warp_loss = fea_warp_stage(outputs[stage_key]['feature'], cams[stage_key], depth_est, stage_idx)
        scalar_outputs["warp_loss_stage{}".format(stage_idx + 1)] = warp_loss
        if depth_loss_weights is not None:
            total_loss = total_loss + depth_loss_weights[stage_idx -1 ] * warp_loss *16
        else:
            total_loss = total_loss + 1.0 * warp_loss

    return total_loss


# 单阶段的跨视角一致性约束
def fea_warp_stage(imgs, cams, depth, stage_idx):
    """
    单阶段的特征warping一致性损失
    
    Args:
        imgs: [B, N, 3, H, W] 输入图像序列
        cams: [B, N, 2, 4, 4] 投影矩阵
        depth: [B, H, W] 深度估计图
        stage_idx: 阶段索引(0, 1, 2)
    
    Returns:
        loss: warp一致性损失
    """
    # imgs = torch.unbind(imgs, 1)   # 拆分为不同视角的列表,将张量降维转化为列表     这里已经拆开了
    cams = torch.unbind(cams, 1)
    assert len(imgs) == len(cams), "Different number of images and projection matrices"
    num_views = len(imgs)
    ref_img = imgs[0]
    # # 对多尺度特征统一分辨率     如果是特征一致性就不应该加了
    # if stage_idx == 0:
    #     ref_img = F.interpolate(ref_img, scale_factor=0.25)
    # elif stage_idx == 1: 
    #     ref_img = F.interpolate(ref_img, scale_factor=0.5)
    # else:
    #     pass
    ref_img = ref_img.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]
    ref_cam = cams[0]

    reconstr_loss = 0
    total_loss = 0
    warped_img_list = []
    mask_list = []
    reprojection_losses = []
    reprojection_volume = []

    for view in range(1, num_views):
        view_img = imgs[view]
        view_cam = cams[view]
        # if stage_idx == 0:
        #     view_img = F.interpolate(view_img, scale_factor=0.25)
        # elif stage_idx == 1:
        #     view_img = F.interpolate(view_img, scale_factor=0.5)
        # else:
        #     pass
        view_img = view_img.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]
        # warp之后的特征图和warp造成的有效区域
        warped_img, mask = inverse_warping(view_img, ref_cam, view_cam, depth) 

        # if mask.sum() == 0:
        #     return torch.tensor(0.0, dtype=torch.float32, device=mask.device)
        warped_img_list.append(warped_img)
        mask_list.append(mask)

        # # # 将warped_img和ref_img可视化并保存
        # save_image(mask.squeeze(), 'result/warped_img_save.png')
        # save_image((warped_img*mask)[0,:,:,2], 'result/warped_img_save.png')
        # save_image((ref_img*mask)[0,:,:,2], 'result/ref_img_save.png')

        # 2. 计算像素级一致性损失
        # 方法1：L1损失    [B, H, W, C]
        pixel_loss = F.l1_loss(warped_img * mask, ref_img * mask, reduction='none')
        # 方法2：余弦相似度
        # cos_sim = F.cosine_similarity(warped_fea, ref_fea, dim=1, eps=1e-8)
        # pixel_loss = 1.0 - cos_sim.unsqueeze(1)  # [B, 1, H, W]
        
        # 3. 平均到通道维度
        mean_pixel_loss = pixel_loss.mean(dim=3, keepdim=True)  # [B, H, W, 1]
        
        # 4. 处理无效区域
        valid_mask = mask > 0.5
        mean_pixel_loss = torch.where(
            valid_mask,
            mean_pixel_loss,
            torch.full_like(mean_pixel_loss, 1e4)  # 无效区域标记为大值
        )
        # 5. 添加到reprojection_volume
        reprojection_volume.append(mean_pixel_loss)


        reconstr_loss = compute_reconstr_loss(warped_img, ref_img, mask, simple=False)  
        total_loss += reconstr_loss
        valid_mask = 1 - mask  # replace all 0 values with INF 
        # 将平均损失映射到特征图尺寸上
        reprojection_losses.append(reconstr_loss + 1e4 * valid_mask)

    reprojection_volume = torch.stack(reprojection_volume).permute(1, 2, 3, 4, 0)  # 将所有视角的重建损失整合
    reprojection_loss_volume = torch.stack(reprojection_losses).permute(1, 2, 3, 4, 0)  # 将所有视角的重建损失整合
    # torch.neg将每一个元素取负值，torch.topk取所有元素中的最大值
    top_vals, top_inds = torch.topk(torch.neg(reprojection_loss_volume), k=1, sorted=False)    # [B,H,W,1,N_src] ([2, 96, 192, 1, 4])   -> [B,H,W,1,1] 既可以作为掩码图，又可以作为重要的参数引导假设深度采样
    top_vals = torch.neg(top_vals)
    top_mask = top_vals < (1e4 * torch.ones_like(top_vals))
    top_mask = top_mask.float()
    top_vals = torch.mul(top_vals, top_mask)
    reconstr_loss = torch.mean(torch.sum(top_vals, dim=-1))  

    return reconstr_loss, reprojection_volume



def inverse_warping(img, left_cam, right_cam, depth):
    # img: [batch_size, height, width, channels]
    # cameras (K, R, t)
    R_left = left_cam[:, 0:1, 0:3, 0:3]  # [B, 1, 3, 3]
    R_right = right_cam[:, 0:1, 0:3, 0:3]  # [B, 1, 3, 3]
    t_left = left_cam[:, 0:1, 0:3, 3:4]  # [B, 1, 3, 1]
    t_right = right_cam[:, 0:1, 0:3, 3:4]  # [B, 1, 3, 1]
    K_left = left_cam[:, 1:2, 0:3, 0:3]  # [B, 1, 3, 3]
    K_right = right_cam[:, 1:2, 0:3, 0:3]  # [B, 1, 3, 3]

    K_left = K_left.squeeze(1)  # [B, 3, 3]
    K_left_inv = torch.inverse(K_left)  # [B, 3, 3]
    R_left_trans = R_left.squeeze(1).permute(0, 2, 1)  # [B, 3, 3]
    R_right_trans = R_right.squeeze(1).permute(0, 2, 1)  # [B, 3, 3]

    R_left = R_left.squeeze(1)
    t_left = t_left.squeeze(1)
    R_right = R_right.squeeze(1)
    t_right = t_right.squeeze(1)

    # estimate egomotion by inverse composing R1,R2 and t1,t2
    R_rel = torch.matmul(R_right, R_left_trans)  # [B, 3, 3]    
    t_rel = t_right - torch.matmul(R_rel, t_left)  # [B, 3, 1]  
    # now convert R and t to transform mat, as in SFMlearner
    batch_size = R_left.shape[0]
    filler = torch.tensor([0.0, 0.0, 0.0, 1.0], device=img.device, dtype=img.dtype).reshape(1, 1, 4)  # [1, 1, 4]
    filler = filler.repeat(batch_size, 1, 1)  # [B, 1, 4]
    transform_mat = torch.cat([R_rel, t_rel], dim=2)  # [B, 3, 4]
    transform_mat = torch.cat([transform_mat.float(), filler.float()], dim=1)  # [B, 4, 4]
    batch_size, img_height, img_width, _ = img.shape
    depth = depth.reshape(batch_size, 1, img_height * img_width)  # [batch_size, 1, height * width]

    grid = _meshgrid_abs(img_height, img_width, img.device, img.dtype)  # [3, height * width]
    grid = grid.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size, 3, height * width]
    cam_coords = _pixel2cam(depth, grid, K_left_inv)  # [batch_size, 3, height * width]
    ones = torch.ones([batch_size, 1, img_height * img_width], device=img.device, dtype=img.dtype)  # [batch_size, 1, height * width]
    cam_coords_hom = torch.cat([cam_coords, ones], dim=1)  # [batch_size, 4, height * width]

    # Get projection matrix for target camera frame to source pixel frame
    hom_filler = torch.tensor([0.0, 0.0, 0.0, 1.0], device=img.device, dtype=img.dtype).reshape(1, 1, 4)  # [1, 1, 4]
    hom_filler = hom_filler.repeat(batch_size, 1, 1)  # [B, 1, 4]
    intrinsic_mat_hom = torch.cat([K_left.float(), torch.zeros([batch_size, 3, 1], device=img.device, dtype=img.dtype)], dim=2)  # [B, 3, 4]
    intrinsic_mat_hom = torch.cat([intrinsic_mat_hom, hom_filler], dim=1)  # [B, 4, 4]
    proj_target_cam_to_source_pixel = torch.matmul(intrinsic_mat_hom, transform_mat)  # [B, 4, 4]
    source_pixel_coords = _cam2pixel(cam_coords_hom, proj_target_cam_to_source_pixel)  # [batch_size, 2, height * width]
    source_pixel_coords = source_pixel_coords.reshape(batch_size, 2, img_height, img_width)   # [batch_size, 2, height, width]
    source_pixel_coords = source_pixel_coords.permute(0, 2, 3, 1)  # [batch_size, height, width, 2]
    warped_right, mask = _spatial_transformer(img, source_pixel_coords)
    return warped_right, mask

def _meshgrid_abs(height, width, device, dtype):
    """Meshgrid in the absolute coordinates."""
    x_t = torch.matmul(
        torch.ones([height, 1], device=device, dtype=dtype),
        torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype).unsqueeze(1).permute(1, 0)
    )  # [height, width]
    y_t = torch.matmul(
        torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype).unsqueeze(1),
        torch.ones([1, width], device=device, dtype=dtype)
    )
    x_t = (x_t + 1.0) * 0.5 * (width - 1)
    y_t = (y_t + 1.0) * 0.5 * (height - 1)
    x_t_flat = x_t.reshape(1, -1)
    y_t_flat = y_t.reshape(1, -1)
    ones = torch.ones_like(x_t_flat)
    grid = torch.cat([x_t_flat, y_t_flat, ones], dim=0)  # [3, height * width]
    return grid


def _pixel2cam(depth, pixel_coords, intrinsic_mat_inv):
    """Transform coordinates in the pixel frame to the camera frame."""
    cam_coords = torch.matmul(intrinsic_mat_inv.float(), pixel_coords.float()) * depth.float()
    return cam_coords


def _cam2pixel(cam_coords, proj_c2p):
    """Transform coordinates in the camera frame to the pixel frame."""
    pcoords = torch.matmul(proj_c2p, cam_coords)  # [batch_size, 4, height * width]
    x = pcoords[:, 0:1, :]  # [batch_size, 1, height * width]
    y = pcoords[:, 1:2, :]  # [batch_size, 1, height * width]
    z = pcoords[:, 2:3, :]  # [batch_size, 1, height * width]
    x_norm = x / (z + 1e-10)
    y_norm = y / (z + 1e-10)
    pixel_coords = torch.cat([x_norm, y_norm], dim=1)
    return pixel_coords  # [batch_size, 2, height * width]


def _spatial_transformer(img, coords):
    """A wrapper over binlinear_sampler(), taking absolute coords as input."""
    # img: [B, H, W, C]
    img_height = img.shape[1]
    img_width = img.shape[2]
    px = coords[:, :, :, :1]  # [batch_size, height, width, 1]
    py = coords[:, :, :, 1:]  # [batch_size, height, width, 1]
    # Normalize coordinates to [-1, 1] to send to _bilinear_sampler.
    px = px / (img_width - 1) * 2.0 - 1.0  # [batch_size, height, width, 1]
    py = py / (img_height - 1) * 2.0 - 1.0  # [batch_size, height, width, 1]
    output_img, mask = _bilinear_sample(img, px, py)
    return output_img, mask

def _spatial_transformer(img, coords):
    """A wrapper over binlinear_sampler(), taking absolute coords as input."""
    # img: [B, H, W, C]
    img_height = img.shape[1]
    img_width = img.shape[2]
    px = coords[:, :, :, :1]  # [batch_size, height, width, 1]
    py = coords[:, :, :, 1:]  # [batch_size, height, width, 1]
    # Normalize coordinates to [-1, 1] to send to _bilinear_sampler.
    px = px / (img_width - 1) * 2.0 - 1.0  # [batch_size, height, width, 1]
    py = py / (img_height - 1) * 2.0 - 1.0  # [batch_size, height, width, 1]
    output_img, mask = _bilinear_sample(img, px, py)
    return output_img, mask


def _bilinear_sample(im, x, y, name='bilinear_sampler'):
    """Perform bilinear sampling on im given list of x, y coordinates.
    Implements the differentiable sampling mechanism with bilinear kernel
    in https://arxiv.org/abs/1506.02025.
    x,y are tensors specifying normalized coordinates [-1, 1] to be sampled on im.
    For example, (-1, -1) in (x, y) corresponds to pixel location (0, 0) in im,
    and (1, 1) in (x, y) corresponds to the bottom right pixel in im.
    Args:
        im: Batch of images with shape [B, h, w, channels].
        x: Tensor of normalized x coordinates in [-1, 1], with shape [B, h, w, 1].
        y: Tensor of normalized y coordinates in [-1, 1], with shape [B, h, w, 1].
        name: Name scope for ops.
    Returns:
        Sampled image with shape [B, h, w, channels].
        Principled mask with shape [B, h, w, 1], dtype:float32.  A value of 1.0
        in the mask indicates that the corresponding coordinate in the sampled
        image is valid.
      """
    x = x.reshape(-1)  # [batch_size * height * width]
    y = y.reshape(-1)  # [batch_size * height * width]

    # Constants.
    batch_size, height, width, channels = im.shape

    x, y = x.float(), y.float()
    max_y = int(height - 1)
    max_x = int(width - 1)

    # Scale indices from [-1, 1] to [0, width - 1] or [0, height - 1].
    x = (x + 1.0) * (width - 1.0) / 2.0
    y = (y + 1.0) * (height - 1.0) / 2.0

    # Compute the coordinates of the 4 pixels to sample from.
    x0 = torch.floor(x).int()
    x1 = x0 + 1
    y0 = torch.floor(y).int()
    y1 = y0 + 1

    mask = (x0 >= 0) & (x1 <= max_x) & (y0 >= 0) & (y0 <= max_y)
    mask = mask.float()

    x0 = torch.clamp(x0, 0, max_x)
    x1 = torch.clamp(x1, 0, max_x)
    y0 = torch.clamp(y0, 0, max_y)
    y1 = torch.clamp(y1, 0, max_y)
    dim2 = width
    dim1 = width * height

    # Create base index.
    base = torch.arange(batch_size) * dim1
    base = base.reshape(-1, 1)
    base = base.repeat(1, height * width)
    base = base.reshape(-1)  # [batch_size * height * width]
    # base = base.long().to(device)
    base = base.long().to(im.device)

    base_y0 = base + y0.long() * dim2
    base_y1 = base + y1.long() * dim2
    idx_a = base_y0 + x0.long()
    idx_b = base_y1 + x0.long()
    idx_c = base_y0 + x1.long()
    idx_d = base_y1 + x1.long()

    # Use indices to lookup pixels in the flat image and restore channels dim.
    im_flat = im.reshape(-1, channels).float()  # [batch_size * height * width, channels]
    pixel_a = im_flat[idx_a]
    pixel_b = im_flat[idx_b]
    pixel_c = im_flat[idx_c]
    pixel_d = im_flat[idx_d]

    wa = (x1.float() - x) * (y1.float() - y)
    wb = (x1.float() - x) * (1.0 - (y1.float() - y))
    wc = (1.0 - (x1.float() - x)) * (y1.float() - y)
    wd = (1.0 - (x1.float() - x)) * (1.0 - (y1.float() - y))
    wa, wb, wc, wd = wa.unsqueeze(1), wb.unsqueeze(1), wc.unsqueeze(1), wd.unsqueeze(1)

    output = wa * pixel_a + wb * pixel_b + wc * pixel_c + wd * pixel_d
    output = output.reshape(batch_size, height, width, channels)
    mask = mask.reshape(batch_size, height, width, 1)
    return output, mask

def compute_reconstr_loss(warped, ref, mask, simple=True):
    if simple:
        return F.smooth_l1_loss(warped*mask, ref*mask, reduction='mean')
    else:
        alpha = 0.5
        ref_dx, ref_dy = gradient(ref * mask)
        warped_dx, warped_dy = gradient(warped * mask)
        photo_loss = F.smooth_l1_loss(warped*mask, ref*mask, reduction='mean')
        grad_loss = F.smooth_l1_loss(warped_dx, ref_dx, reduction='mean') + \
                    F.smooth_l1_loss(warped_dy, ref_dy, reduction='mean')
        return (1 - alpha) * photo_loss + alpha * grad_loss

def gradient(pred):
    D_dy = pred[:, 1:, :, :] - pred[:, :-1, :, :]
    D_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    return D_dx, D_dy





