"""MAE decoder modules for masked reconstruction."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MAEDecoder(nn.Module):
    """Three-branch CNN decoder that reconstructs the reference RGB image."""

    def __init__(self, in_channels_list=[8, 16, 32], out_channels=3):
        super(MAEDecoder, self).__init__()
        fuse_in_channels = out_channels * 3
        self.branch1 = self._make_branch(in_channels_list[0], out_channels, up_scale=1)
        self.branch2 = self._make_branch(in_channels_list[1], out_channels, up_scale=2)
        self.branch3 = self._make_branch(in_channels_list[2], out_channels, up_scale=1)
        self.fusion = self._make_fusion_layer(fuse_in_channels, out_channels)

    def _make_branch(self, in_channels, out_channels, up_scale):
        layers = []
        if up_scale > 1:
            layers.append(nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1))
        layers.extend([
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        ])
        return nn.Sequential(*layers)

    def _make_fusion_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, out_channels, 3, padding=1),
        )

    def forward(self, features_dict, ref_img=None):
        f1 = features_dict["stage1"]
        f2 = features_dict["stage2"]
        f3 = features_dict["stage3"]
        target_h, target_w = f3.shape[2], f3.shape[3]

        out1 = self.branch1(f1)
        out1 = F.interpolate(out1, size=(target_h, target_w), mode="bilinear", align_corners=False)
        out2 = self.branch2(f2)
        out2 = F.interpolate(out2, size=(target_h, target_w), mode="bilinear", align_corners=False)
        out3 = self.branch3(f3)

        output = self.fusion(torch.cat([out1, out2, out3], dim=1))
        mae_loss = F.mse_loss(output, ref_img) if ref_img is not None else None
        return output, mae_loss


class MAEDecoderWithMask(nn.Module):
    """Densify sparse multi-scale features and compute masked-region MAE loss."""

    def __init__(self, in_channels_list=[8, 16, 32], out_channels=3, feat_scale=4):
        super(MAEDecoderWithMask, self).__init__()
        self.feat_scale = feat_scale
        self.dense_proj_list = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, in_ch * 2, 1),
                nn.ReLU(inplace=True),
            )
            for in_ch in in_channels_list
        ])
        self.mask_tokens = nn.ParameterList([
            nn.Parameter(torch.zeros(1, in_ch * 2, 1, 1))
            for in_ch in in_channels_list
        ])
        self.decoder = MAEDecoder(
            in_channels_list=[ch * 2 for ch in in_channels_list],
            out_channels=out_channels,
        )
        self.collect_visuals = False
        self.visual_cache = {}

    def forward(self, features_dict, ref_img, masks_dict=None):
        dense_features = {}
        self.visual_cache = {}
        for i, stage_key in enumerate(["stage1", "stage2", "stage3"]):
            feat = features_dict[stage_key]
            dense_feat = self.dense_proj_list[i](feat)
            if masks_dict is not None and stage_key in masks_dict:
                mask = masks_dict[stage_key]
                if mask.shape[-2:] != dense_feat.shape[-2:]:
                    mask = F.interpolate(mask, size=dense_feat.shape[-2:], mode="nearest")
                dense_feat = torch.where(
                    mask.bool().expand_as(dense_feat),
                    dense_feat,
                    self.mask_tokens[i].expand_as(dense_feat),
                )
            dense_features[stage_key] = dense_feat
            if self.collect_visuals:
                self.visual_cache[f"dense_{stage_key}"] = dense_feat.detach()

        output, _ = self.decoder(dense_features, ref_img)
        if masks_dict is not None and "stage3" in masks_dict:
            mae_loss = mae_loss_function(output, ref_img, masks_dict["stage3"])
        else:
            mae_loss = F.mse_loss(output, ref_img)
        return output, mae_loss


def mae_loss_function(output, target, mask):
    """Mean squared error on masked pixels only.

    The active mask uses 1 for visible pixels and 0 for masked pixels.
    """
    if mask.shape[-2:] != output.shape[-2:]:
        mask = F.interpolate(mask, size=output.shape[-2:], mode="nearest")
    masked_region = (1.0 - mask).to(dtype=output.dtype, device=output.device)
    diff = (output - target) ** 2
    valid_count = masked_region.sum() * output.shape[1] + 1e-8
    return (diff * masked_region).sum() / valid_count


if __name__ == "__main__":
    b, c, h, w = 2, 3, 128, 160
    features = {
        "stage1": torch.randn(b, 8, h // 4, w // 4),
        "stage2": torch.randn(b, 16, h // 2, w // 2),
        "stage3": torch.randn(b, 32, h, w),
    }
    ref_img = torch.randn(b, c, h, w)
    masks = {
        "stage1": torch.ones(b, 1, h // 4, w // 4),
        "stage2": torch.ones(b, 1, h // 2, w // 2),
        "stage3": torch.ones(b, 1, h, w),
    }
    masks["stage3"][:, :, h // 4:h // 2, w // 4:w // 2] = 0
    decoder = MAEDecoderWithMask([8, 16, 32], 3)
    output, loss = decoder(features, ref_img, masks)
    print(f"MAEDecoderWithMask: output={output.shape}, loss={loss.item():.4f}")
