"""
Sparse CNN components for the multi-task MVS reproduction.

Mask convention:
- 1 / True: active, visible, valid feature positions.
- 0 / False: masked positions that must not contribute to sparse feature flow.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


class SparseConv2d(nn.Module):
    """Standard 2D convolution gated by a fixed active mask at output scale."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, mask_value=0.0):
        super(SparseConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.mask_value = mask_value
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)

    def compute_output_mask(self, mask, output_size=None):
        if mask is None:
            return None

        with torch.no_grad():
            out_mask = (mask[:, :1] > 0).to(dtype=mask.dtype)
            if output_size is not None and out_mask.shape[-2:] != tuple(output_size):
                out_mask = F.interpolate(out_mask, size=output_size, mode="nearest")
        return out_mask

    def forward(self, x, mask=None):
        if mask is None:
            return self.conv(x)

        out = self.conv(x)
        out_mask = self.compute_output_mask(mask, out.shape[-2:]).to(device=out.device, dtype=out.dtype)
        if self.mask_value != 0.0:
            return out * out_mask + self.mask_value * (1.0 - out_mask)
        return out * out_mask


class PseudoSparseBatchNorm2d(nn.Module):
    """BatchNorm over active spatial positions only, per channel."""

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(PseudoSparseBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward(self, x, mask=None):
        if mask is None:
            return F.batch_norm(
                x, self.running_mean, self.running_var,
                self.weight, self.bias,
                self.training, self.momentum, self.eps
            )

        active = mask[:, :1].bool()
        if active.shape[-2:] != x.shape[-2:]:
            active = F.interpolate(active.float(), size=x.shape[-2:], mode="nearest").bool()

        active_idx = active.squeeze(1).nonzero(as_tuple=True)
        if active_idx[0].numel() == 0:
            return torch.zeros_like(x)

        bhwc = x.permute(0, 2, 3, 1)
        active_values = bhwc[active_idx]
        training = self.training and active_values.shape[0] > 1

        if self.training:
            self.num_batches_tracked += 1

        normed = F.batch_norm(
            active_values, self.running_mean, self.running_var,
            self.weight, self.bias, training, self.momentum, self.eps
        )

        out = torch.zeros_like(bhwc)
        out[active_idx] = normed
        return out.permute(0, 3, 1, 2)


class SparseConvBlock(nn.Module):
    """SparseConv + sparse BN + ReLU."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SparseConvBlock, self).__init__()
        self.sparse_conv = SparseConv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False
        )
        self.bn = PseudoSparseBatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, mask=None):
        out = self.sparse_conv(x, mask)
        new_mask = self.sparse_conv.compute_output_mask(mask, out.shape[-2:]) if mask is not None else None
        out = self.bn(out, new_mask)
        out = self.relu(out)
        return out, new_mask


class SparseConvBlockV2(nn.Module):
    """Sparse feature extraction block used by SparseFeatureNet."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 use_sparse_bn=True):
        super(SparseConvBlockV2, self).__init__()
        self.conv = SparseConv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False
        )
        self.use_sparse_bn = use_sparse_bn
        self.stride = stride
        self.bn = PseudoSparseBatchNorm2d(out_channels) if use_sparse_bn else nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, mask=None):
        out = self.conv(x, mask)
        new_mask = self.conv.compute_output_mask(mask, out.shape[-2:]) if mask is not None else None
        if self.use_sparse_bn:
            out = self.bn(out, new_mask)
        else:
            out = self.bn(out)
        out = self.relu(out)
        return out, new_mask


class MaskGenerator(nn.Module):
    """Generate active masks at all feature scales."""

    def __init__(self, mask_ratio=0.1):
        super(MaskGenerator, self).__init__()
        self.mask_ratio = mask_ratio

    def forward(self, x):
        b, _, h, w = x.shape
        h4, w4 = h // 4, w // 4
        mask_base = (torch.rand(b, 1, h4, w4, device=x.device, dtype=x.dtype) > self.mask_ratio).float()
        masks = {
            "stage1": mask_base,
            "stage2": F.interpolate(mask_base, size=(h // 2, w // 2), mode="nearest"),
            "stage3": F.interpolate(mask_base, size=(h, w), mode="nearest"),
        }
        return masks, x * masks["stage3"]


class DenseProjection(nn.Module):
    """Densify sparse features before decoding."""

    def __init__(self, in_channels, out_channels):
        super(DenseProjection, self).__init__()
        self.sparse_bn = PseudoSparseBatchNorm2d(in_channels)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, mask):
        return self.conv(self.sparse_bn(x, mask))


if __name__ == "__main__":
    x = torch.randn(2, 16, 32, 32)
    mask = torch.ones(2, 1, 32, 32)
    mask[:, :, 8:24, 8:24] = 0
    sparse_conv = SparseConv2d(16, 32, 3, padding=1)
    out = sparse_conv(x, mask)
    out_mask = sparse_conv.compute_output_mask(mask, out.shape[-2:])
    print(f"SparseConv2d: input {x.shape} -> output {out.shape}, mask {out_mask.shape}")
