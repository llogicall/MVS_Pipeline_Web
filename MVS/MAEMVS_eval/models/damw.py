"""Dynamic adaptive multi-task loss weighting.

This module follows the GradNorm idea used by the reference implementation:
task weights must participate in the gradient-norm objective, otherwise they
cannot be optimized by the balancing loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


class DAMWOptimizer(nn.Module):
    """Compatibility wrapper around learnable two-task weights."""

    def __init__(self, initial_w_mvs=1.0, initial_w_mae=1.0, lr=0.05, beta1=0.9, beta2=0.999):
        super(DAMWOptimizer, self).__init__()
        self.weights = nn.Parameter(torch.tensor([initial_w_mvs, initial_w_mae], dtype=torch.float32))
        self.optimizer = Adam([self.weights], lr=lr, betas=(beta1, beta2))

    @property
    def w_mvs(self):
        return self.weights[0]

    @property
    def w_mae(self):
        return self.weights[1]

    def positive_weights(self):
        return F.softplus(self.weights) + 1e-6

    def get_weights(self):
        weights = self.positive_weights().detach()
        return {"w_mvs": float(weights[0]), "w_mae": float(weights[1])}

    def forward(self, gradients_dict):
        norms_mvs = []
        norms_mae = []
        weights = self.positive_weights()
        for key, grad in gradients_dict.items():
            if grad is None:
                continue
            if "mvs" in key.lower():
                norms_mvs.append(torch.norm(weights[0] * grad))
            elif "mae" in key.lower():
                norms_mae.append(torch.norm(weights[1] * grad))

        device = self.weights.device
        avg_mvs = torch.stack(norms_mvs).mean() if norms_mvs else torch.tensor(0.0, device=device)
        avg_mae = torch.stack(norms_mae).mean() if norms_mae else torch.tensor(0.0, device=device)
        target = torch.stack([avg_mvs.detach(), avg_mae.detach()]).mean()
        grad_loss = torch.abs(avg_mvs - target) + torch.abs(avg_mae - target)
        return grad_loss, self.get_weights()

    def step(self, gradients_dict):
        self.optimizer.zero_grad()
        grad_loss, _ = self(gradients_dict)
        grad_loss.backward()
        self.optimizer.step()
        self.renormalize()
        return float(grad_loss.detach()), self.get_weights()

    def renormalize(self):
        with torch.no_grad():
            positive = self.positive_weights()
            target_sum = torch.tensor(2.0, device=positive.device, dtype=positive.dtype)
            normalized = positive * target_sum / positive.sum().clamp_min(1e-8)
            self.weights.copy_(torch.log(torch.expm1(normalized.clamp_min(1e-6))))


class GradientTracker(nn.Module):
    """Small compatibility cache for older training code paths."""

    def __init__(self, num_stages=3):
        super(GradientTracker, self).__init__()
        self.num_stages = num_stages
        self.mvs_gradients = {}
        self.mae_gradients = {}

    def register_gradients(self, features_dict, loss_name="mvs"):
        target = self.mvs_gradients if loss_name == "mvs" else self.mae_gradients
        target.clear()
        for stage_key, feat in features_dict.items():
            if isinstance(feat, torch.Tensor) and feat.grad is not None:
                target[stage_key] = feat.grad.detach().clone()

    def get_combined_gradients(self):
        gradients = {}
        for stage_key, grad in self.mvs_gradients.items():
            gradients[f"{stage_key}_mvs"] = grad
        for stage_key, grad in self.mae_gradients.items():
            gradients[f"{stage_key}_mae"] = grad
        return gradients

    def compute_grad_norms(self):
        return {key: float(torch.norm(value)) for key, value in self.get_combined_gradients().items()}

    def clear(self):
        self.mvs_gradients.clear()
        self.mae_gradients.clear()


class MultiTaskLoss(nn.Module):
    """Weighted MVS + MAE loss with GradNorm-style DAMW updates."""

    def __init__(self, use_damw=True, initial_w_mvs=12.0, initial_w_mae=1.0, lr=0.05):
        super(MultiTaskLoss, self).__init__()
        self.use_damw = use_damw
        self.weight_sum = 2.0
        if use_damw:
            self.raw_weights = nn.Parameter(torch.tensor([initial_w_mvs, initial_w_mae], dtype=torch.float32))
        else:
            self.register_buffer("fixed_weights", torch.tensor([initial_w_mvs, initial_w_mae], dtype=torch.float32))

    def _positive_weights(self):
        if self.use_damw:
            return F.softplus(self.raw_weights) + 1e-6
        return self.fixed_weights

    def forward(self, loss_mvs, loss_mae, gradients_dict=None):
        weights = self._positive_weights().to(device=loss_mvs.device, dtype=loss_mvs.dtype)
        total_loss = weights[0] * loss_mvs + weights[1] * loss_mae
        loss_dict = {
            "total_loss": total_loss,
            "loss_mvs": loss_mvs,
            "loss_mae": loss_mae,
            "grad_loss": torch.zeros((), device=loss_mvs.device, dtype=loss_mvs.dtype),
            "w_mvs": float(weights[0].detach()),
            "w_mae": float(weights[1].detach()),
        }
        return total_loss, loss_dict

    def _task_grad_norm(self, loss, weight, shared_features):
        norms = []
        for _, feat in shared_features.items():
            if not isinstance(feat, torch.Tensor) or not feat.requires_grad:
                continue
            grad = torch.autograd.grad(
                loss, feat, retain_graph=True, create_graph=False, allow_unused=True
            )[0]
            if grad is not None:
                norms.append(torch.norm(weight * grad.detach()))
        if not norms:
            return weight * 0.0
        return torch.stack(norms).mean()

    def compute_gradnorm_loss(self, loss_mvs, loss_mae, shared_features):
        device = loss_mvs.device
        weights = self._positive_weights().to(device=device)
        norm_mvs = self._task_grad_norm(loss_mvs, weights[0], shared_features)
        norm_mae = self._task_grad_norm(loss_mae, weights[1], shared_features)
        mean_norm = torch.stack([norm_mvs.detach(), norm_mae.detach()]).mean()
        grad_loss = torch.abs(norm_mvs - mean_norm) + torch.abs(norm_mae - mean_norm)
        return grad_loss

    def assign_gradnorm_gradients(self, loss_mvs, loss_mae, shared_features):
        if not self.use_damw:
            return 0.0, self.get_weights()

        if self.raw_weights.grad is not None:
            self.raw_weights.grad.detach_()
            self.raw_weights.grad.zero_()

        grad_loss = self.compute_gradnorm_loss(loss_mvs, loss_mae, shared_features)
        self.raw_weights.grad = torch.autograd.grad(grad_loss, self.raw_weights, retain_graph=True)[0]
        return float(grad_loss.detach()), self.get_weights()

    def update_weights(self, loss_mvs, loss_mae, shared_features):
        return self.assign_gradnorm_gradients(loss_mvs, loss_mae, shared_features)

    def renormalize_weights(self):
        if not self.use_damw:
            return self.get_weights()
        with torch.no_grad():
            positive = self._positive_weights()
            target_sum = torch.tensor(self.weight_sum, device=positive.device, dtype=positive.dtype)
            normalized = positive * target_sum / positive.sum().clamp_min(1e-8)
            self.raw_weights.copy_(torch.log(torch.expm1(normalized.clamp_min(1e-6))).to(self.raw_weights.device))
        return self.get_weights()

    def step(self, gradients_dict):
        optimizer = DAMWOptimizer(initial_w_mvs=self.get_weights()["w_mvs"], initial_w_mae=self.get_weights()["w_mae"])
        return optimizer.step(gradients_dict)

    def get_weights(self):
        weights = self._positive_weights().detach()
        return {"w_mvs": float(weights[0]), "w_mae": float(weights[1])}


def gradient_balanced_loss(grad_norms, loss_mvs, loss_mae):
    all_norms = [float(norm.detach()) if isinstance(norm, torch.Tensor) else float(norm)
                 for norm in grad_norms.values()]
    avg_norm = sum(all_norms) / max(len(all_norms), 1) + 1e-8
    return loss_mvs / avg_norm + loss_mae / avg_norm


if __name__ == "__main__":
    loss_module = MultiTaskLoss(use_damw=True, initial_w_mvs=2.0, initial_w_mae=1.0)
    feat = torch.randn(1, 2, 4, 4, requires_grad=True)
    loss_mvs = (feat * 2).pow(2).mean()
    loss_mae = (feat * 0.2).pow(2).mean()
    print(loss_module.update_weights(loss_mvs, loss_mae, {"stage1": feat}))
