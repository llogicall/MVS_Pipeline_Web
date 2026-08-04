import os

import torch
from torchvision.utils import save_image


def _first_image(tensor):
    if not isinstance(tensor, torch.Tensor):
        return None
    tensor = tensor.detach().float().cpu()
    if tensor.dim() == 4:
        tensor = tensor[:1]
    elif tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    else:
        return None
    return tensor


def _to_visual_image(tensor):
    tensor = _first_image(tensor)
    if tensor is None:
        return None

    if tensor.shape[1] > 3:
        tensor = tensor.mean(dim=1, keepdim=True)
    elif tensor.shape[1] == 2:
        pad = torch.zeros(tensor.shape[0], 1, tensor.shape[2], tensor.shape[3])
        tensor = torch.cat([tensor, pad], dim=1)

    min_val = tensor.amin(dim=(1, 2, 3), keepdim=True)
    max_val = tensor.amax(dim=(1, 2, 3), keepdim=True)
    return (tensor - min_val) / (max_val - min_val).clamp_min(1e-8)


def _flatten_visuals(visuals, prefix=""):
    flat = {}
    for key, value in visuals.items():
        name = f"{prefix}_{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_visuals(value, name))
        else:
            flat[name] = value
    return flat


def save_visualization_bundle(visuals, out_dir, step, prefix="train"):
    os.makedirs(out_dir, exist_ok=True)
    saved_paths = []
    for name, tensor in _flatten_visuals(visuals).items():
        image = _to_visual_image(tensor)
        if image is None:
            continue
        safe_name = name.replace("/", "_").replace("\\", "_").replace(" ", "_")
        path = os.path.join(out_dir, f"{prefix}_step{int(step):08d}_{safe_name}.png")
        save_image(image, path)
        saved_paths.append(path)
    return saved_paths


def collect_mvs_visuals(outputs, model=None):
    visuals = {}

    if "masks" in outputs:
        for stage_key, mask in outputs["masks"].items():
            visuals[f"mask_{stage_key}"] = mask

    if "mae_output" in outputs and outputs["mae_output"] is not None:
        visuals["mae_output"] = outputs["mae_output"]

    if model is not None:
        feature = getattr(model, "feature", None)
        if feature is not None:
            visuals.update(getattr(feature, "visual_cache", {}))
        mae_decoder = getattr(model, "mae_decoder", None)
        if mae_decoder is not None:
            visuals.update(getattr(mae_decoder, "visual_cache", {}))

    return visuals
