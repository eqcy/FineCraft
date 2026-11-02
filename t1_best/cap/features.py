"""Identical predicted inputs for action training and deployment; no object state."""
import torch
from .schema import VISUAL_DIM


def visual_features(model, rgb):
    prediction = model(rgb)
    valid = prediction['visibility_logits'].sigmoid()[..., None]
    points = torch.cat((prediction['uv'] * valid, valid), -1)
    # Retain the exact old 48-feature order before appending new cap information.
    legacy = points[:, :, :8].flatten(1)
    cap = points[:, :, 8:].flatten(1)
    centers = prediction['uv'][:, :, [9, 5]]  # cap middle, TCP
    offsets = rgb.new_tensor([(x, y) for y in [-2/28, 0, 2/28]
                              for x in [-2/28, 0, 2/28]])
    grid = (centers[:, :, :, None] + offsets).reshape(len(rgb) * 2, 2, 9, 2)
    maps = prediction['spatial_maps'].reshape(len(rgb) * 2, 128, 56, 56)
    local = torch.nn.functional.grid_sample(maps, grid, mode='bilinear',
                    padding_mode='border', align_corners=False).mean(-1)
    local = local.transpose(1, 2).reshape(len(rgb), 2, 2, 128)
    context = torch.cat((local, prediction['context'][:, :, None]), 2).flatten(1)
    result = torch.cat((legacy, cap, context), 1)
    assert result.shape == (len(rgb), VISUAL_DIM)
    return result
