# coding=utf-8
# Copyright 2026 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F
from typing import Tuple, Dict

def flow_sample(flow, target):
    h, w = target.shape[-2:]

    # --- Step 2: Create a sampling grid for frame t+1 ---
    y_coords, x_coords = torch.meshgrid(
        torch.arange(h, device=target.device, dtype=torch.float32),
        torch.arange(w, device=target.device, dtype=torch.float32),
        indexing='ij'
    )
    grid_t = torch.stack([x_coords, y_coords], dim=-1)
    flow_t_to_t1_permuted = flow.permute(0, 2, 3, 1)

    grid_t1 = grid_t + flow_t_to_t1_permuted * h / 2

    # Normalize the grid to [-1, 1] for grid_sample
    grid_t1_norm = grid_t1.clone()
    grid_t1_norm[Ellipsis, 0] = 2.0 * grid_t1[Ellipsis, 0] / (w - 1) - 1.0
    grid_t1_norm[Ellipsis, 1] = 2.0 * grid_t1[Ellipsis, 1] / (h - 1) - 1.0

    # --- Step 3: Sample the depth map of frame t+1 at the new locations ---
    target_sampled = F.grid_sample(
        target,
        grid_t1_norm,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )

    return target_sampled

def compute_uniconf(flow_fwd, flow_bwd, depth, valid_depth_fn, temp):
    """
    Helper function to compute confidence for a single direction.

    Args:
        flow_fwd (torch.Tensor): Flow starting from current frame (e.g., t -> t+1).
        flow_bwd (torch.Tensor): Flow starting from target frame (e.g., t+1 -> t).
        depth (torch.Tensor): Depth map for current frame.
        temp (float): Temperature scaling.

    Returns:
        torch.Tensor: 3-channel confidence map (B, 3, H, W).
    """
    # 1. Warp the secondary flow to align with the current frame
    # We sample 'flow_bwd' using the coordinates from 'flow_fwd'
    flow_bwd_warped = flow_sample(flow_fwd, flow_bwd)

    # 2. Calculate cycle consistency error vector
    # Ideally: flow_fwd + flow_bwd_warped = 0
    consistency_vec = flow_fwd + flow_bwd_warped

    # 3. L2 Norm and Temperature scaling
    consistency_err = torch.norm(consistency_vec, p=2, dim=1, keepdim=True)
    flow_conf = (torch.exp(-consistency_err / temp) > 0.5).float()

    # 4. Depth Validity (Handle if depth is (B, H, W) or (B, 1, H, W))
    if depth.dim() == 3:
        depth = depth.unsqueeze(1)

    depth_conf = valid_depth_fn(depth).float()

    # 5. Stack channels: [Flow, Flow, Depth]
    return torch.cat([flow_conf, flow_conf, depth_conf], dim=1)


def compute_confidence(flow_fwd, flow_bwd, depth_t, depth_t1, valid_depth_fn, temp=1.0):
    """
    Computes 3-channel confidence masks for frame t and t+1.

    Args:
        flow_fwd (torch.Tensor): Flow from t -> t+1 (B, 2, H, W).
        flow_bwd (torch.Tensor): Flow from t+1 -> t (B, 2, H, W).
        depth_t (torch.Tensor): Depth map for t.
        depth_t1 (torch.Tensor): Depth map for t+1.
        temp (float): Temperature for sensitivity.

    Returns:
        tuple: (confidence_t, confidence_t1)
    """
    # Compute confidence for frame t
    # Primary flow is t->t+1, Secondary is t+1->t
    conf_t = compute_uniconf(flow_fwd, flow_bwd, depth_t, valid_depth_fn, temp)

    # Compute confidence for frame t+1
    # Primary flow is t+1->t, Secondary is t->t+1
    conf_t1 = compute_uniconf(flow_bwd, flow_fwd, depth_t1, valid_depth_fn, temp)

    return conf_t, conf_t1