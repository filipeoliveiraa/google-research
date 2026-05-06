# coding=utf-8
import torch
import torch.nn.functional as F
import torch.nn as nn

#
# Optimized Helper Modules (No changes needed)
#

class _GatherAll9(nn.Module):
    """
    Gathers all 9 pixels in a 3x3 (dilated) neighborhood.
    """
    def __init__(self, dilations=[1]):
        super(_GatherAll9, self).__init__()
        self.dilations = dilations

        # Kernel to gather all 9 pixels
        weight = torch.zeros(9, 1, 3, 3)
        weight[0, 0, 0, 0] = 1
        weight[1, 0, 0, 1] = 1
        weight[2, 0, 0, 2] = 1
        weight[3, 0, 1, 0] = 1
        weight[4, 0, 1, 1] = 1  # Center pixel
        weight[5, 0, 1, 2] = 1
        weight[6, 0, 2, 0] = 1
        weight[7, 0, 2, 1] = 1
        weight[8, 0, 2, 2] = 1
        self.register_buffer('kernel', weight)

    def forward(self, x):
        # x: [B, K, H, W]
        B, K, H, W = x.size()
        x = x.view(B * K, 1, H, W)

        x_affs = []
        for d in self.dilations:
            x_pad = F.pad(x, [d] * 4, mode='replicate')
            x_aff = F.conv2d(x_pad, self.kernel, dilation=d)
            x_affs.append(x_aff)

        x_aff = torch.cat(x_affs, 1)
        # Reshape to [B, K, P, H, W]
        return x_aff.view(B, K, -1, H, W)

class _Gather8Neighbors(nn.Module):
    """
    Gathers the 8 neighbor pixels in a 3x3 (dilated) neighborhood.
    """
    def __init__(self, dilations=[1]):
        super(_Gather8Neighbors, self).__init__()
        self.dilations = dilations

        # Kernel to gather 8 neighbor pixels
        weight = torch.zeros(8, 1, 3, 3)
        weight[0, 0, 0, 0] = 1
        weight[1, 0, 0, 1] = 1
        weight[2, 0, 0, 2] = 1
        weight[3, 0, 1, 0] = 1
        # No center pixel (index 4)
        weight[4, 0, 1, 2] = 1
        weight[5, 0, 2, 0] = 1
        weight[6, 0, 2, 1] = 1
        weight[7, 0, 2, 2] = 1
        self.register_buffer('kernel', weight)

    def forward(self, x):
        # x: [B, K, H, W]
        B, K, H, W = x.size()
        x = x.reshape(B * K, 1, H, W)

        x_affs = []
        for d in self.dilations:
            x_pad = F.pad(x, [d] * 4, mode='replicate')
            x_aff = F.conv2d(x_pad, self.kernel, dilation=d)
            x_affs.append(x_aff)

        x_aff = torch.cat(x_affs, 1)
        # Reshape to [B, K, P_8, H, W]
        return x_aff.view(B, K, -1, H, W)

#
# Revised Main Module (Iterative Guided Refiner)
#
class PAMR(nn.Module):
    """
    Refines a target_map to align with the structures of a guide_map.

    This module implements a local, iterative, affinity-based refinement
    process, equivalent to a mean-field approximation for a Conditional
    Random Field (CRF).
    """

    def __init__(self, num_iter=1, dilations=[1], resize=None, sigma=0.1):
        super(PAMR, self).__init__()

        self.num_iter = num_iter
        self.dilations = dilations
        self.resize = resize
        self.sigma = sigma
        self.num_dilations = len(dilations)
        self.num_patches_9 = 9 * self.num_dilations # P_9 (total patches from 9-gather)

        # Module to get 9-pixel neighborhood (for std and affinity)
        self.gather_9 = _GatherAll9(dilations)

        # Module to get 8-pixel neighborhood (for target map update)
        self.gather_8 = _Gather8Neighbors(dilations)

        # Create indices to separate center and neighbors from the 9-pixel patch

        # Center pixels are at indices 4, 13, 22, ... (0-indexed)
        center_inds = torch.arange(4, self.num_patches_9, 9, dtype=torch.long)

        # Neighbor pixels are all others
        neighbor_inds_from_9 = torch.tensor(
            [i for i in range(self.num_patches_9) if i % 9 != 4], dtype=torch.long
        )

        self.register_buffer('center_inds', center_inds, persistent=False)
        self.register_buffer('neighbor_inds_from_9', neighbor_inds_from_9, persistent=False)

    @torch.no_grad()
    @torch.amp.autocast(device_type="cuda", enabled=True)
    def forward(self, guide_map, target_map):
        """
        Refines the target_map using the guide_map.

        Args:
            guide_map (torch.Tensor): The guidance signal (e.g., features).
                                      Shape: [B, K, H, W]
            target_map (torch.Tensor): The map to be refined (e.g., mask).
                                       Shape: [B, C, H, W]

        Returns:
            torch.Tensor: The refined target_map. Shape: [B, C, H, W]
        """
        resize_to = guide_map.size()[-2:] if self.resize is None else self.resize

        # Ensure target_map matches guide_map spatial dimensions
        target_map = F.interpolate(target_map, size=resize_to,
                                   mode="bilinear", align_corners=True)

        guide_map = F.interpolate(guide_map, size=resize_to,
                                  mode="bilinear", align_corners=True)

        B, K, H, W = guide_map.size()
        _, C, _, _ = target_map.size()

        # --- 1. Compute Affinities (from guide_map) ---

        # Gather 9-pixel neighborhoods: [B, K, P_9, H, W]
        g_neighbors_9 = self.gather_9(guide_map)

        # --- Calculate std dev ---
        # g_std: [B, K, 1, H, W]
        g_std = g_neighbors_9.std(2, keepdim=True)

        # --- Calculate affinities ---

        # Get center pixels: [B, K, D, H, W] (D = num_dilations)
        g_center = torch.index_select(g_neighbors_9, 2, self.center_inds)

        # Get 8-neighbor pixels: [B, K, P_8, H, W]
        g_8_neighbors = torch.index_select(g_neighbors_9, 2, self.neighbor_inds_from_9)

        # Reshape for broadcasted subtraction
        # [B, K, D, 1, H, W]
        g_center_rs = g_center.view(B, K, self.num_dilations, 1, H, W)
        # [B, K, D, 8, H, W]
        g_8_neighbors_rs = g_8_neighbors.view(B, K, self.num_dilations, 8, H, W)

        # Compute abs(center - neighbor)
        # Output: [B, K, D, 8, H, W]
        aff_val = torch.abs(g_center_rs - g_8_neighbors_rs)

        # Reshape back to [B, K, P_8, H, W]
        aff = aff_val.view(B, K, -1, H, W)

        # --- Final affinity calculation ---
        aff = -aff / (1e-5 + self.sigma * g_std)

        # aff: [B, 1, P_8, H, W]
        aff = aff.mean(1, keepdim=True)
        aff = F.softmax(aff, 2)
        # aff now contains the normalized affinities, fixed for all iterations

        # --- 2. Iterative Refinement (using einsum) ---
        for _ in range(self.num_iter):
            # Gather 8-neighbor pixels from *current* target_map
            # t_neighbors: [B, C, P_8, H, W]
            t_neighbors = self.gather_8(target_map)

            # --- Fused Multiply & Sum using einsum ---
            # This replaces: target_map = (t_neighbors * aff).sum(2)
            # 'bcphw': target_map neighbors [B, C, P, H, W]
            # 'biphw': affinity map [B, 1, P, H, W] (i is the singleton dim)
            # 'bchw': output map [B, C, H, W]
            # This fuses the broadcasted multiply and the sum over 'p'
            target_map = torch.einsum('bcphw, biphw -> bchw', t_neighbors, aff)

        return target_map
