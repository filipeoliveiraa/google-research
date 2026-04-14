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

import abc
import os

from matplotlib import colormaps
import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
from torchvision import transforms


@torch.no_grad()
def convert2pca(x, n_components=3, norm=True):
  B, C, H, W = x.shape
  if C < 3:
    x = x.repeat(1, 3, 1, 1)
    C = x.shape[1]

  x_reshaped = x.movedim(1, -1).reshape(B * H * W, C).cpu().detach()

  # Apply PCA
  pca = PCA(n_components=n_components)
  features_pca = pca.fit_transform(x_reshaped.numpy())

  # Reshape back to [3, H, W]
  x_pca = (
      torch.tensor(features_pca, dtype=torch.float32)
      .reshape(B, H, W, n_components)
      .movedim(-1, 1)
  )

  if norm:
    x_pca = (x_pca - x_pca.min()) / (x_pca.max() - x_pca.min()).clamp(1e-8)

  return (
      x_pca,
      torch.from_numpy(pca.components_),
      torch.from_numpy(pca.singular_values_),
  )


class DisplayBase:
  __metaclass__ = abc.ABCMeta

  def __init__(self, alpha):
    self.alpha = alpha
    self.imgs = []
    self.to_pil = transforms.ToPILImage()
    self.cmap = self.color_map()

  def _overlay_mask(
      self, image, mask, collate=False
  ):  # , contour_thickness=None):

    if image.shape != mask.shape:
      raise ValueError('Image and mask must have the same dimensions')

    blend = self.alpha * image + (1 - self.alpha) * mask

    blend = blend.astype(np.uint8)

    if collate:
      return np.concatenate([image, blend], -2)

    # img = im.copy()
    # img[ann > 0] = fg[ann > 0]

    # if contour_thickness:  # pragma: no cover
    #    import cv2
    #    for obj_id in np.unique(mask[mask > 0]):
    #        contours = cv2.findContours((mask == obj_id).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    #        cv2.drawContours(blend, contours[0], -1, colors[obj_id].tolist(), contour_thickness)

    return blend

  def embed(self, images, ann, pred2rgb=False, maskout=None, **kwargs):
    eps = 1e-6

    if images.shape != ann.shape or pred2rgb:
      ann = self.pred2rgb(ann)
      if not maskout is None:
        ann = ann * maskout.type_as(ann)

    assert (
        ann.shape == images.shape
    ), f'Shape mismatch: {ann.shape} vs {images.shape}'

    if images.is_floating_point():
      assert images.min() + eps >= 0.0 and images.max() - eps <= 1.0, (
          f'Images are not normalized [{images.min().item()},'
          f' {images.max().item()}]'
      )
      images = (255.0 * images).to(torch.uint8)

    if ann.is_floating_point():
      assert (
          ann.min() >= 0.0 and ann.max() <= 1.0
      ), f'Images are not normalized [{ann.min().item()}, {ann.max().item()}]'
      ann = (255.0 * ann).to(torch.uint8)

    images = images.movedim(1, -1)
    ann = ann.movedim(1, -1)

    for image, mask in zip(images, ann):
      image = np.asarray(image, dtype=np.uint8)
      mask = np.asarray(mask, dtype=np.uint8)
      self.imgs += [self._overlay_mask(image, mask, **kwargs)]

  def reset(self):
    self.imgs = []

  def __getitem__(self, index):
    return self.imgs[index]

  def show(self, index):
    img = self.to_pil(self.imgs[index])
    display(img)

  def tensor_images(self):
    return torch.stack([torch.from_numpy(x).movedim(-1, 0) for x in self.imgs])

  def save(self, output_dir, offset=0, suffix=''):
    if not os.path.isdir(output_dir):
      os.makedirs(output_dir)

    for ii, img in enumerate(self.imgs):
      output_fn = os.path.join(output_dir, f'{(ii+offset):03d}{suffix}.png')
      img = self.to_pil(self.imgs[ii])
      img.save(output_fn)


class DisplaySeg(DisplayBase):

  def pred2rgb(self, pred):
    ann = pred.squeeze(1)
    ann = torch.from_numpy(self.cmap[ann]).movedim(-1, 1)
    return ann

  def color_map(self, N=256, normalized=False):
    """Python implementation of the color map function for the PASCAL VOC data set.

    Official Matlab version can be found in the PASCAL VOC devkit
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
    """

    def bitget(byteval, idx):
      return (byteval & (1 << idx)) != 0

    cmap = np.zeros((N, 3), dtype='uint8')
    for i in range(N):
      r = g = b = 0
      c = i
      for j in range(8):
        r = r | (bitget(c, 0) << 7 - j)
        g = g | (bitget(c, 1) << 7 - j)
        b = b | (bitget(c, 2) << 7 - j)
        c = c >> 3

      cmap[i] = np.array([r, g, b])

    cmap = np.asarray(cmap, dtype=np.uint8)
    cmap[-1] = np.array([0, 0, 0])
    return cmap


class DisplayFeats(DisplayBase):

  def pred2rgb(self, pred):
    outs_rgb, _, _ = convert2pca(pred)
    return outs_rgb

  def color_map(self, N=256, normalized=False):
    pass


class DisplayConf(DisplayBase):

  def normalize(self, confd):
    confd = confd.clamp(0, 1)
    confd = np.asarray(confd, dtype=np.float32)
    return (255 * confd).astype(np.int32)

  def pred2rgb(self, pred):
    prednorm = self.normalize(pred)
    ann = prednorm.squeeze(1)
    ann = torch.from_numpy(self.cmap[ann]).movedim(-1, 1)
    return ann

  def color_map(self):
    cmap = np.asarray(colormaps.get_cmap('inferno').colors)
    # cmap = np.flip(cmap, axis=0)
    cmap = (256 * cmap).astype(np.uint8)
    cmap[0] = np.array([0, 0, 0])
    return cmap


class DisplayDepth(DisplayBase):

  def normalize(self, depth, maxval):
    # expecting depth between 0 and 10
    # converting to [0, 255]
    depth = depth.clamp(0, maxval - 1e-3) / maxval
    depth = np.asarray(depth, dtype=np.float32)
    return (255 * depth).astype(np.int32)

  def pred2rgb(self, pred):
    prednorm = self.normalize(pred, maxval=1)
    ann = prednorm.squeeze(1)
    ann = torch.from_numpy(self.cmap[ann]).movedim(-1, 1)
    return ann

  def color_map(self):
    cmap = np.asarray(colormaps.get_cmap('inferno').colors)
    cmap = np.flip(cmap, axis=0)
    cmap = (256 * cmap).astype(np.uint8)
    cmap[0] = np.array([0, 0, 0])
    return cmap


class DisplayNormals(DisplayBase):
  """Equivalent class for visualizing 3D surface normals.

  It assumes the input 'pred' is a PyTorch tensor with shape (N, 3, H, W)
  where channels 0, 1, and 2 correspond to the X, Y, and Z components
  of the normal vectors, and each component is normalized in the range [-1, 1].

  This class maps these components directly to RGB values for visualization:
  - X component [-1, 1] -> R channel [0, 255]
  - Y component [-1, 1] -> G channel [0, 255]
  - Z component [-1, 1] -> B channel [0, 255]
  """

  def color_map(self):
    return None

  def pred2rgb(self, pred, **kwargs):
    """Converts a normal map prediction tensor to an RGB visualization tensor.

    Args:
        pred (torch.Tensor): Normal map tensor with shape (N, 3, H, W) and
          values expected in the range [-1, 1].

    Returns:
        torch.Tensor: RGB visualization tensor with shape (N, 3, H, W)
                      and values as torch.uint8 in the range [0, 255].
    """
    pred = F.normalize(pred, dim=1, p=2)

    # 1. Map the tensor from range [-1, 1] to range [0, 1]
    # (pred * 0.5) maps to [-0.5, 0.5]
    # + 0.5 maps to [0, 1]
    rgb_tensor = (pred * 0.5) + 0.5

    # 2. Scale from [0, 1] to [0, 255]
    rgb_tensor = rgb_tensor * 255.0

    # 3. Clamp values to ensure they are within the valid [0, 255] range
    #    (in case the input slightly exceeded the [-1, 1] bounds)
    rgb_tensor = torch.clamp(rgb_tensor, 0, 255)

    # 4. Convert to uint8 data type to match the output type of the
    #    DisplayDepth class (which gets uint8 from the colormap)
    return rgb_tensor.to(torch.uint8)


class DisplayFlow(DisplayBase):

  def pred2rgb(self, flow):
    return flow_to_image(flow)

  def color_map(self):
    pass


# Flow visualization code used from https://github.com/tomrunia/OpticalFlow_Visualization


# MIT License
#
# Copyright (c) 2018 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-08-03


def make_colorwheel():
  """Generates a color wheel for optical flow visualization as presented in:

      Baker et al. "A Database and Evaluation Methodology for Optical Flow"
      (ICCV, 2007)
      URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

  Code follows the original C++ source code of Daniel Scharstein.
  Code follows the the Matlab source code of Deqing Sun.

  Returns:
      torch.Tensor: Color wheel
  """

  RY = 15
  YG = 6
  GC = 4
  CB = 11
  BM = 13
  MR = 6

  ncols = RY + YG + GC + CB + BM + MR
  colorwheel = torch.zeros((ncols, 3))
  col = 0

  # RY
  colorwheel[0:RY, 0] = 255
  colorwheel[0:RY, 1] = torch.floor(255 * torch.arange(0, RY) / RY)
  col = col + RY
  # YG
  colorwheel[col : col + YG, 0] = 255 - torch.floor(
      255 * torch.arange(0, YG) / YG
  )
  colorwheel[col : col + YG, 1] = 255
  col = col + YG
  # GC
  colorwheel[col : col + GC, 1] = 255
  colorwheel[col : col + GC, 2] = torch.floor(255 * torch.arange(0, GC) / GC)
  col = col + GC
  # CB
  colorwheel[col : col + CB, 1] = 255 - torch.floor(255 * torch.arange(CB) / CB)
  colorwheel[col : col + CB, 2] = 255
  col = col + CB
  # BM
  colorwheel[col : col + BM, 2] = 255
  colorwheel[col : col + BM, 0] = torch.floor(255 * torch.arange(0, BM) / BM)
  col = col + BM
  # MR
  colorwheel[col : col + MR, 2] = 255 - torch.floor(255 * torch.arange(MR) / MR)
  colorwheel[col : col + MR, 0] = 255
  return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
  """Applies the flow color wheel to (possibly clipped) flow components u and v.

  According to the C++ source code of Daniel Scharstein
  According to the Matlab source code of Deqing Sun

  Args:
      u (np.ndarray): Input horizontal flow of shape [H,W]
      v (np.ndarray): Input vertical flow of shape [H,W]
      convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to
        False.

  Returns:
      np.ndarray: Flow visualization image of shape [H,W,3]
  """
  flow_image = torch.zeros(u.shape[0], 3, u.shape[1], u.shape[2])
  colorwheel = make_colorwheel()  # shape [55x3]
  ncols = colorwheel.shape[0]
  rad = torch.sqrt(torch.square(u) + torch.square(v))
  a = torch.arctan2(-v, -u) / np.pi
  fk = (a + 1) / 2 * (ncols - 1)
  k0 = torch.floor(fk).int()
  k1 = k0 + 1
  k1[k1 == ncols] = 0
  f = fk - k0
  for i in range(colorwheel.shape[1]):
    tmp = colorwheel[:, i]
    col0 = tmp[k0] / 255.0
    col1 = tmp[k1] / 255.0
    col = (1 - f) * col0 + f * col1
    idx = rad <= 1
    col[idx] = 1 - rad[idx] * (1 - col[idx])
    col[~idx] = col[~idx] * 0.75  # out of range
    # Note the 2-i => BGR instead of RGB
    ch_idx = 2 - i if convert_to_bgr else i
    flow_image[:, ch_idx, :, :] = col

  return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
  """Expects a two dimensional flow image of shape.

  Args:
      flow_uv (np.ndarray): Flow UV image of shape [B,H,W,2]
      clip_flow (float, optional): Clip maximum of flow values. Defaults to
        None.
      convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to
        False.

  Returns:
      np.ndarray: Flow visualization image of shape [B,H,W,3]
  """
  assert flow_uv.ndim == 4, 'input flow must have three dimensions'
  assert flow_uv.shape[-1] == 2, 'input flow must have shape [B,H,W,2]'
  flow_uv = flow_uv.cpu().detach()

  if clip_flow is not None:
    flow_uv = flow_uv.clamp(0, clip_flow)

  u = flow_uv[:, :, :, 0]
  v = flow_uv[:, :, :, 1]
  rad = torch.sqrt(torch.square(u) + torch.square(v))
  rad_max = torch.max(rad)
  epsilon = 1e-5
  u = u / (rad_max + epsilon)
  v = v / (rad_max + epsilon)

  return flow_uv_to_colors(u, v, convert_to_bgr)


def overlay_masks(images, masks, lut, alpha=0.5):
  """Overlays segmentation masks on images with transparency and white boundaries.

  Args:
      images: Tensor [B, 3, H, W], float [0, 1].
      masks:  Tensor [B, H, W], integer indices.
      lut:    Tensor [256, 3], RGB colors.
      alpha:  Float, transparency of the mask (0=image only, 1=mask only).

  Returns:
      Tensor [B, 3, H, W] with the visualization.
  """
  # 1. Colorize the integer masks
  # lut[masks] gives [B, H, W, 3], permute to [B, 3, H, W]
  masks_rgb = lut.to(masks.device)[masks.long()].permute(0, 3, 1, 2)
  masks_rgb = masks_rgb / 255.0

  # 2. Blend mask with original image
  blended = alpha * masks_rgb + (1.0 - alpha) * images

  # 3. Detect Boundaries using Morphological Gradient
  # We need [B, 1, H, W] float for max_pool2d
  masks_float = masks.unsqueeze(1).float()

  # Dilation: max value in 3x3 neighborhood
  dilated = F.max_pool2d(masks_float, kernel_size=3, stride=1, padding=1)

  # Erosion: min value in 3x3 neighborhood (implemented as -max(-x))
  eroded = -F.max_pool2d(-masks_float, kernel_size=3, stride=1, padding=1)

  # Edges exist where the max neighbor != min neighbor (indices changed)
  # This creates a boundary roughly 2 pixels wide for good visibility
  edges = (dilated - eroded) > 0

  # 4. Apply white boundaries
  # Expand edges to 3 channels to match image
  edges = edges.repeat(1, 3, 1, 1)

  # Clone to avoid modifying gradients/inputs in place
  final_vis = blended.clone()
  final_vis[edges] = 1.0  # Set boundary pixels to White

  return final_vis
