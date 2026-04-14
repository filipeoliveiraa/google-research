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

import math

import cv2
from func_utils import compute_confidence
from hydra.utils import instantiate
from raft.raft import RAFT
from raft.utils.utils import InputPadder
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import Compose

from .dinov2 import DINOv2
from .pamr import PAMR
from .util.blocks import FeatureFusionBlock, _make_scratch
from .util.transform import NormalizeImage, PrepareForNet, Resize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def edge_l1(x, y, sigma=1.0):
  return (1.0 - torch.exp(-y / sigma)) * torch.abs(x - y)


def flow_boundary_loss(pred_flow, gt_flow, conf, sigma=0.1, eps=1e-5):
  """Computes flow boundary loss"""

  grad_x = lambda x: torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
  grad_y = lambda x: torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])

  pred_dx = grad_x(pred_flow)
  pred_dy = grad_y(pred_flow)

  gt_dx = grad_x(gt_flow)
  gt_dy = grad_y(gt_flow)

  loss_dx = edge_l1(pred_dx, gt_dx, sigma)
  loss_dy = edge_l1(pred_dy, gt_dy, sigma)

  valid_dx = (conf[:, :, :, :-1] > 0) & (conf[:, :, :, 1:] > 0)
  valid_dy = (conf[:, :, :-1, :] > 0) & (conf[:, :, 1:, :] > 0)

  return loss_dx[valid_dx].mean() + loss_dy[valid_dy].mean()


class DepthCore:

  def __init__(self, scale=1.0, scale_disp=0.01, depth_tol=0.9, eps=0.1):
    self.scale = scale
    self.scale_disp = scale_disp
    self.eps = eps
    self.depth_tol = depth_tol

  def valid_depth(self, depth):
    return depth < self.depth_tol

  def convert2depth(self, scaled_disp):
    depth = torch.log(1.0 + self.scale / (scaled_disp + self.eps))
    return depth

  def disp2depth(self, disp):
    scaled_disp = disp * self.scale_disp
    depth = self.convert2depth(scaled_disp)
    disp_ignore = torch.zeros_like(disp)
    return depth / self.convert2depth(disp_ignore)


class SEARAFT_Args:

  name = "Tartan-C-T-TSKH432x960-M"
  dataset = "TSKH"
  gpus = [0, 1, 2, 3, 4, 5, 6, 7]

  use_var = True
  var_min = 0
  var_max = 10
  pretrain = "resnet34"
  initial_dim = 64
  block_dims = [64, 128, 256]
  radius = 4
  dim = 128
  num_blocks = 2
  iters = 4

  def __contains__(self, value):
    return hasattr(self, value)


class SeaFlow(nn.Module):

  def __init__(self, denorm=None):
    super().__init__()
    self.denorm = nn.Identity() if denorm is None else denorm
    self.args = SEARAFT_Args()
    self.model = RAFT(self.args)

    state_dict = torch.load(
        "/workdir/checkpoints/Tartan-C-T-TSKH432x960-M.pth",
        map_location=torch.device("cpu"),
        weights_only=True,
    )
    self.model.load_state_dict(state_dict, strict=False)

  @torch.no_grad()
  def forward(self, image1, image2):
    im1_255 = self.denorm(image1) * 255.0
    im2_255 = self.denorm(image2) * 255.0

    assert (
        im1_255.min().item() >= -1e-3 and im1_255.max().item() <= 255.0 + 1e-3
    )
    assert (
        im2_255.min().item() >= -1e-3 and im2_255.max().item() <= 255.0 + 1e-3
    )

    padder = InputPadder(im1_255.shape)

    im1_255pad, im2_255pad = padder.pad(im1_255, im2_255)
    output = self.model(
        im1_255pad, im2_255pad, iters=self.args.iters, test_mode=True
    )

    flow_final = output["flow"][-1]
    flow = padder.unpad(flow_final)

    flow[:, 0] *= 2 / flow.shape[-1]
    flow[:, 1] *= 2 / flow.shape[-2]

    return flow


def _make_fusion_block(features, use_bn, size=None):
  return FeatureFusionBlock(
      features,
      nn.ReLU(False),
      deconv=False,
      bn=use_bn,
      expand=False,
      align_corners=True,
      size=size,
  )


from einops import rearrange, repeat, reduce, pack, unpack


def cdist(x, y):
  x2 = reduce(x**2, "b n d -> b n", "sum")
  y2 = reduce(y**2, "b n d -> b n", "sum")
  xy = torch.einsum("b i d, b j d -> b i j", x, y) * -2
  return (
      (rearrange(x2, "b i -> b i 1") + rearrange(y2, "b j -> b 1 j") + xy)
      .clamp(min=0)
      .sqrt()
  )


class DPTHead(nn.Module):

  def __init__(
      self,
      in_channels,
      features=256,
      use_bn=False,
      out_channels=[256, 512, 1024, 1024],
      use_clstoken=False,
      out_dim=1,
      use_relu=True,
  ):
    super(DPTHead, self).__init__()

    self.use_clstoken = use_clstoken

    self.projects = nn.ModuleList([
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channel,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        for out_channel in out_channels
    ])

    self.resize_layers = nn.ModuleList([
        nn.ConvTranspose2d(
            in_channels=out_channels[0],
            out_channels=out_channels[0],
            kernel_size=4,
            stride=4,
            padding=0,
        ),
        nn.ConvTranspose2d(
            in_channels=out_channels[1],
            out_channels=out_channels[1],
            kernel_size=2,
            stride=2,
            padding=0,
        ),
        nn.Identity(),
        nn.Conv2d(
            in_channels=out_channels[3],
            out_channels=out_channels[3],
            kernel_size=3,
            stride=2,
            padding=1,
        ),
    ])

    if use_clstoken:
      self.readout_projects = nn.ModuleList()
      for _ in range(len(self.projects)):
        self.readout_projects.append(
            nn.Sequential(nn.Linear(2 * in_channels, in_channels), nn.GELU())
        )

    self.scratch = _make_scratch(
        out_channels,
        features,
        groups=1,
        expand=False,
    )

    self.scratch.stem_transpose = None

    self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
    self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
    self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
    self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

    head_features_1 = features
    head_features_2 = 32

    self.scratch.output_conv1 = nn.Conv2d(
        head_features_1,
        head_features_1 // 2,
        kernel_size=3,
        stride=1,
        padding=1,
    )
    self.scratch.output_conv2 = nn.Sequential(
        nn.Conv2d(
            head_features_1 // 2,
            head_features_2,
            kernel_size=3,
            stride=1,
            padding=1,
        ),
        nn.ReLU(True),
        nn.Conv2d(head_features_2, out_dim, kernel_size=1, stride=1, padding=0),
        nn.ReLU(True) if use_relu else nn.Identity(),
    )

  def forward_head(self, path_1, patch_h, patch_w):
    out = self.scratch.output_conv1(path_1)
    out = F.interpolate(
        out,
        (int(patch_h * 14), int(patch_w * 14)),
        mode="bilinear",
        align_corners=True,
    )
    out = self.scratch.output_conv2(out)

    return out

  def forward(self, out_features, patch_h, patch_w):
    out = []
    for i, x in enumerate(out_features):
      if self.use_clstoken:
        x, cls_token = x[0], x[1]
        readout = cls_token.unsqueeze(1).expand_as(x)
        x = self.readout_projects[i](torch.cat((x, readout), -1))
      else:
        x = x[0]

      x = x.permute(0, 2, 1).reshape(
          (x.shape[0], x.shape[-1], patch_h, patch_w)
      )

      x = self.projects[i](x)
      x = self.resize_layers[i](x)

      out.append(x)

    layer_1, layer_2, layer_3, layer_4 = out

    layer_1_rn = self.scratch.layer1_rn(layer_1)
    layer_2_rn = self.scratch.layer2_rn(layer_2)
    layer_3_rn = self.scratch.layer3_rn(layer_3)
    layer_4_rn = self.scratch.layer4_rn(layer_4)

    path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
    path_3 = self.scratch.refinenet3(
        path_4, layer_3_rn, size=layer_2_rn.shape[2:]
    )
    path_2 = self.scratch.refinenet2(
        path_3, layer_2_rn, size=layer_1_rn.shape[2:]
    )
    path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

    out = self.scratch.output_conv1(path_1)
    out = F.interpolate(
        out,
        (int(patch_h * 14), int(patch_w * 14)),
        mode="bilinear",
        align_corners=True,
    )
    out = self.scratch.output_conv2(out)

    return out, path_1


class SmallDPTHead(nn.Module):

  def __init__(
      self,
      in_channels,
      features=256,
      use_bn=False,
      out_channels=[256, 512, 1024, 1024],
      use_clstoken=False,
      out_dim=1,
      use_relu=True,
      patch_size=14,
  ):
    super(SmallDPTHead, self).__init__()

    self.patch_size = patch_size
    self.use_clstoken = use_clstoken

    self.projects = nn.ModuleList([
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channel,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        for out_channel in out_channels
    ])

    self.resize_layers = nn.ModuleList([
        nn.ConvTranspose2d(
            in_channels=out_channels[0],
            out_channels=out_channels[0],
            kernel_size=4,
            stride=4,
            padding=0,
        ),
        nn.ConvTranspose2d(
            in_channels=out_channels[1],
            out_channels=out_channels[1],
            kernel_size=2,
            stride=2,
            padding=0,
        ),
        nn.Identity(),
        nn.Conv2d(
            in_channels=out_channels[3],
            out_channels=out_channels[3],
            kernel_size=3,
            stride=2,
            padding=1,
        ),
    ])

    if use_clstoken:
      self.readout_projects = nn.ModuleList()
      for _ in range(len(self.projects)):
        self.readout_projects.append(
            nn.Sequential(nn.Linear(2 * in_channels, in_channels), nn.GELU())
        )

    self.scratch = _make_scratch(
        out_channels,
        features,
        groups=1,
        expand=False,
    )

    # head_features_1 = features
    head_features_2 = features // 2

    self.scratch.stem_transpose = None

    self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
    self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
    self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
    self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

    self.scratch.output_conv1 = nn.Conv2d(
        features, features // 2, kernel_size=3, stride=1, padding=1
    )
    self.scratch.output_conv2 = nn.Sequential(
        nn.Conv2d(
            features // 2, head_features_2, kernel_size=3, stride=1, padding=1
        ),
        nn.ReLU(True),
        nn.Conv2d(
            head_features_2, head_features_2, kernel_size=1, stride=1, padding=0
        ),
    )

  def forward(self, out_features, patch_h, patch_w):
    out = []
    for i, x in enumerate(out_features):
      if self.use_clstoken:
        x, cls_token = x[0], x[1]
        readout = cls_token.unsqueeze(1).expand_as(x)
        x = self.readout_projects[i](torch.cat((x, readout), -1))
      else:
        x = x[0]

      x = x.permute(0, 2, 1).reshape(
          (x.shape[0], x.shape[-1], patch_h, patch_w)
      )

      x = self.projects[i](x)
      x = self.resize_layers[i](x)

      out.append(x)

    layer_1, layer_2, layer_3, layer_4 = out

    layer_1_rn = self.scratch.layer1_rn(layer_1)
    layer_2_rn = self.scratch.layer2_rn(layer_2)
    layer_3_rn = self.scratch.layer3_rn(layer_3)
    layer_4_rn = self.scratch.layer4_rn(layer_4)

    path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
    path_3 = self.scratch.refinenet3(
        path_4, layer_3_rn, size=layer_2_rn.shape[2:]
    )
    path_2 = self.scratch.refinenet2(
        path_3, layer_2_rn, size=layer_1_rn.shape[2:]
    )
    path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

    out = self.scratch.output_conv1(path_1)
    out = F.interpolate(
        out,
        (int(patch_h * self.patch_size), int(patch_w * self.patch_size)),
        mode="bilinear",
        align_corners=True,
    )
    out = self.scratch.output_conv2(out)

    return out


class DepthAnythingV2(nn.Module):

  def __init__(
      self,
      encoder="vitl",
      features=256,
      out_channels=[256, 512, 1024, 1024],
      use_bn=False,
      use_clstoken=False,
  ):
    super(DepthAnythingV2, self).__init__()

    self.intermediate_layer_idx = {
        "vits": [2, 5, 8, 11],
        "vitb": [2, 5, 8, 11],
        "vitl14": [4, 11, 17, 23],
        "vitg": [9, 19, 29, 39],
    }

    self.encoder = encoder
    self.pretrained = DINOv2(model_name=encoder)
    self.depth_head = DPTHead(
        self.pretrained.embed_dim,
        features,
        use_bn,
        out_channels=out_channels,
        use_clstoken=use_clstoken,
    )

  def forward(self, x):
    patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14

    with torch.no_grad():
      features = self.pretrained.get_intermediate_layers(
          x, self.intermediate_layer_idx[self.encoder], return_class_token=True
      )

    depth, dec_feats = self.depth_head(features, patch_h, patch_w)
    depth = F.relu(depth)

    return depth.squeeze(1), dec_feats

  def forward_head(self, x, y):
    patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
    depth = self.depth_head.forward_head(y, patch_h, patch_w)
    return depth.squeeze(1)

  @torch.no_grad()
  def infer_image(self, raw_image, input_size=518):
    image, (h, w) = self.image2tensor(raw_image, input_size)

    depth = self.forward(image)

    depth = F.interpolate(
        depth[:, None], (h, w), mode="bilinear", align_corners=True
    )[0, 0]

    return depth.cpu().numpy()

  def image2tensor(self, raw_image, input_size=518):
    transform = Compose([
        Resize(
            width=input_size,
            height=input_size,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method="lower_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    h, w = raw_image.shape[:2]

    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

    image = transform({"image": image})["image"]
    image = torch.from_numpy(image).unsqueeze(0)

    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    image = image.to(DEVICE)

    return image, (h, w)


def load_depth_model(encoder="vitl", dev="cuda:0"):

  DEVICE = (
      "cuda"
      if torch.cuda.is_available()
      else "mps"
      if torch.backends.mps.is_available()
      else "cpu"
  )

  model_configs = {
      "vits": {
          "encoder": "vits14",
          "features": 64,
          "out_channels": [48, 96, 192, 384],
      },
      "vitb": {
          "encoder": "vitb14",
          "features": 128,
          "out_channels": [96, 192, 384, 768],
      },
      "vitl": {
          "encoder": "vitl14",
          "features": 256,
          "out_channels": [256, 512, 1024, 1024],
      },
      "vitg": {
          "encoder": "vitg14",
          "features": 384,
          "out_channels": [1536, 1536, 1536, 1536],
      },
  }

  # encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

  model = DepthAnythingV2(**model_configs[encoder])
  model.load_state_dict(
      torch.load(
          f"../checkpoints/depth_anything_v2_{encoder}.pth",
          map_location="cpu",
          weights_only=True,
      )
  )
  model = model.to(dev).eval()

  return model


from collections import OrderedDict
from typing import Literal, Optional, List, Tuple
from functools import partial


def solve_weighted_ridge_regression(
    X, Y, C, lambda_val
):
  """Solves the weighted ridge regression problem for a batch of data.

  This function finds the optimal linear projection matrix W* for each item
  in the batch that minimizes: ||C_mat^(1/2) * (XW - Y)||^2 + lambda * ||W||^2,
  where X are features, Y are targets, and C_mat is a diagonal matrix of
  confidences.

  Args:
      X (torch.Tensor): The input feature tensor.
                        Shape: [B, N, D], where B is batch size, N is the number
                          of elements, and D is the feature dimension.
      Y (torch.Tensor): The target 3D vector map.
                        Shape: [B, N, 3].
      C (torch.Tensor): A confidence score [0, 1] for each target element.
                        Shape: [B, N, 1].
      lambda_val (float): The regularization strength (lambda) for the ridge
        regression.

  Returns:
      torch.Tensor: The optimal weight matrix W* for each batch item.
                    Shape: [B, D, 3].
  """
  # --- 1. Input Validation and Shape Preparation ---
  B, N, D = X.shape
  assert Y.shape[:2] == (B, N), "Targets shape must be [B, N, 3]"
  assert C.shape == (B, N, 1), "Confidences shape must be [B, N]"
  device = X.device

  # To implement the weighting by the confidence matrix, we multiply X and Y by sqrt(C).
  # This is numerically equivalent and much more efficient than forming a diagonal matrix.
  C_sqrt = torch.sqrt(C)
  X_weighted = X * C_sqrt  # Shape: [B, N, D]
  Y_weighted = Y * C_sqrt  # Shape: [B, N, 3]

  # --- 3. Construct the Normal Equation Components (X^T*C*X and X^T*C*Y) ---
  # Transpose the weighted features to get X^T. Shape: [B, D, N]
  X_weighted_t = X_weighted.transpose(1, 2)

  # Compute X^T * C * X for each item in the batch. Shape: [B, D, D]
  # This is equivalent to (X*sqrt(C))^T * (X*sqrt(C))
  A = torch.bmm(X_weighted_t, X_weighted) / N

  # Compute X^T * C * Y for each item in the batch. Shape: [B, D, 3]
  # This is equivalent to (X*sqrt(C))^T * (Y*sqrt(C))
  B_vec = torch.bmm(X_weighted_t, Y_weighted) / N

  # --- 4. Add Regularization and Solve the Linear System ---
  # Create a batch of identity matrices for the ridge regularization term.
  I = torch.eye(D, device=device, dtype=X.dtype).unsqueeze(0).expand(B, -1, -1)

  # The left-hand side of the system is (X^T*C*X + lambda*I)
  A_regularized = A + lambda_val * I

  # We solve the linear system A_reg * W = B_vec for W.
  # This is more numerically stable and efficient than computing the inverse.
  W_star = torch.linalg.solve(A_regularized, B_vec)  # Shape: [B, D, 3]

  return W_star


class RFFEncoder(nn.Module):
  """Encodes a low-dimensional input vector into a high-dimensional

  Random Fourier Feature representation.
  """

  def __init__(self, scale_flow = 1.0, scale_depth = 1.0):
    """Args:

    input_dim (int): The dimensionality of the input vector (e.g., 3 for
    scene flow).
    mapping_size (int): The number of random frequencies to use. The output
        dimensionality will be 2 * mapping_size.
    scale (float): The standard deviation of the Gaussian distribution for
        sampling the random frequencies. This controls the "waviness"
        of the basis functions.
    """
    super().__init__()
    self.scale_flow = scale_flow
    self.scale_depth = scale_depth
    self.tokenise = lambda x: x.flatten(-2).movedim(1, -1)

  def forward(
      self, x, y, z
  ):

    if x.ndim == 4:
      return self.forward_bchw(x, y, z)

    assert x.ndim == 3, f"Expected shape B,N,D, but got {x.shape}"

    return self.forward_bnd(x, y, z)

  def seed(self, x, y):
    if x.ndim == 4:
      x = self.tokenise(x)
      y = self.tokenise(y)

    self.x_std, self.x_mean = torch.std_mean(x, dim=(0, 1, 2), keepdim=True)
    self.y_std, self.y_mean = torch.std_mean(y, dim=(0, 1, 2), keepdim=True)

  def renorm(self, x, y):
    return (x - self.x_mean) / (1e-3 + self.x_std), (y - self.y_mean) / (
        1e-3 + self.y_std
    )

  def forward_bnd(
      self, x, y, z
  ):
    """Applies the RFF encoding to the input tensor.

    Args:
        x (torch.Tensor): The input tensor.
            Shape: [..., input_dim], where ... is any number of batch
              dimensions.

    Returns:
        torch.Tensor: The high-dimensional RFF encoding.
            Shape: [..., output_dim].
    """

    x, y = self.renorm(x, y)

    z = F.normalize(z, p=2, dim=-1)

    x_scaled = self.scale_flow * x
    y_scaled = self.scale_depth * y

    return torch.cat([x_scaled, y_scaled, z], -1)

  def full_conf(self, conf_flowdepth, num_fts):
    conf_self = torch.ones_like(conf_flowdepth[:, :1]).expand(
        -1, num_fts, -1, -1
    )
    return torch.cat([conf_flowdepth, conf_self], 1)

  def forward_bchw(
      self, x, y, z
  ):
    b, c, h, w = x.shape
    x_bnd = self.tokenise(x)  # b,n,c
    y_bnd = self.tokenise(y)
    z_bnd = self.tokenise(z)

    x_feats = self.forward_bnd(x_bnd, y_bnd, z_bnd)
    return x_feats.unflatten(1, (h, w)).movedim(-1, 1)


def load_encoder(
    encoder_name, snapshots_dir = "/workdir/checkpoints"
):
  """Loads a specified vision encoder model from a local snapshot file.

  Args:
      encoder_name: String identifier for the model (e.g., "dinov2_vits").

  Returns:
      The loaded PyTorch model (nn.Module) with loaded weights.
  """

  model_config: Dict[str, Dict[str, Union[callable, str]]] = {
      # Format: {model_name: {"builder": function_to_create_model, "filename": "checkpoint_file.pth"}}
      "dinov2_vits14": partial(DINOv2, model_name="vits14"),
      "dinov2_vitb14": partial(DINOv2, model_name="vitb14"),
      "dinov2_vitl14": partial(DINOv2, model_name="vitl14"),
      "dinov2_vitg14": partial(DINOv2, model_name="vitg14"),
      "dinov2reg_vits14": partial(DINOv2, with_reg=True, model_name="vits14"),
      "dinov2reg_vitb14": partial(DINOv2, with_reg=True, model_name="vitb14"),
      "dinov2reg_vitl14": partial(DINOv2, with_reg=True, model_name="vitl14"),
      "dinov2reg_vitg14": partial(DINOv2, with_reg=True, model_name="vitg14"),
  }

  if encoder_name not in model_config:
    raise ValueError(
        f"Unknown encoder: {encoder_name}. Available models:"
        f" {list(model_config.keys())}"
    )

  print(f"Loading architecture for {encoder_name}...")
  # 2. Instantiate the model architecture
  builder_func = model_config[encoder_name]
  model = builder_func()

  # frozen backbone
  model.eval()
  return model


class LILA(nn.Module):

  def __init__(
      self,
      cfg,
      denorm,
      features=256,
      out_channels=[256, 512, 1024, 1024],
      intermediate_layer_idx=[2, 5, 8, 11],
      patch_size=14,
      use_bn=False,
      use_clstoken=False,
  ):
    super(LILA, self).__init__()

    self.cfg = cfg
    self.denorm = denorm
    self.patch_size = patch_size
    self.intermediate_layer_idx = intermediate_layer_idx
    self.dec_dim = features // 2

    self.pretrained = load_encoder(cfg.encoder)
    self.latent_head = SmallDPTHead(
        self.pretrained.embed_dim,
        features,
        use_bn,
        out_channels=out_channels,
        use_clstoken=use_clstoken,
        patch_size=self.patch_size,
    )

    self.fnorm = nn.LayerNorm(self.dec_dim)

  def norm_feats(self, x):
    h, w = x.shape[-2:]
    feats_bnd = x.flatten(-2).movedim(1, -1)
    feats_bnd = self.fnorm(feats_bnd)
    feats = feats_bnd.movedim(1, -1).unflatten(-1, (h, w))
    return feats

  @torch.no_grad()
  def encode(self, x, norm=True):
    return self.pretrained.get_intermediate_layers(
        x, self.intermediate_layer_idx, return_class_token=True, norm=norm
    )

  def sem_head(self, x, with_norm=True, with_enc_norm=True):
    b, c, h, w = x.shape
    assert c == 3, "Expected RGB images"

    patch_h, patch_w = (
        x.shape[-2] // self.patch_size,
        x.shape[-1] // self.patch_size,
    )

    with torch.no_grad():
      features_enc = self.encode(x, with_enc_norm)

    feats_proj = self.latent_head(features_enc, patch_h, patch_w)

    if with_norm:
      feats_proj = self.norm_feats(feats_proj)

    return feats_proj, features_enc


class TrainLILA(LILA):

  def __init__(self, cfg, denorm, **kwargs):
    super(TrainLILA, self).__init__(cfg, denorm, **kwargs)

    self.depth_net = load_depth_model()
    self.depth_core = instantiate(self.cfg.depthcore)

    self.flow = SeaFlow(denorm)
    self.rff = instantiate(self.cfg.rff)

    self.pamr = nn.ModuleList([
        torch.compile(PAMR(20, [1, 3, 5], (56, 56), sigma=self.cfg.pamr.sigma)),
        torch.compile(
            PAMR(10, [1, 3, 5], (112, 112), sigma=self.cfg.pamr.sigma)
        ),
        torch.compile(PAMR(5, [1, 3, 5], None, sigma=self.cfg.pamr.sigma)),
    ])

  def sample_from(self, x, sample_grid):
    return F.grid_sample(x, sample_grid, mode="bilinear", align_corners=False)

  def refine(self, image, fmap):
    for pamr in self.pamr:
      fmap = pamr(image, fmap)
    return fmap

  def forward_cross(self, features0, features1, scene_data, batch_data):

    # helper functions
    add_one = lambda x: torch.cat([x, torch.ones_like(x[:, :1])], 1)
    rescale_as = lambda x, y: F.interpolate(
        x, y.shape[-2:], mode="bilinear", align_corners=False
    )
    lowres = lambda x: F.adaptive_avg_pool2d(x, (32, 32))
    to_bnd = lambda x: x.flatten(-2).movedim(1, -1)

    features0 = self.norm_feats(features0)
    features1 = self.norm_feats(features1)

    # handling constant signals
    features0 = add_one(features0)
    features1 = add_one(features1)

    ### For in-context learner ###
    encft0_v1 = scene_data["enc_feats0"]
    depth0_v1 = self.sample_from(scene_data["depth0"], batch_data["tf1"])
    flow01_v1 = self.sample_from(scene_data["flow01"], batch_data["tf1"])
    conf01_v1 = self.sample_from(scene_data["conf01"], batch_data["tf1"])

    encft1_v2 = scene_data["enc_feats1"]
    flow10_v2 = self.sample_from(scene_data["flow10"], batch_data["tf2"])
    conf10_v2 = self.sample_from(scene_data["conf10"], batch_data["tf2"])
    depth1_v2 = self.sample_from(scene_data["depth1"], batch_data["tf2"])

    ###
    self.rff.seed(
        torch.cat([flow01_v1, -flow10_v2]), torch.cat([depth0_v1, depth1_v2])
    )

    g_query = self.rff(flow01_v1, depth0_v1, encft0_v1)
    g_context = self.rff(-flow10_v2, depth1_v2, encft1_v2)

    ############### Joint solution for frames 0 and 1 ##########################
    g_context_lstsq = to_bnd(lowres(g_context))
    feats_bnd = to_bnd(lowres(features1))
    confs_bnd = to_bnd(lowres(conf10_v2))

    confs_bnd = torch.prod(confs_bnd, -1, keepdim=True)
    confs_bnd[confs_bnd < 0.5] = 0.0

    wopt = solve_weighted_ridge_regression(
        feats_bnd, g_context_lstsq, confs_bnd, self.cfg.ridge
    )
    ############################################################################

    # rendered flow
    g01_pred = torch.einsum("bchw,bcd->bdhw", features0, wopt)
    g10_pred = torch.einsum("bchw,bcd->bdhw", features1, wopt)

    full_conf = self.rff.full_conf(conf01_v1, self.cfg.num_fts)

    losses = {}

    flow_loss = torch.abs(g01_pred - g_query)
    losses["flow"] = flow_loss[full_conf > 0].mean()
    losses["edge"] = flow_boundary_loss(
        g01_pred, g_query, full_conf, sigma=self.cfg.sigma_edge
    )

    outs = {}
    outs["wopt"] = F.interpolate(
        wopt.movedim(-1, 1)[:, None], scale_factor=8, mode="nearest"
    )

    outs["flow01"] = flow01_v1.movedim(1, -1)
    outs["flow01_enc"] = g_query
    outs["flow01_pred"] = g01_pred
    outs["conf01"] = torch.prod(conf01_v1, dim=1, keepdim=True)

    outs["flow10"] = flow10_v2.movedim(1, -1)
    outs["flow10_enc"] = g_context
    outs["flow10_pred"] = g10_pred

    outs["depth0"] = depth0_v1
    outs["depth1"] = depth1_v2

    outs["feats0"] = features0
    outs["feats1"] = features1

    outs["feats0up"] = encft0_v1
    outs["feats1up"] = encft1_v2

    outs["conf10"] = torch.prod(conf10_v2, dim=1, keepdim=True)

    return outs, losses

  def preproc(self, batch):

    with torch.no_grad():
      x0, x1 = batch["frames"][:, 0], batch["frames"][:, 1]
      x01 = torch.cat([x0, x1])
      x10 = torch.cat([x1, x0])
      flow01, flow10 = self.flow(x01, x10).chunk(2)

      frames_batch = batch["frames"].flatten(0, 1)

      depths, _ = self.depth_net(frames_batch)
      depths = depths.unflatten(0, (-1, 2))
      depths = depths.unsqueeze(2)

      h, w = frames_batch.shape[-2:]
      patch_h, patch_w = h // self.patch_size, w // self.patch_size

    depths_norm = self.depth_core.disp2depth(depths)
    depth0norm, depth1norm = depths_norm[:, 0], depths_norm[:, 1]

    conf01, conf10 = compute_confidence(
        flow01,
        flow10,
        depth0norm,
        depth1norm,
        self.depth_core.valid_depth,
        self.cfg.conf_temp,
    )

    return {
        "flow01": flow01,
        "flow10": flow10,
        "conf01": conf01,
        "conf10": conf10,
        "depth0": depth0norm,
        "depth1": depth1norm,
    }

  def train_forward(self, batch_data):

    # x, disp, rand_crop
    for key in batch_data.keys():
      batch_data[key] = batch_data[key].cuda()

    scene_data = self.preproc(batch_data)

    h, w = batch_data["frames"].shape[-2:]
    patch_h, patch_w = h // self.patch_size, w // self.patch_size

    x_batch = torch.cat([batch_data["view1"], batch_data["view2"]])  # B = bxt

    features_all = self.encode(x_batch)
    features_dec = self.latent_head(features_all, patch_h, patch_w)

    # latents
    features0, features1 = features_dec.chunk(2)

    # encoder self-distillation
    enc_features = features_all[-1][0].unflatten(0, (2, -1))  # 2,B,HW,C
    enc_features = self.random_subset(enc_features)
    enc_features = enc_features.movedim(2, -1).unflatten(-1, (patch_h, patch_w))

    # refining
    enc_features = self.refine(self.denorm(x_batch), enc_features.flatten(0, 1))

    scene_data["enc_feats0"], scene_data["enc_feats1"] = enc_features.chunk(2)

    # randomly selecting a subspace
    outs, losses = self.forward_cross(
        features0, features1, scene_data, batch_data
    )

    return outs, losses

  def random_subset(self, x):
    """Perform per-sample random masking by per-sample shuffling.

    Per-sample shuffling is done by argsort random noise. x: [N, L, D], sequence
    """

    N, B, L, D = x.shape  # batch, length, dim

    noise = torch.rand(1, B, 1, D, device=x.device)  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = torch.argsort(
        noise, dim=-1
    )  # ascend: small is keep, large is remove

    # keep the first subset
    ids_keep = ids_shuffle[Ellipsis, : self.cfg.num_fts]
    x_masked = torch.gather(x, dim=-1, index=ids_keep.expand(N, -1, L, -1))

    return x_masked

  def parameters(self):
    return list(self.latent_head.parameters())
