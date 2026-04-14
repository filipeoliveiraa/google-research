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

import glob
import os
import random
import time

import hydra
from hydra.utils import instantiate
from lila.config import lila_config
from lila.dpt_flowfeat import FlowFeat
from lila.dpt_lila import LILA
import numpy as np
from omegaconf import DictConfig, OmegaConf
from PIL import Image
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import tensorboard
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TVF
from tqdm.auto import tqdm


class PatchResample:

  def __init__(self, h, w, resize_min, patch_size):
    if w < h:
      # If width is the smaller side, set it to resize_min
      new_w = resize_min
      # Calculate the new height to maintain the aspect ratio
      new_h = int(h * resize_min / w)
      # Adjust the new height (the larger dimension) to be the nearest
      # multiple of patch_size.
      new_h = int(round(new_h / patch_size) * patch_size)
    else:
      # If height is the smaller or equal side, set it to resize_min
      new_h = resize_min
      # Calculate the new width to maintain the aspect ratio
      new_w = int(w * resize_min / h)
      # Adjust the new width (the larger dimension) to be the nearest
      # multiple of patch_size.
      new_w = int(round(new_w / patch_size) * patch_size)

    self.norm_hw = (new_h, new_w)
    self.orig_hw = (h, w)
    self.patch_size = patch_size

  def adjust_to_patch_size(self, x):
    return F.interpolate(x, self.norm_hw, mode="bilinear", align_corners=False)


class EvalBase:

  def __init__(self, cfg):
    self.cfg = cfg

    arch_config = lila_config[cfg.model.encoder]
    self.model = self._init_model(cfg.model, cfg.eval.random_dpt, **arch_config)
    self.patch_size = arch_config["patch_size"]
    self.eval_dir = os.path.join(cfg.runtime.root, cfg.runtime.name)

  def _denorm(self, x):
    return x

  def _setup_v(self, name):  # setting up visualisation
    self.root_v = os.path.join(self.job_root, name)
    if not os.path.isdir(self.root_v):
      os.makedirs(self.root_v)

  @staticmethod
  def _rescale(x, HW):
    return F.interpolate(x, HW, mode="bilinear", align_corners=False)

  @staticmethod
  def _rescale_as(x, yref):
    return F.interpolate(
        x, yref.shape[-2:], mode="bilinear", align_corners=False
    )

  def rectify(self, feats, orig_size):
    h = orig_size[0] // self.patch_size
    w = orig_size[1] // self.patch_size
    return feats.movedim(1, -1).unflatten(-1, (h, w))

  @torch.no_grad()
  def encode(self, x):
    HW = x.shape[-2:]
    HW_orig = x.shape[-2:]

    if self.cfg.eval.resize_input:
      HW = list(self.cfg.eval.resize_input_size)
      x = self._rescale(x, HW)

    feats, enc_feats = self.model.sem_head(
        x.cuda(),
        with_norm=self.cfg.eval.dec_norm,
        with_enc_norm=self.cfg.eval.enc_norm,
    )
    # flowfeat
    # feats_enc_hw, feats = self.model(x.cuda()) #, with_norm=self.cfg.eval.dec_norm, with_enc_norm=self.cfg.eval.enc_norm)

    enc_feats_last = enc_feats[self.cfg.eval.enc_layer_idx][0]
    feats_enc_hw = self.rectify(enc_feats_last, HW)

    return feats_enc_hw, feats

  def _init_model(
      self, cfg_model, random_dpt, **arch_config
  ):  # or 'vits', 'vitb', 'vitg'
    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model = LILA(cfg_model, denorm=self._denorm, **arch_config)

    if not random_dpt:
      model_path = f"/gcs/araslanov-71167094c93f/logs/codelab_v2/{cfg_model.name}/{cfg_model.checkpoint}"
      print(f"Loading {model_path}")
      model_weights = torch.load(
          model_path, weights_only=True, map_location="cpu"
      )

      if "pretrained.pos_embed" in model_weights:
        del model_weights["pretrained.pos_embed"]

      shorten_keys = lambda keys: sorted(
          list(set(k.split(".")[0] + ".*" for k in keys))
      )

      incompatible = model.load_state_dict(model_weights, strict=False)
      print(
          "Missing Keys (shortened):", shorten_keys(incompatible.missing_keys)
      )
      print(
          "Unexpected Keys (shortened):",
          shorten_keys(incompatible.unexpected_keys),
      )

    model = model.to(DEVICE).eval()

    for param in model.parameters():
      param.requires_grad = False

    return model
