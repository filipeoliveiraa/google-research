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

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from lila.config import lila_config
from lila.dpt_lila import LILA, load_lila_weights, shorten_incompatible_keys


class PatchResample:
    """Resize a frame so the smaller side matches `resize_min` and stays patch-aligned."""

    def __init__(self, h, w, resize_min, patch_size):
        if w < h:
            new_w = resize_min
            new_h = int(h * resize_min / w)
            new_h = int(round(new_h / patch_size) * patch_size)
        else:
            new_h = resize_min
            new_w = int(w * resize_min / h)
            new_w = int(round(new_w / patch_size) * patch_size)

        self.norm_hw = (new_h, new_w)

    def adjust_to_patch_size(self, x):
        return F.interpolate(x, self.norm_hw, mode="bilinear", align_corners=False)


class EvalBase:
    def __init__(self, cfg):
        self.cfg = cfg

        arch_config = lila_config[cfg.model.encoder]
        self.model = self._init_model(cfg.model, cfg.eval.random_dpt, **arch_config)
        self.patch_size = arch_config["patch_size"]
        self.eval_dir = os.path.join(cfg.runtime.root, cfg.runtime.name)

    @staticmethod
    def _rescale(x, hw):
        return F.interpolate(x, hw, mode="bilinear", align_corners=False)

    @staticmethod
    def _rescale_as(x, yref):
        return F.interpolate(x, yref.shape[-2:], mode="bilinear", align_corners=False)

    def rectify(self, feats, orig_size):
        h = orig_size[0] // self.patch_size
        w = orig_size[1] // self.patch_size
        return feats.movedim(1, -1).unflatten(-1, (h, w))

    @torch.no_grad()
    def encode(self, x):
        if self.cfg.eval.resize_input:
            resize_size = self.cfg.eval.resize_input_size
            if isinstance(resize_size, int):
                resize_hw = (resize_size, resize_size)
            else:
                resize_hw = tuple(resize_size)
            x = self._rescale(x, resize_hw)

        feats, enc_feats = self.model.sem_head(
            x.cuda(),
            with_norm=self.cfg.eval.dec_norm,
            with_enc_norm=self.cfg.eval.enc_norm,
        )
        enc_feats_last = enc_feats[self.cfg.eval.enc_layer_idx][0]
        feats_enc_hw = self.rectify(enc_feats_last, x.shape[-2:])

        return feats_enc_hw, feats

    def _init_model(self, cfg_model, random_dpt, **arch_config):
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        model = LILA(cfg_model, denorm=nn.Identity(), **arch_config)

        if not random_dpt:
            incompatible = load_lila_weights(
                model,
                checkpoint_path=cfg_model.checkpoint_path or None,
                checkpoints_dir=cfg_model.checkpoints_dir or None,
                model_name=None if cfg_model.checkpoint_path else cfg_model.name,
                checkpoint_name=cfg_model.checkpoint,
                strict=False,
            )
            print("Missing Keys (shortened):", shorten_incompatible_keys(incompatible.missing_keys))
            print("Unexpected Keys (shortened):", shorten_incompatible_keys(incompatible.unexpected_keys))

        model = model.to(device).eval()
        for param in model.parameters():
            param.requires_grad = False

        return model
