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

"""Surface Normal Evaluation Module.

This module handles the training and validation of Surface Normal estimation
using a Linear Probe on top of a frozen encoder (DPT/ViT).
"""

import os
from typing import Any
from typing import Dict
from typing import List

# Local imports
import dataload
from eval_base import EvalBase
import hydra
from metrics import NormalsEval
from omegaconf import DictConfig
from probes.linear import LinearCat as LinProbe
import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from utils.vis import DisplayFeats
from utils.vis import DisplayNormals


def angular_loss(
    pred_normals,
    gt_normals,
    mask,
    uncertainty_aware = False,
    eps = 1e-4,
):
  """Computes Angular loss with an optional uncertainty-aware component.

  Based on Bae et al.

  Args:
      pred_normals: Predicted normals (Batch x Channels x H x W).
      gt_normals: Ground truth normals.
      mask: Valid pixel mask.
      uncertainty_aware: Whether to use uncertainty loss.
      eps: Epsilon for numerical stability.

  Returns:
      Mean loss over valid pixels.
  """
  # Ensure mask is float and batch x height x width
  if mask.ndim != 4:
    raise ValueError(
        f"Mask should be (batch x height x width), got {mask.shape}"
    )

  mask = mask.squeeze(1).float()

  # Compute correct loss
  if uncertainty_aware:
    if pred_normals.shape[1] != 4:
      raise ValueError(
          "Prediction must have 4 channels for uncertainty aware loss"
      )

    loss_ang = torch.cosine_similarity(pred_normals[:, :3], gt_normals, dim=1)
    loss_ang = loss_ang.clamp(min=-1 + eps, max=1 - eps).acos()

    # Apply elu and add 1.01 to have a min kappa of 0.01 (similar to paper)
    kappa = F.elu(pred_normals[:, 3]) + 1.01
    kappa_reg = (1 + (-kappa * torch.pi).exp()).log() - (kappa.pow(2) + 1).log()

    loss = kappa_reg + kappa * loss_ang
  else:
    if pred_normals.shape[1] != 3:
      raise ValueError(
          "Prediction must have 3 channels for standard angular loss"
      )

    loss_ang = torch.cosine_similarity(pred_normals, gt_normals, dim=1)
    loss = loss_ang.clamp(min=-1 + eps, max=1 - eps).acos()

  # Compute loss over valid positions
  loss_mean = loss[mask.bool()].mean()

  return loss_mean


class EvalNormal(EvalBase):
  """Evaluation class for Surface Normals.

  Inherits from EvalBase to handle generic encoder steps.
  """

  def __init__(self, cfg):
    super().__init__(cfg)

    self.train_dataset = dataload.NormalsDataset(
        "/workdir/data/NYUv2", "train", output_size=(448, 448)
    )
    self.val_dataset = dataload.NormalsDataset(
        "/workdir/data/NYUv2", "val", output_size=(504, 504)
    )

    if self.val_dataset.num_classes() != self.train_dataset.num_classes():
      raise ValueError("Unmatched number of train/val classes")

    self.probe = LinProbe(
        cfg.probe,
        cfg.probe.enc_fdim,
        cfg.probe.dec_fdim,
        self.train_dataset.num_classes(),
    )
    self.probe.cuda()

    self.job_root = os.path.join(cfg.runtime.root, cfg.runtime.name)
    self.writer = SummaryWriter(self.job_root)
    self.opt = optim.AdamW(self.parameters(), lr=cfg.eval.lr)

    self.best_val = self._init_metrics()
    self._setup_v("normals")

    # Pylint suppression for attributes defined in parent class EvalBase
    # pylint: disable=no-member
    self.model = getattr(self, "model", None)
    self.root_v = getattr(self, "root_v", None)

  @staticmethod
  def _init_metrics():
    """Initialize metric dictionary."""
    return {"rmse": 100.0, "d1": 0.0, "d2": 0.0, "d3": 0.0}

  def parameters(self):
    """Returns parameters for the optimizer.

    Handles specific parameter groups if finetuning is enabled.
    """
    if self.cfg.eval.finetune:
      # Finetuning DPT
      dec_params = list(self.model.latent_head.parameters()) + list(
          self.model.fnorm.parameters()
      )
      # Using parameter groups
      base_lr = self.cfg.eval.lr
      base_wd = self.cfg.eval.weight_decay
      finetune_lr = self.cfg.eval.finetune_lr_mult * base_lr
      finetune_wd = self.cfg.eval.finetune_wd_mult * base_wd
      param_groups = [
          {
              "params": self.probe.parameters(),
              "lr": base_lr,
              "weight_decay": base_wd,
          },
          {
              "params": dec_params,
              "lr": finetune_lr,
              "weight_decay": finetune_wd,
          },
      ]
      return param_groups

    return self.probe.parameters()

  def _save_result(self, results, with_aux = True):
    """Saves visualization of results (Prediction, GT, Features).

    Args:
        results: List of tuples containing batch data.
        with_aux: If True, saves auxiliary images (GT, Features).
    """
    offset = 0
    for batch_data in results:
      # Unpack batch data
      (im_batch, pred_batch, enc_feats, dec_feats, mask, gt_batch) = batch_data

      im_norm = self.val_dataset.denorm(im_batch)

      disp_pred = DisplayNormals(0.0)
      disp_pred.embed(im_norm, pred_batch, pred2rgb=True, collate=False)
      disp_pred.save(self.root_v, offset=offset, suffix="_pred")

      if with_aux:
        disp_img = DisplayNormals(1.0)
        disp_img.embed(im_norm, pred_batch, pred2rgb=True, collate=False)
        disp_img.save(self.root_v, offset=offset, suffix="_image")

        disp_gt = DisplayNormals(0.0)
        disp_gt.embed(
            im_norm, gt_batch, pred2rgb=True, maskout=mask, collate=False
        )
        disp_gt.save(self.root_v, offset=offset, suffix="_gt")

        disp_feats_enc = DisplayFeats(0.0)
        # Resize features to match image size for visualization
        enc_feats_interp = F.interpolate(
            enc_feats, im_norm.shape[-2:], mode="bilinear", align_corners=False
        )

        disp_feats_enc.embed(
            im_norm, enc_feats_interp, pred2rgb=False, collate=False
        )
        disp_feats_enc.save(self.root_v, offset=offset, suffix="_feats_enc")

        disp_feats_dec = DisplayFeats(0.0)
        disp_feats_dec.embed(im_norm, dec_feats, pred2rgb=False, collate=False)
        disp_feats_dec.save(self.root_v, offset=offset, suffix="_feats_dec")

      offset += im_batch.shape[0]

  def validate(self, epoch, giter):
    """Run validation loop."""
    eval_stats = NormalsEval()

    dataloader = self.val_dataset.get_loader(
        batch_size=8,
        shuffle=False,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )

    pbar = tqdm(dataloader)
    results = []
    losses = []

    for niter, batch in enumerate(pbar):
      # Move to device
      images, depth, gt_normals = [x.cuda(non_blocking=True) for x in batch]

      with torch.no_grad():
        enc_feats, feats = self.encode(images)
        pred_normals = self.probe(enc_feats, feats, images.shape[-2:])
        mask = depth.cuda() > 0.0
        loss = angular_loss(pred_normals, gt_normals, mask)
        losses.append(loss.item())

      mask = (depth > 0).cuda()
      eval_stats.add_prediction(pred_normals, gt_normals, mask)

      if niter % 10 == 0:  # Saving every 10th
        results.append((
            images.to("cpu", non_blocking=True),
            pred_normals.to("cpu", non_blocking=True),
            enc_feats.to("cpu", non_blocking=True),
            feats.to("cpu", non_blocking=True),
            mask.to("cpu", non_blocking=True),
            gt_normals.to("cpu", non_blocking=True),
        ))

    avg_loss = sum(losses) / len(losses) if losses else 0.0
    self.writer.add_scalar("losses_normals/val_loss", avg_loss, giter)

    eval_dict = eval_stats.dict()
    for key, val in eval_dict.items():
      self.writer.add_scalar(f"normals/{key}", val, epoch)

    if eval_dict["rmse"] < self.best_val["rmse"]:
      with_aux = self.best_val["rmse"] == 100.0
      print(f"Best RMSE: {eval_dict['rmse']}. Saving (with GT={with_aux})...")
      # self._save_result(results, with_aux=with_aux)
      self.best_val.update(eval_dict)

    return eval_stats

  def train(self):
    """Run training loop."""
    dataloader = self.train_dataset.get_loader(
        batch_size=self.cfg.eval.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    giter = 0
    for epoch in range(self.cfg.eval.num_epochs):
      print(f"Epoch {epoch}")

      # --- Training Loop ---
      pbar = tqdm(dataloader)

      for niter, batch in enumerate(pbar):
        images, depth, gt_normals = [x.cuda(non_blocking=True) for x in batch]

        enc_feats, feats = self.encode(images)
        pred_normals = self.probe(enc_feats, feats, images.shape[-2:])
        mask = depth.cuda() > 0.0
        loss = angular_loss(pred_normals, gt_normals, mask)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        loss_val = loss.item()
        pbar.set_description(f"Iter {niter:03d} | Loss={loss_val:4.3f}")

        if niter % 10 == 0:
          self.writer.add_scalar("losses_normals/train_loss", loss_val, giter)

        giter += 1

      if epoch % 1 == 0:
        stats = self.validate(epoch, giter)
        stats.print_metrics()

    self.writer.add_hparams(
        {
            "batch_size": self.cfg.eval.batch_size,
            "lr": self.cfg.eval.lr,
            "dec": self.cfg.probe.alpha_dec,
            "enc": self.cfg.probe.alpha_enc,
        },
        self.best_val,
        run_name=".",
    )


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg):
  """Entry point for the evaluation script."""
  eval_engine = EvalNormal(cfg)
  eval_engine.train()


if __name__ == "__main__":
  main()
