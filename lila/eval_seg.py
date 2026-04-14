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

import dataload
from eval_base import EvalBase
import hydra
from hydra.utils import instantiate
from metrics import CocoEval
import numpy as np
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from probes.attention import AttentionProbe as Probe
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import tensorboard
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TVF
from tqdm.auto import tqdm
from utils.vis import DisplayFeats, DisplaySeg


class EvalSeg(EvalBase):

  def __init__(self, cfg):
    super().__init__(cfg)

    self.train_dataset = dataload.SegDataset(
        "/workdir/data/COCOStuff",
        "train2017",
        "splits/train_Coco164kFull_Stuff_Coarse.txt",
        preload=False,
    )
    self.val_dataset = dataload.SegDataset(
        "/workdir/data/COCOStuff",
        "val2017",
        "splits/val_Coco164kFull_Stuff_Coarse.txt",
        preload=False,
        val=True,
    )

    assert (
        self.val_dataset.num_classes() == self.train_dataset.num_classes()
    ), "Unmatched number of train/val classes"

    self.probe = Probe(
        cfg.probe,
        cfg.probe.enc_fdim,
        cfg.probe.dec_fdim,
        self.train_dataset.num_classes(),
    )
    self.probe.cuda()

    self.job_root = os.path.join(cfg.runtime.root, cfg.runtime.name)
    self.writer = tensorboard.SummaryWriter(self.job_root)
    self.opt = optim.AdamW(self.parameters(), lr=cfg.eval.lr)
    self.best_val = self._init_metrics()
    self._setup_v("seg")

  def _init_metrics(self):
    return {"mIoU": 0.0}

  def parameters(self):

    if self.cfg.eval.finetune:  # finetuning DPT
      # using parameter groups
      base_lr, base_wd = self.cfg.eval.lr, self.cfg.eval.weight_decay
      param_groups = [
          {
              "params": self.probe.parameters(),
              "lr": base_lr,
              "weight_decay": base_wd,
          },
          {
              "params": self.model.latent_head.parameters(),
              "lr": self.cfg.eval.finetune_lr_mult * base_lr,
              "weight_decay": self.cfg.eval.finetune_wd_mult * base_wd,
          },
      ]
      return param_groups
    else:
      return self.probe.parameters()

  def _save_result(self, results, alpha=0.1, with_aux=True):

    offset = 0
    for im_batch, pred_batch, enc_feats, dec_feats, gt_batch in results:
      im_norm = self.val_dataset.denorm(im_batch)

      disp_pred = DisplaySeg(alpha)
      disp_pred.embed(im_norm, pred_batch)
      disp_pred.save(self.root_v, offset=offset, suffix="_pred")

      if with_aux:
        disp_img = DisplaySeg(1.0)
        disp_img.embed(im_norm, pred_batch)
        disp_img.save(self.root_v, offset=offset, suffix="_image")

        disp_gt = DisplaySeg(alpha)
        disp_gt.embed(im_norm, gt_batch)
        disp_gt.save(self.root_v, offset=offset, suffix="_gt")

        disp_feats_enc = DisplayFeats(0.0)
        enc_feats = F.interpolate(
            enc_feats, im_norm.shape[-2:], mode="bilinear", align_corners=False
        )

        disp_feats_enc.embed(im_norm, enc_feats, pred2rgb=False, collate=False)
        disp_feats_enc.save(self.root_v, offset=offset, suffix="_feats_enc")

        disp_feats_dec = DisplayFeats(0.0)
        disp_feats_dec.embed(im_norm, dec_feats, pred2rgb=False, collate=False)
        disp_feats_dec.save(self.root_v, offset=offset, suffix="_feats_dec")

      offset += im_batch.shape[0]

  def validate(self, epoch):

    eval_stats = CocoEval(self.train_dataset.num_classes())

    dataloader = self.val_dataset.get_loader(
        batch_size=self.cfg.eval.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
    )

    pbar = tqdm(dataloader)

    results = []
    for niter, batch in enumerate(pbar):
      images, gt_masks, _ = [x.cuda(non_blocking=True) for x in batch]

      with torch.no_grad():
        enc_feats, feats = self.encode(images)
        pred_masks = self.probe(enc_feats, feats, images.shape[-2:])
        pred_masks_i = pred_masks.argmax(1)

      for pred_mask, gt_mask in zip(pred_masks_i, gt_masks):
        eval_stats.add_prediction(pred_mask, gt_mask)

      if niter % 10 == 0:  # saving every 10th
        results.append((
            images.to("cpu", non_blocking=True),
            pred_masks_i.to("cpu", non_blocking=True),
            enc_feats.to("cpu", non_blocking=True),
            feats.to("cpu", non_blocking=True),
            gt_masks.to("cpu", non_blocking=True),
        ))

    eval_dict = eval_stats.dict()
    for key, val in eval_dict.items():
      self.writer.add_scalar(f"seg/{key}", val, epoch)

    if eval_dict["mIoU"] > self.best_val["mIoU"]:
      with_aux = self.best_val["mIoU"] == 0
      print(f"Best mIoU: {eval_dict['mIoU']}. Saving (with GT={with_aux})...")
      # self._save_result(results, with_aux=with_aux)
      self.best_val.update(eval_dict)

    return eval_stats

  def train(self):

    dataloader = self.train_dataset.get_loader(
        batch_size=self.cfg.eval.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    giter = 0
    for epoch in range(self.cfg.eval.num_epochs):
      print(f"Epoch {epoch}")

      # --- Training on the First Frame ---
      pbar = tqdm(dataloader)

      for niter, batch in enumerate(pbar):
        images, masks, _ = [x.cuda(non_blocking=True) for x in batch]

        self.opt.zero_grad()
        enc_feats, feats = self.encode(images)
        pred_mask = self.probe(enc_feats, feats, images.shape[-2:])
        loss = F.cross_entropy(
            pred_mask,
            masks.to(torch.long),
            ignore_index=self.train_dataset.ignore_index,
        )
        loss.backward()
        self.opt.step()
        pbar.set_description(
            f"Iter {niter:03d} |" + f" Loss={loss.item():4.3f}"
        )

        if niter % 10 == 0:
          self.writer.add_scalar("losses_seg/train_loss", loss.item(), giter)

        giter += 1

      if epoch % 1 == 0:
        stats = self.validate(epoch)
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

  eval_engine = EvalSeg(cfg)
  eval_engine.train()


if __name__ == "__main__":
  main()
