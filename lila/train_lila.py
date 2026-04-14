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
import random
import dataload
import hydra
from hydra.utils import instantiate
import lightning as L
from lila.config import lila_config
from lila.dpt_lila import TrainLILA
import metrics
from metrics import BaseMetric
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import tensorboard
import torchvision
from tqdm import tqdm
import utils.davis_eval as davis_eval
from utils.vis import *


class LinProbe(nn.Module):

  def __init__(self, idim, odim):
    super().__init__()
    self.probe = nn.Conv2d(idim, odim, 1)

  def forward(self, feats, scale_to):
    logits = self.probe(feats)
    logits = F.interpolate(
        logits, scale_to, mode="bilinear", align_corners=False
    )
    return logits


class EvalVOS(nn.Module):

  def __init__(self, fdim):
    super().__init__()
    self.metrics = davis_eval.DavisEvaluate()
    self.fdim = fdim

  def fit(self, feats, masks, num_epochs=2000):
    torch.set_grad_enabled(True)

    num_classes = masks.max().item() + 1
    probe = LinProbe(self.fdim, num_classes).cuda()
    opt = optim.Adam(probe.parameters(), lr=0.0003, weight_decay=0.0001)

    # --- Training on the First Frame ---
    pbar = tqdm(range(num_epochs))

    for epoch in pbar:
      opt.zero_grad()
      pred_mask = probe(feats, masks.shape[-2:])
      loss = F.cross_entropy(pred_mask, masks.long())
      loss.backward()
      opt.step()
      pbar.set_description(f"Epoch {epoch:03d} |" + f" Loss={loss.item():4.3f}")

    torch.set_grad_enabled(False)

    return probe

  def forward(self, feats, masks):
    probe = self.fit(feats[:1].detach(), masks[:1].cuda())

    with torch.no_grad():
      pred_masks = probe(feats, masks.shape[-2:]).argmax(1)

    self.metrics.add_sequence(masks.numpy(), pred_masks.cpu().detach().numpy())


class TrainModel(L.LightningModule):

  def __init__(self, cfg):
    super().__init__()
    self.cfg = cfg

    self.train_data = instantiate(cfg.train_dataset)
    self.val_data = instantiate(cfg.val_dataset)

    # model and probe shared across tasks
    self.showroom = cfg.showroom
    self.eval_vos = cfg.eval_vos
    # DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    self.model_config = lila_config[
        cfg.model.encoder
    ]  # or 'vits', 'vitb', 'vitg'

    model = TrainLILA(
        cfg.model, denorm=self.train_data.denorm, **self.model_config
    )
    # model.load_state_dict(torch.load(f'../checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'), strict=False)
    self.model = model.cuda()

    self.logdir = os.path.join(os.environ["TB_DIR"] + "_v2", cfg.runtime.name)
    self.writer_train = tensorboard.SummaryWriter(
        os.path.join(self.logdir, "train")
    )
    self.writer_val = tensorboard.SummaryWriter(
        os.path.join(self.logdir, "val")
    )

    self.base_metrics = BaseMetric()

    self.step = 0
    self.first_val = True
    self.best_val = 0

  def train_loader(self, **args):
    return self.train_data.get_loader(**args)

  def val_loaders(self, **args):
    return self.val_data.get_loader(**args)

  def training_step(self, batch, batch_idx):
    outs, losses = self.model.train_forward(batch)

    total_loss = losses["flow"] + self.cfg.model.w_edge * losses["edge"]

    losses["total_loss"] = total_loss

    self.log(f"total", total_loss.item(), prog_bar=True)

    print(f"Loss: flow={losses['flow']:4.3f}, edge={losses['edge']:4.3f}")

    if batch_idx % 50 == 0:
      for key, value in losses.items():
        self.writer_train.add_scalar(f"losses/{key}", value.item(), self.step)

    self.step += 1
    return total_loss

  def on_train_epoch_end(self):
    current_lr = self.optimizers().param_groups[0]["lr"]
    self.writer_train.add_scalar("learning_rate", current_lr, self.step)

    sch = self.lr_schedulers()
    sch.step()

    self.writer_train.flush()

  def validation_step(self, batch, batch_idx):
    outs, losses = self.model.train_forward(batch)

    #### Adding metrics ###
    for loss_key, loss_val in losses.items():
      self.base_metrics.add_to_metric(loss_key, loss_val)

  def on_validation_epoch_end(self):
    for key, value in self.base_metrics.dict().items():
      self.writer_val.add_scalar(f"losses/{key}", value, self.step)

    print("Base metrics: ")
    self.base_metrics.print_metrics()
    self.base_metrics.reset()

    eval_vos = EvalVOS(self.model_config["features"] // 2)
    for seqname in self.eval_vos:
      frames, masks = self.val_data.get_sequence(seqname)

      feats = self.model.sem_head(frames.cuda())[0]
      eval_vos(feats, masks)

    eval_vos.metrics.print_metrics()
    vos_metrics = eval_vos.metrics.dict()
    for key, value in vos_metrics.items():
      self.writer_val.add_scalar(f"vos/{key}", value, self.step)

    torch.save(self.model.state_dict(), f"{self.logdir}/last_checkpoint.pt")

    if vos_metrics["JF"] > self.best_val:
      self.best_val = vos_metrics["JF"]
      print(f"Saving best model: JF = {self.best_val:4.3f}")
      torch.save(self.model.state_dict(), f"{self.logdir}/best_checkpoint.pt")

    self.writer_val.flush()
    self.visualise(self.showroom)

  @torch.no_grad()
  def visualise(self, showroom):
    def collate(samples):
      batch = {}
      for key, val in samples[0].items():
        batch[key] = torch.stack([sample[key] for sample in samples])

      return batch

    samples = [self.train_data[i] for i in showroom]
    batch = collate(samples)

    dn = self.train_data.denorm
    fetch_image = lambda idx: dn(
        batch["view1"] if idx == 0 else batch["view2"]
    ).cpu()

    outs, losses = self.model.train_forward(batch)

    def add_depth(idx, key, alpha):
      disp = DisplayDepth(alpha)
      disp.embed(fetch_image(idx), outs[key].cpu().detach())
      return disp.tensor_images()

    def add_feats(idx, key):
      outs_rgb, _, _ = convert2pca(outs[key])
      disp = DisplaySeg(0.0)
      disp.embed(fetch_image(idx), outs_rgb)
      return disp.tensor_images()

    def add_feats_joint(idx1, key1, idx2, key2):
      feats_joint = torch.cat([outs[key1], outs[key2]])
      outs_rgbs, _, _ = convert2pca(feats_joint)
      outs_rgbs = torch.cat(outs_rgbs.chunk(2), -2)
      imagecat = torch.cat([fetch_image(idx1), fetch_image(idx2)], -2)
      disp = DisplaySeg(0.0)
      disp.embed(imagecat, outs_rgbs)
      return disp.tensor_images()

    def add_sflow(idx, key):
      disp = DisplaySeg(0.0)
      disp.embed(fetch_image(idx), outs[key].cpu())
      return disp.tensor_images()

    def add_conf(idx, key):
      disp = DisplayConf(0.0)
      disp.embed(fetch_image(idx), outs[key].cpu())
      return disp.tensor_images()

    def add_flow(idx, key):
      disp = DisplayFlow(0.0)
      disp.embed(fetch_image(idx), outs[key])
      return disp.tensor_images()

    concat_all = torch.cat(
        [
            add_depth(0, "depth0", 1.0),
            add_depth(0, "depth0", 0.05),
            add_feats_joint(0, "feats0", 1, "feats1"),
            add_flow(0, "flow01"),
            add_feats(0, "flow01_enc"),
            add_feats(0, "flow01_pred"),
            add_conf(0, "conf01"),
            add_depth(1, "depth1", 1.0),
            add_depth(1, "depth1", 0.05),
            add_flow(1, "flow10"),
            add_feats(0, "flow10_enc"),
            add_feats(0, "flow10_pred"),
            add_conf(1, "conf10"),
        ],
        -2,
    )

    grid = torchvision.utils.make_grid(concat_all)
    self.writer_val.add_image("vis", grid, self.step)

    # visualising W*
    w = (0.5 + 1000.0 * outs["wopt"]).clamp(0, 1)
    grid = torchvision.utils.make_grid(w, nrow=1)
    self.writer_val.add_image("w", grid, self.step)

  def configure_optimizers(self):
    # print(optimizer)

    optimizer = instantiate(self.cfg.opt, self.model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.99
    )

    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            # 'epoch' updates the scheduler after every epoch
            # 'step' updates the scheduler after every batch
            "interval": "epoch",
            "frequency": 1,
        },
    }


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg):
  print(OmegaConf.to_yaml(cfg))

  os.environ["PJRT_DEVICE"] = cfg.runtime.PJRT_DEVICE

  # model
  model = TrainModel(cfg)
  train_loader = model.train_loader(**dict(cfg.train_loader))
  val_loaders = model.val_loaders(**dict(cfg.val_loader))

  # training
  trainer = L.Trainer(
      devices=1, accelerator="gpu", max_epochs=50, limit_train_batches=2000
  )
  trainer.fit(
      model=model, train_dataloaders=train_loader, val_dataloaders=val_loaders
  )


if __name__ == "__main__":
  main()
