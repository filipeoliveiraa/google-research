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
from eval_base import EvalBase, PatchResample
import hydra
from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from probes.linear import LinearCat as LinProbe
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import tensorboard
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TVF
from torchvision.utils import save_image
from tqdm.auto import tqdm
import utils.davis_eval as davis_eval
from utils.vis import DisplaySeg, convert2pca, overlay_masks


class EvalVOS(EvalBase):

  def __init__(self, cfg):
    super().__init__(cfg)

    self.dataset = dataload.VideoDataset(
        "/workdir/data/DAVIS2017",
        "splits/val_davis.txt",
        (0.5, 1.0),
        (1.0, 1.0),
    )

    self.writer = tensorboard.SummaryWriter(self.eval_dir)

    self.davis_metrics = davis_eval.DavisEvaluate()

  def _denorm(self, x):
    return self.dataset.denorm(x)

  def _fit_frames(self, cfg_eval, probe, feats_enc, feats, masks):

    opt = optim.AdamW(
        probe.parameters(), lr=cfg_eval.lr, weight_decay=cfg_eval.weight_decay
    )

    # --- Training on the First Frame ---
    pbar = tqdm(range(cfg_eval.num_epochs))

    for epoch in pbar:
      opt.zero_grad()
      pred_mask = probe(feats_enc, feats, masks.shape[-2:])
      loss = F.cross_entropy(pred_mask, masks.long())
      loss.backward()
      opt.step()
      pbar.set_description(f"Epoch {epoch:03d} |" + f" Loss={loss.item():4.3f}")

  @torch.no_grad()
  def _save_results(self, seq_name, frames, feats, pred_masks, gt_masks):
    min_float = lambda x: f"{x.min().item():4.3f}"
    max_float = lambda x: f"{x.max().item():4.3f}"

    disp_seg = DisplaySeg(0)
    lut = torch.from_numpy(disp_seg.cmap).to(pred_masks.device)

    frames_norm = self.dataset.denorm(frames).cpu()

    feats_rgb, _, _ = convert2pca(feats)
    if feats_rgb.shape[-2:] != frames.shape[-2:]:
      feats_rgb = F.interpolate(feats_rgb, frames.shape[-2:], mode="bilinear")

    pred_rgb = overlay_masks(frames_norm, pred_masks.cpu(), lut)
    ground_rgb = overlay_masks(frames_norm, gt_masks.long().cpu(), lut)

    # 1. Define and create the output directory
    vis_dir = os.path.join(self.eval_dir, "vis", seq_name)
    os.makedirs(vis_dir, exist_ok=True)

    # 2. Group tensors for cleaner iteration
    data_map = {
        "frame": frames_norm[::5],
        "feats": feats_rgb[::5],
        "pred": pred_rgb[::5],
        "ground": ground_rgb[::5],
    }

    # 3. Iterate over the batch dimension
    batch_size = data_map["frame"].shape[0]

    for b in range(batch_size):
      for prefix, tensor in data_map.items():
        # Construct filename: e.g., "frame_0001.png"
        # using :04d ensures correct sorting in file explorers
        filename = f"{b:04d}_{prefix}.png"
        file_path = os.path.join(vis_dir, filename)

        # Save the specific image from the batch
        # tensor[b] has shape [3, H, W]
        save_image(tensor[b], file_path)

    print(f"Saved {batch_size * len(data_map)} images to {vis_dir}")

    # print("Images: ", frames_norm.shape, min_float(frames_norm), max_float(frames_norm))
    # print("Feats: ",  feats_rgb.shape, min_float(feats_rgb), max_float(feats_rgb))
    # print("SegPR: ",  pred_rgb.shape, min_float(pred_rgb), max_float(pred_rgb))
    # print("SegGT: ",  ground_rgb.shape, min_float(ground_rgb), max_float(ground_rgb))

  @torch.no_grad()
  def _predict_with_probe(self, probe, frames, HW):  # feats_enc, feats, masks):

    outputs = []
    # feats_list = []
    for frame in frames.split(1):
      feats_enc, feats = self.encode(frame)
      pred_masks = probe(feats_enc, feats, HW)
      outputs += [pred_masks.argmax(1)]
      # feats_list += [feats_enc.cpu()]

    return torch.cat(outputs)  # , torch.cat(feats_list)

  def eval_sequence(self, frames, masks, seq):
    HW = frames.shape[-2:]

    # number of objects
    num_classes = masks.max().item() + 1

    resampler = PatchResample(
        HW[0],
        HW[1],
        self.cfg.eval.resize_input_size,
        patch_size=self.patch_size,
    )
    frames_sm = resampler.adjust_to_patch_size(frames)

    with torch.no_grad():
      enc_feats, feats = self.encode(frames_sm[:1])

    # feats = enc_feats[-1][0].movedim(1, -1).unflatten(-1, (16, 16))
    probe = LinProbe(
        self.cfg.probe,
        enc_feats.shape[1],
        feats.shape[1],
        num_classes,
        kernel_size=1,
        padding=0,
        with_bn=False,
    )
    probe.cuda()

    self._fit_frames(
        self.cfg.eval.vos,
        probe,
        enc_feats[:1].detach(),
        feats[:1].detach(),
        masks[:1].cuda(),
    )

    # predicting result
    pred_masks = self._predict_with_probe(probe, frames_sm, masks.shape[-2:])

    # self._save_results(seq, frames, feats, pred_masks, masks)

    self.davis_metrics.add_sequence(
        masks.numpy(), pred_masks.cpu().detach().numpy()
    )

  def evaluate(self):

    for ii, seq in enumerate(tqdm(self.dataset.videos)):
      frames, masks = self.dataset.get_sequence(seq, val=True)
      print(f">>> Loading {seq} / {frames.shape[0]}")
      self.eval_sequence(frames, masks, seq)

    self.davis_metrics.print_metrics()
    self.writer.add_hparams(
        dict(self.cfg.eval.vos), self.davis_metrics.dict(), run_name="."
    )
    self.writer.flush()


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg):

  eval_engine = EvalVOS(cfg)
  eval_engine.evaluate()


if __name__ == "__main__":
  main()
