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

import hydra
import torch
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import DictConfig
from torch.utils import tensorboard
from tqdm.auto import tqdm

import dataload
import utils.davis_eval as davis_eval

from eval_base import EvalBase, PatchResample
from probes.linear import LinearCat as LinProbe


class EvalVOS(EvalBase):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.dataset = dataload.VideoDataset(
            cfg.data.davis_root,
            "splits/val_davis.txt",
            (0.5, 1.0),
            (1.0, 1.0),
        )
        self.writer = tensorboard.SummaryWriter(self.eval_dir)
        self.davis_metrics = davis_eval.DavisEvaluate()

    def _fit_frames(self, cfg_eval, probe, feats_enc, feats, masks):
        opt = optim.AdamW(probe.parameters(), lr=cfg_eval.lr, weight_decay=cfg_eval.weight_decay)
        pbar = tqdm(range(cfg_eval.num_epochs))

        for epoch in pbar:
            opt.zero_grad()
            pred_mask = probe(feats_enc, feats, masks.shape[-2:])
            loss = F.cross_entropy(pred_mask, masks.long())
            loss.backward()
            opt.step()
            pbar.set_description(f"Epoch {epoch:03d} | Loss={loss.item():4.3f}")

    @torch.no_grad()
    def _predict_with_probe(self, probe, frames, hw):
        outputs = []
        for frame in frames.split(1):
            feats_enc, feats = self.encode(frame)
            pred_masks = probe(feats_enc, feats, hw)
            outputs.append(pred_masks.argmax(1))
        return torch.cat(outputs)

    def eval_sequence(self, frames, masks):
        num_classes = masks.max().item() + 1

        resampler = PatchResample(
            frames.shape[-2],
            frames.shape[-1],
            self.cfg.eval.resize_input_size,
            patch_size=self.patch_size,
        )
        frames_sm = resampler.adjust_to_patch_size(frames)

        with torch.no_grad():
            enc_feats, feats = self.encode(frames_sm[:1])

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

        pred_masks = self._predict_with_probe(probe, frames_sm, masks.shape[-2:])
        self.davis_metrics.add_sequence(masks.numpy(), pred_masks.cpu().detach().numpy())

    def evaluate(self):
        for seq in tqdm(self.dataset.videos):
            frames, masks = self.dataset.get_sequence(seq, val=True)
            print(f">>> Loading {seq} / {frames.shape[0]}")
            self.eval_sequence(frames, masks)

        self.davis_metrics.print_metrics()
        self.writer.add_hparams(dict(self.cfg.eval.vos), self.davis_metrics.dict(), run_name=".")
        self.writer.flush()


@hydra.main(version_base=None, config_path="configs", config_name="eval")
def main(cfg):
    eval_engine = EvalVOS(cfg)
    eval_engine.evaluate()


if __name__ == "__main__":
    main()
