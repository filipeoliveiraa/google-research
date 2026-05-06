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

import hydra
import torch
import torch.optim as optim
from omegaconf import DictConfig
from torch.utils import tensorboard
from tqdm.auto import tqdm

import dataload

from eval_base import EvalBase
from metrics import NormalsEval
from probes.linear import LinearCat as LinProbe


def angular_loss(snorm_pr, snorm_gt, mask, uncertainty_aware=False, eps=1e-4):
    """Angular loss with an optional uncertainty-aware term."""
    assert mask.ndim == 4, f"mask should be (batch x height x width) not {mask.shape}"
    mask = mask.squeeze(1).float()

    if uncertainty_aware:
        assert snorm_pr.shape[1] == 4
        loss_ang = torch.cosine_similarity(snorm_pr[:, :3], snorm_gt, dim=1)
        loss_ang = loss_ang.clamp(min=-1 + eps, max=1 - eps).acos()

        kappa = torch.nn.functional.elu(snorm_pr[:, 3]) + 1.01
        kappa_reg = (1 + (-kappa * torch.pi).exp()).log() - (kappa.pow(2) + 1).log()
        loss = kappa_reg + kappa * loss_ang
    else:
        assert snorm_pr.shape[1] == 3
        loss_ang = torch.cosine_similarity(snorm_pr, snorm_gt, dim=1)
        loss = loss_ang.clamp(min=-1 + eps, max=1 - eps).acos()

    return loss[mask.bool()].mean()


class EvalNorml(EvalBase):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.train_dataset = dataload.NormalsDataset(cfg.data.nyuv2_root, "train", output_size=(448, 448))
        self.val_dataset = dataload.NormalsDataset(cfg.data.nyuv2_root, "val", output_size=(504, 504))

        assert self.val_dataset.num_classes() == self.train_dataset.num_classes(), "Unmatched number of train/val classes"

        self.probe = LinProbe(cfg.probe, cfg.probe.enc_fdim, cfg.probe.dec_fdim, self.train_dataset.num_classes())
        self.probe.cuda()

        self.job_root = os.path.join(cfg.runtime.root, cfg.runtime.name)
        self.writer = tensorboard.SummaryWriter(self.job_root)
        self.opt = optim.AdamW(self.parameters(), lr=cfg.eval.lr)
        self.best_val = {"rmse": 100.0, "d1": 0.0, "d2": 0.0, "d3": 0.0}

    def parameters(self):
        if self.cfg.eval.finetune:
            dec_params = list(self.model.latent_head.parameters()) + list(self.model.fnorm.parameters())
            base_lr = self.cfg.eval.lr
            base_wd = self.cfg.eval.weight_decay
            return [
                {"params": self.probe.parameters(), "lr": base_lr, "weight_decay": base_wd},
                {
                    "params": dec_params,
                    "lr": self.cfg.eval.finetune_lr_mult * base_lr,
                    "weight_decay": self.cfg.eval.finetune_wd_mult * base_wd,
                },
            ]
        return self.probe.parameters()

    def validate(self, epoch, giter):
        eval_stats = NormalsEval()
        dataloader = self.val_dataset.get_loader(
            batch_size=8,
            shuffle=False,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
        )

        losses = []
        for batch in tqdm(dataloader):
            images, depth, gt_normals = [x.cuda(non_blocking=True) for x in batch]

            with torch.no_grad():
                enc_feats, feats = self.encode(images)
                pred_normals = self.probe(enc_feats, feats, images.shape[-2:])
                mask = depth.cuda() > 0.0
                loss = angular_loss(pred_normals, gt_normals, mask)
                losses.append(loss.item())

            mask = (depth > 0).cuda()
            eval_stats.add_prediction(pred_normals, gt_normals, mask)

        self.writer.add_scalar("losses_normals/val_loss", sum(losses) / len(losses), giter)

        eval_dict = eval_stats.dict()
        for key, val in eval_dict.items():
            self.writer.add_scalar(f"normals/{key}", val, epoch)

        if eval_dict["rmse"] < self.best_val["rmse"]:
            print(f"Best RMSE: {eval_dict['rmse']}.")
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
                pbar.set_description(f"Iter {niter:03d} | Loss={loss.item():4.3f}")

                if niter % 10 == 0:
                    self.writer.add_scalar("losses_normals/train_loss", loss.item(), giter)

                giter += 1

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


@hydra.main(version_base=None, config_path="configs", config_name="eval")
def main(cfg):
    eval_engine = EvalNorml(cfg)
    eval_engine.train()


if __name__ == "__main__":
    main()
