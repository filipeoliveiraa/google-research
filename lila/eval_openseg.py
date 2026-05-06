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
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import DictConfig
from torch.utils import tensorboard
from tqdm.auto import tqdm

import dataload

from eval_base import EvalBase
from metrics import CocoEvalZeroShot
from probes.linear import LinearZeroShot as Probe


class EvalSeg(EvalBase):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.train_dataset = dataload.ZShotSegDataset(
            cfg.data.coco_root,
            "train2017",
            "splits/train_list.npy",
            preload=False,
        )
        self.val_dataset = dataload.ZShotSegDataset(
            cfg.data.coco_root,
            "val2017",
            "splits/test_list.npy",
            preload=False,
            val=True,
        )

        label_embeddings = torch.load(cfg.data.coco_embeddings)
        assert self.val_dataset.num_classes() == self.train_dataset.num_classes(), "Unmatched number of train/val classes"

        self.probe = Probe(
            cfg.probe,
            cfg.probe.enc_fdim,
            cfg.probe.dec_fdim,
            label_embeddings,
        )
        self.probe.cuda()

        self.job_root = os.path.join(cfg.runtime.root, cfg.runtime.name)
        self.writer = tensorboard.SummaryWriter(self.job_root)
        self.opt = optim.AdamW(
            self.parameters(),
            lr=cfg.eval.lr,
            weight_decay=cfg.eval.weight_decay,
        )
        self.best_val = {"mIoU_harmonic": 0.0}

    def parameters(self):
        if self.cfg.eval.finetune:
            base_lr = self.cfg.eval.lr
            base_wd = self.cfg.eval.weight_decay
            return [
                {"params": self.probe.parameters(), "lr": base_lr, "weight_decay": base_wd},
                {
                    "params": self.model.latent_head.parameters(),
                    "lr": self.cfg.eval.finetune_lr_mult * base_lr,
                    "weight_decay": self.cfg.eval.finetune_wd_mult * base_wd,
                },
            ]
        return self.probe.parameters()

    def validate(self, epoch):
        eval_stats = CocoEvalZeroShot(self.train_dataset.num_classes())
        dataloader = self.val_dataset.get_loader(
            batch_size=self.cfg.eval.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=4,
            pin_memory=True,
        )

        for batch in tqdm(dataloader):
            images, gt_masks, _ = [x.cuda(non_blocking=True) for x in batch]

            with torch.no_grad():
                enc_feats, feats = self.encode(images)
                pred_masks = self.probe(enc_feats, feats, images.shape[-2:])
                pred_masks_i = pred_masks.argmax(1)

            for pred_mask, gt_mask in zip(pred_masks_i, gt_masks):
                eval_stats.add_prediction(pred_mask, gt_mask)

        eval_dict = eval_stats.dict(self.train_dataset.ignore_cls_list)
        for key, val in eval_dict.items():
            self.writer.add_scalar(f"seg/{key}", val, epoch)

        if eval_dict["mIoU_harmonic"] > self.best_val["mIoU_harmonic"]:
            print(f"Best mIoU harmonic: {eval_dict['mIoU_harmonic']}.")
            self.best_val.update(eval_dict)

        return eval_stats

    def train(self):
        dataloader = self.train_dataset.get_loader(
            batch_size=self.cfg.eval.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        seen_ids = self.train_dataset.seen_ids

        giter = 0
        for epoch in range(self.cfg.eval.num_epochs):
            print(f"Epoch {epoch}")
            pbar = tqdm(dataloader)

            for niter, batch in enumerate(pbar):
                images, masks, _ = [x.cuda(non_blocking=True) for x in batch]

                self.opt.zero_grad()
                enc_feats, feats = self.encode(images)

                pred_mask = self.probe(enc_feats, feats, images.shape[-2:])
                gt_masks = masks.to(torch.long)
                loss = F.cross_entropy(
                    pred_mask,
                    gt_masks,
                    ignore_index=self.train_dataset.ignore_index,
                )

                # Encourage the model to suppress seen categories on ignored pixels.
                loss_neg = torch.tensor(0.0, device=images.device)
                ignore_mask = gt_masks == self.train_dataset.ignore_index
                if ignore_mask.any():
                    probs = F.softmax(pred_mask, dim=1)
                    p_unseen = probs[:, seen_ids, :, :].sum(dim=1)
                    loss_neg = (p_unseen[ignore_mask] ** 2).mean()
                    loss = 0.1 * loss + loss_neg

                loss.backward()
                self.opt.step()
                pbar.set_description(
                    f"Iter {niter:03d} | Loss={loss.item():4.3f}, Neg={loss_neg.item():4.3f}"
                )

                if niter % 10 == 0:
                    self.writer.add_scalar("losses_seg/train_loss", loss.item(), giter)

                giter += 1

            stats = self.validate(epoch)
            stats.print_metrics(self.train_dataset.ignore_cls_list)

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
    eval_engine = EvalSeg(cfg)
    eval_engine.train()


if __name__ == "__main__":
    main()
