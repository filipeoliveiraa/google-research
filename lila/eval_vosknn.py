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
from omegaconf import DictConfig
from torch.utils import tensorboard
from tqdm.auto import tqdm

import dataload
import utils.davis_eval as davis_eval

from eval_base import EvalBase, PatchResample


class EvalKnnVOS(EvalBase):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.dataset = dataload.VideoDataset(
            cfg.data.davis_root,
            "splits/val_davis.txt",
            (0.5, 1.0),
            (1.0, 1.0),
        )
        self.writer = tensorboard.SummaryWriter(os.path.join(cfg.runtime.root, cfg.runtime.name))
        self.davis_metrics = davis_eval.DavisEvaluate()

    def _pad_and_unfold(self, x):
        d, h, w = x.shape[-3:]
        attn_window = self.cfg.eval.vos_knn.attn_window
        padding = attn_window // 2
        x_padded = F.pad(x, (padding, padding, padding, padding), mode="replicate")
        x_unfolded = F.unfold(x_padded, kernel_size=attn_window)
        return x_unfolded.unflatten(1, (d, attn_window**2))

    @torch.compile(mode="reduce-overhead")
    def _propagate_labels(self, feats_current, feats_context_list, masks_context_list):
        window_size = self.cfg.eval.vos_knn.attn_window
        h, w = feats_current.shape[-2:]

        feats_context = torch.stack(feats_context_list)
        masks_context = torch.stack(masks_context_list)

        q_normed = feats_current.movedim(1, -1)
        k_normed = feats_context.movedim(1, -1)
        padding = window_size // 2

        with torch.amp.autocast("cuda", enabled=True):
            v_patches = self._pad_and_unfold(masks_context)
            v_patches = v_patches.permute(3, 0, 2, 1).flatten(1, 2)
            k_padded = F.pad(k_normed, (0, 0, padding, padding, padding, padding), "replicate")

            patches_list = []
            for u_offset in range(window_size):
                for v_offset in range(window_size):
                    k_slice = k_padded[:, u_offset : u_offset + h, v_offset : v_offset + w, :]
                    patches_list.append((q_normed * k_slice).sum(dim=3))

            all_sims_stacked = torch.stack(patches_list, dim=0).flatten(-2)
            sim = all_sims_stacked.permute(2, 1, 0).flatten(1, 2)

            topk, topk_idxs = sim.topk(self.cfg.eval.vos_knn.num_nn, -1)
            topk_vals = v_patches.gather(1, topk_idxs.unsqueeze(-1).expand(-1, -1, v_patches.shape[-1]))

            out_flat = torch.einsum("nk,nkc->cn", F.softmax(topk, 1), topk_vals)
            out_grid = out_flat.unflatten(-1, (h, w))

        return out_grid

    @staticmethod
    def _to_onehot(mask_hw):
        num_classes = int(mask_hw.max()) + 1
        one_hot = F.one_hot(mask_hw.long(), num_classes=num_classes)
        return one_hot.movedim(-1, 1).float()

    def _rescale_grid(self, x):
        h, w = x.shape[-2:]
        feat_grid_size = self.cfg.eval.vos_knn.feat_grid_size

        if h < w:
            h_min = feat_grid_size
            ratio = h_min / h
            w_min = int(w * ratio)
        else:
            w_min = feat_grid_size
            ratio = w_min / w
            h_min = int(h * ratio)

        return self._rescale(x, (h_min, w_min))

    def _predict_with_knn(self, cfg, frames, ref_mask):
        def l2norm(x):
            return F.normalize(x, dim=1, p=2) / self.cfg.eval.vos_knn.temp ** 0.5

        feats_context = {"enc": [], "dec": []}
        masks_context = []
        pred_masks = []

        enc_feats0, dec_feats0 = self.encode(frames[:1])
        feats0_rescaled = {
            "enc": l2norm(self._rescale_grid(enc_feats0)),
            "dec": l2norm(self._rescale_grid(dec_feats0)),
        }

        ref_mask0_rescaled = self._rescale_as(ref_mask, feats0_rescaled["enc"])
        feats_weight = {"enc": self.cfg.probe.alpha_enc, "dec": self.cfg.probe.alpha_dec}

        for frame_current in tqdm(frames[1:]):
            enc_feats, dec_feats = self.encode(frame_current[None])
            context_masks = [ref_mask0_rescaled[0]] + masks_context
            mask_out = torch.zeros_like(context_masks[0])

            for feats, ftype in zip([enc_feats, dec_feats], ["enc", "dec"]):
                if feats_weight[ftype] == 0:
                    continue

                context_feats = [feats0_rescaled[ftype][0]] + feats_context[ftype]
                feats_current = l2norm(self._rescale_grid(feats))
                mask = self._propagate_labels(feats_current, context_feats, context_masks)
                mask_out += feats_weight[ftype] * mask

                feats_context[ftype].append(feats_current[0])
                feats_context[ftype] = feats_context[ftype][-cfg.context_size:]

            masks_context.append(mask_out)
            masks_context = masks_context[-cfg.context_size:]
            pred_masks.append(mask_out)

        pred_masks = self._rescale_as(torch.stack(pred_masks), ref_mask)
        return pred_masks.argmax(1)

    @torch.no_grad()
    def eval_sequence(self, frames, masks):
        resampler = PatchResample(frames.shape[-2], frames.shape[-1], 476, patch_size=14)
        frames_sm = resampler.adjust_to_patch_size(frames)
        ref_mask_onehot = self._to_onehot(masks[:1].cuda())

        pred_masks_but_first = self._predict_with_knn(self.cfg.eval.vos_knn, frames_sm, ref_mask_onehot)
        pred_masks = torch.cat([masks[:1], pred_masks_but_first.cpu().detach()])

        self.davis_metrics.add_sequence(masks.numpy(), pred_masks.numpy())

    def evaluate(self):
        for seq in tqdm(self.dataset.videos):
            frames, masks = self.dataset.get_sequence(seq, val=True)
            print(f">>> Loading {seq} / {frames.shape[0]}")
            self.eval_sequence(frames, masks)

        self.davis_metrics.print_metrics()
        self.writer.add_hparams(dict(self.cfg.eval.vos_knn), self.davis_metrics.dict(), run_name=".")
        self.writer.flush()


@hydra.main(version_base=None, config_path="configs", config_name="eval")
def main(cfg):
    eval_engine = EvalKnnVOS(cfg)
    eval_engine.evaluate()


if __name__ == "__main__":
    main()
