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

import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear(nn.Module):

    def __init__(self, encoder, idim, odim):
        super().__init__()
        #assert hasattr(encoder, "idim"), "Encoder missing attribute 'idim'"
        self.probe = nn.Conv2d(idim, odim, 1)
        self.encoder = encoder

    def parameters(self):
        return self.probe.parameters()

    def forward(self, x, scale_to):

        with torch.no_grad():
            feats, _, _, _ = self.encoder.sem_head(x)

        logits = self.probe(feats)
        logits = F.interpolate(logits, scale_to, mode="bilinear", align_corners=False)
        return logits


class LinearCat(nn.Module):

    def __init__(self, cfg, idim_enc, idim_dec, odim, kernel_size=5, stride=1, padding=2, with_bn=False):
        super().__init__()

        self.alpha_enc = cfg.alpha_enc
        self.alpha_dec = cfg.alpha_dec

        self.probe_enc = nn.Conv2d(idim_enc, odim, 1)

        if with_bn:
            self.probe_enc = nn.Sequential(nn.BatchNorm2d(idim_enc), self.probe_enc)

        self.probe_dec = nn.Conv2d(idim_dec, odim, \
                                   kernel_size, stride=stride, padding=padding)


    def parameters(self):
        return list(self.probe_dec.parameters()) + \
                list(self.probe_enc.parameters())

    def forward(self, feats_enc, feats_dec, scale_to):

        # decoder features
        logits_dec = self.probe_dec(feats_dec)
        if logits_dec.shape[-2:] != scale_to:
            logits_dec = F.interpolate(logits_dec, scale_to, mode="bilinear", align_corners=False)

        # encoder features
        logits_enc = self.probe_enc(feats_enc)
        logits_enc = F.interpolate(logits_enc, scale_to, mode="bilinear", align_corners=False)

        return self.alpha_dec * logits_dec + self.alpha_enc * logits_enc


def orthogonality_loss(conv):
    # conv.weight shape: [out_c, in_c, 1, 1]
    W = conv.weight.squeeze()  # [out_c, in_c]
    out_c, in_c = W.shape

    if out_c <= in_c:
        # Enforce W W^T = I
        M = W @ W.t()                     # [out_c, out_c]
        I = torch.eye(out_c, device=W.device, dtype=W.dtype)
    else:
        # Enforce W^T W = I
        M = W.t() @ W                     # [in_c, in_c]
        I = torch.eye(in_c, device=W.device, dtype=W.dtype)

    return F.mse_loss(M, I)

class LinearZeroShot(nn.Module):

    def __init__(self, cfg, idim_enc, idim_dec, text_embeddings, kernel_size=1, stride=1, padding=0, with_bn=False):
        super().__init__()

        self.alpha_enc = cfg.alpha_enc
        self.alpha_dec = cfg.alpha_dec

        weight_tensor = text_embeddings.unsqueeze(-1).unsqueeze(-1)

        # 3. Create a 1x1 Conv layer
        odim, emb_dim = text_embeddings.shape

        # Ensure your visual backbone outputs the same dimension as CLIP (e.g., 512)
        # If not, you need a projection layer before this.
        self.classifier = nn.Conv2d(emb_dim, odim, kernel_size=1, bias=False)

        # 4. ASSIGN AND FREEZE WEIGHTS
        self.classifier.weight = nn.Parameter(weight_tensor.float())
        self.classifier.weight.requires_grad = False

        self.enc_proj = nn.Conv2d(idim_enc, emb_dim, 1, bias=False)

        if True: #with_bn:
            self.enc_proj = nn.Sequential(nn.BatchNorm2d(idim_enc), self.enc_proj)

        self.dec_proj = nn.Conv2d(idim_dec, emb_dim, kernel_size, stride=stride, padding=padding, bias=False)

        # 3. Logit Scale (Temperature)
        # CLIP uses a learnable temperature, usually initialized to a high value (e.g. 100 or exp(4.6))
        # This helps gradients flow better through the normalized dot product.
        self.logit_scale = nn.Parameter(torch.ones([]) * 4.6052) # ln(100) = 4.6052

    def reg(self):
        return 1000. * orthogonality_loss(self.enc_proj)

    def parameters(self):
        return list(self.enc_proj.parameters()) + list(self.dec_proj.parameters()) + [self.logit_scale]

    def forward_feats(self, x, proj):
        # --- Decoder Branch ---
        # 1. Project to Embedding Dimension
        y = proj(x)

        # 2. L2 NORMALIZE (Critical for Zero-Shot)
        y_norm = F.normalize(y, p=2, dim=1)

        # 3. Classify (Cosine Similarity)
        logits = self.classifier(y_norm)

        return logits

    def forward(self, feats_enc, feats_dec, scale_to):

        # decoder features
        logits_dec = self.forward_feats(feats_dec, self.dec_proj)

        if logits_dec.shape[-2:] != scale_to:
            logits_dec = F.interpolate(logits_dec, scale_to, mode="bilinear", align_corners=False)

        # encoder features
        logits_enc = self.forward_feats(feats_enc, self.enc_proj)
        logits_enc = F.interpolate(logits_enc, scale_to, mode="bilinear", align_corners=False)

        # combine
        logits = self.alpha_dec * logits_dec + self.alpha_enc * logits_enc

        return logits * self.logit_scale.exp()