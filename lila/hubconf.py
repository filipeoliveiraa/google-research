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

dependencies = ["torch", "torchvision"]

from lila.dpt_lila import load_lila_model


def lila(
    encoder="dinov2_vitb14",
    pretrained=False,
    checkpoint_path=None,
    checkpoint_url=None,
    checkpoints_dir=None,
    model_name=None,
    checkpoint_name="best_checkpoint.pt",
    strict=False,
    device=None,
):
    """Load a LILA model from Torch Hub."""
    return load_lila_model(
        encoder_name=encoder,
        pretrained=pretrained,
        checkpoint_path=checkpoint_path,
        checkpoint_url=checkpoint_url,
        checkpoints_dir=checkpoints_dir,
        model_name=model_name,
        checkpoint_name=checkpoint_name,
        strict=strict,
        device=device,
    )


def lila_dinov2_vits14(**kwargs):
    return lila(encoder="dinov2_vits14", **kwargs)


def lila_dinov2_vitb14(**kwargs):
    return lila(encoder="dinov2_vitb14", **kwargs)


def lila_dinov2_vitl14(**kwargs):
    return lila(encoder="dinov2_vitl14", **kwargs)


def lila_dinov2reg_vits14(**kwargs):
    return lila(encoder="dinov2reg_vits14", **kwargs)


def lila_dinov2reg_vitb14(**kwargs):
    return lila(encoder="dinov2reg_vitb14", **kwargs)


def lila_dinov2reg_vitl14(**kwargs):
    return lila(encoder="dinov2reg_vitl14", **kwargs)
