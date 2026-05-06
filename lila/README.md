### Disclaimer: This is not an officially supported Google product.

# Featurising Pixels from Dynamic 3D Scenes with Linear In-Context Learners
**Nikita Araslanov**, **Martin Sundermeyer**, **Hidenobu Matsuki**, **David Joseph Tan**, **Federico Tombari**

✨ CVPR 2026 (oral presentation) ✨

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)

[[Paper]](https://drive.google.com/file/d/1qXzxGk5JU8rvQwYTB6N8PVFR8YtlIqLt/preview) | [[Supplemental Material]](https://drive.google.com/uc?export=download&id=1mGHhhILMktr-179boTiyTJIPWfVzKaCj)

---

<p align="center">
  <img src="./assets/preview.gif" alt="Preview video" width="60%">
  <br>
  LILA learns dense visual representations from video using optical flow and monocular depth cues.
</p>


* * *

## Overview

This repository contains the reference implementation of LILA.

It includes code for model training, evaluation on DAVIS / COCO-Stuff / NYUv2, and Torch Hub entrypoints for loading released checkpoints.

* * *

## 🚀 Usage

> Note: Currently we only support loading local snapshots. Please manually download the snapshots below.

You can load and run LILA directly via PyTorch Hub or from a local clone of this repository.

For an end-to-end image demo, see [notebooks/lila_demo.ipynb](notebooks/lila_demo.ipynb). It shows how to load a local snapshot, run inference on an example image, and visualise a cosine-similarity heatmap from the decoder features.

🔹 Load from PyTorch Hub

```python
import torch

model = torch.hub.load(
    "<github-user>/<repo>",
    "lila_dinov2_vitb14",
    pretrained=True,
    checkpoint_url="<checkpoint-url>",
)

model.eval()
```

🔹 Load from a local clone

```python
import torch

model = torch.hub.load(
    ".",
    "lila_dinov2_vitb14",
    pretrained=True,
    checkpoint_path="./checkpoints/lila_dinov2_vitb14.pt",
    source="local",
)

model.eval()
```

🔹 Supported Torch Hub entrypoints

```text
lila_dinov2_vits14
lila_dinov2_vitb14
lila_dinov2_vitl14
```

🔹 Example inference

```python
import torch

x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    y_enc, y_dec = model(x)

print(y_enc.shape)  # encoder feature grid, e.g. (1, 768, 16, 16)
print(y_dec.shape)  # decoder feature grid, e.g. (1, 192, 224, 224)
```

## 🧰 Pre-trained Models

The Torch Hub entrypoints accept either `checkpoint_path` or `checkpoint_url`.

| Torch Hub entrypoint | Encoder | Decoder dim | Download |
| --- | --- | ---: | --- |
| `lila_dinov2_vitb14` | `dinov2_vitb14` | 192 | [Download (1.8GB)](https://drive.usercontent.google.com/download?id=13rkBLFDoDTt7kHIwmsfZcmIq7PQjWjEx&export=download&authuser=0&confirm=t&uuid=bbce813b-e1c7-4c26-b411-3878035e82d2&at=ALBwUgnRVLXb6xms0_3IIuerA2Pz%3A1777218601920) |
| `lila_dinov2_vitl14` | `dinov2_vitl14` | 256 | [Download (1.5GB)](https://drive.usercontent.google.com/download?id=1mxmjc1-IrEz1c8dO74VKcdqLqVJ2cVRx&export=download&authuser=0&confirm=t&uuid=88134464-902d-4183-8b97-9dc40e574971&at=ALBwUgk-MW6ncK48ESRs_yjoWE90:1777218719244) |


## 🏋️‍♀️ Training Instructions

### Step 0. Clone the repositories

```bash
git clone <repo-url>
cd <repo-dir>
git clone https://github.com/princeton-vl/SEA-RAFT.git raft
```

### Step 1. Set up the environment

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision lightning hydra-core omegaconf tensorboard tqdm albumentations opencv-python pillow timm imageio scipy scikit-image scikit-learn matplotlib
```

`xformers` is optional. The DINOv2 backbone falls back to standard PyTorch attention when it is unavailable.

### Step 2. Set up checkpoints and data

Place checkpoints under `./checkpoints/` or set `LILA_CHECKPOINTS_DIR` to a custom directory.

Keep the SEA-RAFT checkout at `./raft` so the optical-flow loader can import `raft/core`.

Training additionally expects:

- `depth_anything_v2_vitl.pth` from the [official repository](https://github.com/DepthAnything/Depth-Anything-V2).
- `Tartan-C-T-TSKH432x960-M.pth` from the [SEA-RAFT Google Drive](https://drive.google.com/drive/folders/1YLovlvUW94vciWvTyLf-p3uWscbOQRWW).

Dataset paths are configured through Hydra overrides. The default training configs use:

- `train_dataset.root_dir` for Ref-YTVOS or DAVIS-2017
- `val_dataset.root_dir` for DAVIS-2017

### Step 3. Run training

Example training command with the launch script:

```bash
bash launch/train.sh ytvos dinov2_vitb14 demo_run \
  train_dataset.root_dir=/path/to/RefYTVOS \
  val_dataset.root_dir=/path/to/DAVIS2017
```

By default, logs and checkpoints are written under `./runs/train`. Set `TB_DIR=/path/to/runs` if you want a different output root.

## 🧪 Evaluation

Available evaluation entrypoints:

```text
vos 	 # video object segmentation with linear probing
vosknn   # video object segmentation with k-nn
seg	 # semantic segmentation
openseg  # zero-shot semantic segmentation
norml    # surface normal estimation
```

Example evaluation command:

```bash
bash launch/eval.sh vos lila_dinov2_vitb14 demo_eval \
  model.encoder=dinov2_vitb14 \
  model.checkpoint_path=/path/to/checkpoints/lila_dinov2_vitb14.pt \
  data.davis_root=/path/to/DAVIS2017 \
  eval.resize_input=True \
  eval.resize_input_size=476
```

Notes:

- `seg` and `openseg` use `data.coco_root`.
- `norml` uses `data.nyuv2_root`.
- `openseg` additionally requires `data.coco_embeddings`, which defaults to `splits/coco_embeddings.pt`.

## 📚Citation

If you use this repository in your work, please cite our paper:
```
@inproceedings{Araslanov:2026:LILA,
  author = {Araslanov, Nikita and Sundermeyer, Martin and Matsuki, Hidenobu and Tan, David Joseph and Tombari, Federico},
  title = {Featurising Pixels from Dynamic 3D Scenes with Linear In-Context Learners},
  booktitle = {CVPR},
  year = {2026},
}
```
<hr>

<sub>
<b>Acknowledgements:</b>
This code builds on DepthAnything V2 and SEA-RAFT. We gratefully acknowledge their developers, as well as the broader open-source community, for foundational libraries such as PyTorch and NumPy that made this work possible.
</sub>
