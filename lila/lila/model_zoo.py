# coding=utf-8
import os
from pathlib import Path
from typing import Dict, Optional, Union

import torch


CHECKPOINTS_ENV_VAR = "LILA_CHECKPOINTS_DIR"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"


BACKBONE_CHECKPOINTS: Dict[str, Dict[str, Optional[str]]] = {
    "dinov2_vits14": {
        "filename": "dinov2_vits14_pretrain.pth",
        "url": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
    },
    "dinov2_vitb14": {
        "filename": "dinov2_vitb14_pretrain.pth",
        "url": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
    },
    "dinov2_vitl14": {
        "filename": "dinov2_vitl14_pretrain.pth",
        "url": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
    },
    "dinov2_vitg14": {
        "filename": "dinov2_vitg14_pretrain.pth",
        "url": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth",
    },
    "dinov2reg_vits14": {
        "filename": "dinov2_vits14_reg4_pretrain.pth",
        "url": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth",
    },
    "dinov2reg_vitb14": {
        "filename": "dinov2_vitb14_reg4_pretrain.pth",
        "url": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth",
    },
    "dinov2reg_vitl14": {
        "filename": "dinov2_vitl14_reg4_pretrain.pth",
        "url": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth",
    },
    "dinov2reg_vitg14": {
        "filename": "dinov2_vitg14_reg4_pretrain.pth",
        "url": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_reg4_pretrain.pth",
    },
}


AUXILIARY_CHECKPOINTS: Dict[str, Dict[str, Optional[str]]] = {
    "depth_anything_v2_vits": {"filename": "depth_anything_v2_vits.pth", "url": None},
    "depth_anything_v2_vitb": {"filename": "depth_anything_v2_vitb.pth", "url": None},
    "depth_anything_v2_vitl": {"filename": "depth_anything_v2_vitl.pth", "url": None},
    "depth_anything_v2_vitg": {"filename": "depth_anything_v2_vitg.pth", "url": None},
    "searaft_tskh432x960_m": {"filename": "Tartan-C-T-TSKH432x960-M.pth", "url": None},
}


def get_checkpoints_dir(checkpoints_dir = None):
    if checkpoints_dir:
        return Path(checkpoints_dir).expanduser().resolve()

    env_dir = os.environ.get(CHECKPOINTS_ENV_VAR)
    if env_dir:
        return Path(env_dir).expanduser().resolve()

    return DEFAULT_CHECKPOINTS_DIR


def resolve_checkpoint_path(
    filename,
    checkpoints_dir = None,
    subdir = None,
    must_exist = True,
):
    root = get_checkpoints_dir(checkpoints_dir)
    path = root / subdir / filename if subdir else root / filename

    if must_exist and not path.is_file():
        raise FileNotFoundError(
            f"Missing checkpoint: {path}. "
            f"Place the file under {root} or set {CHECKPOINTS_ENV_VAR}."
        )

    return path


def resolve_model_checkpoint(
    model_name,
    checkpoint_name = "best_checkpoint.pt",
    checkpoints_dir = None,
):
    if not model_name:
        raise ValueError("model_name must be set when checkpoint_path is not provided.")

    return resolve_checkpoint_path(
        checkpoint_name,
        checkpoints_dir=checkpoints_dir,
        subdir=model_name,
    )


def _load_state_dict_from_spec(
    registry,
    name,
    checkpoints_dir = None,
    map_location = "cpu",
):
    if name not in registry:
        raise KeyError(f"Unknown checkpoint entry: {name}")

    spec = registry[name]
    local_path = resolve_checkpoint_path(
        spec["filename"],
        checkpoints_dir=checkpoints_dir,
        must_exist=False,
    )
    if local_path.is_file():
        return torch.load(local_path, map_location=map_location, weights_only=True)

    if spec["url"]:
        return torch.hub.load_state_dict_from_url(
            spec["url"],
            map_location=map_location,
            file_name=spec["filename"],
        )

    raise FileNotFoundError(
        f"Missing checkpoint: {local_path}. "
        f"This file is not auto-downloadable yet, so add it manually or set {CHECKPOINTS_ENV_VAR}."
    )


def load_backbone_state_dict(
    name,
    checkpoints_dir = None,
    map_location = "cpu",
):
    return _load_state_dict_from_spec(
        BACKBONE_CHECKPOINTS,
        name,
        checkpoints_dir=checkpoints_dir,
        map_location=map_location,
    )


def load_auxiliary_state_dict(
    name,
    checkpoints_dir = None,
    map_location = "cpu",
):
    return _load_state_dict_from_spec(
        AUXILIARY_CHECKPOINTS,
        name,
        checkpoints_dir=checkpoints_dir,
        map_location=map_location,
    )


def load_lila_state_dict(
    checkpoint_path = None,
    checkpoint_url = None,
    checkpoints_dir = None,
    model_name = None,
    checkpoint_name = "best_checkpoint.pt",
    map_location = "cpu",
):
    if checkpoint_path:
        return torch.load(
            Path(checkpoint_path).expanduser(),
            map_location=map_location,
            weights_only=True,
        )

    if checkpoint_url:
        file_name = Path(checkpoint_name).name
        return torch.hub.load_state_dict_from_url(
            checkpoint_url,
            map_location=map_location,
            file_name=file_name,
        )

    if model_name:
        checkpoint_path = resolve_model_checkpoint(
            model_name,
            checkpoint_name=checkpoint_name,
            checkpoints_dir=checkpoints_dir,
        )
        return torch.load(checkpoint_path, map_location=map_location, weights_only=True)

    raise ValueError(
        "Provide checkpoint_path, checkpoint_url, or model_name when pretrained weights are requested."
    )
