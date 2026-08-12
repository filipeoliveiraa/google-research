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

"""MMS engine for composed massively multilingual speech CTC SFT."""

from collections.abc import Mapping, Sequence
from typing import Any, Callable

from absl import logging
import ml_collections
import peft
import torch
from torch import nn
import transformers

from Uboreshaji_Modeli.common import audio_utils
from Uboreshaji_Modeli.common import data
from . import base
from . import decoders


# TODO: b/517531260 - Refactor MMS preprocessor to be for all audio tasks.
class MmsPreprocessor(base.DataPreprocessor):
  """Prepares audio raw wave datasets for massively multilingual speech CTC tasks."""

  def get_transform_fn(
      self,
      processor,
      cfg = None,
      *,
      is_train = False,
      **kwargs,
  ):
    """Returns a transform function converting audiowave signals to MMS inputs.

    Args:
      processor: AutoProcessor instance to extract features.
      cfg: Optional configuration dictionary with preprocessing overrides.
      is_train: Whether to process datasets in SFT training mode.
      **kwargs: Additional keyword parameters.

    Returns:
      A transform callable processing inputs into wav2vec2 feature vectors.
    """
    del self  # Unused.

    def transform_fn(
        batch,
    ):
      batch_dict = dict(batch)
      audio_column = batch_dict["audio"]
      if isinstance(audio_column, dict):
        audio_inputs = [audio_column]
      else:
        audio_inputs = audio_column

      target_sr = processor.feature_extractor.sampling_rate

      valid_audio_arrays = []
      valid_indices = []
      for i, x in enumerate(audio_inputs):
        arr = audio_utils.get_audio_array(x, target_sr=target_sr)
        if arr is not None:
          valid_audio_arrays.append(arr)
          valid_indices.append(i)
        else:
          logging.warning("Skipping bad audio record at index %d", i)

      if not valid_audio_arrays:
        return {k: [] for k in batch_dict.keys()}

      # Filter other columns in batch if some samples were skipped.
      if len(valid_indices) < len(audio_inputs):
        for k, v in batch_dict.items():
          if isinstance(v, list) and len(v) == len(audio_inputs):
            batch_dict[k] = [v[i] for i in valid_indices]

      inputs = processor(
          valid_audio_arrays,
          sampling_rate=processor.feature_extractor.sampling_rate,
          return_tensors="pt",
          padding=True,
      )

      if isinstance(audio_column, dict):
        batch_dict["input_values"] = inputs.input_values[0]
      else:
        batch_dict["input_values"] = inputs.input_values

      if "text" in batch_dict:
        text_key = "text"
      elif "transcription" in batch_dict:
        text_key = "transcription"
      else:
        text_key = None
      if text_key is not None:
        text_column = batch_dict[text_key]
        if isinstance(text_column, str):
          labels = processor.tokenizer([text_column]).input_ids
          batch_dict["labels"] = labels[0]
        else:
          labels = processor.tokenizer(text_column).input_ids
          batch_dict["labels"] = labels

      return batch_dict

    return transform_fn

  def get_collate_fn(
      self,
      cfg = None,
      **kwargs,
  ):
    """Returns a collation function to dynamically pad token and audio dimensions.

    Args:
      cfg: Optional configuration dictionary.
      **kwargs: Config options including the model's processor.

    Returns:
      A collation callable representing the padded loader batch compiler.
    """
    del self  # Unused.
    processor = kwargs.get("processor")
    if processor is None:
      raise ValueError(
          "processor is required in kwargs for MmsPreprocessor collate_fn."
      )

    device_type = cfg.training.get("device", "cuda") if cfg else "cuda"
    if device_type == "tpu":
      padding = "max_length"
      max_length = cfg.get("max_seq_length", 256) if cfg else 256
    else:
      padding = True
      max_length = None

    return data.DataCollatorCTCWithPadding(
        processor=processor,
        padding=padding,
        max_length=max_length,
    )

  def get_sft_config_overrides(
      self, cfg
  ):
    """Returns SFTConfig overrides for the custom MMS preprocessor."""
    del cfg  # Unused.
    return {
        "dataset_kwargs": {"skip_prepare_dataset": True},
    }


class MmsEngine(base.ModelEngine):
  """Coordinated speech loader settings and CTC model layers for the MMS pipeline."""

  def __init__(self):
    """Initializes the MMS engine instance with preprocessor and CTC decoder."""
    super().__init__(
        preprocessor=MmsPreprocessor(),
        loss_handler=None,
        decoder=decoders.MmsDecoder(),
    )

  @property
  def is_ctc(self):
    """Returns True indicating that MMS works with CTC SFT loss objectives."""
    return True

  def load_model_and_processor(
      self,
      model_id,
      device,
      **kwargs,
  ):
    """Loads massively multilingual speech Wav2Vec2-CTC checkpoints.

    Args:
      model_id: Pretrained repository ID or local path.
      device: Hardware PyTorch mapper execution device.
      **kwargs: Optional execution settings.

    Returns:
      A tuple (model, processor), where model is the initialized
      AutoModelForCTC model on the device, and processor is the AutoProcessor
      instance.
    """
    del self  # Unused.
    processor = transformers.AutoProcessor.from_pretrained(model_id)
    model = transformers.AutoModelForCTC.from_pretrained(model_id)

    cfg = kwargs.get("cfg")

    if cfg and cfg.training.get("freeze_feature_extractor", True):
      logging.info("Freezing MMS feature extractor.")
      if hasattr(model, "freeze_feature_extractor"):
        model.freeze_feature_extractor()

    if cfg and cfg.training.get("freeze_base_model", False):
      logging.info(
          "Freezing MMS base model. Only adapter/head weights will be trained."
      )
      if hasattr(model, "wav2vec2"):
        for param in model.wav2vec2.parameters():
          param.requires_grad = False

    if cfg and cfg.training.get("use_lora", False):
      lora_cfg = cfg.training.lora
      peft_config = peft.LoraConfig(
          r=lora_cfg.r,
          lora_alpha=lora_cfg.alpha,
          lora_dropout=lora_cfg.dropout,
          target_modules=list(lora_cfg.target_modules),
      )
      logging.info(
          "Wrapping MMS model with PEFT (LoRA) config: %s", peft_config
      )
      model = peft.get_peft_model(model, peft_config)

    model.to(device)
    return model, processor
