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

"""Configuration for Gemma text SFT (instruction tuning)."""

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import ml_collections  # pylint: disable=g-import-not-at-top
from Uboreshaji_Modeli.common import config
from Uboreshaji_Modeli.common import config_utils


def get_config():
  """Returns the experiment configuration."""
  cfg = config.get_base_config()

  cfg.experiment_name = "gemma_text_sft"
  cfg.model_flavor = config.ModelFlavor.GEMMA_3_TEXT
  cfg.task_modality = config.TaskModality.TEXT
  cfg.task_type = config.TaskType.TEXT_SFT

  # Clear model_base and model_name to prevent derive_paths from clobbering
  # model_id.
  cfg.model_base = ""
  cfg.model_name = ""
  cfg.model_id = "google/gemma-3-270m"

  # Data overrides
  # Note: The dataset must contain a 'messages' field with chat turns.
  cfg.dataset.dataset_uri = "/path/to/dataset"

  cfg.output_dir = "/path/to/output"


  cfg.max_seq_length = 256

  cfg.platform.hardware = "a100=1"
  cfg.training.batch_size = 4
  cfg.training.num_train_epochs = 3
  cfg.training.learning_rate = 2e-4

  cfg.training.lora = ml_collections.ConfigDict()
  cfg.training.use_lora = True
  cfg.training.merge_lora_on_save = True
  cfg.training.lora.r = 16
  cfg.training.lora.alpha = 32
  cfg.training.lora.dropout = 0.05
  cfg.training.lora.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

  return cfg
