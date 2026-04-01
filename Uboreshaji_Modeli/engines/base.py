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

"""Base class for model-specific engines."""

import abc
from collections.abc import Mapping
from typing import Any, Callable

import ml_collections
import torch
from torch import nn


class ModelEngine(abc.ABC):
  """Abstract base class for model-specific logic."""

  @abc.abstractmethod
  def load_model_and_processor(
      self, model_id, device
  ):
    """Loads the model and processor."""

  @abc.abstractmethod
  def get_transform_fn(
      self,
      processor,
      text_inputs,
      dataset_id2label,
      model_label2id,
      cfg = None,
      is_train = False,
  ):
    """Returns the transformation function for the dataset."""

  @abc.abstractmethod
  def get_collate_fn(
      self,
      cfg = None,
  ):
    """Returns the collation function for the data loader."""

  @abc.abstractmethod
  def get_criterion(
      self,
      num_classes,
      cfg,
      device,
  ):
    """Returns the loss criterion and weight dictionary."""
