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

"""Tests for data utilities."""

import os
import sys

from absl import flags
from absl.testing import absltest
import datasets
import ml_collections

from Uboreshaji_Modeli.common import data


FLAGS = flags.FLAGS
if not FLAGS.is_parsed():
  FLAGS(sys.argv[:1])


class GetDatasetTest(absltest.TestCase):

  def test_returns_loaded_dataset(self):
    temp_dir = self.create_tempdir().full_path
    fake_dataset_dict = {"col1": [1, 2], "col2": ["a", "b"]}
    fake_dataset = datasets.Dataset.from_dict(fake_dataset_dict)
    fake_dataset.save_to_disk(temp_dir)
    cfg = ml_collections.ConfigDict()
    cfg.dataset = ml_collections.ConfigDict()
    cfg.dataset.dataset_path = temp_dir
    result = data.get_dataset(cfg)
    self.assertEqual(list(result), list(fake_dataset))

  def test_propagates_error(self):
    temp_dir = self.create_tempdir().full_path
    cfg = ml_collections.ConfigDict()
    cfg.dataset = ml_collections.ConfigDict()
    cfg.dataset.dataset_path = os.path.join(temp_dir, "nonexistent")
    with self.assertRaises(FileNotFoundError):
      data.get_dataset(cfg)


if __name__ == "__main__":
  absltest.main()
