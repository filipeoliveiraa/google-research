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

from unittest import mock

from absl.testing import absltest
import ml_collections
import transformers

from Uboreshaji_Modeli.engines import whisper


class WhisperEngineTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.engine = whisper.WhisperEngine()

  def test_get_sft_config_overrides(self):
    cfg = ml_collections.ConfigDict()
    overrides = self.engine.get_sft_config_overrides(cfg)
    self.assertEqual(
        overrides,
        {"dataset_kwargs": {"skip_prepare_dataset": True}},
    )

  def test_get_collate_fn(self):
    mock_processor = mock.create_autospec(
        transformers.WhisperProcessor, instance=True
    )
    collate_fn = self.engine.get_collate_fn(processor=mock_processor)
    self.assertIsNotNone(collate_fn)


if __name__ == "__main__":
  absltest.main()
