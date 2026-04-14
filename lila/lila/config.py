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

class LILAConfig:

  _configs = {
      'vits': {
          'features': 2 * 128,
          'out_channels': [48, 96, 192, 384],
          'intermediate_layer_idx': [2, 5, 8, 11],
      },
      'vitb': {
          'features': 2 * 192,
          'out_channels': [96, 192, 384, 768],
          'intermediate_layer_idx': [2, 5, 8, 11],
      },
      'vitl': {
          'features': 2 * 256,
          'out_channels': [256, 512, 1024, 1024],
          'intermediate_layer_idx': [4, 11, 17, 23],
      },
      'vitg': {
          'features': 2 * 384,
          'out_channels': [1536, 1536, 1536, 1536],
          'intermediate_layer_idx': [9, 19, 29, 39],
      },
  }

  def __getitem__(self, encoder_name):
    base_config = self._configs[encoder_name[-6:-2]]
    base_config['patch_size'] = int(encoder_name[-2:])
    return base_config


lila_config = LILAConfig()
