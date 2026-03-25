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

"""Simple tests for box_utils to verify coverage."""

from absl.testing import absltest
import torch
from Uboreshaji_Modeli.common import box_utils


class BoxUtilsTest(absltest.TestCase):

  def test_rescale_bboxes(self):
    bboxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32)
    size = (1000.0, 800.0)
    rescaled = box_utils.rescale_bboxes(bboxes, size)
    expected = torch.tensor([[320, 400, 480, 600]], dtype=torch.float32)
    torch.testing.assert_close(rescaled, expected)

  def test_box_iou(self):
    boxes1 = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=torch.float32)
    boxes2 = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)
    iou, _ = box_utils.box_iou(boxes1=boxes1, boxes2=boxes2)
    self.assertEqual(iou.shape, (2, 1))
    self.assertEqual(iou[0, 0], 1.0)
    self.assertAlmostEqual(iou[1, 0].item(), 1 / 7)

  def test_generalized_box_iou(self):
    boxes1 = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)
    boxes2 = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)
    giou = box_utils.generalized_box_iou(boxes1=boxes1, boxes2=boxes2)
    self.assertAlmostEqual(giou[0, 0].item(), 1.0)


if __name__ == "__main__":
  absltest.main()
