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

import collections

from absl.testing import absltest
import numpy as np

from Uboreshaji_Modeli.common import metrics

EvalPrediction = collections.namedtuple(
    "EvalPrediction", ["predictions", "label_ids"]
)


class MetricsTest(absltest.TestCase):

  def _create_eval_prediction(
      self,
      logits,
      boxes,
      num_gt_boxes,
      gt_boxes,
      class_labels,
  ):
    """Helper to create EvalPrediction objects with correct numpy shapes."""
    boxes_arr = np.array([boxes], dtype=np.float32)

    if num_gt_boxes == 0:
      gt_boxes_arr = np.array([[]], dtype=np.float32).reshape(1, 0, 4)
      gt_classes_arr = np.array([[]], dtype=np.int64).reshape(1, 0)
    else:
      gt_boxes_arr = np.array([gt_boxes], dtype=np.float32)
      gt_classes_arr = np.array([class_labels], dtype=np.int64)

    return EvalPrediction(
        predictions=(np.array([logits], dtype=np.float32), boxes_arr),
        label_ids={
            "num_boxes": np.array([num_gt_boxes]),
            "boxes": gt_boxes_arr,
            "class_labels": gt_classes_arr,
        },
    )

  def test_perfect_match_single_box(self):
    compute_metrics = metrics.create_compute_metrics_fn(
        resize_to=100, score_threshold=0.5
    )
    eval_pred = self._create_eval_prediction(
        logits=[[10.0, -10.0]],
        boxes=[[0.5, 0.5, 0.2, 0.2]],
        num_gt_boxes=1,
        gt_boxes=[[0.5, 0.5, 0.2, 0.2]],
        class_labels=[0],
    )

    result = compute_metrics(eval_pred)

    self.assertAlmostEqual(result["map_50"], 1.0)
    self.assertAlmostEqual(result["map"], 1.0)
    self.assertAlmostEqual(result["mar_100"], 1.0)

  def test_perfect_match_multiple_boxes(self):
    compute_metrics = metrics.create_compute_metrics_fn(
        resize_to=100, score_threshold=0.5
    )
    eval_pred = self._create_eval_prediction(
        logits=[[10.0, -10.0], [-10.0, 10.0]],
        boxes=[[0.5, 0.5, 0.2, 0.2], [0.8, 0.8, 0.1, 0.1]],
        num_gt_boxes=2,
        gt_boxes=[[0.5, 0.5, 0.2, 0.2], [0.8, 0.8, 0.1, 0.1]],
        class_labels=[0, 1],
    )

    result = compute_metrics(eval_pred)

    self.assertAlmostEqual(result["map_50"], 1.0)

  def test_zero_match_wrong_class(self):
    compute_metrics = metrics.create_compute_metrics_fn(
        resize_to=100, score_threshold=0.5
    )
    eval_pred = self._create_eval_prediction(
        logits=[[10.0, -10.0]],
        boxes=[[0.5, 0.5, 0.2, 0.2]],
        num_gt_boxes=1,
        gt_boxes=[[0.5, 0.5, 0.2, 0.2]],
        class_labels=[1],
    )

    result = compute_metrics(eval_pred)

    self.assertAlmostEqual(result["map_50"], 0.0)

  def test_low_confidence_predictions_filtered_out(self):
    compute_metrics = metrics.create_compute_metrics_fn(
        resize_to=100, score_threshold=0.9
    )
    eval_pred = self._create_eval_prediction(
        logits=[[1.0, -1.0]],
        boxes=[[0.5, 0.5, 0.2, 0.2]],
        num_gt_boxes=1,
        gt_boxes=[[0.5, 0.5, 0.2, 0.2]],
        class_labels=[0],
    )

    result = compute_metrics(eval_pred)

    self.assertAlmostEqual(result["map_50"], 0.0)

  def test_empty_ground_truth_returns_undefined(self):
    compute_metrics = metrics.create_compute_metrics_fn(
        resize_to=100, score_threshold=0.5
    )
    eval_pred = self._create_eval_prediction(
        logits=[[10.0, -10.0]],
        boxes=[[0.5, 0.5, 0.2, 0.2]],
        num_gt_boxes=0,
        gt_boxes=[],
        class_labels=[],
    )

    result = compute_metrics(eval_pred)

    self.assertAlmostEqual(result["map_50"], -1.0)


if __name__ == "__main__":
  absltest.main()
