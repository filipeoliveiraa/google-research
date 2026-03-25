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

"""Metrics for object detection."""

from typing import Callable

import torch
from torchmetrics.detection import mean_ap

from Uboreshaji_Modeli.common import box_utils


def create_compute_metrics_fn(
    resize_to, score_threshold = 0.0
):
  """Creates compute_metrics function for HF Trainer."""
  map_metric = mean_ap.MeanAveragePrecision(
      box_format="xyxy", max_detection_thresholds=[10, 100, 1000]
  )

  def compute_metrics(eval_pred):
    logits_arr, boxes_arr = eval_pred.predictions
    labels_dict = eval_pred.label_ids
    map_metric.reset()

    all_logits = torch.from_numpy(logits_arr).float()
    all_boxes = torch.from_numpy(boxes_arr).float()

    preds, targets = [], []

    for i, logits in enumerate(all_logits):
      n_tgt = int(labels_dict["num_boxes"][i].item())

      probs = logits.sigmoid()
      scores, pred_labels = probs.max(-1)

      keep = scores > score_threshold

      pred_boxes = box_utils.box_cxcywh_to_xyxy(all_boxes[i][keep]) * resize_to

      preds.append({
          "boxes": pred_boxes,
          "scores": scores[keep],
          "labels": pred_labels[keep],
      })

      tgt_boxes_raw = torch.from_numpy(labels_dict["boxes"][i, :n_tgt]).float()
      tgt_boxes = box_utils.box_cxcywh_to_xyxy(tgt_boxes_raw) * resize_to

      targets.append({
          "boxes": tgt_boxes,
          "labels": (
              torch.from_numpy(labels_dict["class_labels"][i, :n_tgt]).long()
          ),
      })

    map_metric.update(preds, targets)
    results = map_metric.compute()

    return {
        "map": results["map"].item(),
        "map_50": results["map_50"].item(),
        "map_75": results["map_75"].item(),
        "map_small": results["map_small"].item(),
        "mar_100": results["mar_100"].item(),
        "mar_small": results["mar_small"].item(),
        "mar_medium": results["mar_medium"].item(),
        "mar_large": results["mar_large"].item(),
    }
  return compute_metrics
