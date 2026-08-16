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

"""Matcher logic adapted from DETR."""

from collections.abc import Mapping
from typing import Any

import scipy.optimize
import torch
from torch import nn

from Uboreshaji_Modeli.common import box_utils
from Uboreshaji_Modeli.common import config as base_config


def _hungarian_matcher(
    cost_matrix, sizes
):
  """Hungarian matcher logic."""
  return [
      scipy.optimize.linear_sum_assignment(c[i])
      for i, c in enumerate(cost_matrix.split(sizes, -1))
  ]


def _greedy_matcher(
    cost_matrix, sizes
):
  """Greedy matcher logic. Adapted from OWL-v2.

  The algorithm proceeds as follows:
  1. Flatten the cost matrix and sort all costs in ascending order.
  2. Map the sorted flat indices back to their original row (query) and column
     (target) indices.
  3. Iterate through the sorted matches:
     a. If a target is padded, skip it.
     b. If neither the row (query) nor the column (target) has been matched yet,
        pair them.
     c. Stop once all possible matches are made (min of num_queries and
        curr_targets).

  Args:
    cost_matrix: A tensor of shape [batch_size, num_queries, num_targets]
      containing the costs of all possible assignments.
    sizes: A list of integers, where each integer is the number of actual
      targets in each element of the batch (handling padding).

  Returns:
    A list of tuples, where each tuple corresponds to an image in the
    batch. Each tuple contains two torch.Tensor: the first contains the
    indices of the predicted queries that are matched, and the second
    contains the indices of the target boxes that they are matched to.
  """
  device = cost_matrix.device
  batch_size, num_queries, num_targets = cost_matrix.shape

  flat_costs = cost_matrix.view(batch_size, -1)
  _, sorted_indices = torch.sort(flat_costs, dim=1, stable=True)

  sorted_rows = torch.div(sorted_indices, num_targets, rounding_mode="floor")
  sorted_cols = sorted_indices % num_targets

  rows_cpu = sorted_rows.cpu().tolist()
  cols_cpu = sorted_cols.cpu().tolist()

  final_results = []

  for b in range(batch_size):
    curr_targets = sizes[b]

    row_used = [False] * num_queries
    col_used = [False] * curr_targets

    res_rows, res_cols = [], []
    match_count = 0
    max_possible = min(num_queries, curr_targets)

    for r, c in zip(rows_cpu[b], cols_cpu[b]):
      if c >= curr_targets:
        continue

      if not row_used[r] and not col_used[c]:
        row_used[r] = True
        col_used[c] = True
        res_rows.append(r)
        res_cols.append(c)
        match_count += 1

        if match_count == max_possible:
          break

    final_results.append((
        torch.tensor(res_rows, device=device, dtype=torch.int64),
        torch.tensor(res_cols, device=device, dtype=torch.int64),
    ))

  return final_results


class Matcher(nn.Module):
  """A matcher for computing optimal assignment between targets and predictions.

  The assignment is computed by solving a global bipartite matching problem,
  which minimizes the total cost based on classification and bounding box
  similarity.

  Attributes:
    cost_class: The weight for the classification cost.
    cost_bbox: The weight for the L1 bounding box cost.
    cost_giou: The weight for the generalized IoU cost.
    matcher_type: The type of matching algorithm to use.
  """

  def __init__(
      self,
      *,
      cost_class = 1.0,
      cost_bbox = 1.0,
      cost_giou = 1.0,
      matcher_type = base_config.MatcherType.HUNGARIAN,
  ):
    """Initializes the instance."""
    super().__init__()
    self.cost_class = cost_class
    self.cost_bbox = cost_bbox
    self.cost_giou = cost_giou
    self.matcher_type = matcher_type
    assert (
        cost_class != 0 or cost_bbox != 0 or cost_giou != 0
    ), "all costs cannot be 0"

  @torch.no_grad()
  def forward(
      self, outputs, targets
  ):
    """Computes the assignment between targets and predictions.

    Args:
        outputs: A dictionary of outputs from the model.
        targets: A list of dictionaries, each containing the ground truth labels
          for an image.

    Returns:
        A list of tuples, where each tuple corresponds to an image in the
        batch. Each tuple contains two torch.Tensor: the first contains the
        indices of the predicted queries that are matched, and the second
        contains the indices of the target boxes that they are matched to.
    """
    bs, num_queries = outputs["logits"].shape[:2]
    device = outputs["logits"].device

    if not targets:
      return []

    if num_queries == 0:
      return [
          (
              torch.as_tensor([], dtype=torch.int64, device=device),
              torch.as_tensor([], dtype=torch.int64, device=device),
          )
          for _ in range(bs)
      ]

    indices = []
    for i in range(bs):
      tgt_labels = targets[i]["class_labels"]
      tgt_boxes = targets[i]["boxes"]
      num_targets_i = tgt_labels.shape[0]

      if num_targets_i == 0:
        indices.append((
            torch.as_tensor([], dtype=torch.int64, device=device),
            torch.as_tensor([], dtype=torch.int64, device=device),
        ))
        continue

      out_prob_i = outputs["logits"][i].sigmoid()  # [num_queries, num_classes]
      out_bbox_i = outputs["pred_boxes"][i]  # [num_queries, 4]

      cost_class = -out_prob_i[:, tgt_labels.long()]
      cost_bbox = torch.cdist(out_bbox_i, tgt_boxes, p=1)
      cost_giou = -box_utils.generalized_box_iou(
          boxes1=box_utils.box_cxcywh_to_xyxy(out_bbox_i),
          boxes2=box_utils.box_cxcywh_to_xyxy(tgt_boxes),
      )

      cost_matrix_i = (
          self.cost_bbox * cost_bbox
          + self.cost_class * cost_class
          + self.cost_giou * cost_giou
      ).cpu()

      if self.matcher_type == base_config.MatcherType.HUNGARIAN:
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(
            cost_matrix_i.numpy()
        )
      elif self.matcher_type == base_config.MatcherType.GREEDY:
        res = _greedy_matcher(cost_matrix_i.unsqueeze(0), [num_targets_i])
        row_ind, col_ind = res[0][0].cpu().numpy(), res[0][1].cpu().numpy()
      else:
        raise ValueError(f"Unsupported matcher type: {self.matcher_type}")

      indices.append((
          torch.as_tensor(row_ind, dtype=torch.int64, device=device),
          torch.as_tensor(col_ind, dtype=torch.int64, device=device),
      ))

    return indices


class HungarianMatcher(Matcher):
  """Hungarian matcher."""

  def __init__(self, **kwargs):
    super().__init__(matcher_type=base_config.MatcherType.HUNGARIAN, **kwargs)


class GreedyMatcher(Matcher):
  """Greedy matcher."""

  def __init__(self, **kwargs):
    super().__init__(matcher_type=base_config.MatcherType.GREEDY, **kwargs)
