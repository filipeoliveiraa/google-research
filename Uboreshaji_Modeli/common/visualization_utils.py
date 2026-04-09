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

"""Utilities for visualizing object detection results using Matplotlib."""

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
import numpy as np
from PIL import Image
import torch


def _draw_rect(ax, x, y, w, h, color, text, linestyle="-", alpha=1.0):
  """Draws a rectangle with a background label box for high visibility."""
  ax.add_patch(
      plt.Rectangle(
          (x, y),
          w,
          h,
          fill=False,
          color=color,
          linewidth=3,
          linestyle=linestyle,
          alpha=alpha,
      )
  )

  ax.text(
      x,
      y,
      text,
      color="white" if color != "white" else "black",
      fontsize=12,
      fontweight="black",
      alpha=alpha,
      bbox=dict(
          facecolor=color,
          alpha=alpha,
          edgecolor="none",
          pad=0,
          boxstyle="square,pad=0",
      ),
  )


def _draw_from_results(
    ax, boxes, labels, scores, class_names, color, linestyle="-", alpha=1.0
):
  """Draws a set of detections from processed results."""
  for b, l, s in zip(boxes, labels, scores):
    x1, y1, x2, y2 = b.tolist()
    label_text = f"{class_names[l.item()]}: {s.item():.2f}"
    _draw_rect(
        ax, x1, y1, x2 - x1, y2 - y1, color, label_text, linestyle, alpha
    )


def create_comparison_plot(
    image_np,
    gt_boxes,
    gt_labels,
    pred_boxes,
    pred_labels,
    pred_scores,
    class_names,
    graded_mask,
    ungraded_mask,
    score_threshold = 0.1,
    gt_box_format = "xywh",
):
  """Plots Ground Truth vs. Predictions side-by-side."""
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9), dpi=150)
  fig.subplots_adjust(wspace=0.1)

  ax1.imshow(image_np)
  ax1.set_title(
      "Ground Truth (Human Labels)",
      fontsize=18,
      fontweight="semibold",
      pad=30,
  )
  ax1.axis("off")

  for box, label in zip(gt_boxes, gt_labels):
    b = box.tolist()
    if gt_box_format == "xywh":
      x, y, bw, bh = b
    else:
      x, y, x2, y2 = b
      bw, bh = x2 - x, y2 - y

    _draw_rect(
        ax1, x, y, bw, bh, "white", class_names[label.item()], linestyle="-"
    )

  ax2.imshow(image_np)
  ax2.set_title(
      f"Predictions (Score Thresh {score_threshold}) | Solid=Graded,"
      " Dash=Ungraded",
      fontsize=18,
      fontweight="semibold",
      pad=30,
  )
  ax2.axis("off")

  if graded_mask.any():
    _draw_from_results(
        ax2,
        pred_boxes[graded_mask],
        pred_labels[graded_mask],
        pred_scores[graded_mask],
        class_names,
        "#0047AB",  # Blue
        linestyle="-",
        alpha=1.0,
    )

  if ungraded_mask.any():
    _draw_from_results(
        ax2,
        pred_boxes[ungraded_mask],
        pred_labels[ungraded_mask],
        pred_scores[ungraded_mask],
        class_names,
        "#FF8C00",  # Orange
        linestyle="--",
        alpha=0.7,
    )

  buf = io.BytesIO()
  plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
  plt.close(fig)
  buf.seek(0)
  return Image.open(buf)
