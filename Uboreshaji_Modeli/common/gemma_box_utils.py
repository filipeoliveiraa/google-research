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

"""Gemma-specific box utilities for detection."""

from collections.abc import Sequence
import json
from typing import TypedDict

from Uboreshaji_Modeli.common import box_utils

_LOC_GRID_SIZE = 1024
_JSON_GRID_SIZE = 1000  # Gemini convention: normalize coordinates to [0, 1000].


class DetectionObjects(TypedDict):
  bbox: Sequence[Sequence[float]]
  category: Sequence[int]


def convert_boxes_to_detection_string(
    bboxes,
    category_ids,
    class_names,
    *,
    image_width,
    image_height,
):
  """Converts bboxes and categories into a Gemma location-token string.

  Each detection becomes `<locYYYY><locXXXX><locYYYY><locXXXX> label`,
  separated by ` ; `. Coordinates are quantized to a 1024-cell grid.

  Args:
    bboxes: COCO-format bounding boxes [x, y, w, h].
    category_ids: Category index per bbox.
    class_names: Human-readable class name list.
    image_width: Source image width in pixels.
    image_height: Source image height in pixels.

  Returns:
    A semicolon-separated detection string.
  """
  detection_strings = []
  for bbox, cat_id in zip(bboxes, category_ids):
    xyxy = box_utils.coco_to_xyxy(bbox)
    x_min, y_min, x_max, y_max = xyxy
    y1 = int(y_min / image_height * (_LOC_GRID_SIZE - 1))
    x1 = int(x_min / image_width * (_LOC_GRID_SIZE - 1))
    y2 = int(y_max / image_height * (_LOC_GRID_SIZE - 1))
    x2 = int(x_max / image_width * (_LOC_GRID_SIZE - 1))

    y1 = max(0, min(_LOC_GRID_SIZE - 1, y1))
    x1 = max(0, min(_LOC_GRID_SIZE - 1, x1))
    y2 = max(0, min(_LOC_GRID_SIZE - 1, y2))
    x2 = max(0, min(_LOC_GRID_SIZE - 1, x2))

    loc_str = f"<loc{y1:04d}><loc{x1:04d}><loc{y2:04d}><loc{x2:04d}>"
    label = class_names[cat_id]
    detection_strings.append(f"{loc_str} {label}")

  return " ; ".join(detection_strings)


def convert_boxes_to_json_string(
    bboxes,
    category_ids,
    class_names,
    *,
    image_width,
    image_height,
):
  """Converts bboxes into Gemini-native JSON detection format.

  Output format (per the Gemini spatial understanding team):
  [{"box_2d": [ymin, xmin, ymax, xmax], "label": "class_name"}, ...]

  Coordinates are integers normalized to [0, 1000].

  Args:
    bboxes: COCO-format bounding boxes [x, y, w, h].
    category_ids: Category index per bbox.
    class_names: Human-readable class name list.
    image_width: Source image width in pixels.
    image_height: Source image height in pixels.

  Returns:
    A JSON string of detections.
  """
  detections = []
  for bbox, cat_id in zip(bboxes, category_ids):
    xyxy = box_utils.coco_to_xyxy(bbox)
    x_min, y_min, x_max, y_max = xyxy
    y1 = int(y_min / image_height * _JSON_GRID_SIZE)
    x1 = int(x_min / image_width * _JSON_GRID_SIZE)
    y2 = int(y_max / image_height * _JSON_GRID_SIZE)
    x2 = int(x_max / image_width * _JSON_GRID_SIZE)

    y1 = max(0, min(_JSON_GRID_SIZE, y1))
    x1 = max(0, min(_JSON_GRID_SIZE, x1))
    y2 = max(0, min(_JSON_GRID_SIZE, y2))
    x2 = max(0, min(_JSON_GRID_SIZE, x2))

    detections.append({
        "box_2d": [y1, x1, y2, x2],
        "label": class_names[cat_id],
    })

  return json.dumps(detections)


def format_objects_to_detection_string(
    objects,
    class_names,
    *,
    image_width,
    image_height,
    detection_format = "loc",
):
  """Sorts objects spatially (top-to-bottom, left-to-right) and formats them.

  Supports two output formats:
    - 'loc': PaLiGemma-style <loc> tokens (e.g., <loc0102><loc0511>... label)
    - 'json': Gemini-native JSON (e.g., [{"box_2d": [...], "label": "..."}])

  Example (loc format):
    For an image of size 100x100 and 1024 grid cells:
    - Box [10, 10, 10, 10] (y=10, x=10) scales to y1=102, x1=102 ->
    <loc0102><loc0102>...
    - Box [50, 10, 10, 10] (y=10, x=50) scales to y1=102, x1=511 ->
    <loc0102><loc0511>...
    - Box [80, 80, 10, 10] (y=80, x=80) scales to y1=818, x1=818 ->
    <loc0818><loc0818>...

    They will be ordered as above in the output string.

  Args:
    objects: Dict with 'bbox' and 'category' lists from the dataset.
    class_names: Human-readable class name list.
    image_width: Source image width in pixels.
    image_height: Source image height in pixels.
    detection_format: Output format, either 'loc' or 'json'.

  Returns:
    A formatted detection string, or empty string if no objects.
  """

  def get_sort_key(item):
    bbox, _ = item
    x, y, _, _ = bbox
    return (y, x)

  indexed_objs = sorted(
      zip(objects["bbox"], objects["category"]), key=get_sort_key
  )

  if not indexed_objs:
    return ""

  sorted_bboxes, sorted_cats = zip(*indexed_objs)
  formatter = (
      convert_boxes_to_json_string
      if detection_format == "json"
      else convert_boxes_to_detection_string
  )
  return formatter(
      list(sorted_bboxes),
      list(sorted_cats),
      class_names,
      image_width=image_width,
      image_height=image_height,
  )
