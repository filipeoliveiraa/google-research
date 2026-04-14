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

#!/usr/bin/python

#
# Evaluate the performance of the Deeplab predictions on COCO-Stuff val set.
#

import os
import imageio
import numpy as np
import scipy.io
import torch
import torch.nn.functional as F


def compute_psnr(im1, im2, maxval=1.0):
  return 20 * torch.log10(maxval / torch.sqrt(F.mse_loss(im1, im2)))


def _computeMetrics(confusion):
  """Compute evaluation metrics given a confusion matrix.

  :param confusion: any confusion matrix :return: tuple (miou, fwiou, macc,
  pacc, ious, maccs)
  """

  # Init
  labelCount = confusion.shape[0]
  ious = np.zeros((labelCount))
  maccs = np.zeros((labelCount))
  ious[:] = np.NAN
  maccs[:] = np.NAN

  # Get true positives, positive predictions and positive ground-truth
  total = confusion.sum()
  if total <= 0:
    raise Exception("Error: Confusion matrix is empty!")
  tp = np.diagonal(confusion)
  posPred = confusion.sum(axis=0)
  posGt = confusion.sum(axis=1)

  # Check which classes have elements
  valid = posGt > 0
  iousValid = np.logical_and(valid, posGt + posPred - tp > 0)

  # Compute per-class results and frequencies
  ious[iousValid] = np.divide(
      tp[iousValid], posGt[iousValid] + posPred[iousValid] - tp[iousValid]
  )
  maccs[valid] = np.divide(tp[valid], posGt[valid])
  freqs = np.divide(posGt, total)

  # Compute evaluation metrics
  miou = np.mean(ious[iousValid])
  fwiou = np.sum(np.multiply(ious[iousValid], freqs[iousValid]))
  macc = np.mean(maccs[valid])
  pacc = tp.sum() / total

  return miou, fwiou, macc, pacc, ious, maccs


def _computeMetricsV2(confusion, unseen_indices=None):
  """Compute evaluation metrics given a confusion matrix.

  Optionally computes metrics for Seen vs Unseen splits if indices are provided.

  :param confusion: any confusion matrix (NxN)
  :param unseen_indices: (Optional) list or array of indices corresponding to
  unseen classes
  :return:
      If unseen_indices is None: tuple (miou, fwiou, macc, pacc, ious, maccs)
      If unseen_indices is set: dict with keys 'seen', 'unseen', 'overall'
  """

  # Init
  labelCount = confusion.shape[0]
  ious = np.zeros((labelCount))
  maccs = np.zeros((labelCount))
  ious[:] = np.NAN
  maccs[:] = np.NAN

  # Get true positives, positive predictions and positive ground-truth
  total = confusion.sum()
  if total <= 0:
    # Return zeros/NaNs instead of crashing if possible, or keep exception
    # Keeping exception as per original logic
    raise Exception("Error: Confusion matrix is empty!")

  tp = np.diagonal(confusion)
  posPred = confusion.sum(axis=0)
  posGt = confusion.sum(axis=1)

  # Check which classes have elements
  # valid: Classes present in Ground Truth
  valid = posGt > 0
  # iousValid: Classes present in GT AND (Union > 0)
  iousValid = np.logical_and(valid, posGt + posPred - tp > 0)

  # Compute per-class results and frequencies
  ious[iousValid] = np.divide(
      tp[iousValid], posGt[iousValid] + posPred[iousValid] - tp[iousValid]
  )
  maccs[valid] = np.divide(tp[valid], posGt[valid])
  freqs = np.divide(posGt, total)

  # --- Internal Helper to aggregate metrics for a specific subset of indices ---
  def aggregate_subset(indices):
    # Create masks for this specific subset
    # We need classes that are in the subset AND are valid for calculation
    subset_mask_iou = np.zeros(labelCount, dtype=bool)
    subset_mask_iou[indices] = True
    subset_mask_iou = np.logical_and(subset_mask_iou, iousValid)

    subset_mask_acc = np.zeros(labelCount, dtype=bool)
    subset_mask_acc[indices] = True
    subset_mask_acc = np.logical_and(subset_mask_acc, valid)

    # Calculate Means (Handle case where subset has no valid classes)
    if np.sum(subset_mask_iou) == 0:
      miou = 0.0
      fwiou = 0.0
    else:
      miou = np.mean(ious[subset_mask_iou])
      # fwIoU: sum(iou * freq) for the subset
      fwiou = np.sum(np.multiply(ious[subset_mask_iou], freqs[subset_mask_iou]))

    if np.sum(subset_mask_acc) == 0:
      macc = 0.0
    else:
      macc = np.mean(maccs[subset_mask_acc])

    # pAcc is usually global, but if needed per subset: sum(tp_subset) / sum(gt_subset)
    # However, standard logic usually keeps pAcc global.
    # Here we return the subset-specific metrics required by standard ZSL protocols.
    return miou, fwiou, macc

  # --- Compute Overall Metrics (Original Logic) ---
  miou, fwiou, macc = aggregate_subset(range(labelCount))
  pacc = tp.sum() / total

  # --- Return Logic ---
  if unseen_indices is None:
    return {"overall": (miou, fwiou, macc, pacc, ious, maccs)}

  # --- Handle Zero-Shot Splits ---
  all_indices = np.arange(labelCount)
  unseen_indices = np.array(unseen_indices, dtype=int)

  # Seen = All - Unseen
  seen_indices = np.setdiff1d(all_indices, unseen_indices)

  miou_s, fwiou_s, macc_s = aggregate_subset(seen_indices)
  miou_u, fwiou_u, macc_u = aggregate_subset(unseen_indices)

  # Calculate Harmonic Mean (Standard ZSL metric)
  if (miou_s + miou_u) > 0:
    h_iou = (2 * miou_s * miou_u) / (miou_s + miou_u)
  else:
    h_iou = 0.0

  return {
      "overall": (miou, fwiou, macc, pacc, ious, maccs),
      "seen": (miou_s, fwiou_s, macc_s),
      "unseen": (miou_u, fwiou_u, macc_u),
      "harmonic_iou": h_iou,
  }


# Settings
# test_set = 'val'
# label_count = 182

# Create path names
# test_set_year = test_set + '2017'
# gt_folder = os.path.join('cocostuff/data/annotations', test_set_year)
# pred_folder = os.path.join('cocostuff/features/deeplabv2_vgg16/model120kimages', test_set, 'fc8')

# Get image list
# images = os.listdir(gt_folder)
# images = [i[:-4] for i in images]


class BaseMetric:

  def __init__(self):
    self.metrics = {}
    self.reset()

  def add_to_metric(self, key, val):
    if not key in self.metrics:
      self.metrics[key] = []

    self.metrics[key].append(val)

  def __getitem__(self, name):
    return self.metrics[name]

  def dict(self):
    metrics = {}
    for key, val in self.metrics.items():
      assert not key in metrics
      val = sum(val) / len(val)
      metrics[key] = val.mean().item()

    return metrics

  def print_metrics(self):
    for key, val in self.dict().items():
      print(f"{key}: {val:4.3f}")

  def reset(self):
    self.metrics = {}

  def check_with_dataset(self, dataset):
    pass


class CocoEval(BaseMetric):

  def __init__(self, label_count):
    self.label_count = label_count
    self.confusion = np.zeros((label_count, label_count))
    self.valid_ids = range(0, label_count)
    self.reset()

  def check_with_dataset(self, dataset):
    assert dataset.num_classes() == self.label_count, (
        f"Dataset has {dataset.num_classes()} classes; defined"
        f" {self.label_count}"
    )

  def add_prediction(self, pred, gt):
    pred = pred.detach().cpu().numpy()
    gt = gt.detach().cpu().numpy()

    # Filter labels (includes 255)
    valid = np.reshape(np.in1d(gt, self.valid_ids), gt.shape)
    valid_gt = gt[valid].astype(int)
    valid_pred = pred[valid].astype(int)

    # Accumulate confusion
    n = self.confusion.shape[0] + 1  # Arbitrary number > labelCount
    map_for_count = valid_gt * n + valid_pred
    vals, cnts = np.unique(map_for_count, return_counts=True)
    for v, c in zip(vals, cnts):
      g = v // n
      d = v % n
      self.confusion[g, d] += c

  def print_metrics(self):
    [miou, fwiou, macc, pacc, ious, maccs] = _computeMetrics(self.confusion)

    # Generate dataframe for the general results
    print(f" mIoU: {miou:4.3f}")
    print(f"fwIoU: {fwiou:4.3f}")
    print(f" mAcc: {macc:4.3f}")
    print(f" pAcc: {pacc:4.3f}")

  def reset(self):
    super().reset()
    self.confusion[:, :] = 0.0

  def dict(self):
    [miou, fwiou, macc, pacc, ious, maccs] = _computeMetrics(self.confusion)
    metrics = {
        "mIoU": miou.item(),
        "fwIoU": fwiou.item(),
        "mAcc": macc.item(),
        "pAcc": pacc.item(),
    }

    for key, val in self.metrics.items():
      assert not key in metrics
      val = sum(val) / len(val)
      metrics[key] = val.mean().item()

    return metrics


class CocoEvalZeroShot(CocoEval):

  def print_metrics(self, unseen_indices=None):
    metrics = _computeMetricsV2(self.confusion, unseen_indices)

    overall = metrics["overall"]

    print("Overall: ")
    miou, fwiou, macc, pacc, ious, maccs = overall
    # Generate dataframe for the general results
    print(f"   mIoU: {miou:4.3f}")
    print(f"  fwIoU: {fwiou:4.3f}")
    print(f"   mAcc: {macc:4.3f}")
    print(f"   pAcc: {pacc:4.3f}")

    if unseen_indices is None:
      return

    seen = metrics["seen"]
    unseen = metrics["unseen"]
    harmonic = metrics["harmonic_iou"]

    print("Seen: ")
    miou_s, fwiou_s, macc_s = seen

    # Generate dataframe for the general results
    print(f"   mIoU: {miou_s:4.3f}")
    print(f"  fwIoU: {fwiou_s:4.3f}")
    print(f"   mAcc: {macc_s:4.3f}")

    print("Unseen: ")
    miou_s, fwiou_s, macc_s = unseen

    # Generate dataframe for the general results
    print(f"   mIoU: {miou_s:4.3f}")
    print(f"  fwIoU: {fwiou_s:4.3f}")
    print(f"   mAcc: {macc_s:4.3f}")

    # harmonic mIoU
    print(f"Harmonic IoU: {harmonic:4.3f}")

  def dict(self, unseen_indices=None):
    all_metrics = _computeMetricsV2(self.confusion, unseen_indices)

    miou, fwiou, macc, pacc, ious, maccs = all_metrics["overall"]
    metrics = {
        "mIoU": miou.item(),
        "fwIoU": fwiou.item(),
        "mAcc": macc.item(),
        "pAcc": pacc.item(),
    }

    if not unseen_indices is None:
      miou_s, fwiou_s, macc_s = all_metrics["seen"]
      miou_u, fwiou_u, macc_u = all_metrics["unseen"]

      metrics.update({
          "mIoU_seen": miou_s.item(),
          "fwIoU_seen": fwiou_s.item(),
          "mAcc_seen": macc_s.item(),
          "mIoU_unseen": miou_u.item(),
          "fwIoU_unseen": fwiou_u.item(),
          "mAcc_unseen": macc_u.item(),
          "mIoU_harmonic": all_metrics["harmonic_iou"].item(),
      })

    for key, val in self.metrics.items():
      assert not key in metrics
      val = sum(val) / len(val)
      metrics[key] = val.mean().item()

    return metrics


# Evaluate performance
# print("Performance on %s: %.2f%% (macc), %.2f%% (pacc), %.2f%% (miou), %.2f%% (fwiou)" % (test_set_year, macc*100, pacc*100, miou*100, fwiou*100))


def depth_rmse(depth_pr, depth_gt, image_average=False):
  assert (
      depth_pr.shape == depth_gt.shape
  ), f"{depth_pr.shape} != {depth_gt.shape}"

  if len(depth_pr.shape) == 4:
    depth_pr = depth_pr.squeeze(1)
    depth_gt = depth_gt.squeeze(1)

  # compute RMSE for each image and then average
  valid = (depth_gt > 0).detach().float()

  # clamp to 1 for empty depth images
  num_valid = valid.sum(dim=(1, 2))
  if (num_valid == 0).any():
    num_valid = num_valid.clamp(min=1)
    logger.warning("GT depth is empty. Clamping to avoid error.")

  # compute pixelwise squared error
  sq_error = (depth_gt - depth_pr).pow(2)
  sum_masked_sqe = (sq_error * valid).sum(dim=(1, 2))
  rmse_image = (sum_masked_sqe / num_valid).sqrt()

  return rmse_image.mean() if image_average else rmse_image


def match_scale_and_shift(prediction, target):
  # based on implementation from
  # https://gist.github.com/dvdhfnr/732c26b61a0e63a0abc8a5d769dbebd0

  assert len(target.shape) == len(prediction.shape)
  if len(target.shape) == 4:
    four_chan = True
    target = target.squeeze(dim=1)
    prediction = prediction.squeeze(dim=1)
  else:
    four_chan = False

  mask = (target > 0).float()

  # system matrix: A = [[a_00, a_01], [a_10, a_11]]
  a_00 = torch.sum(mask * prediction * prediction, (1, 2))
  a_01 = torch.sum(mask * prediction, (1, 2))
  a_11 = torch.sum(mask, (1, 2))

  # right hand side: b = [b_0, b_1]
  b_0 = torch.sum(mask * prediction * target, (1, 2))
  b_1 = torch.sum(mask * target, (1, 2))

  # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 *
  # a_10) . b
  det = a_00 * a_11 - a_01 * a_01
  valid = det.nonzero()

  # compute scale and shift
  scale = torch.ones_like(b_0)
  shift = torch.zeros_like(b_1)
  scale[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[
      valid
  ]
  shift[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[
      valid
  ]

  scale = scale.view(-1, 1, 1).detach()
  shift = shift.view(-1, 1, 1).detach()
  prediction = prediction * scale + shift

  return prediction[:, None, :, :] if four_chan else prediction


class DepthEval(BaseMetric):

  def __init__(self, scale_invariant):
    self.scale_invariant = scale_invariant
    self.reset()

  def add_prediction(self, depth_pr, depth_gt):
    # credit: https://github.com/mbanani/probe3d/blob/main/evals/utils/metrics.py

    assert (
        depth_pr.shape == depth_gt.shape
    ), f"{depth_pr.shape} != {depth_gt.shape}"

    if len(depth_pr.shape) == 4:
      depth_pr = depth_pr.squeeze(1)
      depth_gt = depth_gt.squeeze(1)

    # if nyu_crop:
    #     # apply NYU crop --- commonly used in many repos for some reason
    #     assert depth_pr.shape[-2] == 480
    #     assert depth_pr.shape[-1] == 640
    #     depth_pr = depth_pr[..., 45:471, 41:601]
    #     depth_gt = depth_gt[..., 45:471, 41:601]

    if self.scale_invariant:
      depth_pr = match_scale_and_shift(depth_pr, depth_gt)

    # zero out invalid pixels
    valid = (depth_gt > 0).detach().float()
    depth_pr = depth_pr * valid

    # get num valid
    num_valid = valid.sum(dim=(1, 2)).clamp(min=1)

    # get recall @ thresholds
    thresh = torch.maximum(
        depth_gt / depth_pr.clamp(min=1e-9), depth_pr / depth_gt.clamp(min=1e-9)
    )

    d1 = ((thresh < 1.25**1).float() * valid).sum(dim=(1, 2)) / num_valid
    d2 = ((thresh < 1.25**2).float() * valid).sum(dim=(1, 2)) / num_valid
    d3 = ((thresh < 1.25**3).float() * valid).sum(dim=(1, 2)) / num_valid

    # compute RMSE
    sse = (depth_gt - depth_pr).pow(2)
    mse = (sse * valid).sum(dim=(1, 2)) / num_valid
    rmse = mse.sqrt()

    self.add_to_metric("d1", d1.cpu())
    self.add_to_metric("d2", d2.cpu())
    self.add_to_metric("d3", d3.cpu())
    self.add_to_metric("rmse", rmse.cpu())

  def _compute_mean(self, key):
    val = sum(self.metrics[key]) / len(self.metrics[key])
    return val.mean().item()

  def print_metrics(self):
    # Generate dataframe for the general results
    print(f"  d1: {self._compute_mean('d1')}")
    print(f"  d2: {self._compute_mean('d2')}")
    print(f"  d3: {self._compute_mean('d3')}")
    print(f"RMSE: {self._compute_mean('rmse')}")

  def dict(self):
    output_metrics = {}
    for key, value in self.metrics.items():
      output_metrics[key] = self._compute_mean(key)

    return output_metrics


class NormalsEval(BaseMetric):

  def __init__(self):
    super().__init__()
    self.num_valid = 0

  def add_prediction(self, snorm_pr, snorm_gt, valid):
    """Metrics to evaluate surface norm based on iDISC (and probably Fouhey et al.

    2016).

    credit: https://github.com/mbanani/probe3d/blob/main/evals/utils/metrics.py
    """

    snorm_pr = snorm_pr[:, :3]
    assert (
        snorm_pr.shape == snorm_gt.shape
    ), f"{snorm_pr.shape} != {snorm_gt.shape}"

    # compute angular error
    cos_sim = torch.cosine_similarity(snorm_pr, snorm_gt, dim=1)
    cos_sim = cos_sim.clamp(min=-1, max=1.0)
    err_deg = torch.acos(cos_sim) * 180.0 / torch.pi

    # zero out invalid errors
    assert len(valid.shape) == 4
    valid = valid.squeeze(1).float()
    err_deg = err_deg * valid
    num_valid = valid.sum(dim=(1, 2)).clamp(min=1)

    # compute rmse
    rmse = (err_deg.pow(2).sum(dim=(1, 2)) / num_valid).sqrt()

    # compute recall at thresholds
    thresh = [11.25, 22.5, 30]
    d1 = ((err_deg < thresh[0]).float() * valid).sum(dim=(1, 2)) / num_valid
    d2 = ((err_deg < thresh[1]).float() * valid).sum(dim=(1, 2)) / num_valid
    d3 = ((err_deg < thresh[2]).float() * valid).sum(dim=(1, 2)) / num_valid

    metrics = {
        "d1": d1.cpu(),
        "d2": d2.cpu(),
        "d3": d3.cpu(),
        "rmse": rmse.cpu(),
    }

    self.add_to_metric("d1", d1.cpu())
    self.add_to_metric("d2", d2.cpu())
    self.add_to_metric("d3", d3.cpu())
    self.add_to_metric("rmse", rmse.cpu())

  def add_to_metric(self, key, vals):
    if not key in self.metrics:
      self.metrics[key] = []

    self.metrics[key] += vals.tolist()

  def _compute_mean(self, key):
    val = sum(self.metrics[key]) / len(self.metrics[key])
    return val

  def print_metrics(self):
    # Generate dataframe for the general results
    print(f"  d1: {self._compute_mean('d1')}")
    print(f"  d2: {self._compute_mean('d2')}")
    print(f"  d3: {self._compute_mean('d3')}")
    print(f"RMSE: {self._compute_mean('rmse')}")

  def dict(self):
    output_metrics = {}
    for key, value in self.metrics.items():
      output_metrics[key] = self._compute_mean(key)

    return output_metrics
