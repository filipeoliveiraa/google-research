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

import glob
import io
import os
from pathlib import Path
import pickle
import random
import time

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import read_image
from torchvision.io import read_video, read_video_timestamps
from torchvision.transforms import v2
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as F_v2


def rand_crop(scale_range, output_size):
  min_scale, max_scale = scale_range

  # random scale
  scale = (min_scale - max_scale) * torch.rand(1) + max_scale

  # affine matrix
  theta = torch.zeros(1, 2, 3)
  theta[0, 0, 0] = scale
  theta[0, 1, 1] = scale

  max_translation = 1.0 - scale
  tx = (2 * max_translation) * torch.rand(1) - max_translation
  ty = (2 * max_translation) * torch.rand(1) - max_translation
  theta[0, 0, 2] = tx
  theta[0, 1, 2] = ty

  # sample grid
  grid = F.affine_grid(
      theta, (1, 1, output_size, output_size), align_corners=False
  )

  return grid[0]


class BaseDataset(Dataset):

  def __init__(self):
    self.base_crop_tf = T.Compose([
        T.Resize(size=480),
        T.CenterCrop(size=480),
        T.RandomHorizontalFlip(p=0.5),
    ])

    self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    self.tf_post = T.Compose([
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    self.tf_noise = T.Compose([
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
        T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    ])

  def denorm(self, image):
    dev = image.device
    std = self.std.to(dev)
    mean = self.mean.to(dev)
    if image.ndim == 4:
      std = std[None, ...]
      mean = mean[None, ...]

    return image * std + mean

  def _load_frame(self, frame_path):
    return read_image(frame_path)

  def _load_depth(self, depth_path):
    return torch.load(depth_path, map_location="cpu", weights_only=True)

  def bytes2tensor(self, bytes):
    # image = Image.open(bytes)
    image = cv2.imdecode(bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = self.tf(image=image)
    return image

  def bytes2mask(self, bytes, lut=None):
    mask_index = cv2.imdecode(bytes, cv2.CV_64F)
    # mask_index = torch.from_numpy(np.array(mask_image, np.int64))

    if not lut is None:
      # remapping
      mask_index = cv2.LUT(mask_index, lut)

    # mask_index = cv2.cvtColor(mask_index, cv2.CV_64F)
    # print(mask_index.shape)
    #
    # if not remap is None:
    #    mask_index = remap.remap(mask_index)
    #
    # H, W = mask_index.shape
    # mask_tensor = torch.zeros(self.max_num_obj, H, W)
    # mask = mask_tensor.scatter(0, mask_index[None], torch.ones(1, H, W))

    # apply the transform
    # mask = self.tf(mask_index)
    # mask_index = mask.argmax(0)

    return mask_index

  def get_loader(self, *args, **kwargs):
    return DataLoader(self, *args, **kwargs)

  def crop_w_tf(self, image, crop_range, aspect_range, with_flip=False):
    # Define the parameters for the smaller random crop.
    # `scale` determines the fraction of area from the input to crop.
    # `ratio` is the aspect ratio of the crop before resizing.
    crop_params = T.RandomResizedCrop.get_params(
        img=image,
        scale=crop_range,  # Crop between 30% and 100% of the base crop area
        ratio=aspect_range,
    )
    top, left, h, w = crop_params

    # Create the 224x224 crop using the obtained parameters.
    crop = F_v2.resized_crop(
        image, top, left, h, w, size=[224, 224], antialias=True
    )

    # --- Calculate the 2x3 affine transformation matrix (theta) ---
    # This matrix defines the transformation from the normalized coordinates of the
    # output grid (224x224) to the normalized coordinates of the input image (480x480).
    H_in, W_in = 480, 480

    # The formula for the affine matrix `theta = [[s_x, a, t_x], [b, s_y, t_y]]` is derived from:
    # x_in = x_out * (w / W_in) + (2 * left + w) / W_in - 1
    # y_in = y_out * (h / H_in) + (2 * top + h) / H_in - 1
    theta = torch.tensor(
        [
            [w / W_in, 0, (2 * left + w) / W_in - 1],
            [0, h / H_in, (2 * top + h) / H_in - 1],
        ],
        dtype=torch.float,
    )

    if with_flip and torch.rand(1) > 0.5:
      crop = F_v2.horizontal_flip(crop)
      theta[:, 0] = -theta[:, 0]

    # Generate the grid from the affine matrix.
    # The grid has shape (1, 224, 224, 2), which we unsqueeze and then squeeze
    # to match the expected (N, H, W, 2) format for grid_sample.
    grid = F.affine_grid(
        theta.unsqueeze(0),
        size=(1, image.shape[0], 224, 224),
        align_corners=False,
    )[0]

    return crop, grid


class LazyLoad:

  def __init__(self, filenames, preload, func, *func_args):
    self.filenames = filenames
    self.func = func
    self.func_args = func_args
    self.cache = {}
    if preload:
      print("Preloading...", end="")
      start = time.time()
      for ii in range(len(filenames)):
        self.cache_index(ii)
      print(f"... done ({(time.time() - start):4.3f}s)")

  def __len__(self):
    return len(self.filenames)

  def __getitem__(self, index):
    bytes = self.cache_index(index)
    return self.func(bytes, *self.func_args)

  def cache_index(self, index):
    if not index in self.cache:
      self.cache[index] = np.fromstring(
          open(self.filenames[index], "rb").read(), np.uint8
      )
    return self.cache[index]

  def load(self):
    return [self[ii] for ii, fn in enumerate(self.filenames)]


class VideoDataset(BaseDataset):

  def __init__(
      self,
      root_dir,
      split,
      crop_range,
      aspect_range,
      filetype="jpg",
      max_num_obj=8,
      preload=False,
  ):
    super().__init__()

    self.T = 3
    self.max_num_obj = max_num_obj
    self.crop_range = crop_range
    self.aspect_range = aspect_range
    self.mult_video = 1000 if "train" in split else 1

    self.video_keys = [x.strip() for x in open(split).readlines()]

    fetch_frames = lambda x, y: sorted(glob.glob(os.path.join(x, f"*.{y}")))

    self.videos = {}
    for seqname in self.video_keys:
      self.videos[seqname] = {"frames": None, "masks": None, "depth": None}

      image_dir = self._image_dir(root_dir, seqname)
      self.videos[seqname]["frames"] = fetch_frames(image_dir, filetype)

      mask_dir = self._mask_dir(root_dir, seqname)
      self.videos[seqname]["masks"] = fetch_frames(mask_dir, "png")

      depth_dir = self._depth_dir(root_dir, seqname)
      self.videos[seqname]["depth"] = fetch_frames(depth_dir, "pt")

    self.tf_A = A.Compose([
        A.SmallestMaxSize(224),
        A.CenterCrop(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        A.ToTensorV2(),
    ])

    self.tf_A_val = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        A.ToTensorV2(),
    ])

  def _load_sample(self, image_path, mask_path, tf=None):

    if tf is None:
      tf = self.tf_A

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = Image.open(mask_path)
    mask = np.array(mask)

    augmented = tf(image=image, mask=mask)
    return augmented["image"], augmented["mask"]

  def _image_dir(self, root_dir, seqname):
    return f"{root_dir}/JPEGImages/480p/{seqname}"

  def _mask_dir(self, root_dir, seqname):
    return f"{root_dir}/Annotations/480p/{seqname}"

  def _depth_dir(self, root_dir, seqname):
    return f"{root_dir}/Depth/480p/{seqname}"

  def get_sequence(self, seqname, val=False):
    tf = self.tf_A_val if val else self.tf_A
    frames = []
    masks = []
    sequence = self.videos[seqname]
    for image_path, mask_path in zip(sequence["frames"], sequence["masks"]):
      frame, mask = self._load_sample(image_path, mask_path, tf)
      frames.append(frame)
      masks.append(mask)

    frames = torch.stack(frames)
    masks = torch.stack(masks)

    return frames, masks

  def __len__(self):
    return self.mult_video * len(self.videos)

  def __getitem__(self, idx):

    video_key = self.video_keys[idx % len(self.video_keys)]
    frames = self.videos[video_key]["frames"]

    N = len(frames)

    idx1 = random.randint(0, N - 1)
    delta = random.choice(list(range(-self.T, 0)) + list(range(1, self.T + 1)))
    idx2 = max(0, min(N - 1, idx1 + delta))

    frame1 = self._load_frame(frames[idx1])
    frame2 = self._load_frame(frames[idx2])
    video_stack = torch.stack([frame1, frame2])

    ### Generating crops ###
    state = torch.get_rng_state()
    base_crops = self.base_crop_tf(video_stack)

    view1, tf1 = self.crop_w_tf(
        base_crops[0], self.crop_range, self.aspect_range
    )
    view2, tf2 = self.crop_w_tf(
        base_crops[1], self.crop_range, self.aspect_range
    )

    ### Post-processing ###

    # 3. Downscale the base crop to 224x224.
    base_crops = F_v2.resize(base_crops, size=[224, 224], antialias=True)

    base_crops = self.tf_post(base_crops)
    view1 = self.tf_post(view1)
    view2 = self.tf_post(view2)

    return {
        "frames": base_crops,
        "view1": view1,
        "view2": view2,
        "tf1": tf1,
        "tf2": tf2,
    }


class YTVOS(BaseDataset):

  def __init__(
      self,
      root_dir,
      split,
      crop_range,
      aspect_range,
      filetype="jpg",
      T=2,
      mask_ratio=0.5,
  ):
    super().__init__()

    self.T = T
    self.mask_ratio = [mask_ratio, mask_ratio]
    self.mult_video = 100 if "train" in split else 1
    self.crop_range = crop_range
    self.aspect_range = aspect_range
    self.video_keys = [x.strip() for x in open(split).readlines()]

    fetch_frames = lambda x, y: sorted(glob.glob(os.path.join(x, f"*.{y}")))

    self.videos = {}
    self.videokeys = []
    for seqname in self.video_keys:

      frame_path = os.path.join(root_dir, seqname)
      self.videos[seqname] = {}

      self.videos[seqname]["frames"] = fetch_frames(frame_path, filetype)

      # depth_dir = os.path.join(root_dir, "Depth", seqname.split("/")[-1])
      # self.videos[seqname]["depths"] = fetch_frames(depth_dir, "pt")

      # nframes = len(self.videos[seqname]["frames"])
      # ndepths = len(self.videos[seqname]["depths"])
      # assert nframes == ndepths, \
      #            f"Expected the same number of frames and depthmaps. Mismatch {nframes} != {ndepths}"

      self.videokeys.append(seqname)

  def __len__(self):
    return self.mult_video * len(self.videos)

  def get_sequence(self, idx):

    video_key = self.videokeys[idx % len(self.videokeys)]
    frames = self.videos[video_key]["frames"]

    frames = [self._load_frame(fn) for fn in frames]
    frames = torch.stack(frames)

    base_crops = self.base_crop_tf(frames)
    base_crops = F_v2.resize(base_crops, size=[224, 224], antialias=True)

    return self.tf_post(base_crops)

  def __getitem__(self, idx):

    video_key = self.videokeys[idx % len(self.videokeys)]
    frames = self.videos[video_key]["frames"]
    N = len(frames)

    idx1 = random.randint(0, N - 1)
    # delta = random.choice(list(range(-self.T, 0)) + list(range(1, self.T + 1)))
    delta = random.choice(list(range(1, self.T + 1)))
    idx2 = max(0, min(N - 1, idx1 + delta))

    frame1 = self._load_frame(frames[idx1])
    frame2 = self._load_frame(frames[idx2])
    video_stack = torch.stack([frame1, frame2])

    ### Generating crops ###
    state = torch.get_rng_state()
    base_crops = self.base_crop_tf(video_stack)

    view1, tf1 = self.crop_w_tf(
        base_crops[0], self.crop_range, [1.0, 1.0], with_flip=True
    )
    view2, tf2 = self.crop_w_tf(base_crops[1], self.mask_ratio, [1.0, 1.0])

    ### Post-processing ###

    # 3. Downscale the base crop to 224x224.
    base_crops = F_v2.resize(base_crops, size=[224, 224], antialias=True)

    base_crops = self.tf_post(base_crops)
    view1 = self.tf_post(view1)
    view2 = self.tf_post(view2)

    return {
        "frames": base_crops,
        "view1": view1,
        "view2": view2,
        "tf1": tf1,
        "tf2": tf2,
    }


class SegDataset(BaseDataset):

  fine_to_coarse = {
      0: 9,
      1: 11,
      2: 11,
      3: 11,
      4: 11,
      5: 11,
      6: 11,
      7: 11,
      8: 11,
      9: 8,
      10: 8,
      11: 8,
      12: 8,
      13: 8,
      14: 8,
      15: 7,
      16: 7,
      17: 7,
      18: 7,
      19: 7,
      20: 7,
      21: 7,
      22: 7,
      23: 7,
      24: 7,
      25: 6,
      26: 6,
      27: 6,
      28: 6,
      29: 6,
      30: 6,
      31: 6,
      32: 6,
      33: 10,
      34: 10,
      35: 10,
      36: 10,
      37: 10,
      38: 10,
      39: 10,
      40: 10,
      41: 10,
      42: 10,
      43: 5,
      44: 5,
      45: 5,
      46: 5,
      47: 5,
      48: 5,
      49: 5,
      50: 5,
      51: 2,
      52: 2,
      53: 2,
      54: 2,
      55: 2,
      56: 2,
      57: 2,
      58: 2,
      59: 2,
      60: 2,
      61: 3,
      62: 3,
      63: 3,
      64: 3,
      65: 3,
      66: 3,
      67: 3,
      68: 3,
      69: 3,
      70: 3,
      71: 0,
      72: 0,
      73: 0,
      74: 0,
      75: 0,
      76: 0,
      77: 1,
      78: 1,
      79: 1,
      80: 1,
      81: 1,
      82: 1,
      83: 4,
      84: 4,
      85: 4,
      86: 4,
      87: 4,
      88: 4,
      89: 4,
      90: 4,
      91: 17,
      92: 17,
      93: 22,
      94: 20,
      95: 20,
      96: 22,
      97: 15,
      98: 25,
      99: 16,
      100: 13,
      101: 12,
      102: 12,
      103: 17,
      104: 17,
      105: 23,
      106: 15,
      107: 15,
      108: 17,
      109: 15,
      110: 21,
      111: 15,
      112: 25,
      113: 13,
      114: 13,
      115: 13,
      116: 13,
      117: 13,
      118: 22,
      119: 26,
      120: 14,
      121: 14,
      122: 15,
      123: 22,
      124: 21,
      125: 21,
      126: 24,
      127: 20,
      128: 22,
      129: 15,
      130: 17,
      131: 16,
      132: 15,
      133: 22,
      134: 24,
      135: 21,
      136: 17,
      137: 25,
      138: 16,
      139: 21,
      140: 17,
      141: 22,
      142: 16,
      143: 21,
      144: 21,
      145: 25,
      146: 21,
      147: 26,
      148: 21,
      149: 24,
      150: 20,
      151: 17,
      152: 14,
      153: 21,
      154: 26,
      155: 15,
      156: 23,
      157: 20,
      158: 21,
      159: 24,
      160: 15,
      161: 24,
      162: 22,
      163: 25,
      164: 15,
      165: 20,
      166: 17,
      167: 17,
      168: 22,
      169: 14,
      170: 18,
      171: 18,
      172: 18,
      173: 18,
      174: 18,
      175: 18,
      176: 18,
      177: 26,
      178: 26,
      179: 19,
      180: 19,
      181: 24,
      255: 27,
  }

  def __init__(self, root_dir, split, filelist, preload=False, val=False):
    """Args:

    root_dir: contains train2017 and val2017 directories
    split: train2017 or val2017
    """
    super().__init__()

    image_fns = []
    mask_fns = []
    with open(filelist, "r") as f:
      img_ids = [fn.rstrip() for fn in f.readlines()]
      for img_id in img_ids:
        image_fns.append(os.path.join(root_dir, split, img_id + ".jpg"))
        mask_fns.append(os.path.join(root_dir, split, img_id + ".png"))

    print(f"Found {len(image_fns)} images")
    for image_fn, mask_fn in zip(image_fns, mask_fns):
      assert (
          Path(image_fn).stem == Path(mask_fn).stem
      ), f"Mask/image filename mismatch: {image_fn} vs {mask_fn}"

    self.images = LazyLoad(image_fns, preload, self.bytes2tensor)

    LUT = [-1] * 256
    for key, value in self.fine_to_coarse.items():
      LUT[key] = value

    self.masks = LazyLoad(
        mask_fns, preload, self.bytes2mask, np.array(LUT, dtype=np.int64)
    )

    if val:
      self.tf = A.Compose([
          A.SmallestMaxSize(224),
          A.CenterCrop(224, 224),
          A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
          ToTensorV2(),
      ])
    else:
      self.tf = A.Compose([
          A.RandomResizedCrop(
              size=(224, 224),
              scale=(0.5, 1.0),  # Crop 50% to 100% of the original image
              ratio=(0.75, 1.33),  # Aspect ratio range
              p=1.0,
          ),
          # --- Geometric Augmentations ---
          A.HorizontalFlip(p=0.5),  # Flipping left-right is realistic
          # --- Final Formatting ---
          A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
          ToTensorV2(),  # Converts image and mask to PyTorch tensors
      ])

  def preload(self):
    for i, (image, mask) in enumerate(zip(self.images, self.masks)):
      image.cache_index(i)
      mask.cache_index(i)

  def num_classes(self):
    num = 0
    for key, value in self.fine_to_coarse.items():
      if value == self.ignore_index:
        continue
      num = max(num, value)
    return num + 1

  @property
  def ignore_index(self):
    return 27

  def _image_dir(self, root_dir, split):
    return [f"{root_dir}/{split}", "*.jpg"]

  def _mask_dir(self, root_dir, split):
    return [f"{root_dir}/{split}", "*.png"]

  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):
    image, mask = self.images[idx], self.masks[idx]

    augmented = self.tf(image=image, mask=mask)

    # image, task labels, depth
    return (
        augmented["image"],
        augmented["mask"],
        torch.zeros_like(augmented["image"][:1]),
    )


class ZShotSegDataset(BaseDataset):

  def __init__(self, root_dir, split, filelist, preload=False, val=False):
    """Args:

    root_dir: contains train2017 and val2017 directories
    split: train2017 or val2017
    """
    super().__init__()

    image_fns = []
    mask_fns = []
    filenames_data = np.load(filelist)
    img_ids = [
        os.path.splitext(os.path.basename(fn))[0] for fn in filenames_data
    ]

    for img_id in img_ids:
      image_fns.append(os.path.join(root_dir, split, img_id + ".jpg"))
      mask_fns.append(os.path.join(root_dir, split, img_id + ".png"))

    print(f"Found {len(image_fns)} images")
    for image_fn, mask_fn in zip(image_fns, mask_fns):
      assert (
          Path(image_fn).stem == Path(mask_fn).stem
      ), f"Mask/image filename mismatch: {image_fn} vs {mask_fn}"

    self.images = LazyLoad(image_fns, preload, self.bytes2tensor)

    # loading classes
    cls_list = list(range(256))
    LUT = np.full(256, 255, dtype=np.int64)
    LUT[cls_list] = cls_list

    if not val:  # training
      self.ignore_cls_list = np.load("splits/novel_cls.npy")
      print(f"Ignoring {len(self.ignore_cls_list)} unseen classes")
      LUT[self.ignore_cls_list] = 255
      self.seen_ids = [
          idx
          for idx in range(self.num_classes())
          if not idx in self.ignore_cls_list
      ]

    self.masks = LazyLoad(
        mask_fns, preload, self.bytes2mask, np.array(LUT, dtype=np.int64)
    )

    if val:
      self.tf = A.Compose([
          A.SmallestMaxSize(224),
          A.CenterCrop(224, 224),
          A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
          ToTensorV2(),
      ])
    else:
      self.tf = A.Compose([
          A.RandomResizedCrop(
              size=(224, 224),
              scale=(0.5, 1.0),  # Crop 50% to 100% of the original image
              ratio=(0.75, 1.33),  # Aspect ratio range
              p=1.0,
          ),
          # --- Geometric Augmentations ---
          A.HorizontalFlip(p=0.5),  # Flipping left-right is realistic
          # --- Final Formatting ---
          A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
          ToTensorV2(),  # Converts image and mask to PyTorch tensors
      ])

  def preload(self):
    for i, (image, mask) in enumerate(zip(self.images, self.masks)):
      image.cache_index(i)
      mask.cache_index(i)

  def num_classes(self):
    return 182  # hard-coded for coco-stuff

  @property
  def ignore_index(self):
    return 255

  def _image_dir(self, root_dir, split):
    return [f"{root_dir}/{split}", "*.jpg"]

  def _mask_dir(self, root_dir, split):
    return [f"{root_dir}/{split}", "*.png"]

  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):
    image, mask = self.images[idx], self.masks[idx]

    augmented = self.tf(image=image, mask=mask)

    # image, task labels, depth
    return (
        augmented["image"],
        augmented["mask"],
        torch.zeros_like(augmented["image"][:1]),
    )


class RandomHFlipWithNormals(nn.Module):
  """Applies a horizontal flip to an image and its corresponding normal map.

  The normal map's X-channel (index 0) is inverted to reflect the flip.

  Accepts: A tuple (image_tensor, normal_tensor)
  Returns: A tuple (flipped_image, flipped_normal)
  """

  def __init__(self, p=0.5):
    super().__init__()
    self.p = p

  def forward(self, sample):
    # We expect the input 'sample' to be a tuple (image, normal)
    image, depth, normal = sample

    # print(f"IM {image.shape} / D {depth.shape} / N {normal.shape}")

    if torch.rand(1).item() < self.p:
      # Apply the spatial flip to both tensors
      image = F_v2.hflip(image)
      depth = F_v2.hflip(depth)
      normal = F_v2.hflip(normal)

      # Invert the X coordinate (channel 0) of the normal vector
      # Normal vector X component points right. When flipped, it must point left.
      # normal[0, :, :] references the first channel (X) of the (C, H, W) tensor.
      normal[0, :, :] = -normal[0, :, :]

    return image, depth, normal


class NormalsDataset(BaseDataset):

  def __init__(self, root_dir, split, output_size=(224, 224)):
    super().__init__()

    self.depth_data = pickle.load(
        open(f"{root_dir}/nyu_{split}_depth.pkl", "rb")
    )
    self.image_data = pickle.load(
        open(f"{root_dir}/nyu_{split}_image.pkl", "rb")
    )
    self.norml_data = pickle.load(
        open(f"{root_dir}/nyu_{split}_normal.pkl", "rb")
    )

    self.output_size = output_size
    self.mode = split.lower()
    self.mult = 10 if split == "train" else 1

    # --- Define ImageNet normalization (applies ONLY to image) ---
    self.image_normalization = v2.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    # --- Define Augmentation Pipelines ---
    if self.mode == "train":
      # Training Pipeline (with randomization)

      # 1. Spatial transforms (applied to BOTH image and normal)
      self.spatial_transform = v2.Compose([
          v2.RandomResizedCrop(
              self.output_size, scale=(0.5, 1.0), antialias=True
          ),
          # Use our custom flip class
          RandomHFlipWithNormals(p=0.5),
      ])

      # 2. Image-only transforms (applied ONLY to image)
      self.image_only_transform = v2.Compose([
          v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
          self.image_normalization,  # Normalize image AFTER color jitter
      ])
    else:
      # Validation Pipeline (deterministic, no augmentation)

      # 1. Spatial transforms (applied to BOTH image and normal)
      # Just resize and center crop.
      self.spatial_transform = v2.Compose([
          v2.Resize(
              int(self.output_size[0] * 1.1), antialias=True
          ),  # Resize slightly larger
          v2.CenterCrop(self.output_size),
      ])

      # 2. Image-only transforms (just normalization)
      self.image_only_transform = self.image_normalization

  def get_loader(self, *args, **kwargs):
    return DataLoader(self, *args, **kwargs)

  def num_classes(self):
    return 3

  def __len__(self):
    return self.mult * len(self.image_data)

  def __getitem__(self, idx):

    idx = idx % len(self.image_data)

    image, depth, norml = (
        self.image_data[idx],
        self.depth_data[idx],
        self.norml_data[idx],
    )

    image = F_v2.to_dtype(F_v2.to_image(image), dtype=torch.float32, scale=True)
    depth = F_v2.to_dtype(
        F_v2.to_image(depth), dtype=torch.float32, scale=False
    )
    norml = F_v2.to_dtype(
        F_v2.to_image(norml), dtype=torch.float32, scale=False
    )

    image, depth, norml = self.spatial_transform((image, depth, norml))

    # Apply the IMAGE-ONLY transforms (color jitter, normalization) just to the image.
    image = self.image_only_transform(image)

    return image, depth, norml


class DepthDataset(BaseDataset):

  def __init__(self, root_dir, split, preload=False):
    """Args:

    root_dir: contains *.pkl files
    """

    super().__init__()

    self.mult = 10 if split == "train" else 1
    self.depth_data = pickle.load(
        open(f"{root_dir}/nyu_{split}_depth.pkl", "rb")
    )
    self.image_data = pickle.load(
        open(f"{root_dir}/nyu_{split}_image.pkl", "rb")
    )
    self.norml_data = pickle.load(
        open(f"{root_dir}/nyu_{split}_normal.pkl", "rb")
    )

    self.tf = A.Compose(
        [A.CenterCrop(224, 224), A.ToFloat(), ToTensorV2()],
        additional_targets={"depth": "image", "normal": "image"},
    )
    self.tf_norm = T.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

  def get_loader(self, *args, **kwargs):
    return DataLoader(self, *args, **kwargs)

  def num_classes(self):
    return 3

  def __len__(self):
    return self.mult * len(self.image_data)

  def __getitem__(self, idx):
    idx = idx % len(self.image_data)

    image, depth, norml = (
        self.image_data[idx],
        self.depth_data[idx],
        self.norml_data[idx],
    )

    augmented = self.tf(image=image, depth=depth, normal=norml)

    # depth_labels = augmented['depth'].clamp(0, 10 - 1e-3) / 10.
    # depth_labels = (256 * depth_labels).long()

    # image, task labels, depth
    return (
        self.tf_norm(augmented["image"]),
        augmented["depth"],
        augmented["normal"],
    )
