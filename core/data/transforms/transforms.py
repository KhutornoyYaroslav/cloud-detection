import torch
import numpy as np


class TransformCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, input, target, roi = None):
        for t in self.transforms:
            input, target, roi = t(input, target, roi)
        roi = 0 if roi is None else roi

        return input, target, roi


class ConvertFromInts:
    def __call__(self, input, target, roi = None):
        input = input.astype(np.float32)
        target = target.astype(np.float32)
        if roi is not None:
            roi = roi.astype(np.float32)

        return input, target, roi


class ConvertToInts:
    def __call__(self, input, target, roi = None):
        input = input.astype(np.uint8)
        target = target.astype(np.uint8)
        if roi is not None:
            roi = roi.astype(np.uint8)

        return input, target, roi


class Clip(object):
    def __init__(self, min: float = 0.0, max: float = 255.0):
        self.min = min
        self.max = max
        assert self.max >= self.min, "max must be >= min"

    def __call__(self, input, target, roi = None):
        input = np.clip(input, self.min, self.max)

        return input, target, roi


class Normalize(object):
    def __init__(self, norm_roi: bool = True):
        self.norm_roi = norm_roi

    def __call__(self, input, target, roi = None):
        input = input.astype(np.float32) / 255.0
        if roi is not None and self.norm_roi:
            roi = roi.astype(np.float32) / 255.0

        return input, target, roi


class Denormalize(object):
    def __init__(self, denorm_roi: bool = True):
        self.denorm_roi = denorm_roi

    def __call__(self, input, target, roi = None):
        input = input.astype(np.float32) * 255.0
        if roi is not None and self.denorm_roi:
            roi = roi.astype(np.float32) * 255.0

        return input, target, roi


class ToTensor:
    def __call__(self, input, target, roi = None):
        input = torch.from_numpy(input.astype(np.float32)).permute(2, 0, 1)
        target = torch.from_numpy(target.astype(np.float32)).permute(2, 0, 1)
        if roi is not None:
            roi = torch.from_numpy(roi.astype(np.float32)).permute(2, 0, 1)

        return input, target, roi


class FromTensor:
    def __init__(self, dtype = np.float32):
        self.dtype = dtype

    def __call__(self, input: torch.Tensor, target, roi = None):
        input = input.permute(1, 2, 0).numpy().astype(self.dtype)
        target = target.permute(1, 2, 0).numpy().astype(self.dtype)
        if roi is not None:
            roi = roi.permute(1, 2, 0).numpy().astype(self.dtype)

        return input, target, roi
