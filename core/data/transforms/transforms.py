import torch
import cv2 as cv
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
    def __init__(self, mean = [0], scale = [255]):
        self.mean = mean
        self.scale = scale

    def __call__(self, input, target, roi = None):
        assert input.shape[-1] == len(self.mean) == len(self.scale)
        input = (input.astype(np.float32) - self.mean) / self.scale

        return input, target, roi

class Denormalize(object):
    def __init__(self, mean = [0], scale = [255]):
        self.mean = mean
        self.scale = scale

    def __call__(self, input, target, roi = None):
        assert input.shape[-1] == len(self.mean) == len(self.scale)
        input = self.scale * input + self.mean

        return input, target, roi


class ToTensor:
    def __call__(self, input, target, roi = None):
        # check channels: (H, W) to (H, W, 1)
        if input.ndim == 2:
            input = np.expand_dims(input, axis=-1)
        if target.ndim == 2:
            target = np.expand_dims(target, axis=-1)
        if roi is not None and roi.ndim == 2:
            roi = np.expand_dims(roi, axis=-1)

        # to tensor
        input = torch.from_numpy(input.astype(np.float32)).permute(2, 0, 1)
        target = torch.from_numpy(target.astype(np.int64)).permute(2, 0, 1)
        if roi is not None:
            roi = torch.from_numpy(roi.astype(np.float32)).permute(2, 0, 1)
            roi = roi / 255.0

        return input, target, roi


class FromTensor:
    def __init__(self, dtype = np.float32):
        self.dtype = dtype

    def __call__(self, input, target, roi = None):
        input = input.permute(1, 2, 0).numpy().astype(self.dtype)
        target = target.permute(1, 2, 0).numpy().astype(self.dtype)
        if roi is not None:
            roi = 255.0 * roi
            roi = roi.permute(1, 2, 0).numpy().astype(self.dtype)

        return input, target, roi


class RandomRotate(object):
    def __init__(self, angle_min: float = -45.0, angle_max: float = 45.0, probability: float = 0.5):
        assert angle_max >= angle_min, "angle max must be >= angle min"
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.probability = np.clip(probability, 0.0, 1.0)

    def __call__(self, input, target, roi = None):
        if np.random.choice([0, 1], size=1, p=[1 - self.probability, self.probability]):
            angle = np.random.uniform(self.angle_min, self.angle_max)
            # input
            center = input.shape[1] / 2, input.shape[0] / 2
            rot_mat = cv.getRotationMatrix2D(center, angle, 1.0)
            input = cv.warpAffine(input, rot_mat, input.shape[1::-1], flags=cv.INTER_CUBIC, borderValue=0)
            # target
            center = target.shape[1] / 2, target.shape[0] / 2
            rot_mat = cv.getRotationMatrix2D(center, angle, 1.0)
            target = cv.warpAffine(target, rot_mat, target.shape[1::-1], flags=cv.INTER_NEAREST, borderValue=0)
            # roi
            if roi is not None:
                center = roi.shape[1] / 2, roi.shape[0] / 2
                rot_mat = cv.getRotationMatrix2D(center, angle, 1.0)
                roi = cv.warpAffine(roi, rot_mat, roi.shape[1::-1], flags=cv.INTER_NEAREST, borderValue=0)          

        return input, target, roi


class RandomCrop(object):
    def __init__(self, min_size: float, max_size: float, probability: float = 0.5, keep_aspect: bool = False):
        self.min_size = np.clip(min_size, 0.0, 1.0)
        self.max_size = np.clip(max_size, 0.0, 1.0)
        self.probability = np.clip(probability, 0.0, 1.0)
        self.keep_aspect = keep_aspect

    def __call__(self, input, target, roi = None):
        if np.random.choice([0, 1], size=1, p=[1 - self.probability, self.probability]):
            # random size
            w_norm, h_norm = np.random.uniform(self.min_size, self.max_size, 2)
            if self.keep_aspect:
                h_norm = w_norm
            x_norm = np.random.random() * (1 - w_norm)
            y_norm = np.random.random() * (1 - h_norm)

            # crop
            h, w = input.shape[0:2]
            input = input[int(y_norm * h):int(y_norm * h) + int(h_norm * h),
                          int(x_norm * w):int(x_norm * w) + int(w_norm * w)]

            h, w = target.shape[0:2]
            target = target[int(y_norm * h):int(y_norm * h) + int(h_norm * h),
                            int(x_norm * w):int(x_norm * w) + int(w_norm * w)]

            if roi is not None:
                h, w = roi.shape[0:2]
                roi = roi[int(y_norm * h):int(y_norm * h) + int(h_norm * h),
                          int(x_norm * w):int(x_norm * w) + int(w_norm * w)]

        return input, target, roi


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, input, target, roi = None):
        input = cv.resize(input, self.size, interpolation=cv.INTER_AREA)
        target = cv.resize(target, self.size, interpolation=cv.INTER_NEAREST)
        if roi is not None:
            roi = cv.resize(roi, self.size, interpolation=cv.INTER_NEAREST)

        return input, target, roi


class RandomMirror(object):
    def __init__(self, horizont_prob: float = 0.5, probability: float = 0.5):
            self.horizont_prob = np.clip(horizont_prob, 0.0, 1.0)
            self.probability = np.clip(probability, 0.0, 1.0)

    def __call__(self, input, target, roi = None):
        if np.random.choice([0, 1], size=1, p=[1-self.probability, self.probability]):
            if np.random.choice([0, 1], size=1, p=[1-self.horizont_prob, self.horizont_prob]):
                input = input[:, ::-1]
                target = target[:, ::-1]
                if roi is not None:
                    roi = roi[:, ::-1]
            else:
                input = input[::-1]
                target = target[::-1]
                if roi is not None:
                    roi = roi[::-1]

        return input, target, roi


class ConvertColor(object):
    def __init__(self, current: str, transform: str):
        self.transform = transform
        self.current = current

    def __call__(self, input, target, roi = None):
        if self.current == 'BGR' and self.transform == 'HSV':
            input = cv.cvtColor(input, cv.COLOR_BGR2HSV)
        elif self.current == 'BGR' and self.transform == 'GRAY':
            input = cv.cvtColor(input, cv.COLOR_BGR2GRAY)
        elif self.current == 'RGB' and self.transform == 'HSV':
            input = cv.cvtColor(input, cv.COLOR_RGB2HSV)
        elif self.current == 'BGR' and self.transform == 'RGB':
            input = cv.cvtColor(input, cv.COLOR_BGR2RGB)
        elif self.current == 'HSV' and self.transform == 'BGR':
            input = cv.cvtColor(input, cv.COLOR_HSV2BGR)
        elif self.current == 'HSV' and self.transform == "RGB":
            input = cv.cvtColor(input, cv.COLOR_HSV2RGB)
        else:
            raise NotImplementedError

        return input, target, roi
    

class RandomGamma(object):
    def __init__(self, lower: float = 0.5, upper: float = 2.0, probability: float = 0.5):
        self.lower = np.clip(lower, 0.0, None)
        self.upper = np.clip(upper, 0.0, None)
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        self.probability = np.clip(probability, 0.0, 1.0)

    def __call__(self, input, target, roi = None):
        assert input.dtype == np.float32, "image dtype must be float"
        if np.random.choice([0, 1], size=1, p=[1-self.probability, self.probability]):
            gamma = np.random.uniform(self.lower, self.upper)
            # if np.mean(input) > 100:
            input = pow(input / 255., gamma) * 255. # TODO: check it

        return input, target, roi
