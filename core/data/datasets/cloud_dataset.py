import os
import cv2 as cv
import numpy as np
from glob import glob
from torch.utils.data import Dataset
from core.data.transforms.transforms import (
    Clip,
    Resize,
    ToTensor,
    Normalize,
    RandomCrop,
    FromTensor,
    Denormalize,
    RandomMirror,
    RandomRotate,
    ConvertColor,
    ConvertToInts,
    ConvertFromInts,
    TransformCompose
)


class CloudDataset(Dataset):
    def __init__(self, cfg, root_dir: str, is_train: bool):
        self.cfg = cfg
        self.root_dir = root_dir
        self.num_classes = len(cfg.DATASET.CLASS_LABELS)
        self.imgs = sorted(glob(os.path.join(root_dir, "*", "img", "*")))
        self.rois = sorted(glob(os.path.join(root_dir, "*", "roi", "*")))
        self.labels = sorted(glob(os.path.join(root_dir, "*", "label", "*")))
        assert len(self.imgs) == len(self.labels) == len(self.rois)
        self.transforms = self.build_transforms(is_train=is_train)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        image = cv.imread(self.imgs[idx], cv.IMREAD_COLOR)
        label = cv.imread(self.labels[idx], cv.IMREAD_GRAYSCALE)
        roi = cv.imread(self.rois[idx], cv.IMREAD_GRAYSCALE)
        assert(image.shape[0:1] == label.shape[0:1] == roi.shape[0:1])

        if self.transforms:
            image, label, roi = self.transforms(image, label, roi)

        return image, label, roi

    def build_transforms(self, is_train: bool = True):
        if self.cfg.INPUT.DEPTH == 1:
            transform = [ConvertColor('BGR', 'GRAY')]
        else:
            transform = [ConvertColor('BGR', 'RGB')]

        if is_train:
            transform += [
                RandomCrop(0.5, 1.0, 0.5, True),
                RandomMirror(0.5, 0.5),
                RandomRotate(-45, 45, 0.5),
                Resize(self.cfg.INPUT.IMAGE_SIZE),
                ConvertFromInts(),
                Clip()
            ]
        else:
            transform += [
                Resize(self.cfg.INPUT.IMAGE_SIZE),
                ConvertFromInts(),
                Clip()
            ]

        transform = transform + [Normalize(self.cfg.INPUT.NORM_MEAN, self.cfg.INPUT.NORM_SCALE),
                                 ToTensor()]

        return TransformCompose(transform)

    def visualize(self, tick_ms: int = 25):
        back_transforms = [
            FromTensor(),
            Denormalize(self.cfg.INPUT.NORM_MEAN, self.cfg.INPUT.NORM_SCALE),
            ConvertToInts()
        ]
        back_transforms = TransformCompose(back_transforms)

        color_step = int(255 / (self.num_classes - 1))
        for idx in range(0, self.__len__()):
            input, label, roi = self.__getitem__(idx)
            input, label, roi = back_transforms(input, label, roi)

            label = color_step * np.concatenate(input.shape[-1] * [label], -1)
            roi = np.concatenate(input.shape[-1] * [roi], -1)
            collage = np.concatenate([input, label, roi], axis=1)

            cv.imshow("input+label+roi", collage)
            if cv.waitKey(tick_ms) & 0xFF == ord('q'):
                break
