import os
import cv2 as cv
import numpy as np
from glob import glob
from typing import Tuple
from torch.utils.data import Dataset
from core.data.transforms.transforms import (
    Clip,
    Resize,
    ToTensor,
    Normalize,
    RandomCrop,
    FromTensor,
    Denormalize,
    RandomGamma,
    RandomMirror,
    RandomRotate,
    ConvertColor,
    ConvertToInts,
    ConvertFromInts,
    TransformCompose
)


class CloudDataset(Dataset):
    def __init__(self, cfg, root_dir: str, is_train: bool):
        self.root_dir = root_dir
        self.imgs = sorted(glob(os.path.join(root_dir, "*", "img", "*")))
        self.rois = sorted(glob(os.path.join(root_dir, "*", "roi", "*")))
        self.labels = sorted(glob(os.path.join(root_dir, "*", "label", "*")))
        assert len(self.imgs) == len(self.labels) == len(self.rois)
        self.transforms = self.build_transforms(img_size=cfg.INPUT.IMAGE_SIZE, is_train=is_train)

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

    def build_transforms(self, img_size: Tuple[int, int], is_train: bool = True):
        if is_train:
            transform = [
                ConvertColor('BGR', 'GRAY'),
                RandomCrop(0.5, 1.0, 0.5, True),
                RandomMirror(0.5, 0.5),
                RandomRotate(-45, 45, 0.5),
                Resize(img_size),
                ConvertFromInts(),
                # RandomGamma(0.5, 2.0, 0.25),
                Clip()
            ]
        else:
            transform = [
                ConvertColor('BGR', 'GRAY'),
                Resize(img_size),
                ConvertFromInts(),
                Clip()
            ]

        transform = transform + [Normalize(norm_roi=True), ToTensor()]

        return TransformCompose(transform)

    def visualize(self, tick_ms: int = 25):
        back_transforms = [
            FromTensor(),
            Denormalize(denorm_roi=True),
            ConvertToInts()
        ]
        back_transforms = TransformCompose(back_transforms)

        for idx in range(0, self.__len__()):
            input, label, roi = self.__getitem__(idx)
            input, label, roi = back_transforms(input, label, roi)

            label = 125 * np.concatenate(input.shape[-1] * [label], -1) # TODO: 125
            roi = np.concatenate(input.shape[-1] * [roi], -1)
            collage = np.concatenate([input, label, roi], axis=1)

            cv.imshow("input+label+roi", collage)
            if cv.waitKey(tick_ms) & 0xFF == ord('q'):
                break
