import torch
from torch import nn
from typing import List, Optional


class Dice():
    def __init__(self, num_classes: int, eps=1e-6):
        super(Dice, self).__init__()
        self.eps = eps
        self.num_classes = num_classes

    def __call__(self,
                 preds: torch.Tensor,
                 targets: torch.Tensor,
                 roi_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculates dice metric for image semantic segmentation task.

        Parameters:
            preds : tensor
                Predicted labels with shape (N, H, W).
            targets : tensor
                Target labels with shape (N, H, W).
            roi_mask : tensor
                Region of interesting binary mask (0 or 1) with shape (N, H, W).
                Used to mask image pixels that are not taken into account when calculating the result metric.

        Returns:
            dice : tensor
                Result dice metric with shape (N, C).
        """
        result = []
        for class_idx in range(self.num_classes):
            preds_ = torch.where(preds == class_idx, 1, 0)
            targets_ = torch.where(targets == class_idx, 1, 0)

            if roi_mask is not None:
                preds_ = roi_mask * preds_
                targets_ = roi_mask * targets_

            intersection = torch.sum(preds_ * targets_, dim=(1, 2))
            dice = (2 * intersection) / (torch.sum(preds_, dim=(1, 2)) + torch.sum(targets_, dim=(1, 2)) + self.eps)
            result.append(dice)

        return torch.stack(result, dim=-1)


class JaccardIndex():
    def __init__(self, num_classes: int, eps=1e-6):
        super(JaccardIndex, self).__init__()
        self.eps = eps
        self.num_classes = num_classes

    def __call__(self,
                 preds: torch.Tensor,
                 targets: torch.Tensor,
                 roi_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculates Jaccard metric for image semantic segmentation task.

        Parameters:
            preds : tensor
                Predicted labels with shape (N, H, W).
            targets : tensor
                Target labels with shape (N, H, W).
            roi_mask : tensor
                Region of interesting binary mask (0 or 1) with shape (N, H, W).
                Used to mask image pixels that are not taken into account when calculating the result metric.

        Returns:
            jaccard_index : tensor
                Result Jaccard metric with shape (N, C).
        """
        result = []
        for class_idx in range(self.num_classes):
            preds_ = torch.where(preds == class_idx, 1, 0)
            targets_ = torch.where(targets == class_idx, 1, 0)

            if roi_mask is not None:
                preds_ = roi_mask * preds_
                targets_ = roi_mask * targets_

            intersection = torch.sum(preds_ * targets_, dim=(1, 2))
            union = torch.sum(preds_, dim=(1, 2)) + torch.sum(targets_, dim=(1, 2)) - intersection
            jaccard = intersection / (union + self.eps)
            result.append(jaccard)

        return torch.stack(result, dim=-1)
