import torch
import numpy as np
from typing import Tuple, List


def create_tensorboard_sample_collage(input: torch.Tensor,
                                      pred_labels: torch.Tensor,
                                      target_labels: torch.Tensor,
                                      class_colors: List[Tuple[int, int, int]],
                                      alpha: float = 0.5) -> torch.Tensor:
    """
    Creates tensorboard sample collage.
    Combines input image, predicted labels and target labels into single collage image.

    Parameters:
        input : tensor
            Input image with shape (C, H, W).
            Can be in grayscale or rgb color format.
        pred_labels : tensor
            Predicted class labels (integers) with shape (H, W).
        target_labels : tensor
            Target class labels (integers) with shape (H, W).
        class_colors : list
            List of colors (R, G, B) for each class.
        alpha : float
            Coefficient for blending images in range [0, 1].
    Returns:
        collage : tensor
            Result collage with shape (C, H, 3 * W).
    """
    assert input.shape[1:] == pred_labels.shape[:2] == target_labels.shape[:2]
    alpha = np.clip(alpha, 0.0, 1.0)

    with torch.no_grad():
        # to rgb
        if input.shape[0] == 1:
            input = input.repeat(3, -1)

        # draw labels
        mask = torch.zeros(size=(2, *input.shape), dtype=torch.float, device=input.device)
        for class_idx, color in enumerate(class_colors):
            color = torch.tensor(color, dtype=mask.dtype, device=mask.device).unsqueeze(1)
            mask[0, :, target_labels.type(torch.long) == class_idx] = color
            mask[1, :, pred_labels.type(torch.long) == class_idx] = color
        labels = (1 - alpha) * torch.stack([input, input], dim=0) + alpha * mask

    return torch.cat([input, labels[0], labels[1]], dim=-1)


def update_tensorboard_image_samples(limit: int,
                                     accumulator: List[Tuple[float, torch.Tensor]],
                                     input: torch.Tensor,
                                     metric: torch.Tensor,
                                     pred_labels: torch.Tensor,
                                     target_labels: torch.Tensor,
                                     min_metric_better: bool,
                                     blending_alpha: float = 0.5,
                                     nonzero_only: bool = True):
    """
    Updates array with best/worst samples images.

    Parameters:
        limit : integer
            Maximum number of samples storing in accumulator.
        accumulator : list
            Accumulator where to save samples. Samples are saved
            if form of pair (metric, image).
        input : tensor
            Input images with shape (N, C, H, W).
        metric : tensor
            Metrics with shape (N, K), where K is number of classes.
        pred_labels : tensor
            Predicted class labels (integers) with shape (N, H, W).
        target_labels : tensor
            Target class labels (integers) with shape (N, H, W).
        min_metric_better : bool
            Whether to choose sample with minimum metric or maximum as best sample.
        blending_alpha : float
            Coefficient for blending images (input and labels).
        nonzero_only : bool
            Wheter to skip samples with only zero class label or not.

    Returns:
        None
    """   
    # TODO: as parameter
    class_colors = [
        (0, 0, 0), # background
        (1, 0, 0.5), # fog
        (0, 0.5, 1), # cloud
    ]

    with torch.no_grad():
        input = input.detach()
        metric = metric.detach()
        pred_labels = pred_labels.detach()
        target_labels = target_labels.detach()

        idxs = torch.arange(input.shape[0], dtype=torch.long, device=input.device)
        if nonzero_only:
            cnz = torch.count_nonzero(target_labels, dim=(1, 2))
            idxs = idxs[cnz > 0]

        if torch.numel(idxs):
            # find best sample
            metric_per_sample = torch.index_select(metric, 0, idxs)
            if min_metric_better:
                choosen_idx = torch.argmin(metric_per_sample).item()
            else:
                choosen_idx = torch.argmax(metric_per_sample).item()
            choosen_loss = metric_per_sample[choosen_idx].item()
    
            # check if better than existing
            need_save = True
            id_to_remove = None
            if len(accumulator) >= limit:
                if min_metric_better:
                    id_to_remove = max(range(len(accumulator)), key=lambda x : accumulator[x][0])
                    if choosen_loss > accumulator[id_to_remove][0]:
                        need_save = False
                else:
                    id_to_remove = min(range(len(accumulator)), key=lambda x : accumulator[x][0])
                    if choosen_loss < accumulator[id_to_remove][0]:
                        need_save = False

            if need_save:
                # prepare tensorboard collage
                best_input = torch.index_select(input, 0, idxs)[choosen_idx]
                best_pred_labels = torch.index_select(pred_labels, 0, idxs)[choosen_idx]
                best_target_labels = torch.index_select(target_labels, 0, idxs)[choosen_idx]
                collage = create_tensorboard_sample_collage(best_input, best_pred_labels, best_target_labels, class_colors, blending_alpha)

                # add to best collages
                if id_to_remove != None:
                    del accumulator[id_to_remove]
                accumulator.append((choosen_loss, collage))
