import torch
import numpy as np
from typing import Tuple, List


def draw_labels(input: torch.Tensor,
                labels: torch.Tensor,
                class_colors: List[Tuple[int, int, int]],
                alpha: float = 0.5) -> torch.Tensor:
    """
    Draw segmentation masks.

    Parameters:
        input : tensor
            Input image with shape (C, H, W).
            Can be in grayscale or rgb color format.
        labels : tensor
            Class labels (integers) with shape (H, W).
        class_colors : list
            List of colors (R, G, B) for each class.
        alpha : float
            Coefficient for blending images in range [0, 1].
    Returns:
        image : tensor
            Result image with shape (3, H, W).
    """
    assert input.shape[1:] == labels.shape
    alpha = np.clip(alpha, 0.0, 1.0)

    with torch.no_grad():
        # to rgb
        if input.shape[0] == 1:
            input = input.expand((3, -1, -1))

        # draw labels
        mask = torch.zeros_like(input)
        for class_idx, color in enumerate(class_colors):
            color = torch.tensor(color, dtype=mask.dtype, device=mask.device).unsqueeze(1)
            mask[:, labels.type(torch.long) == class_idx] = color
        result = (1 - alpha) * input + alpha * mask

    return result


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

    with torch.no_grad():
        # to rgb
        if input.shape[0] == 1:
            input = input.expand((3, -1, -1))

        # draw labels
        pred_image = draw_labels(input, pred_labels, class_colors, alpha)
        target_image = draw_labels(input, target_labels, class_colors, alpha)

    return torch.cat([input, target_image, pred_image], dim=-1)


def update_tensorboard_image_samples(limit: int,
                                     accumulator: List[Tuple[float, torch.Tensor]],
                                     input: torch.Tensor,
                                     metric: torch.Tensor,
                                     pred_labels: torch.Tensor,
                                     target_labels: torch.Tensor,
                                     min_metric_better: bool,
                                     blending_alpha: float = 0.5,
                                     nonzero_factor: float = 0.0):
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
        nonzero_factor : bool
            Percent threshold to select samples that contain > 'nonzero_factor' nonzero class pixels.
            Useful to skip samples with dominating background class.

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

        # select only nonzero samples
        idxs = torch.arange(input.shape[0], dtype=torch.long, device=input.device)
        nonzero_factor = np.clip(nonzero_factor, 0.0, 1.0)
        cnz = torch.count_nonzero(target_labels, dim=(1, 2))
        ctot = target_labels.shape[1] * target_labels.shape[2]
        idxs = idxs[cnz / ctot > nonzero_factor]

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
