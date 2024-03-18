import torch
from tqdm import tqdm
from core.utils.colors import get_rgb_colors
from core.utils.tensorboard import update_tensorboard_image_samples
from core.engine.metrics import Dice, JaccardIndex, Precision, Recall


def do_validation(cfg, model, data_loader, device):
    # create metrics
    cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
    num_classes = len(cfg.DATASET.CLASS_LABELS)
    dice_metric = Dice(num_classes)
    jaccard_metric = JaccardIndex(num_classes)
    precision_metric = Precision(num_classes)
    recall_metric = Recall(num_classes)

    # create stats
    stats = {
        'loss_sum': 0,
        'dice_sum': 0,
        'jaccard_sum': 0,
        'precision_sum': 0,
        'recall_sum': 0,
        'best_samples': [],
        'worst_samples': [],
        'iterations': 0
    }

    # gather stats
    for data_entry in tqdm(data_loader):
        with torch.no_grad():
            # get data
            input, target_labels, roi = data_entry
            input = input.to(device)                                                # (n, c, h, w)
            target_labels = target_labels.squeeze(1).type(torch.long).to(device)    # (n, h, w)
            roi = roi.squeeze(1).to(device)                                         # (n, h, w)

            # forward model
            output = model(input) # (n, k, h, w)

            # calculate loss
            loss = cross_entropy.forward(output, target_labels)     # (n, h, w)
            loss = loss * roi                                       # (n, h, w)

            # calculate metrics
            pred_labels = torch.softmax(output, dim=1).argmax(dim=1)        # (n, h, w)
            dice = dice_metric(pred_labels, target_labels, roi)             # (n, k)
            jaccard = jaccard_metric(pred_labels, target_labels, roi)       # (n, k)
            precision = precision_metric(pred_labels, target_labels, roi)   # (n, k)
            recall = recall_metric(pred_labels, target_labels, roi)         # (n, k)

            # update stats
            stats['loss_sum'] += torch.mean(loss).item()
            stats['dice_sum'] += torch.mean(dice, 0).cpu().numpy()
            stats['jaccard_sum'] += torch.mean(jaccard, 0).cpu().numpy()
            stats['precision_sum'] += torch.mean(precision, 0).cpu().numpy()
            stats['recall_sum'] += torch.mean(recall, 0).cpu().numpy()
            stats['iterations'] += 1

            # update best samples
            update_tensorboard_image_samples(limit=cfg.TENSORBOARD.BEST_SAMPLES_NUM,
                                             accumulator=stats['best_samples'], 
                                             input=input,
                                             metric=torch.mean(loss, dim=(1, 2)), 
                                             pred_labels=pred_labels, 
                                             target_labels=target_labels,
                                             min_metric_better=True,
                                             class_colors=get_rgb_colors(num_classes, mean=cfg.INPUT.NORM_MEAN, scale=cfg.INPUT.NORM_SCALE),
                                             blending_alpha=cfg.TENSORBOARD.ALPHA_BLENDING,
                                             nonzero_factor=0.1)

            # update worst samples
            update_tensorboard_image_samples(limit=cfg.TENSORBOARD.WORST_SAMPLES_NUM,
                                             accumulator=stats['worst_samples'], 
                                             input=input,
                                             metric=torch.mean(loss, dim=(1, 2)), 
                                             pred_labels=pred_labels, 
                                             target_labels=target_labels,
                                             min_metric_better=False,
                                             class_colors=get_rgb_colors(num_classes, mean=cfg.INPUT.NORM_MEAN, scale=cfg.INPUT.NORM_SCALE),
                                             blending_alpha=cfg.TENSORBOARD.ALPHA_BLENDING,
                                             nonzero_factor=0.1)

    return stats
