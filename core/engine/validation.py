import torch
import logging
from tqdm import tqdm
from core.engine.metrics import Dice, JaccardIndex
from core.utils.tensorboard import update_tensorboard_image_samples


def do_validation(cfg, model, data_loader, device):
    logger = logging.getLogger("CORE")

    # create metrics
    cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
    dice_metric = Dice(len(cfg.MODEL.CLASS_LABELS))
    jaccard_metric = JaccardIndex(len(cfg.MODEL.CLASS_LABELS))

    # create stats
    stats = {
        'loss_sum': 0,
        'dice_sum': 0,
        'jaccard_sum': 0,
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
            pred_labels = torch.softmax(output, dim=1).argmax(dim=1)    # (n, h, w)
            dice = dice_metric(pred_labels, target_labels, roi)         # (n, k)
            jaccard = jaccard_metric(pred_labels, target_labels, roi)   # (n, k)

            # update stats
            stats['loss_sum'] += torch.mean(loss).item()
            stats['dice_sum'] += torch.mean(dice, 0).cpu().numpy()
            stats['jaccard_sum'] += torch.mean(jaccard, 0).cpu().numpy()
            stats['iterations'] += 1

            # update best samples
            update_tensorboard_image_samples(limit=cfg.TENSORBOARD.BEST_SAMPLES_NUM,
                                             accumulator=stats['best_samples'], 
                                             input=input,
                                             metric=torch.mean(loss, dim=(1, 2)), 
                                             pred_labels=pred_labels, 
                                             target_labels=target_labels,
                                             min_metric_better=True,
                                             blending_alpha=cfg.TENSORBOARD.ALPHA_BLENDING,
                                             nonzero_factor=0.25)

            # update worst samples
            update_tensorboard_image_samples(limit=cfg.TENSORBOARD.WORST_SAMPLES_NUM,
                                             accumulator=stats['worst_samples'], 
                                             input=input,
                                             metric=torch.mean(loss, dim=(1, 2)), 
                                             pred_labels=pred_labels, 
                                             target_labels=target_labels,
                                             min_metric_better=False,
                                             blending_alpha=cfg.TENSORBOARD.ALPHA_BLENDING,
                                             nonzero_factor=0.25)

    return stats
