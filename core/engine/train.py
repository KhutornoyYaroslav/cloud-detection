import os
import torch
import logging
import numpy as np
from tqdm import tqdm
from core.engine.validation import do_validation
from torch.utils.tensorboard import SummaryWriter
from core.engine.metrics import Dice, JaccardIndex, Precision, Recall
from core.utils.tensorboard import update_tensorboard_image_samples


def update_summary_writer(summary_writer, stats, iterations, optimizer, global_step, class_labels, is_train: bool = True):
    domen = "train" if is_train else "val"
    with torch.no_grad():
        # scalars
        if is_train:
            summary_writer.add_scalar('optimizer/lr', optimizer.param_groups[0]['lr'], global_step=global_step)

        dice_dict = {}
        jaccard_dict = {}
        precision_dict = {}
        recall_dict = {}
        for idx, label in enumerate(class_labels):
            dice_dict[f"dice_{label}"] = stats['dice_sum'][idx] / iterations
            jaccard_dict[f"jaccard_{label}"] = stats['jaccard_sum'][idx] / iterations
            precision_dict[f"precision_{label}"] = stats['precision_sum'][idx] / iterations
            recall_dict[f"recall_{label}"] = stats['recall_sum'][idx] / iterations

        summary_writer.add_scalars(domen + '/dice', dice_dict, global_step=global_step)
        summary_writer.add_scalars(domen + '/jaccard', jaccard_dict, global_step=global_step)
        summary_writer.add_scalars(domen + '/precision', precision_dict, global_step=global_step)
        summary_writer.add_scalars(domen + '/recall', recall_dict, global_step=global_step)
        summary_writer.add_scalar(domen + '/loss', stats['loss_sum'] / iterations, global_step=global_step)

        # images
        if len(stats['best_samples']):
            tb_images = [s[1] for s in stats['best_samples']]
            image_grid = torch.concatenate(tb_images, dim=1)
            summary_writer.add_image(domen + '/best_samples', image_grid, global_step=global_step)
        if len(stats['worst_samples']):
            tb_images = [s[1] for s in stats['worst_samples']]
            image_grid = torch.concatenate(tb_images, dim=1)
            summary_writer.add_image(domen + '/worst_samples', image_grid, global_step=global_step)

        # save
        summary_writer.flush()


def do_train(cfg,
             model,
             data_loader_train,
             data_loader_val,
             optimizer,
             scheduler,
             checkpointer,
             device,
             arguments,
             args):
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    logger = logging.getLogger("CORE")
    logger.info("Start training ...")

    # set model to train mode
    model.train()

    # create tensorboard writer
    if args.use_tensorboard:
        summary_writer = SummaryWriter(log_dir=os.path.join(cfg.OUTPUT_DIR, 'tf_logs'))
    else:
        summary_writer = None

    # prepare to train
    iters_per_epoch = len(data_loader_train)
    total_steps = iters_per_epoch * cfg.SOLVER.MAX_EPOCH
    start_epoch = arguments["epoch"]
    end_epoch = cfg.SOLVER.MAX_EPOCH
    logger.info("Iterations per epoch: {0}. Total steps: {1}. Start epoch: {2}".format(iters_per_epoch, total_steps, start_epoch))

    # create metrics
    cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
    dice_metric = Dice(len(cfg.MODEL.CLASS_LABELS))
    jaccard_metric = JaccardIndex(len(cfg.MODEL.CLASS_LABELS))
    precision_metric = Precision(len(cfg.MODEL.CLASS_LABELS))
    recall_metric = Recall(len(cfg.MODEL.CLASS_LABELS))

    # epoch loop
    for epoch in range(start_epoch, end_epoch):
        arguments["epoch"] = epoch + 1

        # create progress bar
        print(('\n' + '%10s' * 4 + '%20s' * 4) % ('epoch', 'gpu_mem', 'lr', 'loss', 'dice', 'jaccard', 'precision', 'recall'))
        pbar = enumerate(data_loader_train)
        pbar = tqdm(pbar, total=len(data_loader_train))

        # create stats
        stats = {
            'loss_sum': 0,
            'dice_sum': 0,
            'jaccard_sum': 0,
            'precision_sum': 0,
            'recall_sum': 0,
            'best_samples': [],
            'worst_samples': []
        }

        # iteration loop
        for iteration, data_entry in pbar:
            global_step = epoch * iters_per_epoch + iteration

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

            # optimize model
            optimizer.zero_grad()
            loss_mean = torch.mean(loss)
            loss_mean.backward()
            optimizer.step()

            # calculate metrics
            pred_labels = torch.softmax(output, dim=1).argmax(dim=1)        # (n, h, w)
            dice = dice_metric(pred_labels, target_labels, roi)             # (n, k)
            jaccard = jaccard_metric(pred_labels, target_labels, roi)       # (n, k)
            precision = precision_metric(pred_labels, target_labels, roi)   # (n, k)
            recall = recall_metric(pred_labels, target_labels, roi)         # (n, k)

            # update stats
            stats['loss_sum'] += loss_mean.item()
            stats['dice_sum'] += torch.mean(dice, 0).cpu().numpy()
            stats['jaccard_sum'] += torch.mean(jaccard, 0).cpu().numpy()
            stats['precision_sum'] += torch.mean(precision, 0).cpu().numpy()
            stats['recall_sum'] += torch.mean(recall, 0).cpu().numpy()

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

            # update progress bar
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)

            dice_avg = stats['dice_sum'] / (iteration + 1)
            dice_avg = [f'{x:.2f}' for x in dice_avg]
            jaccard_avg = stats['jaccard_sum'] / (iteration + 1)
            jaccard_avg = [f'{x:.2f}' for x in jaccard_avg]
            precision_avg = stats['precision_sum'] / (iteration + 1)
            precision_avg = [f'{x:.2f}' for x in precision_avg]
            recall_avg = stats['recall_sum'] / (iteration + 1)
            recall_avg = [f'{x:.2f}' for x in recall_avg]

            s = ('%10s' * 2 + '%10.4g' * 2 + '%20s' * 4) % ('%g/%g' % (epoch + 1, end_epoch),
                                               mem,
                                               optimizer.param_groups[0]["lr"],
                                               stats['loss_sum'] / (iteration + 1),
                                               ", ".join(dice_avg),
                                               ", ".join(jaccard_avg),
                                               ", ".join(precision_avg),
                                               ", ".join(recall_avg))
            pbar.set_description(s)

        # update learning rate
        if scheduler is not None:
            scheduler.step()

        # do validation
        if (args.val_step > 0) and (epoch % args.val_step == 0) and (data_loader_val is not None):
            print('\n')
            logger.info("Start validation ...")

            torch.cuda.empty_cache()
            model.eval()
            val_stats = do_validation(cfg, model, data_loader_val, device)
            torch.cuda.empty_cache()
            model.train()

            val_loss = val_stats['loss_sum'] / val_stats['iterations']
            val_dice = val_stats['dice_sum'] / val_stats['iterations']
            val_dice = [f'{x:.2f}' for x in val_dice]
            val_jaccard = val_stats['jaccard_sum'] / val_stats['iterations']
            val_jaccard = [f'{x:.2f}' for x in val_jaccard]
            val_precision = val_stats['precision_sum'] / val_stats['iterations']
            val_precision = [f'{x:.2f}' for x in val_precision]
            val_recall = val_stats['recall_sum'] / val_stats['iterations']
            val_recall = [f'{x:.2f}' for x in val_recall]

            log_preamb = 'Validation results: '
            print((log_preamb + '%10s' * 1 + '%20s' * 2) % ('loss', 'dice', 'jaccard'))
            print((len(log_preamb) * ' ' + '%10.4g' * 1 + '%20s' * 4) % (val_loss,
                                                                         ", ".join(val_dice),
                                                                         ", ".join(val_jaccard),
                                                                         ", ".join(val_precision),
                                                                         ", ".join(val_recall)))
            print('\n')

            if summary_writer:
                update_summary_writer(summary_writer,
                                      val_stats,
                                      val_stats['iterations'],
                                      optimizer,
                                      global_step,
                                      cfg.MODEL.CLASS_LABELS,
                                      False)

        # save epoch results
        if epoch % args.save_step == 0:
            checkpointer.save("model_{:06d}".format(global_step), **arguments)
            if summary_writer:
                update_summary_writer(summary_writer,
                                      stats,
                                      iteration + 1,
                                      optimizer,
                                      global_step,
                                      cfg.MODEL.CLASS_LABELS,
                                      True)

    # save final model
    checkpointer.save("model_final", **arguments)

    return model
