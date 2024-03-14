import os
import torch
import logging
import numpy as np
from tqdm import tqdm
# from .validation import eval_dataset
# from torchvision.utils import make_grid
# from core.data import make_data_loader
from torch.utils.tensorboard import SummaryWriter
from core.engine.metrics import Dice, JaccardIndex
from core.utils.tensorboard import update_tensorboard_image_samples


# def do_eval(cfg, model, forward_method, loss_dist_key, loss_rate_keys, p_frames):
#     torch.cuda.empty_cache()
#     if isinstance(model, torch.nn.parallel.DistributedDataParallel):
#         model = model.module

#     data_loader = make_data_loader(cfg, False)
#     model.eval()
#     device = torch.device(cfg.MODEL.DEVICE)
#     result_dict = eval_dataset(forward_method, loss_dist_key, loss_rate_keys, p_frames, data_loader, device, cfg)

#     torch.cuda.empty_cache()
#     return result_dict


def do_train(cfg,
             model,
             data_loader,
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
    iters_per_epoch = len(data_loader)
    total_steps = iters_per_epoch * cfg.SOLVER.MAX_EPOCH
    start_epoch = arguments["epoch"]
    end_epoch = cfg.SOLVER.MAX_EPOCH
    logger.info("Iterations per epoch: {0}. Total steps: {1}. Start epoch: {2}".format(iters_per_epoch, total_steps, start_epoch))

    # create metrics
    cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
    dice_metric = Dice(len(cfg.MODEL.CLASS_LABELS))
    jaccard_metric = JaccardIndex(len(cfg.MODEL.CLASS_LABELS))

    # epoch loop
    for epoch in range(start_epoch, end_epoch):
        arguments["epoch"] = epoch + 1

        # create progress bar
        print(('\n' + '%10s' * 4 + '%20s' * 2) % ('epoch', 'gpu_mem', 'lr', 'loss', 'dice', 'jaccard'))
        pbar = enumerate(data_loader)
        pbar = tqdm(pbar, total=len(data_loader))

        # create stats
        stats = {
            'loss_sum': 0,
            'dice_sum': 0,
            'jaccard_sum': 0,
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
            pred_labels = torch.softmax(output, dim=1).argmax(dim=1)    # (n, h, w)
            dice = dice_metric(pred_labels, target_labels, roi)         # (n, k)
            jaccard = jaccard_metric(pred_labels, target_labels, roi)   # (n, k)

            # update stats
            stats['loss_sum'] += loss_mean.item()
            stats['dice_sum'] += torch.mean(dice, 0).cpu().numpy()
            stats['jaccard_sum'] += torch.mean(jaccard, 0).cpu().numpy()

            # update best samples
            update_tensorboard_image_samples(limit=cfg.TENSORBOARD.BEST_SAMPLES_NUM,
                                             accumulator=stats['best_samples'], 
                                             input=input,
                                             metric=torch.mean(loss, dim=(1, 2)), 
                                             pred_labels=pred_labels, 
                                             target_labels=target_labels,
                                             min_metric_better=True,
                                             blending_alpha=cfg.TENSORBOARD.ALPHA_BLENDING)

            # update worst samples
            update_tensorboard_image_samples(limit=cfg.TENSORBOARD.WORST_SAMPLES_NUM,
                                             accumulator=stats['worst_samples'], 
                                             input=input,
                                             metric=torch.mean(loss, dim=(1, 2)), 
                                             pred_labels=pred_labels, 
                                             target_labels=target_labels,
                                             min_metric_better=False,
                                             blending_alpha=cfg.TENSORBOARD.ALPHA_BLENDING)

            # update progress bar
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)

            dice_avg = stats['dice_sum'] / (iteration + 1)
            dice_avg = [f'{x:.2f}' for x in dice_avg]
            jaccard_avg = stats['jaccard_sum'] / (iteration + 1)
            jaccard_avg = [f'{x:.2f}' for x in jaccard_avg]

            s = ('%10s' * 2 + '%10.4g' * 2 + '%20s' * 2) % ('%g/%g' % (epoch + 1, end_epoch),
                                               mem,
                                               optimizer.param_groups[0]["lr"],
                                               stats['loss_sum'] / (iteration + 1),
                                               ", ".join(dice_avg),
                                               ", ".join(jaccard_avg))
            pbar.set_description(s)

        # update learning rate
        if scheduler is not None:
            scheduler.step()

        # # do evaluation
        # if (args.eval_step > 0) and (epoch % args.eval_step == 0) and len(cfg.DATASET.TEST_ROOT_DIRS):
        #     print('\nEvaluation ...')
        #     result_dict = do_eval(cfg,
        #                           model,
        #                           stage_params['forward_method'],
        #                           stage_params['loss_dist_key'],
        #                           stage_params['loss_rate_keys'],
        #                           stage_params['p_frames'])

        #     print(('\n' + 'Evaluation results:' + '%12s' * 2 + '%25s' * 2) % ('loss', 'mse', 'bpp', 'psnr'))
        #     bpp_print = [f'{x:.2f}' for x in result_dict['bpp']]
        #     psnr = 10 * np.log10(1.0 / (result_dict['psnr']))
        #     psnr_print = [f'{x:.1f}' for x in psnr]
        #     print('                   ' + ('%12.4g' * 2 + '%25s' * 2) %
        #           (result_dict['loss_sum'],
        #            result_dict['mse_sum'],
        #            ", ".join(bpp_print),
        #            ", ".join(psnr_print))
        #           )

        #     add_metrics(cfg, summary_writer, result_dict, global_step, is_train=False)

        #     model.train()

        # save epoch results
        if epoch % args.save_step == 0:
            checkpointer.save("model_{:06d}".format(global_step), **arguments)
            if summary_writer:
                with torch.no_grad():
                    # scalars
                    summary_writer.add_scalar('train/loss', stats['loss_sum'] / (iteration + 1), global_step=global_step)
                    summary_writer.add_scalar('optimizer/lr', optimizer.param_groups[0]['lr'], global_step=global_step)

                    dice_dict = {}
                    jaccard_dict = {}
                    for idx, label in enumerate(cfg.MODEL.CLASS_LABELS):
                        dice_dict[f"dice_{label}"] = stats['dice_sum'][idx] / (iteration + 1)
                        jaccard_dict[f"jaccard_{label}"] = stats['jaccard_sum'][idx] / (iteration + 1)

                    summary_writer.add_scalars(f'train/dice', dice_dict, global_step=global_step)
                    summary_writer.add_scalars(f'train/jaccard', jaccard_dict, global_step=global_step)

                    # images
                    if len(stats['best_samples']):
                        tb_images = [s[1] for s in stats['best_samples']]
                        image_grid = torch.concatenate(tb_images, dim=1)
                        summary_writer.add_image('train/best_samples', image_grid, global_step=global_step)

                    if len(stats['worst_samples']):
                        tb_images = [s[1] for s in stats['worst_samples']]
                        image_grid = torch.concatenate(tb_images, dim=1)
                        summary_writer.add_image('train/worst_samples', image_grid, global_step=global_step)

                    # save
                    summary_writer.flush()

    # save final model
    checkpointer.save("model_final", **arguments)

    return model
