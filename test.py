import os
import torch
import shutil
import logging
import argparse
import cv2 as cv
import numpy as np
from torch import nn
from glob import glob
from core.config import cfg as cfg
from core.models import build_model
from core.utils.logger import setup_logger
from core.data.transforms.functional import image_to_tiles, tiles_to_image
from core.utils.checkpoint import CheckPointer
from core.utils.tensorboard import draw_labels


_LOGGER_NAME = "MODEL TEST"


def test_model(cfg,
               src_path: str,
               dst_path: str,
               tile_size: int,
               resize_factor: float):
    # create device
    device = torch.device(cfg.MODEL.DEVICE)

    # create model
    model = build_model(cfg)
    model.to(device)
    model.eval()

    # load weights
    checkpointer = CheckPointer(model, None, None, cfg.OUTPUT_DIR)
    checkpointer.load(cfg.MODEL.PRETRAINED_WEIGHTS)

    # read image
    image = cv.imread(src_path, cv.IMREAD_COLOR)

    # split to tiles
    tiles_pad = 0
    tiles = image_to_tiles(image, tile_size, tiles_pad)

    # TODO: as parameter
    class_colors = [
        (0, 0, 0), # background
        (1, 0, 0.5), # fog
        (0, 0.5, 1), # cloud
    ]

    # process tiles
    result_tiles = []
    with torch.no_grad():
        for tile in tiles:
            # prepare input
            input = torch.from_numpy(tile).type(torch.float).to(device) # TODO: check input depth
            input = input / 255
            input = input.permute(2, 0, 1) # (C, H, W)

            # do prediction
            output = model.forward(input.unsqueeze(0)) # (1, K, H, W)
            output = output.squeeze(0) # (K, H, W)
            label = torch.softmax(output, dim=0).argmax(dim=0) # (H, W)

            # draw labels
            result = draw_labels(input, label, class_colors, 0.15)

            # to numpy
            result = result.permute(1, 2, 0).cpu().numpy()
            result = (255 * result).astype(np.uint8)
            result_tiles.append(result)

            # # debug show
            # cv.imshow('tile', tile)
            # cv.imshow('result', result)
            # if cv.waitKey(0) & 0xFF == ord('q'):
            #     break

    # result tiles to image
    result_tiles = np.stack(result_tiles, axis=0)
    result_image = tiles_to_image(result_tiles, image.shape[0], image.shape[1], tiles_pad)
    cv.imwrite(dst_path, result_image)
            

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Cloud Detection In Satellite Images Model Testing With PyTorch')
    parser.add_argument("--config-file", dest="config_file", required=False, type=str, default="configs/cfg.yaml",
                        help="Path to config file")
    # TODO: src-path (pattern)
    parser.add_argument('--src-path', dest="src_path", required=False, type=str, default="data/tests/0001_0001_34715_1_34690_01_ORT_10_01_09.png",
                        help='Path to source image to segment')
    parser.add_argument('--dst-path', dest="dst_path", required=False, type=str, default="data/results/0001_0001_34715_1_34690_01_ORT_10_01_09.png",
                        help='Path where to save segmentation result')
    parser.add_argument('--tile-size', dest='tile_size', type=int, default=1536,
                        help="Size of output frames")
    # parser.add_argument('--resize-factor', dest='resize_factor', type=float, default=1.0,
    #                     help="Factor to resize tiles before feeding to model")
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")
    args = parser.parse_args()

    # create logger
    logger = setup_logger(_LOGGER_NAME, distributed_rank=0)
    logger.info(args)

    # read config
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # test model
    test_model(cfg, args.src_path, args.dst_path, args.tile_size, None) # TODO: resize_factor


if __name__ == "__main__":
    main()