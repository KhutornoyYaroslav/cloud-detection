import os
import torch
import argparse
import cv2 as cv
import numpy as np
from tqdm import tqdm
from glob import glob
from core.config import cfg as cfg
from core.models import build_model
from core.utils.logger import setup_logger
from core.utils.colors import get_rgb_colors
from core.utils.tensorboard import draw_labels
from core.utils.checkpoint import CheckPointer
from core.data.transforms.functional import image_to_tiles, tiles_to_image


_LOGGER_NAME = "MODEL TEST"


def test_model(cfg,
               src_path: str,
               dst_root: str,
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

    # create label colors
    num_classes = len(cfg.DATASET.CLASS_LABELS)
    colors_rgb = get_rgb_colors(num_classes, mean=cfg.INPUT.NORM_MEAN, scale=cfg.INPUT.NORM_SCALE)

    # scan sources
    src_files = sorted(glob(src_path))
    for src_file in tqdm(src_files):
        # read image
        image = cv.imread(src_file, cv.IMREAD_COLOR)

        # resize image
        if resize_factor != 1.0:
            image = cv.resize(image, dsize=None, fx=resize_factor, fy=resize_factor, interpolation=cv.INTER_AREA)

        if cfg.INPUT.DEPTH == 1:
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            image = np.expand_dims(image, axis=-1)

        # split to tiles
        tiles_pad = 0
        tiles = image_to_tiles(image, tile_size, tiles_pad)

        # process tiles
        result_tiles = []
        with torch.no_grad():
            for tile in tiles:
                # prepare input
                tile = (tile - cfg.INPUT.NORM_MEAN) / cfg.INPUT.NORM_SCALE
                input = torch.from_numpy(tile).type(torch.float).to(device)
                input = input.permute(2, 0, 1) # (c, h, w)

                # do prediction
                output = model.forward(input.unsqueeze(0))          # (1, k, h, w)
                output = output.squeeze(0)                          # (k, h, w)
                label = torch.softmax(output, dim=0).argmax(dim=0)  # (h, w)

                # draw labels
                result = draw_labels(input, label, colors_rgb, 0.25)

                # to numpy
                result = result.permute(1, 2, 0).cpu().numpy()
                result = result * cfg.INPUT.NORM_SCALE + cfg.INPUT.NORM_MEAN
                result = result.astype(np.uint8)
                result_tiles.append(result)

        # result tiles to image
        os.makedirs(dst_root, exist_ok=True)
        dst_file = os.path.join(dst_root, os.path.basename(os.path.normpath(src_file)))

        result_tiles = np.stack(result_tiles, axis=0)
        result_image = tiles_to_image(result_tiles, image.shape[0], image.shape[1], tiles_pad)
        cv.imwrite(dst_file, result_image)
            

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Cloud Detection In Satellite Images Model Testing With PyTorch')
    parser.add_argument("--config-file", dest="config_file", required=False, type=str, default="configs/cfg.yaml",
                        help="Path to config file")
    parser.add_argument('--src-path', dest="src_path", required=False, type=str, default="data/tests/*",
                        help='Pattern-like path to source images to segment')
    parser.add_argument('--dst-root', dest="dst_root", required=False, type=str, default="data/results",
                        help='Path where to save results')
    parser.add_argument('--tile-size', dest='tile_size', type=int, default=1536,
                        help="Size of output frames")
    parser.add_argument('--resize-factor', dest='resize_factor', type=float, default=1.0,
                        help="Factor to resize tiles before feeding to model")
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
    test_model(cfg, args.src_path, args.dst_root, args.tile_size, args.resize_factor)


if __name__ == "__main__":
    main()
