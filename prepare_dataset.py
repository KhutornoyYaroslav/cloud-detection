import os
import shutil
import logging
import argparse
import cv2 as cv
import numpy as np
from glob import glob
from core.utils.logger import setup_logger
from core.data.transforms.functional import image_to_tiles


_LOGGER_NAME = "DATASET PREP"


def resize_and_split(image: np.ndarray, tile_size: int, resize_factor: float = 1.0):
    if resize_factor != 1.0:
        image_resized = cv.resize(image, dsize=None, fx=resize_factor, fy=resize_factor, interpolation=cv.INTER_AREA)
    else:
        image_resized = image

    return image_to_tiles(image_resized, tile_size=tile_size)


def create_multiclass_label(cloud: np.ndarray, fog: np.ndarray):
    h, w = cloud.shape[:2]
    assert cloud.shape == fog.shape, "Arrays must have equal shapes"

    label = np.full(shape=(h, w), fill_value=0, dtype=np.uint8)
    label[fog > 0] = 1
    label[cloud > 0] = 2

    return label


def prepare_dataset(src_path: str,
                    dst_root: str,
                    tile_size: int,
                    resize_factor: float = 1.0,
                    val_perc: float = 0.2,
                    skip_zero: bool = False,
                    debug_show: bool = False,
                    filename_template: str = "tile_%05d.png"):
    logger = logging.getLogger(_LOGGER_NAME)

    val_perc = np.clip(val_perc, 0.0, 1.0)

    # scan dataset directories
    src_dirs = glob(src_path)
    for src_dir in src_dirs:
        logger.info(f"Processing '{src_dir}' ...")

        # scan files
        img_path = glob(os.path.join(src_dir, "img", "*.*"))
        cloud_path = glob(os.path.join(src_dir, "mask_cloud", "*.*"))
        fog_path = glob(os.path.join(src_dir, "mask_fog", "*.*"))

        files_valid = len(img_path) == len(cloud_path) == len(fog_path) == 1
        if not files_valid:
            logger.warning(f"Source files invalid. Skip it.")
            continue

        # image
        img = cv.imread(img_path[0], cv.IMREAD_COLOR)
        img_tiles = resize_and_split(img, tile_size, resize_factor)

        # label
        cloud = cv.imread(cloud_path[0], cv.IMREAD_GRAYSCALE)
        fog = cv.imread(fog_path[0], cv.IMREAD_GRAYSCALE)
        label = create_multiclass_label(cloud, fog)
        label_tiles = resize_and_split(label, tile_size, resize_factor)

        # roi
        roi = np.full_like(cloud, fill_value=255)
        roi_tiles = resize_and_split(roi, tile_size, resize_factor)

        assert img_tiles.shape[:2] == label_tiles.shape[:2] == roi_tiles.shape[:2]

        # save to disk
        dst_train_subdir = os.path.join(dst_root, "train", os.path.basename(os.path.normpath(src_dir)))
        shutil.rmtree(dst_train_subdir, True)
        dst_val_subdir = os.path.join(dst_root, "val", os.path.basename(os.path.normpath(src_dir)))
        shutil.rmtree(dst_val_subdir, True)

        saved_tile_cnt = 0
        for tile_idx, (img_tile, label_tile, roi_tile) in enumerate(zip(img_tiles, label_tiles, roi_tiles)):
            if skip_zero and np.count_nonzero(label_tile) == 0:
                continue

            dst_subdir = dst_val_subdir if np.random.choice(2, p=[1 - val_perc, val_perc]) else dst_train_subdir

            dst_img_dir = os.path.join(dst_subdir, "img")
            os.makedirs(dst_img_dir, exist_ok=True)
            cv.imwrite(os.path.join(dst_img_dir, filename_template % tile_idx), img_tile)

            dst_label_dir = os.path.join(dst_subdir, "label")
            os.makedirs(dst_label_dir, exist_ok=True)
            cv.imwrite(os.path.join(dst_label_dir, filename_template % tile_idx), label_tile)

            dst_roi_dir = os.path.join(dst_subdir, "roi")
            os.makedirs(dst_roi_dir, exist_ok=True)
            cv.imwrite(os.path.join(dst_roi_dir, filename_template % tile_idx), roi_tile)
            saved_tile_cnt += 1

            if debug_show:
                c = img_tile.shape[-1]
                debug_label = 125 * np.stack(c * [label_tile], -1)
                debug_roi = np.stack(c * [roi_tile], -1)
                debug_img = np.concatenate([img_tile, debug_label, debug_roi], axis=1)
                cv.imshow('debug', debug_img)
                if cv.waitKey(0) & 0xFF == ord('q'):
                    return

        logger.info(f"Saved {saved_tile_cnt} tiles")

    logger.info(f"Done.")


def str2bool(s):
    return s.lower() in ('true', 'yes', 'y', '1')


def main():
    # create argument parser
    parser = argparse.ArgumentParser(description='Dataset Preparing For Cloud Detection In Satellite Images')
    parser.add_argument('--src-path', dest='src_path', type=str, default="/media/yaroslav/SSD/khutornoy/data/clouds/sources/*/*/",
                        help="Pattern-like path to dataset directories")
    parser.add_argument('--dst-root', dest='dst_root', type=str, default="/media/yaroslav/SSD/khutornoy/data/clouds/outputs/t512_rf1_se",
                        help="Path where to save result dataset")
    parser.add_argument('--tile-size', dest='tile_size', type=int, default=512,
                        help="Size of output frames")
    parser.add_argument('--resize-factor', dest='resize_factor', type=float, default=1.0,
                        help="Factor for pre-resizing frames")
    parser.add_argument('--val-perc', dest='val_perc', type=float, default=0.2,
                        help="Size of validation data as a percentage of total size")
    parser.add_argument('--skip-zero', dest='skip_zero', default=True, type=str2bool,
                        help="Whether to skip tiles without class indexes > 0")
    parser.add_argument('--debug-show', dest='debug_show', default=False, type=str2bool,
                        help="Whether to show resulting tiles or not")
    args = parser.parse_args()

    # create logger
    logger = setup_logger(_LOGGER_NAME, distributed_rank=0)
    logger.info(args)

    # prepare dataset
    prepare_dataset(args.src_path, args.dst_root, args.tile_size, args.resize_factor, args.val_perc, args.skip_zero, args.debug_show)


if __name__ == "__main__":
    main()
