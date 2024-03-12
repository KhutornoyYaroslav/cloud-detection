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


def prepare_dataset(src_root: str,
                    dst_root: str,
                    tile_size: int,
                    resize_factor: float = 1.0,
                    skip_empty: bool = False,
                    debug_show: bool = False,
                    filename_template: str = "tile_%05d.png"):
    logger = logging.getLogger(_LOGGER_NAME)

    # scan dataset sub-directories
    src_dirs = glob(os.path.join(src_root, "*/"))
    for src_dir in src_dirs:
        logger.info(f"Processing '{src_dir}' ...")

        img_path = glob(os.path.join(src_dir, "img", "*.png"))
        cloud_path = glob(os.path.join(src_dir, "mask_cloud", "*.png"))
        fog_path = glob(os.path.join(src_dir, "mask_fog", "*.png"))
        assert len(img_path) == len(cloud_path) == len(fog_path) == 1, "Number of images and corresponding labels must be equal to 1"

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
        dst_subdir = os.path.join(dst_root, os.path.basename(os.path.normpath(src_dir)))

        shutil.rmtree(dst_subdir, True)
        dst_img_dir = os.path.join(dst_subdir, "img")
        dst_label_dir = os.path.join(dst_subdir, "label")
        dst_roi_dir = os.path.join(dst_subdir, "roi")

        saved_tile_cnt = 0
        for tile_idx, (img_tile, label_tile, roi_tile) in enumerate(zip(img_tiles, label_tiles, roi_tiles)):
            if skip_empty and np.count_nonzero(label_tile) == 0:
                continue

            os.makedirs(dst_img_dir, exist_ok=True)
            cv.imwrite(os.path.join(dst_img_dir, filename_template % tile_idx), img_tile)
            os.makedirs(dst_label_dir, exist_ok=True)
            cv.imwrite(os.path.join(dst_label_dir, filename_template % tile_idx), label_tile)
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
    parser.add_argument('--src-root', dest='src_root', type=str, default="/media/yaroslav/SSD/khutornoy/data/clouds/sources/0001_0001_35146_1_35122_03_ORT_10_00",
                        help="Path to dataset root directory")
    parser.add_argument('--dst-root', dest='dst_root', type=str, default="/media/yaroslav/SSD/khutornoy/data/clouds/outputs/0001_0001_35146_1_35122_03_ORT_10_00",
                        help="Path where to save result dataset")
    parser.add_argument('--tile-size', dest='tile_size', type=int, default=512,
                        help="Size of output frames")
    parser.add_argument('--resize-factor', dest='resize_factor', type=float, default=1.0,
                        help="Factor for pre-resizing frames")
    parser.add_argument('--skip-empty', dest='skip_empty', default=True, type=str2bool,
                        help="Whether to skip tiles without clouds or not")
    parser.add_argument('--debug-show', dest='debug_show', default=False, type=str2bool,
                        help="Whether to show resulting tiles or not")
    args = parser.parse_args()

    # create logger
    logger = setup_logger(_LOGGER_NAME, distributed_rank=0)
    logger.info(args)

    # prepare dataset
    prepare_dataset(args.src_root, args.dst_root, args.tile_size, args.resize_factor, args.skip_empty, args.debug_show)


if __name__ == "__main__":
    main()
