import argparse
from core.config import cfg as cfg
from core.data.datasets import build_dataset


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='Dataset Testing')
    parser.add_argument("--config-file", dest="config_file", required=False, type=str, default="configs/cfg.yaml",
                        help="Path to config file")
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")
    args = parser.parse_args()

    # read config
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # test dataset
    datasets = build_dataset(cfg, cfg.DATASET.TRAIN_ROOT_DIRS[0], True)
    datasets.visualize(0)


if __name__ == '__main__':
    main()
