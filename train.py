import os
import torch
import logging
import argparse
from core.config import cfg as cfg
from core.models import build_model
from core.data import make_data_loader
from core.solver import make_optimizer
from core.engine.train import do_train
from core.utils.logger import setup_logger
from core.utils.checkpoint import CheckPointer

def train_model(cfg, args):
    logger = logging.getLogger('CORE')
    device = torch.device(cfg.MODEL.DEVICE)

    # create model
    model = build_model(cfg)
    model.to(device)

    # create data loaders
    data_loader_train = make_data_loader(cfg, is_train=True)
    if data_loader_train is None:
        logger.error(f"Failed to create train dataset loader.")
        return None

    data_loader_val = make_data_loader(cfg, is_train=False)

    # create optimizer
    optimizer = make_optimizer(cfg, model)
    scheduler = None

    # create checkpointer
    arguments = {"epoch": 0}
    checkpointer = CheckPointer(model, optimizer, scheduler, cfg.OUTPUT_DIR, True, logger)
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.PRETRAINED_WEIGHTS)
    arguments.update(extra_checkpoint_data)

    # Train model
    model = do_train(cfg, model, data_loader_train, data_loader_val, optimizer, scheduler, checkpointer, device, arguments, args)

    return model


def str2bool(s):
    return s.lower() in ('true', '1')


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Cloud Detection In Satellite Images Model Training With PyTorch')
    parser.add_argument("--config-file", dest="config_file", required=False, type=str, default="configs/cfg.yaml",
                        help="Path to config file")
    parser.add_argument('--save-step', dest="save_step", required=False, type=int, default=1,
                        help='Save checkpoint every save_step')
    parser.add_argument('--val-step', dest="val_step", required=False, type=int, default=1,
                        help='Validate model every val_step, disabled when val_step <= 0')
    parser.add_argument('--use-tensorboard', dest="use_tensorboard", required=False, default=True, type=str2bool,
                        help='Use tensorboard summary writer')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")
    args = parser.parse_args()

    # enable cudnn auto-tuner
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # read config
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # create output directory
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # create logger
    logger = setup_logger("CORE", 0, cfg.OUTPUT_DIR)
    logger.info(args)
    logger.info("Loaded configuration file {}".format(args.config_file))

    # create config backup
    with open(os.path.join(cfg.OUTPUT_DIR, 'cfg.yaml'), "w") as cfg_dump:
        cfg_dump.write(str(cfg))

    # train model
    model = train_model(cfg, args)


if __name__ == '__main__':
    main()
