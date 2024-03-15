import time
import torch
import logging
from .datasets import build_dataset
from torch.utils.data import (
    Dataset,
    DataLoader,
    BatchSampler,
    ConcatDataset,
    RandomSampler,
    SequentialSampler
)


def create_loader(dataset:Dataset,
                  shuffle:bool,
                  batch_size:int,
                  num_workers:int = 1,
                  pin_memory:bool = True):
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(int(time.time()))
        sampler = RandomSampler(dataset, generator=generator)
    else:
        sampler = SequentialSampler(dataset)

    batch_sampler = BatchSampler(sampler=sampler, batch_size=batch_size, drop_last=False)
    data_loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=num_workers, pin_memory=pin_memory)

    return data_loader


def make_data_loader(cfg, is_train: bool = True) -> DataLoader:
    logger = logging.getLogger('CORE')

    if is_train:
        dataset_roots = cfg.DATASET.TRAIN_ROOT_DIRS
    else:
        dataset_roots = cfg.DATASET.VAL_ROOT_DIRS

    # create dataset
    datasets = []
    for root_dir in dataset_roots:
        dataset = build_dataset(cfg, root_dir, is_train)
        logger.info(f"Loaded dataset from '{root_dir}'. Size: {len(dataset)}")
        datasets.append(dataset)

    if not datasets:
        return None

    dataset = ConcatDataset(datasets)

    # create dataloader
    shuffle = is_train
    data_loader = create_loader(dataset, shuffle, cfg.SOLVER.BATCH_SIZE, cfg.DATA_LOADER.NUM_WORKERS, cfg.DATA_LOADER.PIN_MEMORY)

    return data_loader
