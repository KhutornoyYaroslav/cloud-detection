from .cloud_dataset import CloudDataset


def build_dataset(cfg, root_dir: str, is_train: bool):
    return CloudDataset(cfg, root_dir, is_train)
