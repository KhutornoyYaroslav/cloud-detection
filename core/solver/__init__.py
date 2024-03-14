import torch


def make_optimizer(cfg, model: torch.nn.Module) -> torch.optim.Optimizer:
    lr = float(cfg.SOLVER.LR)

    params_to_train = []
    for p in model.parameters():
        if p.requires_grad:
            params_to_train.append(p)

    optimizer = torch.optim.Adam(params=params_to_train, lr=lr)

    return optimizer
