from torch import nn
from core.models.unet.res_unet_plus_plus import ResUnetPlusPlus


_MODEL_ARCHITECTURES = {
    "ResUnetPlusPlus": ResUnetPlusPlus,
}


def build_model(cfg) -> nn.Module:
    model = _MODEL_ARCHITECTURES[cfg.MODEL.ARCHITECTURE]
    return model(cfg)
