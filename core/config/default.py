from yacs.config import CfgNode as CN


_CFG = CN()

# ---------------------------------------------------------------------------- #
# Model
# ---------------------------------------------------------------------------- #
_CFG.MODEL = CN()
_CFG.MODEL.ARCHITECTURE = "ResUnetPlusPlus"
_CFG.MODEL.CLASS_LABELS = ["fog", "cloud"]
_CFG.MODEL.DEVICE = "cpu"
_CFG.MODEL.PRETRAINED_WEIGHTS = ""

# -----------------------------------------------------------------------------
# Input
# -----------------------------------------------------------------------------
_CFG.INPUT = CN()
_CFG.INPUT.DEPTH = 3
_CFG.INPUT.IMAGE_SIZE = [512, 512]

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_CFG.DATASET = CN()
_CFG.DATASET.TRAIN_ROOT_DIRS = []
_CFG.DATASET.VAL_ROOT_DIRS = []

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_CFG.DATA_LOADER = CN()
_CFG.DATA_LOADER.NUM_WORKERS = 1
_CFG.DATA_LOADER.PIN_MEMORY = True

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_CFG.SOLVER = CN()
_CFG.SOLVER.TYPE = 'Adam'
_CFG.SOLVER.MAX_EPOCH = 100
_CFG.SOLVER.BATCH_SIZE = 32
_CFG.SOLVER.LR = 1e-3

# ---------------------------------------------------------------------------- #
# Output options
# ---------------------------------------------------------------------------- #
_CFG.OUTPUT_DIR = 'outputs/test'

# ---------------------------------------------------------------------------- #
# Tensorboard
# ---------------------------------------------------------------------------- #
_CFG.TENSORBOARD = CN()
_CFG.TENSORBOARD.ALPHA_BLENDING = 0.15
_CFG.TENSORBOARD.BEST_SAMPLES_NUM = 32
_CFG.TENSORBOARD.WORST_SAMPLES_NUM = 32
