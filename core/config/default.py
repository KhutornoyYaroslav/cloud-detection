from yacs.config import CfgNode as CN


_CFG = CN()

# ---------------------------------------------------------------------------- #
# Model
# ---------------------------------------------------------------------------- #
_CFG.MODEL = CN()
_CFG.MODEL.DEVICE = "cpu"
_CFG.MODEL.META_ARCHITECTURE = 'MulticlassSegmentator'

# ---------------------------------------------------------------------------- #
# Backbone
# ---------------------------------------------------------------------------- #
_CFG.MODEL.BACKBONE = CN()
_CFG.MODEL.BACKBONE.NAME = ''
_CFG.MODEL.BACKBONE.PRETRAINED = True
_CFG.MODEL.BACKBONE.FREEZE = True

# ---------------------------------------------------------------------------- #
# Head
# ---------------------------------------------------------------------------- #
_CFG.MODEL.HEAD = CN()
_CFG.MODEL.HEAD.NAME = ''
_CFG.MODEL.HEAD.INPUT_DEPTH = [64, 128, 256, 384]
_CFG.MODEL.HEAD.HIDDEN_DEPTH = 64
_CFG.MODEL.HEAD.PRETRAINED = True
_CFG.MODEL.HEAD.FREEZE = True
_CFG.MODEL.HEAD.DROPOUT = 0.5
_CFG.MODEL.HEAD.CLASS_LABELS = []

# -----------------------------------------------------------------------------
# Input
# -----------------------------------------------------------------------------
_CFG.INPUT = CN()
_CFG.INPUT.IMAGE_SIZE = [512, 512]
_CFG.INPUT.MAKE_DIVISIBLE_BY = 8

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_CFG.DATASETS = CN()
_CFG.DATASETS.TRAIN_ROOT_DIRS = []
_CFG.DATASETS.TEST_ROOT_DIRS = []

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
# Specific test options
# ---------------------------------------------------------------------------- #
_CFG.TEST = CN()
_CFG.TEST.BATCH_SIZE = 32

# ---------------------------------------------------------------------------- #
# Output options
# ---------------------------------------------------------------------------- #
_CFG.OUTPUT_DIR = 'outputs/test'

# ---------------------------------------------------------------------------- #
# Tensorboard
# ---------------------------------------------------------------------------- #
_CFG.TENSORBOARD = CN()
_CFG.TENSORBOARD.BEST_SAMPLES_NUM = 32
_CFG.TENSORBOARD.WORST_SAMPLES_NUM = 32
_CFG.TENSORBOARD.METRICS_BIN_THRESHOLD = 0.85