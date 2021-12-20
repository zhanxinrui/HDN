from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from yacs.config import CfgNode as CN

__C = CN()

cfg = __C

__C.META_ARC = "hdn_r50_l234"

__C.CUDA = True

#--------------------------------------------------------------------------#
#Base dirs setting
#--------------------------------------------------------------------------#
__C.BASE = CN()
__C.BASE.PROJ_PATH = '/home/zhanxinrui/SOT/HDN/' #
__C.BASE.BASE_PATH = '/home/zhanxinrui/SOT/'#base path of other resources
__C.BASE.DATA_PATH = '/home/zhanxinrui/data1/'
# __C.BASE.DATA_PATH = '/home/zhanxinrui/Downloads/Dataset/SOT/POT/'#POT base path
__C.BASE.DATA_ROOT = '/home/zhanxinrui/data1/'
# __C.BASE.DATA_ROOT = '/home/zhanxinrui/Downloads/Dataset/'#other datasets base path

# ------------------------------------------------------------------------ #
# Training options
# ------------------------------------------------------------------------ #
__C.TRAIN = CN()

# optimize obj
__C.TRAIN.OBJ = 'ALL' #LP, NM, ALL


# Number of negative
__C.TRAIN.NEG_NUM = 16

# Number of positive
__C.TRAIN.POS_NUM = 16

# Number of anchors per images
__C.TRAIN.TOTAL_NUM = 64

__C.TRAIN.WEIGHTED_MAP_LP = False

__C.TRAIN.EXEMPLAR_SIZE = 127

__C.TRAIN.SEARCH_SIZE = 255

__C.TRAIN.BASE_SIZE = 8

__C.TRAIN.OUTPUT_SIZE = 25

# __C.TRAIN.OUTPUT_SIZE_LP = 25
__C.TRAIN.OUTPUT_SIZE_LP = 13

__C.TRAIN.RESUME = ''

__C.TRAIN.PRETRAINED = ''

__C.TRAIN.DISTRIBUTED = False

__C.TRAIN.LOG_DIR = './logs'

__C.TRAIN.SNAPSHOT_DIR = './snapshot'

__C.TRAIN.EPOCH = 20

__C.TRAIN.START_EPOCH = 0

__C.TRAIN.HOMO_START_LR = 0.001

__C.TRAIN.START_EP = 0

__C.TRAIN.BATCH_SIZE = 32

__C.TRAIN.NUM_WORKERS = 4

__C.TRAIN.MOMENTUM = 0.9

__C.TRAIN.MODEL_TYPE = 'E2E'

__C.TRAIN.WEIGHT_DECAY = 0.0001

__C.TRAIN.CLS_WEIGHT = 1.0

__C.TRAIN.LOC_WEIGHT = 1.0

__C.TRAIN.FEATURE_DIS_WEIGHT = 0.2

__C.TRAIN.PRINT_FREQ = 100

__C.TRAIN.LOG_GRADS = False

__C.TRAIN.GRAD_CLIP = 10.0

__C.TRAIN.BASE_LR = 0.005

__C.TRAIN.LR = CN()

__C.TRAIN.LR.TYPE = 'log'

__C.TRAIN.LR.KWARGS = CN(new_allowed=True)

__C.TRAIN.LR_WARMUP = CN()

__C.TRAIN.LR_WARMUP.WARMUP = True

__C.TRAIN.LR_WARMUP.TYPE = 'step'

__C.TRAIN.LR_WARMUP.EPOCH = 5

__C.TRAIN.LR_WARMUP.KWARGS = CN(new_allowed=True)

__C.TRAIN.REFINE_DIS_THRSH = 10

__C.TRAIN.REFINE_CLS_POS_THRESH = 0.6

__C.TRAIN.GPU_NUM = 2

__C.TRAIN.VAL_ENABLE = False

__C.TRAIN.SAVE_LOGS = False

__C.TRAIN.HOMO_LR_RATIO = 1


# ------------------------------------------------------------------------ #
# Dataset options
# ------------------------------------------------------------------------ #
__C.DATASET = CN(new_allowed=True)

# Augmentation
# for template
__C.DATASET.TEMPLATE = CN()

# Random shift see [SiamPRN++](https://arxiv.org/pdf/1812.11703)
# for detail discussion
__C.DATASET.TEMPLATE.RHO = 10

__C.DATASET.TEMPLATE.SHIFT = 4

__C.DATASET.TEMPLATE.SCALE = 0.05

__C.DATASET.TEMPLATE.BLUR = 0.0

__C.DATASET.TEMPLATE.FLIP = 0.0

__C.DATASET.TEMPLATE.COLOR = 1.0

__C.DATASET.TEMPLATE.IMG_COMP_ALPHA = 1.0

__C.DATASET.TEMPLATE.IMG_COMP_BETA = 0.0

__C.DATASET.TEMPLATE.IMG_COMP_GAMMA = 0.0

__C.DATASET.TEMPLATE.ROTATION = 0.0

__C.DATASET.TEMPLATE.DISTORTION = 0.0

__C.DATASET.TEMPLATE.AFFINE_A = 0.0

__C.DATASET.TEMPLATE.AFFINE_C = 0.0

__C.DATASET.TEMPLATE.AFFINE_D = 0.0


__C.DATASET.SEARCH = CN()

__C.DATASET.SEARCH.RHO = 10

__C.DATASET.SEARCH.SHIFT = 64

__C.DATASET.SEARCH.SCALE = 0.18

__C.DATASET.SEARCH.BLUR = 0.0

__C.DATASET.SEARCH.FLIP = 0.0

__C.DATASET.SEARCH.IMG_COMP_ALPHA = 1.0

__C.DATASET.SEARCH.IMG_COMP_BETA = 0.0

__C.DATASET.SEARCH.IMG_COMP_GAMMA = 0.0

__C.DATASET.SEARCH.COLOR = 1.0

__C.DATASET.SEARCH.ROTATION = 0.0

__C.DATASET.SEARCH.DISTORTION = 0.001

__C.DATASET.SEARCH.AFFINE_A = 0.2

__C.DATASET.SEARCH.AFFINE_C = 0.1

__C.DATASET.SEARCH.AFFINE_D = 0.0

#Unsupervised Augmentation
__C.DATASET.TEMPLATE.UNSUPERVISED = CN()

__C.DATASET.TEMPLATE.UNSUPERVISED.RHO = 10

__C.DATASET.TEMPLATE.UNSUPERVISED.SHIFT = 4

__C.DATASET.TEMPLATE.UNSUPERVISED.SCALE = 0.05

__C.DATASET.TEMPLATE.UNSUPERVISED.BLUR = 0.0

__C.DATASET.TEMPLATE.UNSUPERVISED.FLIP = 0.0

__C.DATASET.TEMPLATE.UNSUPERVISED.COLOR = 1.0

__C.DATASET.TEMPLATE.UNSUPERVISED.ROTATION = 0.0

__C.DATASET.TEMPLATE.UNSUPERVISED.DISTORTION = 0.0

__C.DATASET.TEMPLATE.UNSUPERVISED.AFFINE_A = 0.0

__C.DATASET.TEMPLATE.UNSUPERVISED.AFFINE_C = 0.0

__C.DATASET.TEMPLATE.UNSUPERVISED.AFFINE_D = 0.0

__C.DATASET.TEMPLATE.UNSUPERVISED.IMG_COMP_ALPHA = 1.0

__C.DATASET.TEMPLATE.UNSUPERVISED.IMG_COMP_BETA = 0.0

__C.DATASET.TEMPLATE.UNSUPERVISED.IMG_COMP_GAMMA = 0.0

__C.DATASET.SEARCH.UNSUPERVISED = CN()

__C.DATASET.SEARCH.UNSUPERVISED.RHO = 10

__C.DATASET.SEARCH.UNSUPERVISED.SHIFT = 64

__C.DATASET.SEARCH.UNSUPERVISED.SCALE = 1.18

__C.DATASET.SEARCH.UNSUPERVISED.BLUR = 0.0

__C.DATASET.SEARCH.UNSUPERVISED.FLIP = 0.0

__C.DATASET.SEARCH.UNSUPERVISED.COLOR = 1.0

__C.DATASET.SEARCH.UNSUPERVISED.ROTATION = 0.0

__C.DATASET.SEARCH.UNSUPERVISED.DISTORTION = 0.001

__C.DATASET.SEARCH.UNSUPERVISED.AFFINE_A = 0.2

__C.DATASET.SEARCH.UNSUPERVISED.AFFINE_C = 0.1

__C.DATASET.SEARCH.UNSUPERVISED.AFFINE_D = 0.0

__C.DATASET.SEARCH.UNSUPERVISED.IMG_COMP_ALPHA = 1.0

__C.DATASET.SEARCH.UNSUPERVISED.IMG_COMP_BETA = 0.0

__C.DATASET.SEARCH.UNSUPERVISED.IMG_COMP_GAMMA = 0.0

# Sample Negative pair see [DaSiamRPN](https://arxiv.org/pdf/1808.06048)
# for detail discussion
__C.DATASET.NEG = 0.2
__C.DATASET.COMP = 0.0 # two images composition
__C.DATASET.LIGHT = 0.0
__C.DATASET.DARK = 0.0
__C.DATASET.OCC = 1.0
__C.DATASET.SAME = 0.5

__C.DATASET.TYPE = 'simi_aug_dataset' #'simi_aug_dataset', simi_aug_homo_dataset, semi_supervised_dataset, unsupervised_datset

# improve tracking performance for otb100
__C.DATASET.GRAY = 0.0

__C.DATASET.NAMES = ('VID', 'YOUTUBEBB', 'DET', 'COCO', 'GOT10K', 'LASOT')

__C.DATASET.VID = CN()
__C.DATASET.VID.ROOT = 'training_dataset/vid/crop511'
__C.DATASET.VID.ANNO = 'training_dataset/vid/train.json'
__C.DATASET.VID.FRAME_RANGE = 100
__C.DATASET.VID.NUM_USE = 100000

__C.DATASET.POT = CN()
__C.DATASET.POT.ROOT = 'training_dataset/pot/crop511'
__C.DATASET.POT.ANNO = 'training_dataset/pot/train.json'
__C.DATASET.POT.FRAME_RANGE = 501
__C.DATASET.POT.NUM_USE = 10000  #it means 100000 video pics A are generated from POT randomly first, there may have several subdatasets for training ,

__C.DATASET.POT_HOMO = CN()
__C.DATASET.POT_HOMO.ROOT = 'training_dataset/pot_homo/crop511'
__C.DATASET.POT_HOMO.ANNO = 'training_dataset/pot_homo/train.json'
__C.DATASET.POT_HOMO.FRAME_RANGE = 0
__C.DATASET.POT_HOMO.NUM_USE = 10000  #it means 100000 video pics A are generated from POT randomly first, there may have several subdatasets for training ,


__C.DATASET.POT_E2E = CN()
__C.DATASET.POT_E2E.ROOT = 'training_dataset/pot_e2e/crop511'
__C.DATASET.POT_E2E.ANNO = 'training_dataset/pot_e2e/train.json'
__C.DATASET.POT_E2E.FRAME_RANGE = 0
__C.DATASET.POT_E2E.IF_UNSUP = False
# __C.DATASET.POT_E2E.IF_UNSUP = True
# __C.DATASET.POT_E2E.UNSUPERVISED_FRAME_RANGE = 3
# __C.DATASET.POT_E2E.SUPERVISED_FRAME_RANGE = 0
# __C.DATASET.POT_E2E.SUPERVISED_NUM_USE = 5000
# __C.DATASET.POT_E2E.UNSUPERVISED_NUM_USE = 5000
__C.DATASET.POT_E2E.NUM_USE = 10000


__C.DATASET.POT_E2E_UNSUP = CN()
__C.DATASET.POT_E2E_UNSUP.ROOT = 'training_dataset/pot_e2e/crop511'
__C.DATASET.POT_E2E_UNSUP.ANNO = 'training_dataset/pot_e2e/train.json'
__C.DATASET.POT_E2E_UNSUP.FRAME_RANGE = 3
__C.DATASET.POT_E2E_UNSUP.IF_UNSUP = True
# __C.DATASET.POT_E2E_UNSUP.IF_UNSUP = False
# __C.DATASET.POT_E2E.UNSUPERVISED_FRAME_RANGE = 3
# __C.DATASET.POT_E2E.SUPERVISED_FRAME_RANGE = 0
# __C.DATASET.POT_E2E.SUPERVISED_NUM_USE = 5000
# __C.DATASET.POT_E2E.UNSUPERVISED_NUM_USE = 5000
__C.DATASET.POT_E2E_UNSUP.NUM_USE = 10000


__C.DATASET.POT_HOMO_VAL = CN()
__C.DATASET.POT_HOMO_VAL.ROOT = 'training_dataset/pot_homo/crop511'
__C.DATASET.POT_HOMO_VAL.ANNO = 'training_dataset/pot_homo/val.json'
__C.DATASET.POT_HOMO_VAL.FRAME_RANGE = 3
__C.DATASET.POT_HOMO_VAL.NUM_USE = 10000  #it means 100000 video pics A are generated from POT randomly first, there may have several subdatasets for training ,

__C.DATASET.YOUTUBEBB = CN()
__C.DATASET.YOUTUBEBB.ROOT = 'training_dataset/yt_bb/crop511'
__C.DATASET.YOUTUBEBB.ANNO = 'training_dataset/yt_bb/train.json'
__C.DATASET.YOUTUBEBB.FRAME_RANGE = 3
__C.DATASET.YOUTUBEBB.NUM_USE = 200000

__C.DATASET.COCO = CN()
__C.DATASET.COCO.ROOT = 'training_dataset/coco/crop511'
__C.DATASET.COCO.ANNO = 'training_dataset/coco/train2017.json'
__C.DATASET.COCO.FRAME_RANGE = 0
__C.DATASET.COCO.NUM_USE = 100000


__C.DATASET.COCO14 = CN()
__C.DATASET.COCO14.ROOT = 'training_dataset/coco14/crop511'
__C.DATASET.COCO14.ANNO = 'training_dataset/coco14/train2014.json'
__C.DATASET.COCO14.IF_UNSUP = False
__C.DATASET.COCO14.FRAME_RANGE = 0
__C.DATASET.COCO14.NUM_USE = 5000


__C.DATASET.DET = CN()
__C.DATASET.DET.ROOT = 'training_dataset/det/crop511'
__C.DATASET.DET.ANNO = 'training_dataset/det/train.json'
__C.DATASET.DET.FRAME_RANGE = 1
__C.DATASET.DET.NUM_USE = 200000

#GOT10K E2E  only setting different frame_range to train the perspective transformation.
__C.DATASET.GOT10K_E2E = CN()
__C.DATASET.GOT10K_E2E.ROOT = 'training_dataset/got_10k/crop511'
__C.DATASET.GOT10K_E2E.ANNO = 'training_dataset/got_10k/train.json'
__C.DATASET.GOT10K_E2E.FRAME_RANGE = 0 #origin100
__C.DATASET.GOT10K_E2E.IF_UNSUP = False
# __C.DATASET.GOT10K_E2E.NUM_USE = 200000
__C.DATASET.GOT10K_E2E.NUM_USE = 10000


__C.DATASET.GOT10K_E2E_UNSUP = CN()
__C.DATASET.GOT10K_E2E_UNSUP.ROOT = 'training_dataset/got_10k/crop511'
__C.DATASET.GOT10K_E2E_UNSUP.ANNO = 'training_dataset/got_10k/train.json'
__C.DATASET.GOT10K_E2E_UNSUP.FRAME_RANGE = 3 #origin100
__C.DATASET.GOT10K_E2E_UNSUP.IF_UNSUP = True
# __C.DATASET.GOT10K_E2E.NUM_USE = 200000
__C.DATASET.GOT10K_E2E_UNSUP.NUM_USE = 10000


__C.DATASET.GOT10K = CN()
__C.DATASET.GOT10K.ROOT = 'training_dataset/got_10k/crop511'
__C.DATASET.GOT10K.ANNO = 'training_dataset/got_10k/train.json'
__C.DATASET.GOT10K.FRAME_RANGE = 10 #origin100
__C.DATASET.GOT10K.NUM_USE = 200000
# __C.DATASET.GOT10K.NUM_USE = 10000  #it means 100000 video pics A are generated from POT randomly first, there may have several subdatasets for training ,
# __C.DATASET.GOT10K.FRAME_RANGE = 20 #origin100

__C.DATASET.GOT_HOMO = CN()
__C.DATASET.GOT_HOMO.ROOT = 'training_dataset/got_homo/crop511'
__C.DATASET.GOT_HOMO.ANNO = 'training_dataset/got_homo/train.json'
__C.DATASET.GOT_HOMO.FRAME_RANGE = 3 #origin100
__C.DATASET.GOT_HOMO.NUM_USE = 200000


__C.DATASET.LASOT = CN()
__C.DATASET.LASOT.ROOT = 'training_dataset/lasot/crop511'
__C.DATASET.LASOT.ANNO = 'training_dataset/lasot/train.json'
__C.DATASET.LASOT.FRAME_RANGE = 100
__C.DATASET.LASOT.NUM_USE = 200000

__C.DATASET.VIDEOS_PER_EPOCH = 1000000
# ------------------------------------------------------------------------ #
# Backbone options
# ------------------------------------------------------------------------ #
__C.BACKBONE = CN()

# Backbone type, current only support resnet18,34,50;alexnet;mobilenet
__C.BACKBONE.TYPE = 'res50'

__C.BACKBONE.KWARGS = CN(new_allowed=True)

# Pretrained backbone weights
__C.BACKBONE.PRETRAINED = ''

# whether use backbone weights
__C.BACKBONE.IF_PRETRAINED = False

# Train layers
__C.BACKBONE.TRAIN_LAYERS = ['layer2', 'layer3', 'layer4']

# Layer LR
__C.BACKBONE.LAYERS_LR = 0.1

# Switch to train layer
__C.BACKBONE.TRAIN_EPOCH = 10


# ------------------------------------------------------------------------ #
# Backbone_Homo options
# ------------------------------------------------------------------------ #
__C.BACKBONE_HOMO = CN()

# Backbone type, current only support resnet18,34,50;alexnet;mobilenet
__C.BACKBONE_HOMO.TYPE = 'res34'

__C.BACKBONE_HOMO.KWARGS = CN(new_allowed=True)

# Pretrained backbone weights
__C.BACKBONE_HOMO.PRETRAINED = ''

# whether use backbone weights
__C.BACKBONE_HOMO.IF_PRETRAINED = False

# Train layers
__C.BACKBONE_HOMO.TRAIN_LAYERS = ['layer2', 'layer3', 'layer4']

# Layer LR
__C.BACKBONE_HOMO.LAYERS_LR = 0.1

# Switch to train layer
__C.BACKBONE_HOMO.TRAIN_EPOCH = 10

# ------------------------------------------------------------------------ #
# Adjust layer options
# ------------------------------------------------------------------------ #
__C.ADJUST = CN()

# Adjust layer
__C.ADJUST.ADJUST = True

__C.ADJUST.KWARGS = CN(new_allowed=True)
__C.ADJUST.HOMO_KWARGS = CN(new_allowed=True)
# Adjust layer type
__C.ADJUST.TYPE = "AdjustAllLayer"

# ------------------------------------------------------------------------ #
# BAN options
# ------------------------------------------------------------------------ #
__C.BAN = CN()

# Whether to use ban head
__C.BAN.BAN = False

# BAN type
__C.BAN.TYPE = 'MultiBAN'

__C.BAN.KWARGS = CN(new_allowed=True)

# ------------------------------------------------------------------------ #
# BAN_LP options
# ------------------------------------------------------------------------ #
__C.BAN_LP = CN()

# Whether to use ban head
__C.BAN_LP.BAN = False

# BAN type
__C.BAN_LP.TYPE = 'MultiCircBAN'

__C.BAN_LP.KWARGS = CN(new_allowed=True)


#--------------------------------------------------------------------------#
# homo Correlation options
#--------------------------------------------------------------------------#
# HOMO_CORR options
__C.HOMO_CORR = CN()

# Whether to use ban head
__C.HOMO_CORR.CORR = False

# BAN type
__C.HOMO_CORR.TYPE = 'HomoCorr'

__C.HOMO_CORR.KWARGS = CN(new_allowed=True)

# ------------------------------------------------------------------------ #
# Point options
# ------------------------------------------------------------------------ #
__C.POINT = CN()

# Point stride
__C.POINT.STRIDE = 8

# Point LP
__C.POINT.STRIDE_LP = 8


# ------------------------------------------------------------------------ #
# Tracker options
# ------------------------------------------------------------------------ #
__C.TRACK = CN()

__C.TRACK.TYPE = 'hdnTracker'

# Scale penalty
__C.TRACK.PENALTY_K = 0.14

__C.TRACK.PENALTY_K_LP = 0.14

#Search enlarge
__C.TRACK.BASE_SC_FAC = 2.0

__C.TRACK.SCALE_SCORE_THRESH = 0.5

# Window influence
__C.TRACK.WINDOW_INFLUENCE = 0.45

# Interpolation learning rate
__C.TRACK.LR = 0.30

# Interpolation learning rate
__C.TRACK.LR_LP = 0.60

# Exemplar size
__C.TRACK.EXEMPLAR_SIZE = 127

# Instance size
__C.TRACK.INSTANCE_SIZE = 255

# Shallow size
__C.TRACK.SHALLOW_SIZE = 127

# Base size
__C.TRACK.BASE_SIZE = 8

# Context amount
__C.TRACK.CONTEXT_AMOUNT = 0.5
__C.TRACK.CONTEXT_MUL = 2.0
__C.TRACK.HOMO_CONTEXT = 1.0

