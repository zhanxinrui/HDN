from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# from hdn.models.head.ban import UPChannelBAN, DepthwiseBAN, MultiBAN
# from hdn.models.head.ban_lp import DepthwiseCircBAN, MultiCircBAN
# from homo_estimator.Deep_homography.Oneline_DLTv1.head.homo_head import
from homo_estimator.Deep_homography.Oneline_DLTv1.preprocess.input_feature_extractor import  PreShareFeature
from homo_estimator.Deep_homography.Oneline_DLTv1.preprocess.input_mask_generator import  MaskGenerator
# from test_ideas.net.unet import  UNet as PreShareFeature_v2
# from test_ideas.net.unet import


"""
preprocessing part of homo-estimator, including the preShareFeature extractor and MaksGenerator.
"""
head = {
    'PreShareFeature': PreShareFeature,
    'MaskGenerator': MaskGenerator,
    # 'PreShareFeature_v2': PreShareFeature_v2,
}


def get_pre(name, **kwargs):
    return head[name](**kwargs)

