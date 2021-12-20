from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from hdn.models.head.ban import UPChannelBAN, DepthwiseBAN, MultiBAN
from hdn.models.head.ban_lp import DepthwiseCircBAN, MultiCircBAN

BANS = {
        'UPChannelBAN': UPChannelBAN,
        'DepthwiseBAN': DepthwiseBAN,
        'MultiBAN': MultiBAN,
        'DepthwiseCircBAN': DepthwiseCircBAN,
        'MultiCircBAN': MultiCircBAN,
       }


def get_ban_head(name, **kwargs):
    return BANS[name](**kwargs)

