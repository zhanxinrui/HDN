#Copyright 2021, XinruiZhan
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from hdn.core.config import cfg
from hdn.tracker.hdn_tracker import hdnTracker
from hdn.tracker.hdn_tracker_proj_e2e import hdnTrackerHomo as hdnTrackerHomoProje2e


TRACKS = {
        'hdnTracker': hdnTracker,
        'hdnTrackerHomoProje2e': hdnTrackerHomoProje2e,
        }


def build_tracker(model):
    return TRACKS[cfg.TRACK.TYPE](model)
