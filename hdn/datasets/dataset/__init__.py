#Copyright 2021, XinruiZhan
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
# from hdn.datasets.dataset.dataset import BANDataset as simi_aug_dataset# similarity augmentation on objects that nothing changed
from hdn.datasets.dataset.unconstrained_v2_dataset import BANDataset as unconstrained_v2_dataset # homo on supervised or simi/homo on unsupervised
from hdn.datasets.dataset.dataset import  SubDataset

DATASETS = {
    # 'simi_aug_dataset': simi_aug_dataset,
    'unconstrained_v2_dataset':unconstrained_v2_dataset,
    'sub_dataset': SubDataset
}


def get_dataset(name, **kwargs):
    return DATASETS[name](**kwargs)
