import json
import os
import numpy as np

from PIL import Image
from tqdm import tqdm
from glob import glob

from .dataset import Dataset
from .video import Video

class POTVideo(Video):
    """
    Args:
        name: video name
        root: dataset root
        video_dir: video directory
        init_rect: init rectangle
        img_names: image names
        gt_rect: groundtruth rectangle
        flag: indicates the status of each frame.
            (0:normal; 1:occluded for more than half; 2: out of view for more than half; 3: heavily blurred.)
        homo: homography transformation (3x3 matrix) that projects the initial four reference points to a given frame.

    """
    def __init__(self,name,root,video_dir,init_rect,img_names,gt_rect,flag,homo,load_img=False, eval_mode=False):
        super(POTVideo,self).__init__(name,root,video_dir,init_rect,img_names,gt_rect,load_img=load_img)
        self.tags={}
        self.tags['flag'] = flag
        self.tags['homo'] = homo
        if eval_mode:
            self.gt_traj = [x for i,x in enumerate(self.gt_traj) if i==0 or i%2==1]
        self.val_ids = np.where(np.array(self.tags['flag']) == '0')

    def load_tracker(self, path, tracker_names=None, store=True):
        """
        eval the results
        :param path: path to result
        :param tracker_names: name of tracker
        :param store:
        :return:
        """
        if not tracker_names:
            tracker_names = [x.split('/')[-1] for x in glob(path)
                    if os.path.isdir(x)]
        if isinstance(tracker_names, str):
            tracker_names = [tracker_names]
        for name in tracker_names:
            # traj_file = os.path.join(path,name,self.name+'.txt')
            traj_file = os.path.join(path, name,self.name+'_'+name+'.txt')
            if os.path.exists(traj_file):
                with open(traj_file, 'r') as f :
                    # pred_traj = [list(map(float, x.strip().split(',')))
                    #         for x in f.readlines()]
                    pred_traj = [list(map(float, x.strip().split(' ')))
                                 for x in f.readlines()]

                    pred_traj = [x for i,x in enumerate(pred_traj) if i==0 or i%2==1]
                    if len(pred_traj) != len(self.gt_traj):
                        print('pred_traj != gt_traj',name, len(pred_traj), len(self.gt_traj), self.name)
                    #remove the blank gt
                    if store:
                        self.pred_trajs[name] = pred_traj
                    else:
                        return pred_traj
            else:
                print(traj_file)
        self.tracker_names = list(self.pred_trajs.keys())

class POTDataset(Dataset):
    def __init__(self, name, dataset_root, load_img=False, eval_mode=False):
        super(POTDataset,self).__init__(name, dataset_root)
        print('json addr', os.path.join(dataset_root,name+'.json'))
        with open(os.path.join(dataset_root,name+'.json'),'r') as f:
            meta_data = json.load(f)
        # load videos
        pbar = tqdm(meta_data.keys(), desc='loading '+name, ncols=100)
        self.videos = {}
        for video in pbar:
            pbar.set_postfix_str(video)
            self.videos[video] = POTVideo(video,
                                          dataset_root,
                                          meta_data[video]['video_dir'],
                                          meta_data[video]['init_rect'],
                                          meta_data[video]['img_names'],
                                          meta_data[video]['gt_rect'],
                                          meta_data[video]['flag'],
                                          meta_data[video]['homography'],
                                          load_img, eval_mode=eval_mode)
        self.tags = ['flag', 'homo']
        # set attr
        attr = []
        for x in self.videos.values():
            attr.append(x.attr)
        attr = set(attr)
        self.attr = {}
        self.attr['ALL'] = list(self.videos.keys())
        for x in attr:
            self.attr[x] = []
        for k, v in self.videos.items():
            for attr_ in v.attr:
                self.attr[attr_].append(k)

