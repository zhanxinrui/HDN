import numpy as np

from colorama import Style, Fore

from ..utils import overlap_ratio, success_overlap, success_error,success_4pts_error, success_poly_overlap, success_centroid_error, robust_poly_overlap

class HomoBenchmark:
    """
    Args:
        result_path: result path of your tracker
                should the same format like VOT
    """
    def __init__(self, dataset):
        self.dataset = dataset
    def convert_poly_to_center(self, poly):
        return np.array((poly[0,:] + poly[2,:])/2,
                        (poly[1,:] + poly[3,:])/2).T # use T?
    def convert_points_to_poly(self, points):
        return np.array(points).reshape(4,2)
    def convert_points_to_center(self, points):
        '''
        k1=(p0-p4)/(p1-p5);
        k2=(p3-p7)/(p2-p6);

        x=(k1*p0-k2*p2+p5-p1)/(k1-k2);
        y=p1+(x-p0)*k1;
        '''

        if (points[1] - points[5]) <= 0.00005 and points[2]-points[6] <=0.00005 :
            x = (points[0]+points[2] + points[4]+points[6])/4
            y = (points[1]+points[3] + points[5]+points[7])/4
        elif (points[1] - points[5]) <= 0.00005:
            k2 = (points[3] - points[7]) / (points[2] - points[6])
            x = (points[0] + points[2])/2
            y = k2 * x + points[3] - k2 * points[2]
            # return np.array([x,y])
        elif  (points[2]-points[6]) <=0.00005:
            k1 = (points[0] - points[4]) / (points[1] - points[5])
            x = (points[0] + points[2])/2
            y = k1 * x + points[3] - k1 * points[2]
            # return np.array([x,y])
        else:
            k1 = (points[0] - points[4]) / (points[1] - points[5])
            k2 = (points[3] - points[7]) / (points[2] - points[6])
            if k1 == k2:
                x = (points[0]+points[2] + points[4]+points[6])/4
                y = (points[1]+points[3] + points[5]+points[7])/4

            if k1 != k2:
                x = (k1*points[0]-k2*points[2]+points[5]-points[1]) / (k1-k2)
                y = points[1] + (x-points[0])*k1
        return np.array([x,y])

    def convert_points_to_mean_center(self, points):
        x = (points[:,0] + points[:,2] + points[:,4] + points[:,6]) / 4
        y = (points[:,1] + points[:,3] + points[:,5] + points[:,7]) / 4
        return np.array([x,y]).T

    def convert_points_to_bbox(self, points):
        try:
            points = np.array(points).reshape(-1, 4,2)
        except:
            print('points', points)
        points_min = np.min(points, axis=1)
        points_max = np.max(points, axis=1)
        points_w_h = (points_max - points_min)
        points = np.concatenate([points_min, points_w_h], axis=1)
        return points
    def eval_bbox_overlap_success(self, eval_trackers=None):
        """
        Args:
            eval_trackers: list of tracker name or single tracker name
        Return:
            res: dict of results
        """
        if eval_trackers is None:
            eval_trackers = self.dataset.tracker_names
        if isinstance(eval_trackers, str):
            eval_trackers = [eval_trackers]

        success_ret = {}
        for tracker_name in eval_trackers:
            success_ret_ = {}
            for video in self.dataset:
                gt_traj = self.convert_points_to_bbox(video.gt_traj)
                if tracker_name not in video.pred_trajs:
                    tracker_traj = video.load_tracker(self.dataset.tracker_path,
                                                      tracker_name, False)
                    tracker_traj = np.array(tracker_traj)
                    tracker_traj = self.convert_points_to_bbox(tracker_traj)

                else:
                    tracker_traj = self.convert_points_to_bbox(video.pred_trajs[tracker_name])
                if hasattr(video, 'absent'):
                    gt_traj = gt_traj[video.absent == 1]
                    tracker_traj = tracker_traj[video.absent == 1]
                if tracker_traj.shape[0] < gt_traj.shape[0]:
                    zero_padd = np.zeros([gt_traj.shape[0]-tracker_traj.shape[0], 4])
                    tracker_traj =  np.concatenate([tracker_traj, zero_padd], axis=0)
                if hasattr(video, 'val_ids'):
                    gt_traj = np.array(gt_traj)[video.val_ids]
                    tracker_traj = np.array(tracker_traj)[video.val_ids]
                n_frame = len(gt_traj)
                success_ret_[video.name] = success_overlap(gt_traj, tracker_traj, n_frame)
            success_ret[tracker_name] = success_ret_
        return success_ret

    def eval_poly_overlap_success(self, eval_trackers=None):
        """
        Args:
            eval_trackers: list of tracker name or single tracker name
        Return:
            res: dict of results
        """
        if eval_trackers is None:
            eval_trackers = self.dataset.tracker_names
        if isinstance(eval_trackers, str):
            eval_trackers = [eval_trackers]

        success_ret = {}
        for tracker_name in eval_trackers:
            success_ret_ = {}
            for video in self.dataset:
                gt_traj = np.array(video.gt_traj)
                if tracker_name not in video.pred_trajs:
                    tracker_traj = video.load_tracker(self.dataset.tracker_path,
                                                      tracker_name, False)
                    tracker_traj = np.array(tracker_traj)

                else:
                    tracker_traj = np.array(video.pred_trajs[tracker_name])
                if tracker_traj.shape[0] < gt_traj.shape[0]:
                    zero_padd = np.zeros([gt_traj.shape[0]-tracker_traj.shape[0], 8])
                    tracker_traj =  np.concatenate([tracker_traj, zero_padd], axis=0)

                if hasattr(video, 'val_ids'):
                    gt_traj = gt_traj[video.val_ids]
                    tracker_traj = tracker_traj[video.val_ids]
                gt_traj = np.array(gt_traj)
                n_frame = gt_traj.shape[0]
                if hasattr(video, 'absent'):
                    gt_traj = gt_traj[video.absent == 1]
                    tracker_traj = tracker_traj[video.absent == 1]
                success_ret_[video.name] = success_poly_overlap(gt_traj, tracker_traj, n_frame)
            success_ret[tracker_name] = success_ret_
        return success_ret



    def eval_robustness(self, eval_trackers=None):
        """
        Args:
            eval_trackers: list of tracker name or single tracker name
        Return:
            res: dict of results
        """
        if eval_trackers is None:
            eval_trackers = self.dataset.tracker_names
        if isinstance(eval_trackers, str):
            eval_trackers = [eval_trackers]

        success_ret = {}
        for tracker_name in eval_trackers:
            success_ret_ = {}
            for video in self.dataset:
                gt_traj = np.array(video.gt_traj)
                if tracker_name not in video.pred_trajs:
                    tracker_traj = video.load_tracker(self.dataset.tracker_path,
                                                      tracker_name, False)
                    tracker_traj = np.array(tracker_traj)
                else:
                    tracker_traj = np.array(video.pred_trajs[tracker_name])
                if tracker_traj.shape[0] < gt_traj.shape[0]:
                    zero_padd = np.zeros([gt_traj.shape[0]-tracker_traj.shape[0], 8])
                    tracker_traj =  np.concatenate([tracker_traj, zero_padd], axis=0)
                gt_traj = np.array(gt_traj)
                n_frame = gt_traj.shape[0]
                if hasattr(video, 'absent'):
                    gt_traj = gt_traj[video.absent == 1]
                    tracker_traj = tracker_traj[video.absent == 1]
                success_ret_[video.name] = robust_poly_overlap(gt_traj, tracker_traj, n_frame)
            success_ret[tracker_name] = success_ret_
        return success_ret

    def eval_4pts_precision(self, eval_trackers=None):
        """
        Args:
            eval_trackers: list of tracker name or single tracker name
        Return:
            res: dict of results
        """
        # pass
        if eval_trackers is None:
            eval_trackers = self.dataset.tracker_names
        if isinstance(eval_trackers, str):
            eval_trackers = [eval_trackers]

        precision_ret = {}
        for tracker_name in eval_trackers:
            print('tracker_name:', tracker_name)
            precision_ret_ = {}
            for video in self.dataset:
                gt_traj = np.array(video.gt_traj)

                if tracker_name not in video.pred_trajs:
                    tracker_traj = video.load_tracker(self.dataset.tracker_path,
                                                      tracker_name, False)
                    tracker_traj = np.array(tracker_traj)
                else:
                    tracker_traj = np.array(video.pred_trajs[tracker_name])
                thresholds = np.arange(0, 51, 1)
                if tracker_traj.shape[0] < gt_traj.shape[0]:
                    zero_padd = np.zeros([gt_traj.shape[0]-tracker_traj.shape[0], 8])
                    tracker_traj =  np.concatenate([tracker_traj, zero_padd], axis=0)
                if hasattr(video, 'val_ids'):
                    gt_traj = np.array(gt_traj)[video.val_ids]
                    tracker_traj = np.array(tracker_traj)[video.val_ids]
                gt_traj = gt_traj[1:]
                tracker_traj = tracker_traj[1:]
                gt_traj = np.array(gt_traj)
                n_frame = gt_traj.shape[0]
                precision_ret_[video.name] = success_4pts_error(gt_traj, tracker_traj,
                                                                thresholds, n_frame)
            precision_ret[tracker_name] = precision_ret_
        return precision_ret

    def eval_centroid_precision(self, eval_trackers=None):
        """
        Args:
            eval_trackers: list of tracker name or single tracker name
        Return:
            res: dict of results
        """
        if eval_trackers is None:
            eval_trackers = self.dataset.tracker_names
        if isinstance(eval_trackers, str):
            eval_trackers = [eval_trackers]

        precision_ret = {}
        for tracker_name in eval_trackers:
            print('tracker_name', tracker_name)
            precision_ret_ = {}
            for video in self.dataset:
                gt_traj = np.array(video.gt_traj)
                if tracker_name not in video.pred_trajs:
                    tracker_traj = video.load_tracker(self.dataset.tracker_path,
                                                      tracker_name, False)
                    tracker_traj = np.array(tracker_traj)
                else:
                    tracker_traj = np.array(video.pred_trajs[tracker_name])
                if tracker_traj.shape[0] < gt_traj.shape[0]:
                    zero_padd = np.zeros([gt_traj.shape[0]-tracker_traj.shape[0], 8])
                    tracker_traj =  np.concatenate([tracker_traj, zero_padd], axis=0)
                if hasattr(video, 'val_ids'):
                    gt_traj = np.array(gt_traj)[video.val_ids]
                    tracker_traj = np.array(tracker_traj)[video.val_ids]
                gt_traj = np.array(gt_traj)
                n_frame = gt_traj.shape[0]
                thresholds = np.arange(0, 51, 1)
                precision_ret_[video.name] = success_centroid_error(gt_traj, tracker_traj,
                                                                    thresholds, n_frame)
            precision_ret[tracker_name] = precision_ret_
        return precision_ret


    def eval_center_norm_precision(self, eval_trackers=None):
        """
        Args:
            eval_trackers: list of tracker name or single tracker name
        Return:
            res: dict of results
        """
        if eval_trackers is None:
            eval_trackers = self.dataset.tracker_names
        if isinstance(eval_trackers, str):
            eval_trackers = [eval_trackers]

        norm_precision_ret = {}
        for tracker_name in eval_trackers:
            norm_precision_ret_ = {}
            for video in self.dataset:
                gt_traj = np.array(video.gt_traj)
                if tracker_name not in video.pred_trajs:
                    tracker_traj = video.load_tracker(self.dataset.tracker_path,
                                                      tracker_name, False)
                    tracker_traj = np.array(tracker_traj)
                else:
                    tracker_traj = np.array(video.pred_trajs[tracker_name])
                n_frame = len(gt_traj)
                if hasattr(video, 'absent'):
                    gt_traj = gt_traj[video.absent == 1]
                    tracker_traj = tracker_traj[video.absent == 1]
                gt_center_norm = self.convert_bb_to_norm_center(gt_traj, gt_traj[:, 2:4])
                tracker_center_norm = self.convert_bb_to_norm_center(tracker_traj, gt_traj[:, 2:4])
                thresholds = np.arange(0, 51, 1) / 100
                norm_precision_ret_[video.name] = success_error(gt_center_norm,
                                                                tracker_center_norm, thresholds, n_frame)
            norm_precision_ret[tracker_name] = norm_precision_ret_
        return norm_precision_ret

    def show_result(self, success_ret=None, precision_ret=None,
                    precision_c_ret=None, show_video_level=False, helight_threshold=0.6):
        """pretty print result
        Args:
            result: returned dict from function eval
        """
        # sort tracker
        tracker_auc = {}
        for tracker_name in success_ret.keys():
            auc = np.mean(list(success_ret[tracker_name].values()))
            tracker_auc[tracker_name] = auc
        tracker_auc_ = sorted(tracker_auc.items(),
                              key=lambda x:x[1],
                              reverse=True)[:20]
        tracker_names = [x[0] for x in tracker_auc_]


        tracker_name_len = max((max([len(x) for x in success_ret.keys()])+2), 12)

        header = ("|{:^"+str(tracker_name_len)+"}|{:^16}|{:^9}|{:^16}|{:^16}|{:^16}|{:^16}").format(
            "Tracker name",  "avg Precision", "Success", "avg centPrec","Precision(e<5)", "Precision(e<10)", "Precsion(e<20)")
        formatter = "|{:^"+str(tracker_name_len)+"}|{:^16.3f}|{:^9.3f}|{:^16.3f}|{:^16.3f}|{:^16.3f}|{:^16.3f}"
        print('-'*len(header))
        print(header)
        print('-'*len(header))
        for tracker_name in tracker_names:
            success = tracker_auc[tracker_name]
            if precision_ret is not None:
                precision_err5 = np.mean(list(precision_ret[tracker_name].values()), axis=0)[5]
                precision_err10 = np.mean(list(precision_ret[tracker_name].values()), axis=0)[10]
                precision_err20 = np.mean(list(precision_ret[tracker_name].values()), axis=0)[20]
                average_precision = np.mean(np.mean(list(precision_ret[tracker_name].values()), axis=0))
            else:
                precision_err5, precision_err10, precision_err20, average_precision = 0, 0, 0, 0
            if precision_c_ret is not None:
                average_centroid_precision = np.mean(np.mean(list(precision_c_ret[tracker_name].values()), axis=0))
            else:
                average_centroid_precision = 0
            print(formatter.format(tracker_name, average_precision, success, average_centroid_precision,precision_err5, precision_err10, precision_err20))
        print('-'*len(header))
