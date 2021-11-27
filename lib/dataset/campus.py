import os
import pickle 
import json
from collections import OrderedDict

import numpy as np
import scipy.io as scio
import torch

#https://github.com/jiahaoLjh/PlaneSweepPose/tree/35d8bc32155dc0488c0a8b37de1afa8a5069117e
# TODO - THIS IS ONLY TO UNDERSTAND HOW THE GENERATED POSES WORK

campus_joints_dict = {
    'Right-Ankle': 0,
    'Right-Knee': 1,
    'Right-Hip': 2,
    'Left-Hip': 3,
    'Left-Knee': 4,
    'Left-Ankle': 5,
    'Right-Wrist': 6,
    'Right-Elbow': 7,
    'Right-Shoulder': 8,
    'Left-Shoulder': 9,
    'Left-Elbow': 10,
    'Left-Wrist': 11,
    'Bottom-Head': 12,
    'Top-Head': 13
}

campus_bones_dict = [
    [13, 12],  # head
    [12, 9], [9, 10], [10, 11],  # left arm
    [12, 8], [8, 7], [7, 6],  # right arm
    [9, 3], [8, 2],  # trunk
    [3, 4], [4, 5],  # left leg
    [2, 1], [1, 0],  # right leg
]

coco_joints_dict = {
    0: 'nose',
    1: 'Leye', 2: 'Reye',
    3: 'Lear', 4: 'Rear',
    5: 'Lsho', 6: 'Rsho',
    7: 'Lelb', 8: 'Relb',
    9: 'Lwri', 10: 'Rwri',
    11: 'Lhip', 12: 'Rhip',
    13: 'Lkne', 14: 'Rkne',
    15: 'Lank', 16: 'Rank',
}
coco_bones_dict = [
    [0, 1], [0, 2], [1, 3], [2, 4],  # head
    [3, 5], [5, 7], [7, 9],  # left arm
    [4, 6], [6, 8], [8, 10],  # right arm
    [5, 11], [6, 12],  # trunk
    [11, 13], [13, 15],  # left leg
    [12, 14], [14, 16],  # right leg
]

class Campus(torch.utils.data.Dataset):
    def __init__(self, config, image_set, is_train):
        super().__init__()
        
        self.num_joints = len(campus_joints_dict)
        self.num_joints_coco = len(coco_joints_dict)
        self.cam_list = [0, 1, 2]
        self.num_views = 3
        self.frame_range = list(range(350, 471)) + list(range(650, 751))
        
        self.is_train = is_train
        
        self.dataset_root = os.path.join(os.path.dirname(__file__), "../../data/Campus")
        self.image_set = image_set
        self.image_width = 360
        self.image_height = 288
        
        self.max_num_persons = 5
        
        self.pred_pose2d = self._get_pose2d()
        self.cameras = self._get_cam()
        self.db = self._get_db()
        
    def _get_pose2d(self):
        file_path = os.path.join(self.dataset_root, "pred_campus_maskrcnn_hrnet_coco.pkl")
        with open(file_path, "rb") as f:
            pred_2d = pickle.load(f)
        return pred_2d
    
    def _get_cam(self):
        cam_file = os.path.join(self.dataset_root, "calibration_campus.json")
        with open(cam_file, "r") as f:
            cameras = json.load(f)
            
        for cam_id, cam in cameras.items():
            for k,v in cam.items():
                cameras[cam_id][k] = np.array(v)
            cameras[cam_id]["id"] = cam_id
            
        return cameras
    
    def _get_db(self):
        # TODO-This will probably be replaced by graph generation function
        # we prbably need some function in utils.py to generate graphs 
        # do other image processing related functionalities
        pass

        #return db
    
    def __len__(self):
        return len(self.db)
    
    def __getitem__(self, idx):
        pass
    