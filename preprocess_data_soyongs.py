import os
import time
import os.path as osp
from glob import glob
from collections import defaultdict

import torch
import joblib
import numpy as np
from tqdm import tqdm

from constants import RAW_DATA_DIR, PROC_DATA_DIR

activity_mapper = {
    'hopping': 0,
    'high_jump': 0,
    'walking': 1,
    'running': 2,
    'ramp_ascending': 3,
    'ramp_descending': 4,
    'stair_ascending': 5,
    'stair_descending': 6,
    'sit_to_stand': 7,
}
sensor_list = ['lshank','rshank','lthigh','rthigh','sacrum']

def preprocess(is_train):
    if is_train:
        subj_list = [f'S{i:02d}' for i in range(1,65)]
    else:
        subj_list = [f'S{i:02d}' for i in range(66,82)]
    
    dataset = defaultdict(list)
    for subj in (sbar := tqdm(sorted(subj_list))):
        sbar.set_description_str(f'Subj: {subj}')
        activity_list = os.listdir(osp.join(RAW_DATA_DIR, subj))
        activity_list = [activity for activity in activity_list if not 'whole' in activity]
        
        for activity in (abar := tqdm(sorted(activity_list), leave=False)):
            abar.set_description_str(f'{activity}')
            
            # ======= Load label ======= #
            label = [val for key, val in activity_mapper.items() if key in activity]
            if len(label) != 1:
                continue
            label = label[0]
            
            # ======= Load IMUs ======= #
            temp_imus = []
            for sensor in sensor_list:
                acc = np.load(osp.join(RAW_DATA_DIR, subj, activity, f'{sensor}_acc.npy'))
                gyr = np.load(osp.join(RAW_DATA_DIR, subj, activity, f'{sensor}_gyr.npy'))
                imu = np.concatenate((acc, gyr), axis=-1)
                temp_imus.append(imu)
            
            temp_imus = torch.from_numpy(np.concatenate(temp_imus, axis=-1)).float()
            crop_l = temp_imus.shape[0] // 10
            temp_imus = temp_imus[crop_l:-crop_l]
            dataset['imu'].append(temp_imus)
            
            # ======= Append data ======= #
            dataset['label'].append(torch.tensor([label] * len(temp_imus)))
            dataset['seq'].append(np.array([f'{subj}_{activity}'] * len(temp_imus)))
    
    if is_train:
        for key, val in dataset.items():
            if isinstance(val, torch.Tensor): 
                dataset[key] = torch.cat(val)
            else:
                dataset[key] = np.concatenate(val)
    
    os.makedirs(PROC_DATA_DIR, exist_ok=True)
    fname = 'train_label.pth' if is_train else 'test_label.pth'
    joblib.dump(dataset, osp.join(PROC_DATA_DIR, fname))

if __name__ == '__main__':
    # preprocess(True)
    preprocess(False)
