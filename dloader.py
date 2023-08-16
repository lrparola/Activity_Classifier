from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from configs import constants as _C
from lib.utils.data_utils import normalize_angle, process_imu_data, normalize_new_data

import torch
import numpy as np
from skimage.util.shape import view_as_windows

from pdb import set_trace as st

class Normalizer:
    def __init__(self, norm_dict):
        self.x_mean = norm_dict['params']['x_mean'].detach().cpu()
        self.x_std = norm_dict['params']['x_std'].detach().cpu()
        self.y_mean = norm_dict['params']['y_mean'].detach().cpu()
        self.y_std = norm_dict['params']['y_std'].detach().cpu()

    def normalize_input(self, x):
        return (x - self.x_mean) / self.x_std

    def normalize_output(self, y):
        return (y - self.y_mean) / self.y_std

    def unnormalize_input(self, x):
        return x * self.x_std + self.x_mean
    
    def unnormalize_output(self, y):
        return y * self.y_std + self.y_mean
        
class Dataset(torch.utils.data.Dataset):
    def __init__(self, label_pth, joint, norm_dict, is_train=True,is_remote=False, **kwargs):
        super(Dataset, self).__init__()

        self.labels = torch.load(label_pth)
        self.joint = joint
        self.is_train = is_train
        self.is_remote = is_remote

        self.normalizer = Normalizer(norm_dict)
        self.prepare_sequence_batch(kwargs.get('input_length', 400))
        self.kwargs =kwargs


    def prepare_sequence_batch(self, seq_length):
        self.seq_indices = []
        seqs = self.labels['seq']

        seqs_unique, group = np.unique(
            seqs, return_index=True)
        perm = np.argsort(group)
        group_perm = group[perm]
        indices = np.split(
            np.arange(0, seqs.shape[0]), group_perm[1:]
        )
        
        for idx in range(len(seqs_unique)):
            indexes = indices[idx]
            if self.is_train:
                if indexes.shape[0] < seq_length: continue
                chunks = view_as_windows(
                    indexes, (seq_length), step=seq_length // 4
                )
                start_finish = chunks[:, (0, -1)].tolist()
                self.seq_indices += start_finish
            else:
                self.seq_indices += indexes[None, (0, -1)].tolist()


    def __len__(self):
        return len(self.seq_indices)

    def __getitem__(self, index):
        if self.is_remote == True:
            x = self.get_single_imu_sequence(index,self.kwargs)
        else:
            x = self.get_single_sequence(index,self.kwargs)
        return x
    def get_single_imu_sequence(self, index,kwargs):
        start_index, end_index = self.seq_indices[index]
        all_imu = self.labels['imu'][start_index:end_index+1]
        

        if self.is_train == False:
            activity_label = self.labels['label'][index]
        else:
            activity_label = 0
        imus, angles = [], []
        
        for side in ['left', 'right']:
            trg_joint = side[0] + self.joint.lower()
            temp_imu = []
            for sensor in _C.DATA.JOINT_IMU_MAPPER[trg_joint]:
                temp_imu.append(all_imu[:, _C.DATA.IMU_LIST.index(sensor)])    

            if kwargs['validation_norm_dict'] == True:
                self.norm_dict = normalize_new_data(*temp_imu)
                
            imu = process_imu_data(*temp_imu) #, self.norm_dict)
            imu = self.normalizer.normalize_input(imu)
            imus.append(imu)


        imus = torch.stack(imus)


        return imus,activity_label
        
    def get_single_sequence(self, index,kwargs):
        start_index, end_index = self.seq_indices[index]
        all_imu = self.labels['imu'][start_index:end_index+1]
        all_angle = self.labels['angle'][start_index:end_index+1]

        if self.is_train == False:
            activity_label = self.labels['label'][index]
        else:
            activity_label = 0
        imus, angles = [], []
        
        for side in ['left', 'right']:
            trg_joint = side[0] + self.joint.lower()
            temp_imu = []
            for sensor in _C.DATA.JOINT_IMU_MAPPER[trg_joint]:
                temp_imu.append(all_imu[:, _C.DATA.IMU_LIST.index(sensor)])    

            if kwargs['validation_norm_dict'] == True:
                self.norm_dict = normalize_new_data(*temp_imu)
                
            imu = process_imu_data(*temp_imu) #, self.norm_dict)
            imu = self.normalizer.normalize_input(imu)
            imus.append(imu)

            angle = all_angle[:, _C.DATA.JOINT_LIST.index(trg_joint)]
            angle = self.normalizer.normalize_output(angle)
            # angle = normalize_angle(angle, self.norm_dict)
            angles.append(angle)

        imus = torch.stack(imus)
        angles = torch.stack(angles)

        return imus, angles,activity_label


def setup_validation_data(norm_dict=None,
                          joint=None,
                          batch_size=None,
                          **kwargs):
    n_workers = 0

    train_dataset = Dataset(_C.PATHS.TRAIN_DATA_LABEL, joint, norm_dict, is_train=True, **kwargs)
    test_dataset = Dataset(_C.PATHS.TEST_DATA_LABEL, joint, norm_dict, is_train=False, **kwargs)
    train_dloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size // 2,
        num_workers=n_workers,
        shuffle=True,
        pin_memory=True,
    )
    
    test_dloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=n_workers,
        shuffle=False,
        pin_memory=True,
    )
    
    return train_dloader, test_dloader
    
    
    
def setup_confirmation_data(norm_dict=None,
                          joint=None,
                          batch_size=None,
                          **kwargs):
    n_workers = 0

    all_dataset = Dataset(_C.PATHS.TEST_DATA_LABEL, joint, norm_dict, is_train=False, **kwargs)

    # acl_dataset = Dataset(_C.PATHS.ACL_DATA_LABEL, joint, norm_dict, is_train=False, **kwargs)
    # healthy_dataset = Dataset(_C.PATHS.HEALTHY_DATA_LABEL, joint, norm_dict, is_train=False, **kwargs)
    all_dloader = torch.utils.data.DataLoader(
        all_dataset,
        batch_size=1,
        num_workers=n_workers,
        shuffle=False,
        pin_memory=True,
    )
    
    # acl_dloader = torch.utils.data.DataLoader(
        # acl_dataset,
        # batch_size=1,
        # num_workers=n_workers,
        # shuffle=False,
        # pin_memory=True,
    # )
    
    # healthy_dloader = torch.utils.data.DataLoader(
        # healthy_dataset,
        # batch_size=1,
        # num_workers=n_workers,
        # shuffle=False,
        # pin_memory=True,
    # )
    
    return all_dloader#, acl_dloader,healthy_dloader
    

def setup_overground_data(norm_dict=None,joint=None, batch_size=None,**kwargs):
    n_workers = 0

    all_dataset = Dataset(_C.PATHS.EXTRA_SUBJECTS, joint, norm_dict, is_train=False,is_remote=True, **kwargs)

    all_dloader = torch.utils.data.DataLoader(
        all_dataset,
        batch_size=1,
        num_workers=n_workers,
        shuffle=False,
        pin_memory=True,
    )
    

    return all_dloader#, acl_dloader,healthy_dloader   
    
    
def setup_remote_data(norm_dict=None,
                          joint=None,
                          batch_size=None,
                          **kwargs):
    n_workers = 0

    all_dataset = Dataset(_C.PATHS.ALL_DATA_LABEL, joint, norm_dict, is_train=False,is_remote=True, **kwargs)
    print(len(all_dataset))

    all_dloader = torch.utils.data.DataLoader(
        all_dataset,
        batch_size=1,
        num_workers=n_workers,
        shuffle=False,
        pin_memory=True,
    )
    
    
    
    return all_dloader
    
    
    
  

