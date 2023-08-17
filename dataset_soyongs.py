import os.path as osp

import torch
import joblib
import numpy as np
from skimage.util.shape import view_as_windows

import constants as _C

class Dataset(torch.utils.data.Dataset):
    def __init__(self, is_train=True, n_frames=100):
        self.is_train = is_train
        fname = 'train_label.pth' if is_train else 'test_label.pth'
        self.labels = joblib.load(osp.join(_C.PROC_DATA_DIR, fname))
        self.n_frames = n_frames
        
        if is_train: 
            self.prepare_seq_batch()
            _, self.counts = np.unique(self.labels['label'], return_counts=True)
            self.counts = self.counts // self.n_frames
        
    def prepare_seq_batch(self):
        self.seq_indices = []
        seq_name = self.labels['seq']
        
        sequence_names_unique, group = np.unique(seq_name, return_index=True)
        perm = np.argsort(group)
        group_perm = group[perm]
        indices = np.split(
            np.arange(0, self.labels['seq'].shape[0]), group_perm[1:]
        )
        for idx in range(len(sequence_names_unique)):
            indexes = indices[idx]
            if indexes.shape[0] < self.n_frames: continue
            chunks = view_as_windows(
                indexes, (self.n_frames), step=self.n_frames
            )
            start_finish = chunks[:, (0, -1)].tolist()
            self.seq_indices += start_finish
            
    def __len__(self):
        if self.is_train:
            return len(self.seq_indices)
        else:
            return len(self.labels['imu'])
        
    def __getitem__(self, index):
        if self.is_train: 
            return self.get_train_data(index)
        else:
            return self.get_test_data(index)
    
    def get_train_data(self, index):
        start, end = self.seq_indices[index]
        label = torch.tensor(self.labels['label'][start])
        imu = torch.from_numpy(self.labels['imu'][start:end+1]).float()
        
        return {'imu': imu, 'label': label}
    
    def get_test_data(self, index):
        
        imu = self.labels['imu'][index].float()
        label = torch.tensor(self.labels['label'][index])
        
        try: imu = torch.stack(torch.split(imu, self.n_frames, dim=0))
        except: imu = torch.stack(torch.split(imu, self.n_frames, dim=0)[:-1])
        
        label = torch.ones_like(label[:len(imu)]) * label[0]
        return {'imu': imu, 'label': label}
    
    
def make_collate_fn():
    def collate_fn(items):
        items = list(filter(lambda x: x is not None , items))
        batch = dict()
        
        for key in items[0].keys():
            try: batch[key] = torch.stack([item[key] for item in items])
            except: pass
        return batch

    return collate_fn
