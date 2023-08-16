from __future__ import absolute_import
from __future__ import print_function
from __future__ import division



import glob
import sys
import os
import os.path as osp

import torch
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

from pdb import set_trace as st

def extract_windows_vectorized(array, clearing_time_index, max_time, sub_window_size):
    start = 0
    import pdb; pdb.set_trace()
    sub_windows = (
        start +
        # expand_dims are used to convert a 1D array to 2D array.
        np.expand_dims(np.arange(sub_window_size), 0) +
        np.expand_dims(np.arange(max_time + 1), 0).T
    )

    return array[sub_windows,:]
def extract_windows(array, clearing_time_index, max_time, sub_window_size):
    examples = []
    start = clearing_time_index + 1 - sub_window_size + 1
    
    for i in range(max_time+1):
        example = array[start+i:start+sub_window_size+i]
        examples.append(np.expand_dims(example, 0))
    import pdb; pdb.set_trace()
    return np.vstack(examples)
def extract_sub_section(array,size):
    counter = round(len(array)//size)
    examples = []

    for i in range(counter):
        example = array[i*size:size*(i+1),:]
        examples.append(np.expand_dims(example, 0))
    return np.vstack(examples)
    
    
def preprocess(subjects,data_loc,sensor_list, is_train,group_label):


    
    imus, labels, subj_act, seqs, seq_idx = [], [], [],[], 0

    for subject in subjects:
        print(subject)

        #subject_dir = glob.glob(osp.join(data_loc, f'{subject}*'))[0]
        subject_dir = osp.join(data_loc, f'{subject}')
        #_, trials, _ = next(os.walk(subject_dir))
        trials = [activity for activity in os.listdir(subject_dir) ]
        for trial in trials:
          if '_motion' not in trial:
            print(trial)
            #print(trial)
            trial_dir = osp.join(subject_dir, trial)
            #print(trial_dir)
            # Load IMU data
            temp_imus = []
            
            for val,sensor in enumerate(sensor_list):
                    # acc = np.load(trial_dir+'\\'+ f'{sensor}_acc.npy')
                    # gyr = np.load(trial_dir+'\\'+ f'{sensor}_gyr.npy')
                    acc = np.load(osp.join(trial_dir, f'{sensor}_acc.npy'),allow_pickle=True)
                    gyr = np.load(osp.join(trial_dir, f'{sensor}_gyr.npy'),allow_pickle=True)
                    imu = torch.from_numpy(np.concatenate((acc, gyr), axis=-1)).float()
                    temp_imus.append(imu.unsqueeze(1))

                # axs[val].plot(gyr[0:600,:],label=sensor+' gyr')
                # axs[val].set_ylabel(sensor+' gyr')
            # fig.savefig('imus.png')

            temp_imus = torch.cat(temp_imus, dim=2)
            
            
            temp = temp_imus.flatten(1)
            #import pdb; pdb.set_trace()
            #print(temp.size())
            # temp = extract_sub_section(temp,100)
            #temp = extract_windows_vectorized(temp,180,len(temp)-120,120)
            #import pdb; pdb.set_trace()
            
            # counter = 1
            # list_temp =[]
            # for val, var for enumerate(temp[):
                
                # if counter == 80:
                
                # if count < 80:
                # list_temp.append(

            imus.append(temp)
            
            # Load Mocap angle data
            if 'walking' in trial:
                label = 1
            if 'running' in trial:
                label = 2
            if 'ramp_ascending' in trial:
                label = 3
            if 'ramp_descending' in trial:
                label = 4  
            if 'stair_ascending' in trial:
                label = 5
            if 'stair_descending' in trial:
                label = 6 
            if 'sit_to_stand' in trial:
                label = 7
            if 'hopping' in trial:
                label = 8
            if 'high_jump' in trial:
                label = 0 

                
            trial_labels =label# * len(temp)
            # import pdb; pdb.set_trace()
            temp_labels = torch.from_numpy(np.array(trial_labels))#.unsqueeze(1).float()
            labels.append(temp_labels)
            subj_act.append(subject+'_'+trial)

            #import pdb; pdb.set_trace()
        
    if is_train == True:
        save_loc = data_loc+'train_data.pt'
    else:
        save_loc = data_loc+'test_data.pt'
    # print(np.shape(imus[1]))
    # print(np.shape(labels))
    # import pdb; pdb.set_trace()
    #import pdb; pdb.set_trace()
    max_length = max(len(seq) for seq in imus)

# Pad sequences with zeros
    padded_data = [np.pad(seq,  ((0,max_length - len(seq)), (0, 0)), mode='constant') for seq in imus]

    imu = torch.from_numpy(np.array(padded_data)) #torch.cat(imus, dim=0)

    label = torch.from_numpy(np.array(labels)) #torch.cat(labels, dim=0)
    torch.save({
        'imu': imu,
        'label': label},
        save_loc
    )


if __name__ == '__main__':
    train_subjects = ['S0'+str(i)  if i < 10 else 'S'+str(i) for i in range(1,65)]
    test_subjects = ['S'+str(i) for i in range(66,82)]
    sensor_list = ['lshank','rshank','lthigh','rthigh','sacrum']
    out_fname = 'F:\\Dome_Pilot\\Synced\\'
    preprocess(train_subjects,out_fname,sensor_list,  True,False)
    preprocess(test_subjects,out_fname, sensor_list, False,False)

