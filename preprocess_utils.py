#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 11:49:57 2022

@author: laurenparola
"""

import sys
import pandas as pd
import os
import argparse
import numpy as np


import xlsxwriter
import matplotlib.pyplot as plt
import tarfile

import os
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt,find_peaks
import matplotlib.pyplot as plt
import sys
import quaternion
from scipy.spatial.transform import Rotation
import csv

# args_default['imu_pth']= 'C:\\Users\\cmumbl\\Box\\CMU_MBL\\Data\\ACLR_Pilot_1\\PS001\\Three Month\\IMU\\in_lab'
#args_default['imu_pth'] = 'C:\\Users\\cmumbl\\Box\\CMU_MBL\\Data\\ACLR_Pilot_1\\PS004\\Three mIMU\\in_lab'

# if 'P' in args_default['subject']:
#     args_default['imu_pth'] = '/Users/laurenparola/Library/CloudStorage/Box-Box/CMU_MBL/Data/ACLR_Pilot_1/'+args_default['subject']+'/Three Month/IMU/in_lab'
def butter_low(data,fs, order, fc):
    '''
    Zero-lag butterworth filter for column data (i.e. padding occurs along axis 0).
    The defaults are set to be reasonable for standard optoelectronic data.
    '''
    # Filter design
    b, a = butter(order, 2*fc/fs, 'low')
    # Make sure the padding is neither overkill nor larger than sequence length permits
    padlen = min(int(0.5*data.shape[0]), 200)
    # Zero-phase filtering with symmetric padding at beginning and end
    filt_data = filtfilt(b, a, data, padlen=padlen, axis=0)
    return filt_data

def highpass_iir(data,fs, order, fc):
    nyq = 0.5 * fs
    normal_cutoff = fc / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    # Make sure the padding is neither overkill nor larger than sequence length permits
    padlen = min(int(0.5*data.shape[0]), 200)
    # Zero-phase filtering with symmetric padding at beginning and end
    filt_data = filtfilt(b, a, data, padlen=padlen, axis=0)
    return filt_data
def align_data(df,position):
    '''
    Aligns MC10 data to standard coordinate system
    X - forward; Y - up; Z - right
    '''
    
    df_old=df.copy()
    
    if 'sacrum' in position:
        y_rot_quat = quaternion.from_euler_angles(0, -np.pi/2, 0)
        rot_quat = y_rot_quat
    else:

        rot_quat = quaternion.from_euler_angles(0, np.pi/2, 0)


  
    sensors = ['Accel', 'Gyro']
    axes = ['X', 'Y', 'Z']
    
    accel_col = [col for col in df.columns if 'Accel' in col]
    data = df[accel_col].values.copy()
    data = quaternion.rotate_vectors(rot_quat, data, axis=1)
    df.loc[:, accel_col] = data

    
    
    gyro_col = [col for col in df.columns if 'Gyro' in col and not 'event_id' in col]
    data = df[gyro_col].values.copy()
    data = quaternion.rotate_vectors(rot_quat, data, axis=1)
    df.loc[:, gyro_col] = data
    
    

    return df

def clean_raw_data(df_raw,segment,fs):

    df_raw = df_raw.interpolate(limit_area='inside')
    df_si = df_raw.copy()
    df_temp = df_raw.copy()
    df_filt = df_raw.copy()
    new_cols = []
    accel_col = [col for col in df_raw.columns if "Accel" in col]

    gyro_col = [col for col in df_raw.columns if "Gyro" in col and not 'event_id' in col]
    

    
    df_si.loc[:,accel_col] = df_raw.loc[:, accel_col] * -9.81
    df_filt.loc[:, accel_col] = butter_low(df_si.loc[:, accel_col].values,fs,order=4,fc=5)

    temp = [col.replace('(g)', '(m/s^2) '+segment) for col in accel_col]

    
    df_filt = df_filt.rename(columns={accel_col[0]: temp[0], accel_col[1]: temp[1],accel_col[2]: temp[2]})
        
    #[new_cols.append(i) for i in temp]
    
    df_si.loc[:,gyro_col] = df_raw.loc[:, gyro_col] / 180 * np.pi

    df_filt.loc[:, gyro_col] = butter_low(df_si.loc[:, gyro_col].values,fs, order=4, fc=5)
    

    temp = [col.replace('(°/s)', '(rad/s) '+segment) for col in gyro_col]


    df_filt = df_filt.rename(columns={gyro_col[0]: temp[0], gyro_col[1]: temp[1],gyro_col[2]: temp[2]})



    df_filt['Datetime'] = pd.to_datetime(df_filt.index*1000)
    df_filt = df_filt.set_index('Datetime')
    return df_filt

def sync_dfs(df_list, resamp_freq):
    '''
    Synchronizes dataframes in df_list to start and stop at same index
    Resamples dataframe to desired frequency
    '''

    # Joins all the dataframes in df_list
    col_len = []
    col_start = [0]
   # df_len = len(df_list)
    for i,key in enumerate(df_list):
        col_len.append(df_list[key].shape[1])
        col_start.append((col_start[i]+col_len[i]))
        if i == 0:
            df = df_list[key]
        else:
            df = df.join(df_list[key], how='outer')

    # Interpolate missing timepoints and remove beginning and end NaNs
    df = df.interpolate(limit_area='inside')
    df = df.dropna()
    # Converts index to datetime if in timestamps
    if df.index.dtype == 'int64':
        df['Datetime'] = pd.to_datetime((df.index*1e6).astype(np.int64))
        df = df.set_index('Datetime')

    # Create separate time index at desired frequency
    freq_in_ms = int(1000/resamp_freq)

    tmp_idx = pd.date_range(start=df.index[0], end=df.index[-1], freq=(str(freq_in_ms)+'ms'))
    d = np.zeros((len(tmp_idx)))

    tmp_df = pd.DataFrame({'temp': d}, index=tmp_idx)
    tmp_idx.name = 'Datetime'
    
    # Join inner and outer dataframe to add new indices
    df = df.join(tmp_df, how='outer')

    # Drop temporary zeros column
    df = df.drop(columns=['temp'])

    # Interpolate data again
    df = df.interpolate(limit_area='inside')

    # Extract only the time points of the desired frequency
    df = df.loc[tmp_idx, :]

    # Creates dataframe list with synchronized and resampled data
    df_list_new = []
    df_add = tmp_df.copy()
    for i,key in enumerate(df_list):
        df_add = tmp_df.copy()

        for j in range(col_len[i]):
            df_add = df_add.join(df.loc[:, df.columns[col_start[i]+j]], how='outer')
            if j == col_len[i]-1:
                df_add = df_add.drop(columns=['temp'])

        df_list_new.append(df_add)

    #time = np.arange(0,len(df_list)/freq_in_ms,1/freq_in_ms)
    return df_list_new

