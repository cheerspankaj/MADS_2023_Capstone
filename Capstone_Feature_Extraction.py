import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

import warnings
warnings.filterwarnings("ignore")

# FLIRT package to derive sensor features e.g. Accelerometer, EDA related features based on raw recorded values
import flirt
import flirt.reader.empatica

from ecgdetectors import Detectors

# mne is the open source library for werable package to derive features for deriving features for EEG,EDA etc. sensors.
# link: https://mne.tools/stable/index.html
import mne

# library to derive eda features
from scipy.signal import argrelextrema
#link: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.argrelextrema.html

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Function to derive statistical feature of a specified columns in a dataframe dataset
# Parameters:
# Input: 
# datasets: dataframe for which statistical features are to be derived
# Output/Return : 
# Statistical Feature Disctionary object with Key as the column name and value as statistical feature value e.g. mean,SD
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def get_stats_features(dataset, sensor_columns):
    # get data frame for list of columns
    data = dataset[sensor_columns]
    stats_feature_dict = {}
    for feature in sensor_columns:
        stats_feature_dict[feature+'_mean'] = data[feature].mean()
        stats_feature_dict[feature+'_std'] = data[feature].std()
        stats_feature_dict[feature+'_min'] = np.min(data[feature])
        stats_feature_dict[feature + '_max'] = np.max(data[feature])
        # stats_feature_dict[feature + '_range'] = np.max(data[feature]) - np.min(data[feature])
        # stats_feature_dict[feature + '_median'] = np.median(data[feature])
        # stats_feature_dict[feature + '_skew'] = data[feature].skew()
        # stats_feature_dict[feature + '_kurtosis'] = data[feature].kurtosis()
        # stats_feature_dict[feature + '_iqr'] = np.percentile(data[feature], 75) - np.percentile(data[feature], 25)
        # stats_feature_dict[feature + '_cv'] = np.std(data[feature]) / np.mean(data[feature])
    
    return stats_feature_dict

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Function to compute statistical ACC features based on the l2-norm of the x-, y-, and z- acceleration
# Parameters:
#
# dataset:  input ACC time series in x-, y-, and z- direction
# window_length: the window size in seconds to consider
# window step: the time step to shift each window
# frequency: the frequency of the input signal
#
# Returns: 
# Dataframe with additonal feature columns (containing statistical aggregation features) added in the input dataset
#
# Reference doc: https://flirt.readthedocs.io/en/latest/api.html#module-flirt.hrv
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def get_acc_features(dataset,window_length=60,window_step_size=1,frequency=700):
    data = dataset.copy()
    data.columns = ['acc_x','acc_y','acc_z']
    
    # get total records to be updated for new feature
    total_records = data.shape[0]
    
    # get batch size to update set of records for a specific feature value
    batch_size = total_records/frequency
    print('Total Number of records to be added for each feature:',total_records)
    print('Batch Size:',batch_size)
    
    # get All ACC features
    acc_features = flirt.get_acc_features(data,
                                          window_length = window_length, 
                                          window_step_size = window_step_size,
                                          data_frequency = frequency)

    #print('Total Number of Unique features derived:',len(acc_features.columns))

    acc_features_list = ['l2_mean','l2_std', 'l2_min', 'l2_max', 'l2_ptp', 'l2_sum', 'l2_energy', 'l2_peaks', 'l2_rms', 
                         'l2_lineintegral','l2_n_above_mean', 'l2_n_below_mean', 'l2_n_sign_changes', 'l2_entropy']
    
    for feature in acc_features_list:
        #print('Adding Feature:',feature)
        # initialize with 0
        data[feature] = 0
        
        # get all values for this feature
        feature_value_list = acc_features[feature].tolist()
        #print('number of values for this feature are:',len(feature_value_list))
        
        # update each value for batch size
        updated_records_count = 0
        try:
            for feature_value in feature_value_list:
                # print('Adding value:',feature_value)
            
                # assign this value to associated records in main dataset
                if batch_size > 1:
                
                    if ((total_records - updated_records_count) > batch_size) :
                        record_start_index = updated_records_count
                        record_end_index = (updated_records_count + batch_size)
                    else:
                        record_start_index =  updated_records_count
                        record_end_index = total_records
                else:
                    record_start_index = updated_records_count
                    record_end_index = total_records
                    
                updated_records_count = (updated_records_count + batch_size)
            
                # Update value for new feature for records set
                # print('Start Index:{} End Index:{}'.format(record_start_index,record_end_index,))
                data.loc[range(int(record_start_index),int(record_end_index)), feature] = feature_value
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise
    
    return data

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Function to compute statistical EDA features based on the signal decompositon into phasic/tonic 
# components of the skin conductivity
# 
# Parameters:
#
# Input: 
# dataset:  input EDA time series data
# window_length: the window size in seconds to consider
# window step: the time step to shift each window
# frequency: the frequency of the input signal
#
# Returns : 
# Dataframe with additonal feature columns(containing statistical aggregation features of the tonic/phasic 
# EDA components) added in the input dataset
#
# Reference doc: https://flirt.readthedocs.io/en/latest/api.html#module-flirt.hrv
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def get_eda_features(dataset,window_length=60,window_step_size=1,frequency=700):
    data = dataset    
    # get total records to be updated for new feature
    total_records = data.shape[0]
    
    # get batch size to update set of records for a specific feature value
    batch_size = total_records/frequency
    print('Total Number of records to be added for each feature:',total_records)
    print('Batch Size:',batch_size)
    
    # get All features    
    eda_features = flirt.get_eda_features(data,
                                          window_length = window_length, 
                                          window_step_size = window_step_size,
                                          data_frequency = frequency)

    #print('Total Number of Unique features:',len(eda_features.columns))

    eda_features_list = ['tonic_mean', 'tonic_std', 'tonic_min', 'tonic_max', 'tonic_energy', 'tonic_peaks', 'tonic_rms','tonic_entropy',
                         'phasic_mean', 'phasic_std','phasic_min', 'phasic_max','phasic_energy','phasic_peaks', 'phasic_rms',
                         'phasic_entropy']
    
    #for feature in eda_features.columns:
    for feature in eda_features_list:
        #print('Adding Feature:',feature)
        # initialize with 0
        data[feature] = 0
        
        # get all values for this feature
        feature_value_list = eda_features[feature].tolist()
        #print('number of values for this feature are:',len(feature_value_list))
        
        # update each value for batch size
        updated_records_count = 0
        for feature_value in feature_value_list:
            #print('Adding value:',feature_value)
            
            # assign this value to associated records in main dataset
            if batch_size > 1:
                
                if ((total_records - updated_records_count) > batch_size) :
                    record_start_index = updated_records_count
                    record_end_index = (updated_records_count + batch_size)
                else:
                    record_start_index =  updated_records_count
                    record_end_index = total_records
            else:
                record_start_index = updated_records_count
                record_end_index = total_records
                    
            updated_records_count = (updated_records_count + batch_size)
            
            # Update value for new feature for records set
            # print('Start Index:{} End Index:{}'.format(record_start_index,record_end_index,))
            data.loc[range(int(record_start_index),int(record_end_index)), feature] = feature_value
    
    return data
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Add function definition
# 
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def get_ECG_feature_column(var_ECG_raw_data, fs=700, n_row_increments=5000):
    var_ECG_df = pd.DataFrame([])
    var_ECG_df["raw_ECG"] = var_ECG_raw_data
    var_ECG_df["ECG_freq"] = np.nan
    
    ### code taken from:  https://dsp.stackexchange.com/questions/58155/how-to-filter-ecg-and-detect-r-peaks
    fs=700 # sample freq

    var_ECG_ff_list = []
    for var_current_row in range(n_row_increments,len(var_ECG_raw_data)+1,n_row_increments):
        var_start = var_current_row-n_row_increments
        var_end = var_current_row
        #print(var_start,var_end)
        heartbeat = var_ECG_raw_data[var_start:var_end]

        detectors = Detectors(fs)

        r_peaks_pan = detectors.pan_tompkins_detector(heartbeat)
        r_peaks_pan= np.asarray(r_peaks_pan)
        
        var_ff = len(r_peaks_pan)
        var_ECG_ff_list.append(var_ff)
        
        var_ECG_df["ECG_freq"][var_start:var_end] = var_ff
    
    # fill in last part of array with previous value here:
    #print(var_current_row)
    #print(var_ff)
    var_start = var_current_row
    var_end = len(var_ECG_raw_data)
    var_ECG_df["ECG_freq"][var_start:var_end] = var_ff
    
    # return tuple:  (dataframe of ECG signal , the frequency list of each measured interval)
  
    return (var_ECG_df, var_ECG_ff_list)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Function to derive EMG feature of a dataset specified columns using MNE package
# Parameters:
# Input: 
# datasets: dataframe for which emg features are to be derived
# Output/Return : 
# Feature Disctionary object with Key as the column name and value as emg feature value e.g. zc
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def extract_emg_features(emg_signal, sampling_rate):
    # Ensure that emg_signal is at least a 2D array
    emg_signal = np.atleast_2d(emg_signal)
    
    # Preprocess the EMG signal
    num_signals = emg_signal.shape[0]
    filtered_signals = []
    for i in range(num_signals):
        filtered_signal = mne.filter.filter_data(emg_signal[i], sfreq=sampling_rate, l_freq=10, h_freq=300)
        filtered_signals.append(filtered_signal)
    filtered_signals = np.array(filtered_signals)
    
    # Extract features
    features = {}
    features['variance'] = np.var(filtered_signals)
    features['rms'] = np.sqrt(np.mean(filtered_signals**2))
    features['wl'] = np.sum(np.abs(np.diff(filtered_signals)))
    features['zc'] = np.sum(np.abs(np.diff(np.sign(filtered_signals)))) / (2 * len(filtered_signals))
    features['mav'] = np.mean(np.abs(filtered_signals))
    
    return features

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Function to derive additional EDA features of a dataset specified columns using scipy package
# Parameters:
# Input: 
# datasets: dataframe for which eda features are to be derived
# Output/Return : 
# Feature Disctionary object with Key as the column name and value as eda feature value e.g. eda_amp_mean
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def eda_features(eda, sampling_rate=700):
    
#     # Compute the mean and standard deviation of the EDA signal
#     eda_mean = np.mean(eda)
#     eda_std = np.std(eda)

#     # Compute the maximum and minimum of the EDA signal
#     eda_max = np.max(eda)
#     eda_min = np.min(eda)

    # Convert EDA signal to conductance (G)
    eda = eda / 1000  # Convert from microsiemens to siemens
    eda = 1 / eda  # Convert from resistance to conductance

    # Compute smoothed EDA signal using a moving average filter
    eda_smoothed = pd.Series(eda).rolling(window=int(sampling_rate * 0.75), center=True).mean()

    # Compute the first derivative of the smoothed EDA signal
    eda_diff = np.diff(eda_smoothed)

    # Compute the SCR onset and peak indices
    eda_peak = argrelextrema(eda_diff, np.greater)[0]
    eda_onset = [np.argmax(eda_smoothed[:i]) for i in eda_peak]

    # Compute the SCR amplitudes
    eda_amp = eda_smoothed[eda_peak] - eda_smoothed[eda_onset]

    # Compute the SCR rise times
    eda_rise = eda_peak - np.array(eda_onset)

    # Compute the SCR recovery times
    eda_recovery = [np.argmax(eda_smoothed[i:] < eda_smoothed[eda_peak[j]] * 0.63) for j, i in enumerate(eda_peak)]
    eda_recovery = np.array(eda_recovery) - eda_peak

    # Compute the mean and standard deviation of the SCR amplitudes, rise times, and recovery times
    eda_amp_mean = np.mean(eda_amp)
    eda_amp_std = np.std(eda_amp)
    eda_rise_mean = np.mean(eda_rise)
    eda_rise_std = np.std(eda_rise)
    eda_recovery_mean = np.mean(eda_recovery)
    eda_recovery_std = np.std(eda_recovery)

    # Compute the number of SCRs
    eda_scr_num = len(eda_peak)

    # Compute the SCR frequency
    eda_scr_freq = eda_scr_num / (eda.size / sampling_rate)

    # Compute the mean and standard deviation of the EDA signal
    # eda_mean = np.mean(eda_smoothed)
    # eda_std = np.std(eda_smoothed)

    # Return a dictionary containing the EDA features
    eda_dict = {
        'eda_amp_mean': eda_amp_mean,
        'eda_amp_std': eda_amp_std,
        'eda_rise_mean': eda_rise_mean,
        'eda_rise_std': eda_rise_std,
        'eda_recovery_mean': eda_recovery_mean,
        'eda_recovery_std': eda_recovery_std,
        'eda_scr_num': eda_scr_num,
        'eda_scr_freq': eda_scr_freq
        #'eda_mean': eda_mean,
        #'eda_std': eda_std
    }

    return eda_dict

