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

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Function to derive statistical feature of a dataset specified columns
# Parameters:
# Input: 
# datasets: dataframe for which statistical features are to be derived
# Output/Return : 
# Feature Disctionary object with Key as the column name and value as statistical feature value e.g. mean,SD
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
        stats_feature_dict[feature + '_range'] = np.max(data[feature]) - np.min(data[feature])
        stats_feature_dict[feature + '_median'] = np.median(data[feature])
        stats_feature_dict[feature + '_skew'] = data[feature].skew()
        stats_feature_dict[feature + '_kurtosis'] = data[feature].kurtosis()
        stats_feature_dict[feature + '_iqr'] = np.percentile(data[feature], 75) - np.percentile(data[feature], 25)
        stats_feature_dict[feature + '_cv'] = np.std(data[feature]) / np.mean(data[feature])
    
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
    data = dataset
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

    print('Total Number of Unique features:',len(acc_features.columns))
    
    for feature in acc_features.columns:
        #print('Adding Feature:',feature)
        # initialize with 0
        data[feature] = 0
        
        # get all values for this feature
        feature_value_list = acc_features[feature].tolist()
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
            # print('Start Index:{} End Index:{}'.format(record_end_index,))
            data.loc[range(int(record_start_index),int(record_end_index)), feature] = feature_value
    
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

    print('Total Number of Unique features:',len(eda_features.columns))
    
    for feature in eda_features.columns:
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