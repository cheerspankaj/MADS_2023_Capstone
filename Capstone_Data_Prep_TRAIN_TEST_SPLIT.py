# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 17:30:05 2023


"""


import pickle
import numpy as np
import pandas as pd


# set variables here:
var_file_pickle_name = 'C:\\Capstone\\data\\S'

var_file_personal_info = 'C:\\Capstone\\subject_personal_info.xlsx'

# all subjects list here
var_subject_ID_list = [2,3,4,5,6,7,8,9,10,11,13,14,15,16,17]
# split dataframe here (training data = 12 subjects)
var_train_subject_list = [2,3,4,5,6,7,8,9,10,11,13,14]
#var_train_subject_list = [2,3,4,5,6,7,8,9,10,11,13,14,17]
# split dataframe here (testing data = 3 subjects)
var_test_subject_list = [15,16,17]
#var_test_subject_list = [15,16]

def Load_Personal_Info(var_personal_file_name):
    var_personal_info_df = pd.read_excel(var_personal_file_name, header=0)  
    print(var_personal_info_df)
    print(var_personal_info_df.shape)
    
    # set male = 1, female = 0
    var_personal_info_df.loc[var_personal_info_df["gender"] == "male", "gender"] = 1
    var_personal_info_df.loc[var_personal_info_df["gender"] == "female", "gender"] = 0
    # set right = 1, left = 0
    var_personal_info_df.loc[var_personal_info_df["dominant_hand"] == "right", "dominant_hand"] = 1
    var_personal_info_df.loc[var_personal_info_df["dominant_hand"] == "left", "dominant_hand"] = 0
    # change subject (remove "S" in front) , instead of S2, it should be 2
    var_personal_info_df["subject"] = var_personal_info_df["subject"].str[1:]
        
    var_personal_info_df = var_personal_info_df.astype('int')
    
    var_personal_info_df = var_personal_info_df.set_index('subject')
    print()
    print(var_personal_info_df)
    print(var_personal_info_df.shape)
    print(var_personal_info_df.dtypes)
    
    return var_personal_info_df


def Load_Subject(str_Path, var_Subject_Number):
    
    var_file_subject = str_Path + str(var_Subject_Number) + ".pkl"
    file = open(var_file_subject, "rb")
    
    #data = pickle.load(file, encoding='bytes')
    data = pickle.load(file, encoding='latin1')
    
    file.close()
            
    
    ########################################################################################
    ########################################################################################
    ########################################################################################
    ########################################################################################
    ########################################################################################
    
    # signal data here (chest)
    # load each sensor from chest into df and then put those df's into a python list
    
    var_chest_sensor_df_list = []
    var_chest_sensor_list = ['ACC','ECG','EMG','EDA','Temp','Resp']
    
    
    for var_sensor_chest in var_chest_sensor_list:
        var_numpy_array_chest_sensor = data['signal']['chest'][var_sensor_chest]
        if var_sensor_chest == 'ACC':
            df_chest_sensor = pd.DataFrame(var_numpy_array_chest_sensor, columns = ['Column_A','Column_B','Column_C'])
        else:
            df_chest_sensor = pd.DataFrame(var_numpy_array_chest_sensor, columns = ['Column_A'])
        var_chest_sensor_df_list.append(df_chest_sensor)

    

    
    ########################################################################################
    ########################################################################################
    ########################################################################################
    ########################################################################################
    ########################################################################################  
    
    # signal data here (wrist)
        
    var_wrist_sensor_df_list = []
    var_wrist_sensor_list = ['ACC','BVP','EDA','TEMP']
    
    for var_sensor_wrist in var_wrist_sensor_list:
        var_numpy_array_wrist_sensor = data['signal']['wrist'][var_sensor_wrist]
        if var_sensor_wrist == 'ACC':
            df_wrist_sensor = pd.DataFrame(var_numpy_array_wrist_sensor, columns = ['Column_A','Column_B','Column_C'])
        else:
            df_wrist_sensor = pd.DataFrame(var_numpy_array_wrist_sensor, columns = ['Column_A'])
        var_wrist_sensor_df_list.append(df_wrist_sensor)
    

    
    
    ########################################################################################
    ########################################################################################
    ########################################################################################
    ########################################################################################
    ########################################################################################
    
    # label data here
    
    var_numpy_array_label = data['label']
    
    df_label = pd.DataFrame(var_numpy_array_label, columns = ['Column_A'])
    
    
    return (var_chest_sensor_df_list, var_wrist_sensor_df_list, df_label)


# load all subjects based on list parameter, then do train test split here
# export train / test / split to csv files
# var_file_name = input file name and directory path to the pickle files for this dataset
def Load_and_Split_Data(var_subject_ID_list, var_train_subject_list, var_test_subject_list, var_file_name, var_personal_info_df):
    # we have to skip subject 12 as it is not in the dataset:
    
    var_total_records = 0
    var_chest_records = 0
    var_wrist_records = 0
    
    var_df_Total_Chest_Data = pd.DataFrame()
    
    for var_subject_ID in var_subject_ID_list:
        
        # function returns a tuple ( data frame for chest, data frame for wrist, data frame for labels)
        # loop through each subject here and collect data frame for chest and label - combine into one file / dataframe 
        #    for chest
        var_DFs_sensor_list = Load_Subject(var_file_name, var_subject_ID)
        
        # go through chest sensors here:
        
        # create dataframe for single subject here:
        var_df_Total_Chest_Data_temp_subject = pd.DataFrame()
        
        for i in var_DFs_sensor_list[0]:
            print("chest", var_subject_ID, np.shape(i))
            var_rows = np.shape(i)[0]
            var_chest_records = var_chest_records + var_rows
            var_total_records = var_total_records + var_rows
            
            var_df_Total_Chest_Data_temp_subject = pd.concat([var_df_Total_Chest_Data_temp_subject, i], axis=1)
            
        # concat subjects vertically
        # first set the subject ID here:
        var_df_Total_Chest_Data_temp_subject["subject"] = var_subject_ID
        
        #######################################################################
        #######################################################################
        #######################################################################
        #######################################################################
        #######################################################################
        # load personal info for each subject here:
        for var_column_personal in var_personal_info_df.columns:
            print(var_subject_ID, var_column_personal)
            var_personal_data = var_personal_info_df.loc[var_subject_ID][var_column_personal]
            print(var_subject_ID, var_column_personal, var_personal_data)
            # set main dataframe column of data here with personal info:
            var_df_Total_Chest_Data_temp_subject[var_column_personal] = var_personal_data
        # end load of personal info
        #######################################################################
        #######################################################################
        #######################################################################
        #######################################################################
        #######################################################################
        
        # set label for this subject here:
        var_df_Total_Chest_Data_temp_subject = pd.concat([var_df_Total_Chest_Data_temp_subject, var_DFs_sensor_list[2]], axis=1)
        # concat subjects vertically here:
        var_df_Total_Chest_Data = pd.concat([var_df_Total_Chest_Data, var_df_Total_Chest_Data_temp_subject], axis=0)
            
        for i in var_DFs_sensor_list[1]:
            print("wrist", var_subject_ID, np.shape(i))
            var_rows = np.shape(i)[0]
            var_wrist_records = var_wrist_records + var_rows
            var_total_records = var_total_records + var_rows
            
            
            
        print("label", var_subject_ID, np.shape(var_DFs_sensor_list[2]))
        
        print()
        
    
    print("total record counts here:")
    print("chest cells of data = ", var_chest_records)
    print("wrist cells of data = ", var_wrist_records)
    print("total cells of data = ", var_total_records) 
    
    
    print("shape of chest total dataframe:  ") 
    print(np.shape(var_df_Total_Chest_Data))
    
    print(var_df_Total_Chest_Data.head(50))
    
    var_column_name_list = ['ACC_1','ACC_2','ACC_3','ECG','EMG','EDA','Temp','Resp','subject','age','height','weight','gender','dominant_hand','label']
    
    var_df_Total_Chest_Data.columns = var_column_name_list
    
    var_df_Total_Chest_Data_ordered = var_df_Total_Chest_Data[['subject','ACC_1','ACC_2','ACC_3','ECG','EMG','EDA','Temp','Resp','age','height','weight','gender','dominant_hand','label']]

    # write out all data in dataframe here to CSV file:
    var_total_num_subjects = len(var_subject_ID_list)
    var_df_Total_Chest_Data_ordered.to_csv('chest_data_with_label_' + str(var_total_num_subjects) + '_subjects_ALL_DATA.csv', encoding='utf-8')

    # write out all TRAIN data in dataframe here to CSV file:
    var_total_num_TRAIN_subjects = len(var_train_subject_list)    
    var_train_df = var_df_Total_Chest_Data_ordered[var_df_Total_Chest_Data_ordered['subject'].isin(var_train_subject_list)]
    var_train_df.to_csv('chest_data_with_label_' + str(var_total_num_TRAIN_subjects) + '_subjects_TRAIN_DATA.csv', encoding='utf-8')
    
    # write out all TEST data in dataframe here to CSV file:
    var_total_num_TEST_subjects = len(var_test_subject_list)     
    var_test_df = var_df_Total_Chest_Data_ordered[var_df_Total_Chest_Data_ordered['subject'].isin(var_test_subject_list)]
    var_test_df.to_csv('chest_data_with_label_' + str(var_total_num_TEST_subjects)+ '_subjects_TEST_DATA.csv', encoding='utf-8')


########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################

### load data here based on variables set at top of this file:

# all subjects list here
####var_subject_ID_list = [2,3,4,5,6,7,8,9,10,11,13,14,15,16,17]
# split dataframe here (training data = 12 subjects)
####var_train_subject_list = [2,3,4,5,6,7,8,9,10,11,13,14]
# split dataframe here (testing data = 3 subjects)
####var_test_subject_list = [15,16,17]


var_personal_info_by_subject_df = Load_Personal_Info(var_file_personal_info)

Load_and_Split_Data(var_subject_ID_list, var_train_subject_list, var_test_subject_list, var_file_pickle_name, var_personal_info_by_subject_df)
    
    