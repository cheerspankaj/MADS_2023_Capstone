import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Function to create downsample records from input samples as per specified frequency
# Parameters:
# Input: 
# datasets: dataframe to be downsampled
# Output/Return : 
# downsampled dataset
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def data_sampling(dataset,sample_rate,down_sample_freq='100ms'):
    print('Shape before down sampling:',dataset.shape)
    max_time = dataset.shape[0]/sample_rate
    time_steps = np.linspace(0, max_time, dataset.shape[0])
    dataset['seconds'] = time_steps

    dataset['time'] = pd.to_datetime(dataset['seconds'], unit='s')
    dataset = dataset.set_index(['subject','label','time'], drop=True)
    
    level_values = dataset.index.get_level_values
    #print('level_values:',level_values)
    result = (dataset.groupby([level_values(i) for i in [0,1]]
                      +[pd.Grouper(freq=down_sample_freq, level=-1)]).mean())


    print("Shape AFTER resample:",result.shape)
    
    return result

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Function to clean the dataset by dropping Drop NaN,Inf, duplicate etc. in the dataset 
# Parameters:
# Input: 
# datasets: dataframe to be cleaned
# Output/Return : 
# cleaned dataset
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def Format_Data(dataset):
    # Load and preprocess the data
    data = dataset
    
    # restructure columns here so that label is the last column:
    var_final_dataset_column_list = data.columns.tolist()
    
    # remove the "label" column here and append it at the end of the list:
    var_final_dataset_column_list.remove("label")
    var_final_dataset_column_list.remove("time")
    var_final_dataset_column_list.remove("tonic_entropy")

    
    var_final_dataset_column_list.append("label")
    data = data[var_final_dataset_column_list]
    
    var_final_dataset_column_list_final = var_final_dataset_column_list.copy()

    for var_column in var_final_dataset_column_list:
        #nan_count = data[var_column].isna().sum()
        nan_count = data[var_column].isna().values.any()
        #if nan_count > 0:
        if nan_count == True:
            # drop this column here:
            #print("column has NaN:")
            #print(var_column,nan_count)
            var_final_dataset_column_list_final.remove(var_column)
            
        #inf_count = data[var_column].map(np.isinf).sum()
        inf_count = np.isinf(data[var_column]).values.sum()
        #print(inf_count)
        if inf_count > 0:
            # drop this column here:
            #print()
            #print("column has Inf:")
            #print(var_column,inf_count)
            if var_column in var_final_dataset_column_list_final:
                var_final_dataset_column_list_final.remove(var_column)
        
        #print(var_column[0:5])
        # remove survey columns here:
        if var_column[0:5] == "PANAS" or var_column[0:4] == "STAI" or var_column[0:4] == "SAM0":
            # drop this column here:
            #print()
            #print("column has Survey Data:")
            #print(var_column)
            if var_column in var_final_dataset_column_list_final:
                var_final_dataset_column_list_final.remove(var_column)
            
    #data.dropna(inplace=True)
    # Remove the first column
    #first_column_name = data.columns[0]
    #data = data.drop(columns=first_column_name)
    
    # remove all columns which have NaN here:
    data = data[var_final_dataset_column_list_final]
    
    data = data[(data['label'] < 5) & (data['label'] > 0)]

    columns_to_drop = ['Unnamed: 0','Unnamed: 0.1','subject','acc_x', 'acc_y', 'acc_z','EDA_1']

    data = data.drop(columns_to_drop, axis=1)

    #data = data.drop(['Unnamed: 0','subject','acc_x', 'acc_y', 'acc_z','EDA.1'],axis=1)
    
    return data

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Function to split the dataset into train, validate and test. 
# Training is created from 80% of the total dataset, Validate dataset is created from 10% of the total dataset
# Testing dataset is created from remaining 10% of the total dataset
# Parameters:
# Input: 
# data_X: dataset with features
# data_y: dataset with target label
# Output/Return : 
# train, validate and test data and corresponding target labels
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def split_data(data_X,data_y):
    stop_train = int(.8 * data_X.shape[0])
    stop_validate = int(.9 * data_X.shape[0])
    X_train =data_X[:stop_train]
    X_validate = data_X[stop_train:stop_validate]
    X_test = data_X[stop_validate:]

    y_train =data_y[:stop_train]
    y_validate = data_y[stop_train:stop_validate]
    y_test = data_y[stop_validate:]
    return X_train,X_validate,X_test,y_train,y_validate,y_test