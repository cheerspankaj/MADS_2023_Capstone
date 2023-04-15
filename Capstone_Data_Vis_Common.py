import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# assign label and code
effective_state_label = {'Baseline':1,'Stress':2,'Amusement':3,'Meditation':4}

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Function to plot line graph for a subject for a specific sensor type passed to the function e.g. "ECG"
# Parameters:
# Input: 
# datasets: dataset with features for which graph is to be created
# Subject: Subject number in the dataset
# num_of_records : Number of records to be used for plot
# sensor_type : name of the sensor to be used for plot
# labels : List of labels to be used for plot e.g. ["Baseline",'Stress"]
# first_last : first N records or last N records to be used for plot
# sample rate: data sampling rate to be used for plot
# Output/Return : 
# line plot for the subject for the labels provided associated with the sensor specified
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def plot_data_for_subject(dataset,
                          subject,
                          num_of_records,
                          sensor_type,
                          labels=effective_state_label,
                          first_last = 'head',
                          sample_rate = 700):
    
    for label in labels:
        subj_data = dataset.loc[(dataset.subject == subject) & (dataset.label == effective_state_label[label])]   
        if first_last == 'first': 
            subject_data = subj_data.head(num_of_records)
        else: 
            subject_data = subj_data.tail(num_of_records)
        
        max_time = subject_data.shape[0]/sample_rate
        time_steps = np.linspace(0, max_time, subject_data.shape[0])

        plt.figure(figsize=(10,5))
        plt.title('Plot for '+ sensor_type +' for label:'+ label +' for subject:'+ str(subject)+
                  ' [Records count:'+str(num_of_records)+']')
        plt.plot(#range(0,subject_data.shape[0]),
                 time_steps,
                 subject_data[sensor_type],
                 color='red')
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.show()

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Function to plot ACC graph for a subject
# Parameters:
# Input: 
# datasets: dataset with features for which graph is to be created
# Subject: Subject number in the dataset
# num_of_records : Number of records to be used for plot
# labels : List of labels to be used for plot e.g. ["Baseline",'Stress"]
# first_last : first N records or last N records to be used for plot
# sample rate: data sampling rate to be used for plot
# Output/Return : 
# ACC X,Y and Z line plot for the subject for the labels provided
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def plot_acc_for_subject(dataset,
                         subject,
                         num_of_records = 1000,
                         labels=effective_state_label,
                         first_last = 'head',
                         sample_rate = 700
                        ):

    for label in labels:
        subj_acc_data = dataset.loc[(dataset.subject == subject) & (dataset.label == effective_state_label[label])]
        if first_last == 'first': 
            subject_acc_data = subj_acc_data[['ACC_1','ACC_2','ACC_3']].head(num_of_records)
        else: 
            subject_acc_data = subj_acc_data[['ACC_1','ACC_2','ACC_3']].tail(num_of_records)
        
        subject_acc_data = subject_acc_data.rename(columns = {'ACC_1':'ACC_X','ACC_2':'ACC_Y','ACC_3':'ACC_Z'})

        max_time = subject_acc_data.shape[0]/sample_rate
        time_steps = np.linspace(0, max_time, subject_acc_data.shape[0])

        plt.figure(figsize=(10,5))
        ACC_X = time_steps
        plt.plot(ACC_X,subject_acc_data.ACC_X,label = "accelerometer channel-X")
        plt.plot(ACC_X,subject_acc_data.ACC_Y,label = "accelerometer channel-Y")
        plt.plot(ACC_X,subject_acc_data.ACC_Z,label = "accelerometer channel-Z")
        plt.title('Accelerometer graph for Subject:'+str(subject) +' for label:'+label + 
                  ' [Records Count:'+str(num_of_records)+']')
        plt.legend(loc=1)
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude [m/s^2]")
        plt.show()

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Function to subplot data distribution across all signals within the datasets 
# Parameters:
# Input: 
# datasets: dataframe for which subplot for each signal data distribution to plot
# Output/Return : 
# bar plots of each sensors like ecg,emg
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def plot_sensor_data(df, sensor_list):
    num_sensors = len(sensor_list)
    num_rows = 2
    num_cols = int(np.ceil(num_sensors / num_rows))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, sensor in enumerate(sensor_list):
        data = df[sensor]
        ax = axes[i]
        hist, bin_edges = np.histogram(data, bins=20)
        ax.bar(bin_edges[:-1], hist, width=np.diff(bin_edges)[0]*0.8)
        ax.set_title(f'{sensor} Distribution')
        ax.set_xlabel(sensor)
        ax.set_ylabel('Count')
        ax.ticklabel_format(style='plain', axis='y')
        
    if num_sensors % 2 != 0:
        axes[-1].remove()
        
    plt.tight_layout()
    plt.show()

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Function to create barplot based on provided specification
# Parameters:
# Input: 
# x: x-axis data
# y: y-axis data
# x_label: label for x-axis
# y_label: label for y-axis
# t_title: title for the graph
# Output/Return : 
# None
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def plot_records_count(x,y,x_label,y_label,t_title,bar_width=0.3,fig_x=15,fig_y=5):
    fig = plt.figure(figsize = (fig_x,fig_y))
    plt.bar(x,y,width = bar_width)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(t_title)
    plt.show()