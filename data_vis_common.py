#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# set file path
file_path = '/scratch/siads699w23_class_root/siads699w23_class/team8_cpastone_2023/'
chest_data_filename = 'chest_All.csv'
# personal_info_filename = 'subject_personal_info.xlsx'

# get chest data
chest_data = pd.read_csv(file_path+chest_data_filename)
# assign label and code
chest_data_label = {'Baseline':1,'Stress':2,'Amusement':3,'Medication':4}

# Function to plot line graph for a subject for a specific sensor type e.g. "ECG"
def plot_chest_data_for_subject(subject,
                                num_of_records,
                                sensor_type,
                                labels=chest_data_label,
                                first_last = 'head',
                                sample_rate = 700
                               ):

    for label in labels:
        subj_chest_data = chest_data.loc[(chest_data.subject == subject) & (chest_data.label == chest_data_label[label])]   
        if first_last == 'first': 
            subject_data = subj_chest_data.head(num_of_records)
        else: 
            subject_data = subj_chest_data.tail(num_of_records)
        
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

# plot_chest_data_for_subject(2,10000,'ECG')
# Function to plot ACC graph for a subject
def plot_acc_for_subject(subject,
                         num_of_records = 1000,
                         labels=chest_data_label,
                         first_last = 'head',
                         sample_rate = 700
                        ):
    for label in labels:
        subj_acc_data = chest_data.loc[(chest_data.subject == subject) & (chest_data.label == chest_data_label[label])]
        if first_last == 'first': 
            subject_acc_data = subj_acc_data[['ACC_1','ACC_2','ACC_3']].head(num_of_records)
        else: 
            subject_acc_data = subj_acc_data[['ACC_1','ACC_2','ACC_3']].tail(num_of_records)
        
        subject_acc_data = subject_acc_data.rename(columns = {'ACC_1':'ACC_X','ACC_2':'ACC_Y','ACC_3':'ACC_Z'})

        max_time = subject_acc_data.shape[0]/sample_rate
        time_steps = np.linspace(0, max_time, subject_acc_data.shape[0])

        plt.figure(figsize=(10,5))
        #ACC_X = range(0,subject_acc_data.shape[0])
        ACC_X = time_steps
        plt.plot(ACC_X,subject_acc_data.ACC_X,label = "accelerometer channel-X")
        plt.plot(ACC_X,subject_acc_data.ACC_Y,label = "accelerometer channel-Y")
        plt.plot(ACC_X,subject_acc_data.ACC_Z,label = "accelerometer channel-Z")
        plt.title('Accelerometer graph for Subject:'+str(subject) +' for label:'+label + 
                  ' [Records Count:'+str(num_of_records)+']')
        plt.legend(loc=1)
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.show()