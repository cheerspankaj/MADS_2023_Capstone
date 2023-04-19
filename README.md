# Capstone Team- 8 Digital Biomaker
## <img width="400" height="400" alt="Screenshot 2023-04-17 at 6 18 57 PM" src="https://user-images.githubusercontent.com/108576861/232935357-dc2101b3-0aac-420d-8aa6-16e0da3f3ed9.png">  <img width="400" height="400" alt="Screenshot 2023-04-17 at 6 29 28 PM" src="https://user-images.githubusercontent.com/108576861/232935597-e05a0f4d-0964-40eb-9456-115461516b6d.png">

Image source Link:https://whiterose.ac.uk/news/digital-health-medical-technologies/

Image source Link: http://pure-oai.bham.ac.uk/ws/files/144982402/IqbalT2021sensitivity.pdf

## Objectives

This machine learning project will assist individuals to accomplish the following tasks.
1) Analyze wearable data signals for different effective states e.g. stress, amusement etc.
2) Obtain insights about the different effective states that exist in the wearable data by running machine learning pipeline along with accuracy score, F1 score, Confusion matrix and Classification report.

## Datasets

The primary dataset contains wearable sensors data for 15 subjects contained in the individual subject folder as pickle file. As part of data preprocessing all pickle file data is combined/merged into two dataset i.e. training dataset and test dataset respectively.
In addition to the primary dataset, two secondary datasets were created using the information provided for individual subjects and subject reported outcomes.

Primary dataset: Publicly available
Link: https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download

Secondary dataset : Created from the publicly available dataset.

1) subject_personal_info.xlsx

2) survey_ref_table_subject_responses.csv

## Data Preparation and Feature Engineering

## ![Screenshot 2023-04-18 at 8 37 15 PM](https://user-images.githubusercontent.com/108576861/232936374-97f2ff63-3c2f-430f-863c-c86e13010ef1.png)

Image source Link :https://flirt.readthedocs.io/en/latest/index.html

To perform the data preprocessing and feature engineering, please follow the below steps.

1. Download the following datasets from Google Drive to expedite the dataset creation step (commented in the master notebook as it takes significant time).

- chest_data_with_label_12_subjects_TRAIN_DATA_with_survey.csv

- chest_data_with_label_3_subjects_TEST_DATA_with_survey.csv

- subject_personal_info.xlsx

- survey_ref_table_subject_responses.csv

- Location:https://drive.google.com/drive/folders/1isguiTfbAOuZ2n0sZ6CGZwz3Dw-ql2A7?usp=share_link
- Format: CSV
- Access Method: Download

2. Download the followings python files and notebook from the project Git repository

- Capstone_Data_Vis_Common.py

- Capstone_Feature_Extraction.py

- Capstone_Data_Prep_TRAIN_TEST_SPLIT_with_survey_data.py

- Capstone_Utilities

- Capstone_StressDetection_Master.ipynb

3. Upload above files in the jupyter notebook environment working directory.

4. Open the notebook and uncomment the commented cell for first time execution to allow python 
libraries be installed and training and test dataset gets created. Creating training and test dataset may take significant time, to expedite this step, please download the data set from Git project repository data folder.

5. Comment the uncommented cells after the first run to avoid re-installation and recreation of training and test datasets.

6. Run all the cells and verify the cell outputs for graphs and final dataset “Capstone_final_dataset_for_ml_50ms.csv” is created in the working directory upon successful execution of the notebook.

7. The final dataset will be used in the Machine learning modeling steps for training different classifiers.

## Machine Learning Models

Following Machine Learning models developed for model selection and uploaded on git repository.

- Feed Forward Neural Network (using pytorch)
- Naive Bayes
- Multi-Layer Perceptron (using sklearn)
- Logistic Regression
- Random Forest 
- Convolutional Neural Network

Link for Notebooks - https://github.com/cheerspankaj/MADS_2023_Capstone

Please follow the below steps to run the individual notebooks for the above mentioned machine learning classifiers.

1. Execute “Data Preparation and Feature Engineering” steps (mentioned in the above section).

2. Download the notebook for above mentioned classifiers from the project git repository.

3. Upload to the current Jupyter environment working directory.

4. Open the notebook for the classifier to be executed.

5. Run all the cells in the notebook.

6. Analyze the outputs for example accuracy, F1 scores, classification report etc.

Project Git Repository link - https://github.com/cheerspankaj/MADS_2023_Capstone
