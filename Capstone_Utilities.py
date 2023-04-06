import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Function to create downsample records from input samples as per specified frequency
# Parameters:
# Input: 
# datasets: dataframe to be downsampled
# Output/Return : 
# downsampled dataset
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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