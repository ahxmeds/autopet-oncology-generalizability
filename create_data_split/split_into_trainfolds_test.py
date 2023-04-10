#%%
import pandas as pd 
import numpy as np 
import os 
import glob 
from sklearn.model_selection import train_test_split
# %%
def split_train_folds_test(data, trn_size=115, savepath=''):
    # patientIDs = list(data['PatientID'].values)
    train_data, test_data = train_test_split(data, train_size=trn_size)
    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)
    
    folds = [0, 1, 2, 3, 4]
    train_extra_column = np.array([[f"TRAIN_{folds[i]}"]*int(len(train_data)/5) for i in range(len(folds))])
    train_extra_column = train_extra_column.flatten()
    train_extra_column_df = pd.DataFrame(data=train_extra_column, columns=['TRAIN/TEST'])
    train_final = pd.concat([train_data, train_extra_column_df], axis=1)
    test_extra_column = [f"TEST"]*int(len(test_data))
    test_extra_column_df = pd.DataFrame(data=test_extra_column, columns=['TRAIN/TEST'])
    test_final = pd.concat([test_data, test_extra_column_df], axis=1)
    df_final = pd.concat([train_final, test_final], axis=0)
    df_final.to_csv(savepath, index=False)
    # %%
autopetdir = '/data/blobfuse/autopet2022_data'
#%%
disease_data = os.path.join(autopetdir, 'metadata_lymphoma.csv')
path = 'metadata_lymphoma.csv'
data = pd.read_csv(disease_data)
split_train_folds_test(data, trn_size=115, savepath=path)
# %%
disease_data = os.path.join(autopetdir, 'metadata_lungcancer.csv')
path = 'metadata_lungcancer.csv'
data = pd.read_csv(disease_data)
split_train_folds_test(data, trn_size=135, savepath=path)

# %%
disease_data = os.path.join(autopetdir, 'metadata_melanoma.csv')
path = 'metadata_melanoma.csv'
data = pd.read_csv(disease_data)
split_train_folds_test(data, trn_size=150, savepath=path)

# %%
