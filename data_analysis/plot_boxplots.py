#%%
import numpy as np 
import pandas as pd 
import SimpleITK as sitk 
import os
import glob 
import matplotlib.pyplot as plt 
import seaborn as sns


#%%
def plot_boxplots_for_specific_train_on_disease_for_all_test_on_disease(logs_fpaths, train_on_disease_value, test_on_disease_values=['Lymphoma', 'LungCancer', 'Melanoma']):
    nrows, ncolumns = np.shape(logs_fpaths)[0], np.shape(logs_fpaths)[1] 
    dfs = [[pd.read_csv(logs_fpaths[i][j]) for j in range(ncolumns)] for i in range(nrows)]
    
    n_test_on_diseases = np.shape(dfs)[0]
    n_foldensembles = np.shape(dfs)[1]

    foldensemble_values = ['fold0', 'fold1', 'fold2', 'fold3', 'fold4', 'avg', 'wtavg']
    
    def add_new_columns(dfs):
        'Here I add columns: TrainDisease', 'TestDisease', 'FoldEnsemble'
        for i in range(n_test_on_diseases):
            for j in range(n_foldensembles):
                test_on_disease_value = test_on_disease_values[i]
                foldensemble_value = foldensemble_values[j]
                train_on_disease_column = [train_on_disease_value]*len(dfs[i][j])
                test_on_disease_column = [test_on_disease_value]*len(dfs[i][j])
                foldensemble_column = [foldensemble_value]*len(dfs[i][j])
                new_columns_data = np.column_stack(
                    (
                        train_on_disease_column,
                        test_on_disease_column,
                        foldensemble_column
                    )
                )
                new_columns_df = pd.DataFrame(data=new_columns_data, columns=['TrainDisease', 'TestDisease', 'FoldEnsemble'])
                dfs[i][j] = pd.concat([dfs[i][j], new_columns_df], axis=1)
        return dfs

    dfs_new = add_new_columns(dfs)
    dfs_for_concat = []
    for i in range(n_test_on_diseases):
        for j in range(n_foldensembles):
            dfs_for_concat.append(dfs_new[i][j])
    
    df_concat = pd.concat(dfs_for_concat, axis=0)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax = sns.boxplot(data = df_concat, x='TestDisease', y='DSC', hue='FoldEnsemble', ax=ax, width=0.5, whis=0.8).set(
        xlabel='TestDisease',
        ylabel='Dice Score'
    )

    return dfs_new


# %%
fold = [0, 1, 2, 3, 4]
network = 'unet'
train_on_disease = 'melanoma'
inputtype = 'ctpt'
inputsize = 'randcrop192'
test_on_disease = ['test_lymphoma', 'test_lungcancer', 'test_melanoma']
experiment_code = [f"{network}_{train_on_disease}_fold{str(f)}_{inputtype}_{inputsize}" for f in fold]
dir = '/data/blobfuse/default/autopet_generalizability_results/saved_testmetrics_folds/segmentation3d'
dir_folds = [os.path.join(dir, 'fold'+str(fold[i]), network, experiment_code[i]) for i in range(len(fold))]
logs_fpaths = [[os.path.join(dir_folds[i], test_on_disease[j], 'testmetrics.csv') for i in range(len(dir_folds))] for j in range(len(test_on_disease))]
# %%
ensembletype = ['avg', 'wtavg']
experiment_code = [f"{network}_{train_on_disease}_{e}_{inputtype}_{inputsize}" for e in ensembletype]
dir_ensembles = [os.path.join(dir, ensembletype[i], network, experiment_code[i]) for i in range(len(ensembletype))]
logs_fpaths_ensembles = [[os.path.join(dir_ensembles[i], test_on_disease[j], 'testmetrics.csv') for i in range(len(dir_ensembles))] for j in range(len(test_on_disease))]
# 
_ = [logs_fpaths[i].extend(logs_fpaths_ensembles[i]) for i in range(len(logs_fpaths))]

# %%

dfs = plot_boxplots_for_specific_train_on_disease_for_all_test_on_disease(logs_fpaths, 'Lymphoma')


# %%
