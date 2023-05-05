#%%
import numpy as np 
import pandas as pd 
import SimpleITK as sitk 
import os
import glob 
import matplotlib.pyplot as plt 
import seaborn as sns


def get_data_averaged_over_folds(log_fpaths):
    nrows, ncolumns = np.shape(logs_fpaths)[0], np.shape(logs_fpaths)[1] 
    dfs = [[pd.read_csv(logs_fpaths[i][j]) for j in range(ncolumns)] for i in range(nrows)]
    df_averaged_final = []
    for i in range(nrows):
        df0, df1, df2, df3, df4 = dfs[i]
        PatientIDs = df0['PatientID'].values
        df0_dscs = np.array(df0['DSC'].values)
        df1_dscs = np.array(df1['DSC'].values)
        df2_dscs = np.array(df2['DSC'].values)
        df3_dscs = np.array(df3['DSC'].values)
        df4_dscs = np.array(df4['DSC'].values)
        df0_jacc = np.array(df0['Jaccard'].values)
        df1_jacc = np.array(df1['Jaccard'].values)
        df2_jacc = np.array(df2['Jaccard'].values)
        df3_jacc = np.array(df3['Jaccard'].values)
        df4_jacc = np.array(df4['Jaccard'].values)
        df0_hd = np.array(df0['95HD'].values)
        df1_hd = np.array(df1['95HD'].values)
        df2_hd = np.array(df2['95HD'].values)
        df3_hd = np.array(df3['95HD'].values)
        df4_hd = np.array(df4['95HD'].values)
        averaged_dscs = (df0_dscs + df1_dscs + df2_dscs + df3_dscs + df4_dscs)/5
        averaged_jacc = (df0_jacc + df1_jacc + df2_jacc + df3_jacc + df4_jacc)/5
        averaged_hd = (df0_hd + df1_hd + df2_hd + df3_hd + df4_hd)/5
        averaged_data = np.column_stack((PatientIDs, averaged_dscs, averaged_jacc, averaged_hd))
        averaged_data_df = pd.DataFrame(averaged_data, columns=['PatientID', 'DSC', 'Jaccard', '95HD'])
        
        df_averaged_final.append(averaged_data_df)
    
    return df_averaged_final, dfs
        


#%%
def plot_boxplots_for_specific_train_on_disease_for_all_test_on_disease(logs_fpaths, train_on_disease_value, test_on_disease_values=['Lymphoma', 'Lung cancer', 'Melanoma']):
    nrows, ncolumns = np.shape(logs_fpaths)[0], np.shape(logs_fpaths)[1] 
    dfs = [[pd.read_csv(logs_fpaths[i][j]) for j in range(ncolumns)] for i in range(nrows)]
    
    n_test_on_diseases = np.shape(dfs)[0]
    n_foldensembles = np.shape(dfs)[1]

    # foldensemble_values = ['fold0', 'fold1', 'fold2', 'fold3', 'fold4', 'avg', 'wtavg', 'vote', 'staple']
    foldensemble_values = ['Averaged DSC over 5 folds', 'Avg', 'WtAvg', 'Vote', 'STAPLE']
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
    ax.set_ylim([0,1])
    sns.set_style("whitegrid")
    boxplot = sns.boxplot(data = df_concat, x='TestDisease', y='DSC', hue='FoldEnsemble', ax=ax, width=0.7, whis=1, showfliers=True)
    start, end = ax.get_ylim()
    ax.yaxis.set_ticks(np.arange(start, end, 0.1))

    ax.set_ylim([-0.1,1])
    sns.move_legend(ax, title='', loc='lower center', bbox_to_anchor=(0.5, -0.29), ncol=5)
    ax.set_xlabel('Test cancer type', fontsize=13)
    ax.set_ylabel('Dice similarity coefficient (patient-level)' , fontsize=13)
    ax.set_title(f'Training cancer type: {train_on_disease_value}', fontsize=14, fontweight='bold')
    fig.savefig(f'{train_on_disease_value}-unet.png', dpi=400, bbox_inches='tight')
    return dfs_new


# %%
fold = [0, 1, 2, 3, 4]
network = 'unet'
train_on_disease = 'lungcancer'
inputtype = 'ctpt'
inputsize = 'randcrop192'
test_on_disease = ['test_lymphoma', 'test_lungcancer', 'test_melanoma']
experiment_code = [f"{network}_{train_on_disease}_fold{str(f)}_{inputtype}_{inputsize}" for f in fold]
dir = '/data/blobfuse/default/autopet_generalizability_results/saved_testmetrics_folds/segmentation3d'
dir_folds = [os.path.join(dir, 'fold'+str(fold[i]), network, experiment_code[i]) for i in range(len(fold))]
logs_fpaths = [[os.path.join(dir_folds[i], test_on_disease[j], 'testmetrics.csv') for i in range(len(dir_folds))] for j in range(len(test_on_disease))]
# %%

data_averaged, dfs = get_data_averaged_over_folds(logs_fpaths)
fnames_averaged = [
    f'{train_on_disease}_lymphoma_averaged.csv',
    f'{train_on_disease}_lungcancer_averaged.csv',
    f'{train_on_disease}_melanoma_averaged.csv'
]
for i in range(len(data_averaged)):
    data_averaged[i].to_csv(fnames_averaged[i], index=False)


averaged_log_fpaths = [
    [fnames_averaged[0]],
    [fnames_averaged[1]],
    [fnames_averaged[2]]
]
#%%
ensembletype = ['avg', 'wtavg', 'vote', 'staple']
experiment_code = [f"{network}_{train_on_disease}_{e}_{inputtype}_{inputsize}" for e in ensembletype]
dir_ensembles = [os.path.join(dir, ensembletype[i], network, experiment_code[i]) for i in range(len(ensembletype))]
logs_fpaths_ensembles = [[os.path.join(dir_ensembles[i], test_on_disease[j], 'testmetrics.csv') for i in range(len(dir_ensembles))] for j in range(len(test_on_disease))]
# 
_ = [averaged_log_fpaths[i].extend(logs_fpaths_ensembles[i]) for i in range(len(averaged_log_fpaths))]

# %%
if train_on_disease == 'lymphoma':
    train_on_disease_value = 'Lymphoma'
elif train_on_disease == 'lungcancer':
    train_on_disease_value = 'Lung cancer'
else:
    train_on_disease_value = 'Melanoma' 
#%%
dfs = plot_boxplots_for_specific_train_on_disease_for_all_test_on_disease(averaged_log_fpaths, train_on_disease_value)


# %%
train_on_disease = 'melanoma'
test_on_disease = 'melanoma'
path = f'{train_on_disease}_{test_on_disease}_averaged.csv'
data = pd.read_csv(path)
dsc_mean = data['DSC'].mean()
dsc_std = data['DSC'].std()
dsc_median = data['DSC'].median()
print(f"Mean: {round(dsc_mean, 4)} +/- {round(dsc_std, 4)}")
print(f"Median: {round(dsc_median, 4)}")

## 

# %%
