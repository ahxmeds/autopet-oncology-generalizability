#%%
import matplotlib.pyplot as plt
import pandas as pd 
import os
import numpy as np 
import seaborn as sb
#%%
def plot_training_validation_metrics(train_loss, val_metrics):
    fig, ax = plt.subplots(1,2, figsize=(10,5))
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(0.7)

    epochs = [i + 1 for i in range(len(train_loss))]
    ax[0].plot(epochs, train_loss, '-')
    min_loss_epoch = 1+np.argmin(train_loss)
    min_loss = np.min(train_loss)
    ax[0].plot(min_loss_epoch, min_loss, '-o', color='red')
    ax[0].legend(['Train loss'])
    ax[0].set_xlabel('Epochs')
    ax[0].set_title('Train loss')
    
    # epochs = [i + 1 for i in range(len(train_metrics))]

    # ax[1].plot(epochs, train_metrics, '-o')
    # ax[1].legend(['Train DSC'])
    # ax[1].set_xlabel('Epochs')
    # ax[1].set_title('Train DSC')

    val_interval = 2
    epochs2 = [val_interval * (i + 1) for i in range(len(val_metrics))]
    ax[1].plot(epochs2, val_metrics, '-')
    max_dsc_epoch = 2*(np.argmax(val_metrics) + 1)
    max_dsc = np.max(val_metrics)
    ax[1].plot(max_dsc_epoch, max_dsc, '-o', color='red')
    ax[1].legend(['Validation DSC'])
    ax[1].set_xlabel('Epochs')
    ax[1].set_title('Valid DSC')
    plt.show()

def plot_training_validation_metrics_given_expcode(save_logs_dir):
    trainlossfilename = os.path.join(save_logs_dir, 'trainloss.csv')
    validdscfilename = os.path.join(save_logs_dir, 'validdice.csv')
    trainlossdata = pd.read_csv(trainlossfilename)
    validdicedata = pd.read_csv(validdscfilename)

    plot_training_validation_metrics(trainlossdata['loss'], validdicedata['Metric'])
    print(np.max(validdicedata['Metric']))
    print(len(validdicedata))
    print(len(trainlossdata))


#%%
fold = 0
network = 'unet'
disease = 'lymphoma'
inputtype = 'ctpt'
inputsize = 'randcrop192'
# extra_features = ''# '_no3delastica
# ugmentation'
# sizes = [96, 128, 160, 192, 224, 256, 288]
experiment_code = f"{network}_{disease}_fold{str(fold)}_{inputtype}_{inputsize}"
save_logs_dir = '/data/blobfuse/default/autopet_generalizability_results/saved_logs_folds/segmentation3d'
save_logs_dir = os.path.join(save_logs_dir, 'fold'+str(fold), network, experiment_code)
# save_logs_dir = '/fold0/unet/unet_lymphoma_fold0_ctpt_randcrop192'
print('Lymphoma')
plot_training_validation_metrics_given_expcode(save_logs_dir)
# %%

fold = 0
network = 'unet'
disease = 'lungcancer'
inputtype = 'ctpt'
inputsize = 'randcrop192'
# extra_features = ''# '_no3delasticaugmentation'
# sizes = [96, 128, 160, 192, 224, 256, 288]
experiment_code = f"{network}_{disease}_fold{str(fold)}_{inputtype}_{inputsize}"
save_logs_dir = '/data/blobfuse/default/autopet_generalizability_results/saved_logs_folds/segmentation3d'
save_logs_dir = os.path.join(save_logs_dir, 'fold'+str(fold), network, experiment_code)
# save_logs_dir = '/fold0/unet/unet_lymphoma_fold0_ctpt_randcrop192'
print('Lymphoma')
plot_training_validation_metrics_given_expcode(save_logs_dir)
# %%
