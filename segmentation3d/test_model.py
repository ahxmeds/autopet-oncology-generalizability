#%%
import logging
import os
import sys
import nibabel as nib
import numpy as np
import torch
import pandas as pd 
import SimpleITK as sitk
import glob
from torch.nn.functional import one_hot 
from monai.metrics import compute_hausdorff_distance
from monai.config import print_config
from monai.data import Dataset, CacheDataset, DataLoader
from monai.engines import EnsembleEvaluator
from monai.handlers import MeanDice, from_engine
from monai.inferers import SlidingWindowInferer
from monai.networks.nets import UNet
import torch.nn as nn
from monai.networks.layers import Norm
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    DeleteItemsd,
    ScaleIntensityd,
    Spacingd,
    ConcatItemsd,
    ScaleIntensityRanged,
    EnsureTyped,
    MeanEnsembled,
    Activationsd,
    AsDiscreted,
    VoteEnsembled,
    SaveImaged,
    Invertd
)
print_config()
#%%
def convert_to_4digits(str_num):
    if len(str_num) == 1:
        new_num = '000' + str_num
    elif len(str_num) == 2:
        new_num = '00' + str_num
    elif len(str_num) == 3:
        new_num = '0' + str_num
    else:
        new_num = str_num
    return new_num
#%%
save_logs_dir = '/data/blobfuse/default/autopet_generalizability_results/saved_logs_folds/segmentation3d'
save_models_dir = '/data/blobfuse/default/autopet_generalizability_results/saved_models_folds/segmentation3d'
train_on_disease = 'lymphoma'
test_on_disease = 'lymphoma'
ensemble_type = 'vote' 
device = torch.device('cpu')
fold = [0]
network = 'unet'
disease = 'lymphoma'
inputtype = 'ctpt'
inputsize = 'randcrop192'
experiment_code = [f"{network}_{disease}_fold{str(f)}_{inputtype}_{inputsize}" for f in fold]
valid_logs_fname = [os.path.join(save_logs_dir, 'fold'+str(fold[i]), network, experiment_code[i], 'validdice.csv') for i in range(len(experiment_code))]
data = [pd.read_csv(fname) for fname in valid_logs_fname]
epoch_max_valid_dsc = [2*(np.argmax(d['Metric']) + 1) for d in data]
max_valid_dsc = [np.max(d['Metric']) for d in data]
best_model_fname = ['model_ep=' + convert_to_4digits(str(epoch)) +'.pth' for epoch in epoch_max_valid_dsc]

save_models_path = [os.path.join(save_models_dir, 'fold'+str(fold[i]), network, experiment_code[i], best_model_fname[i]) for i in range(len(best_model_fname))]
    
# %%
PATH = save_models_path[0]

# %%
def get_base_model(device):    
    model = UNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=2,
        channels=(16, 32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    )
    # model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    return model
# %%
model = get_base_model(device)
# %%
model_state_dict = torch.load(PATH, map_location=device)

#%%
model.load_state_dict(model_state_dict)

#%%
model_state_dict = {k.removeprefix('module.'): v for k, v in model_state_dict.items()}
model.load_state_dict(model_state_dict, strict=False)

#%%
model_base = get_base_model(device)
torch.save(model_base.state_dict(), 'model.pth')
state_dict = torch.load('model.pth', map_location=device)
keys_base = list(state_dict.keys())
model_base.load_state_dict(state_dict)
#%%
model = get_base_model(device)
model_state_dict = torch.load(PATH, map_location=device)
model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
keys = list(state_dict.keys())
model.load_state_dict(model_state_dict)
# %%
