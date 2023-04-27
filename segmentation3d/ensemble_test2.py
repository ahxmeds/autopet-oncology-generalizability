#%%
from abc import ABC, abstractmethod
import logging
import os
import tempfile
import shutil
import sys

import nibabel as nib
import numpy as np
import torch
import pandas as pd 
from monai.apps import CrossValidation
from monai.config import print_config
from monai.data import CacheDataset, DataLoader, create_test_image_3d
from monai.engines import EnsembleEvaluator, SupervisedEvaluator, SupervisedTrainer
from monai.handlers import MeanDice, StatsHandler, ValidationHandler, from_engine
from monai.inferers import SimpleInferer, SlidingWindowInferer
from monai.losses import DiceLoss
from monai.networks.nets import UNet
from monai.transforms import (
    Activationsd,
    EnsureChannelFirstd,
    AsDiscreted,
    Compose,
    LoadImaged,
    MeanEnsembled,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ScaleIntensityd,
    EnsureTyped,
    VoteEnsembled,
)
import torch.nn as nn
from monai.networks.layers import Norm
from monai.utils import set_determinism
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    DeleteItemsd,
    ScaleIntensityd,
    Spacingd,
    RandAffined,
    Rand3DElasticd,
    ConcatItemsd,
    ScaleIntensityRanged,
    EnsureTyped,
    MeanEnsembled,
    Activationsd,
    AsDiscreted,
    VoteEnsembled,
)
print_config()
# %%
directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)
# %%
set_determinism(seed=0)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
device = torch.device("cuda:0")

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
def get_base_model(device):    
    model = UNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=2,
        channels=(16, 32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    return model
#%%
def get_all_CVmodels(save_logs_dir, save_models_dir, train_on_disease='lymphoma'):
    models = []
    fold = [0, 1, 2, 3, 4]
    network = 'unet'
    disease = train_on_disease
    inputtype = 'ctpt'
    inputsize = 'randcrop192'
    experiment_code = [f"{network}_{disease}_fold{str(f)}_{inputtype}_{inputsize}" for f in fold]
    valid_logs_fname = [os.path.join(save_logs_dir, 'fold'+str(fold[i]), network, experiment_code[i], 'validdice.csv') for i in range(len(experiment_code))]
    data = [pd.read_csv(fname) for fname in valid_logs_fname]
    epoch_max_valid_dsc = [2*(np.argmax(d['Metric']) + 1) for d in data]
    max_valid_dsc = [np.max(d['Metric']) for d in data]
    best_model_fname = ['model_ep=' + convert_to_4digits(str(epoch)) +'.pth' for epoch in epoch_max_valid_dsc]
    
    save_models_path = [os.path.join(save_models_dir, 'fold'+str(fold[i]), network, experiment_code[i], best_model_fname[i]) for i in range(len(best_model_fname))]
    models = [get_base_model(torch.device('cuda:0')) for _ in range(5)]
    for i in range(5):
        models[i].load_state_dict(torch.load(save_models_path[i]))
        # base_models[i] = base_models[i].module
    models = [models[i].module for i in range(len(models))]
    return models, max_valid_dsc
save_logs_dir = '/data/blobfuse/default/autopet_generalizability_results/saved_logs_folds/segmentation3d'
save_models_dir = '/data/blobfuse/default/autopet_generalizability_results/saved_models_folds/segmentation3d'
train_on_disease = 'lymphoma'
test_on_disease = 'lymphoma'
models, max_dscs = get_all_CVmodels(save_logs_dir, save_models_dir, train_on_disease)
#%%

def create_ctpaths_ptpaths_gtpaths(patientIDs, images_dir, labels_dir):
    ctpaths_test = []
    ptpaths_test = []
    gtpaths_test = []

    for ptid in patientIDs:
        ctpath = os.path.join(images_dir, ptid+'_0000.nii.gz')
        ptpath = os.path.join(images_dir, ptid+'_0001.nii.gz')
        gtpath = os.path.join(labels_dir, ptid+'.nii.gz')

        ctpaths_test.append(ctpath)
        ptpaths_test.append(ptpath)
        gtpaths_test.append(gtpath)
    
    return ctpaths_test, ptpaths_test, gtpaths_test

def create_dictionary_ctptgt(ctpaths, ptpaths, gtpaths):
    data = []
    for i in range(len(gtpaths)):
        ctpath = ctpaths[i]
        ptpath = ptpaths[i]
        gtpath = gtpaths[i]
        data.append({'CT':ctpath, 'PT':ptpath, 'label':gtpath})
    return data
test_on_disease = 'lymphoma'
autopet_diagnosis_csvfpath = f'/home/jhubadmin/Projects/autopet-oncology-generalizability/create_data_split/metadata_{test_on_disease}.csv'
autopet_images_dir = '/data/blobfuse/default/autopet2022_data/images'
autopet_labels_dir = '/data/blobfuse/default/autopet2022_data/labels'
diagnosis_df = pd.read_csv(autopet_diagnosis_csvfpath)
diagnosis_df_test = diagnosis_df[diagnosis_df['TRAIN/TEST'] == 'TEST']
patientIDs = list(diagnosis_df_test['PatientID'])
ctpaths_test, ptpaths_test, gtpaths_test = create_ctpaths_ptpaths_gtpaths(patientIDs, autopet_images_dir, autopet_labels_dir)
test_data = create_dictionary_ctptgt(ctpaths_test, ptpaths_test, gtpaths_test)
#%%
mod_keys = ['CT', 'PT', 'label']
test_transforms = Compose(
    [
        LoadImaged(keys=mod_keys),
        EnsureChannelFirstd(keys=mod_keys),
        ScaleIntensityRanged(keys=['CT'], a_min=-1024, a_max=1024, b_min=0, b_max=1, clip=True),
        ScaleIntensityd(keys=['PT'], minv=0, maxv=1),
        CropForegroundd(keys=mod_keys, source_key='PT'),
        Orientationd(keys=mod_keys, axcodes="RAS"),
        Spacingd(keys=mod_keys, pixdim=(2.0, 2.0, 2.0), mode=('bilinear', 'bilinear', 'nearest')),
        ConcatItemsd(keys=['CT', 'PT'], name='image', dim=0),
        DeleteItemsd(keys=['CT', 'PT']),
        EnsureTyped(keys=["image", "label"]),
    ]
)

dataset_test = CacheDataset(data=test_data[0:2], transform=test_transforms, cache_rate=0, num_workers=16)
dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=16)

# %%
device = torch.device('cuda:0')
def ensemble_evaluate(post_transforms, models):
    evaluator = EnsembleEvaluator(
        device=device,
        val_data_loader=dataloader_test,
        pred_keys=["pred0", "pred1", "pred2", "pred3", "pred4"],
        networks=models,
        inferer=SlidingWindowInferer(roi_size=(96, 96, 96), sw_batch_size=4, overlap=0.5),
        postprocessing=post_transforms,
        key_val_metric={
            "test_mean_dice": MeanDice(
                include_background=True,
                output_transform=from_engine(["pred", "label"]),
            )
        },
    )
    evaluator.run()
# %%
mean_post_transforms = Compose(
    [
        EnsureTyped(keys=["pred0", "pred1", "pred2", "pred3", "pred4"]),
        MeanEnsembled(
            keys=["pred0", "pred1", "pred2", "pred3", "pred4"],
            output_key="pred",
            # in this particular example, we use validation metrics as weights
            weights=[0.95, 0.94, 0.95, 0.94, 0.90],
        ),
        Activationsd(keys="pred", sigmoid=True),
        AsDiscreted(keys="pred", threshold=0.5),
    ]
)
ensemble_evaluate(mean_post_transforms, models)
# %%
