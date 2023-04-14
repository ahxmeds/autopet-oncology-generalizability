#%%
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
)
from monai.networks.layers import Norm
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet, DynUNet, SwinUNETR, UNETR
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
import torch
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
# from torchsummary import summary

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

def create_dictionary_ctptgt(ctpaths, ptpaths, gtpaths):
    data = []
    for i in range(len(gtpaths)):
        ctpath = ctpaths[i]
        ptpath = ptpaths[i]
        gtpath = gtpaths[i]
        data.append({'CT':ctpath, 'PT':ptpath, 'GT':gtpath})
    return data

def create_dictionary_ptgt(ptpaths, gtpaths):
    data = []
    for i in range(len(gtpaths)):
        ptpath = ptpaths[i]
        gtpath = gtpaths[i]
        data.append({'PT':ptpath, 'GT':gtpath})
    return data

def get_ctptgt_paths_from_df(df):
    ctpaths, ptpaths, gtpaths = [],[],[]
    autopet_datadir =  '/data/blobfuse/default/autopet2022_data'
    images_dir = os.path.join(autopet_datadir, 'images')
    labels_dir = os.path.join(autopet_datadir, 'labels')
    
    for index, row in df.iterrows():
        patientid = row['PatientID']
        ctpath = os.path.join(images_dir, f"{patientid}_0000.nii.gz")
        ptpath = os.path.join(images_dir, f"{patientid}_0001.nii.gz")
        gtpath = os.path.join(labels_dir, f"{patientid}.nii.gz")
        ctpaths.append(ctpath)
        ptpaths.append(ptpath)
        gtpaths.append(gtpath)
    return ctpaths, ptpaths, gtpaths

def get_train_valid_paths_dictionary(df, fold):
    valid_df = df[df['TRAIN/TEST'] == f'TRAIN_{str(fold)}']
    train_df = df[(df['TRAIN/TEST'] != f'TRAIN_{str(fold)}') & (df['TRAIN/TEST'] != f'TEST')]

    ctpaths_train, ptpaths_train, gtpaths_train = get_ctptgt_paths_from_df(train_df)
    ctpaths_valid, ptpaths_valid, gtpaths_valid = get_ctptgt_paths_from_df(valid_df)
    
    train_data = train_data = create_dictionary_ctptgt(ctpaths_train, ptpaths_train, gtpaths_train)
    valid_data = create_dictionary_ctptgt(ctpaths_valid, ptpaths_valid, gtpaths_valid)
    return train_data, valid_data

#%%
fold = 0
network = 'unet'
disease = 'lungcancer'
inputtype = 'ctpt'
inputsize = 'randcrop192'
# extrafeature = 'nopmbclbccv'
experiment_code = f"{network}_{disease}_fold{str(fold)}_{inputtype}_{inputsize}"
trsz = 192
spatialsize = (trsz, trsz, trsz)
swsz = 192
#%%
metadata_path = f'/home/jhubadmin/Projects/autopet-oncology-generalizability/create_data_split/metadata_{disease}.csv'
metadata_df = pd.read_csv(metadata_path)
train_data, valid_data = get_train_valid_paths_dictionary(metadata_df, fold)

#%%
mod_keys = ['CT', 'PT', 'GT']
train_transforms = Compose(
    [
        LoadImaged(keys=mod_keys),
        EnsureChannelFirstd(keys=mod_keys),
        ScaleIntensityRanged(keys=['CT'], a_min=-1024, a_max=1024, b_min=0, b_max=1, clip=True),
        ScaleIntensityd(keys=['PT'], minv=0, maxv=1),
        CropForegroundd(keys=mod_keys, source_key='PT'),
        Orientationd(keys=mod_keys, axcodes="RAS"),
        Spacingd(keys=mod_keys, pixdim=(2.0, 2.0, 2.0), mode=('bilinear', 'bilinear', 'nearest')),
        RandCropByPosNegLabeld(
            keys=mod_keys,
            label_key='GT',
            spatial_size = spatialsize,
            pos=5,
            neg=1,
            num_samples=1,
            image_key='PT',
            image_threshold=0,
        ),
        RandAffined(
            keys=mod_keys,
            mode=('bilinear', 'bilinear', 'nearest'),
            prob=0.5,
            spatial_size = spatialsize,
            translate_range=(10,10,10),
            rotate_range=(0, 0, np.pi/15),
            scale_range=(0.1, 0.1, 0.1)),
        Rand3DElasticd(
            keys=mod_keys,
            sigma_range=(0.0, 1.0),
            magnitude_range=(0.0, 1.0),
            spatial_size = spatialsize,
            prob=0.4,
        ),
        ConcatItemsd(keys=['CT', 'PT'], name='CTPT', dim=0),
        DeleteItemsd(keys=['CT', 'PT'])
    ]
)
# %%
valid_transforms = Compose(
    [
        LoadImaged(keys=mod_keys),
        EnsureChannelFirstd(keys=mod_keys),
        ScaleIntensityRanged(keys=['CT'], a_min=-1024, a_max=1024, b_min=0, b_max=1, clip=True),
        ScaleIntensityd(keys=['PT'], minv=0, maxv=1),
        CropForegroundd(keys=mod_keys, source_key='PT'),
        Orientationd(keys=mod_keys, axcodes="RAS"),
        Spacingd(keys=mod_keys, pixdim=(2.0, 2.0, 2.0), mode=('bilinear', 'bilinear', 'nearest')),
        ConcatItemsd(keys=['CT', 'PT'], name='CTPT', dim=0),
        DeleteItemsd(keys=['CT', 'PT'])
    ]
)
#%%
dataset_train = CacheDataset(data=train_data, transform=train_transforms, cache_rate=1, num_workers=16)
dataloader_train = DataLoader(dataset_train, batch_size=8, shuffle=True, num_workers=16)

#%%
dataset_valid = CacheDataset(data=valid_data, transform=valid_transforms, cache_rate=1, num_workers=16)
dataloader_valid = DataLoader(dataset_valid, batch_size=1, shuffle=False, num_workers=16)


#%%
device = torch.device("cuda:0")
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


loss_function = DiceLoss(to_onehot_y=True, softmax=True) 
optimizer = torch.optim.Adam(model.parameters(), 1e-4)
dice_metric = DiceMetric(include_background=False, reduction="mean")
scheduler = CosineAnnealingLR(optimizer, T_max=1200, eta_min=0)

#%%
save_models_dir = '/data/blobfuse/default/autopet_generalizability_results/saved_models_folds/segmentation3d/'
save_models_dir = os.path.join(save_models_dir, 'fold'+str(fold), network, experiment_code)
os.makedirs(save_models_dir, exist_ok=True)
#%%
max_epochs = 1200
val_interval = 2
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
post_label = Compose([AsDiscrete(to_onehot=2)])

train_metric_values = []

save_logs_dir = '/data/blobfuse/default/autopet_generalizability_results/saved_logs_folds/segmentation3d/'
save_logs_dir = os.path.join(save_logs_dir, 'fold'+str(fold), network, experiment_code)
os.makedirs(save_logs_dir, exist_ok=True)

trainlog_fpath = os.path.join(save_logs_dir, 'trainloss.csv')
validlog_fpath = os.path.join(save_logs_dir, 'validdice.csv')
#%%
experiment_start_time = time.time()

for epoch in range(max_epochs):
    epoch_start_time = time.time()
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in dataloader_train:
        step += 1
        inputs, labels = (
            batch_data['CTPT'].to(device),
            batch_data['GT'].to(device),
        )
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(
            f"{step}/{len(dataset_train) // dataloader_train.batch_size}, "
            f"train_loss: {loss.item():.4f}")
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    # steps forward the CosineAnnealingLR scheduler
    scheduler.step()

    epoch_loss_values_df = pd.DataFrame(data=epoch_loss_values, columns=['loss'])
    epoch_loss_values_df.to_csv(trainlog_fpath, index=False)


    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            for val_data in dataloader_valid:
                val_inputs, val_labels = (
                    val_data['CTPT'].to(device),
                    val_data['GT'].to(device),
                )
                roi_size = (swsz, swsz, swsz)
                sw_batch_size = 4
                val_outputs = sliding_window_inference(
                    val_inputs, roi_size, sw_batch_size, model)
                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                # compute metric for current iteration
                dice_metric(y_pred=val_outputs, y=val_labels)

            # aggregate the final mean dice result
            metric = dice_metric.aggregate().item()
            # reset the status for next validation round
            dice_metric.reset()

            metric_values.append(metric)
            metric_values_df = pd.DataFrame(data=metric_values, columns=['Metric'])
            metric_values_df.to_csv(validlog_fpath, index=False)

            torch.save(model.state_dict(), os.path.join(save_models_dir, "model_ep="+convert_to_4digits(str(int(epoch+1)))+".pth"))

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f"\nbest mean dice: {best_metric:.4f} "
                f"at epoch: {best_metric_epoch}"
            )
    epoch_elapsed_time = time.time() - epoch_start_time
    print(f"Time taken for epoch {epoch + 1}: {epoch_elapsed_time/60} mins")
# %%
experiment_elapsed_time = time.time() - experiment_start_time
print(f"Time taken for the experiment: {experiment_elapsed_time/(60*60)} hrs")