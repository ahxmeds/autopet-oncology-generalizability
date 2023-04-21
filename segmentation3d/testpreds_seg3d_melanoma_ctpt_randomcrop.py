#%%
import numpy as np 
import glob
import os 
import pandas as pd 
import shutil
import SimpleITK as sitk
from torch.nn.functional import one_hot 
import torch
from monai.metrics import compute_hausdorff_distance
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityRanged,
    ScaleIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    ConcatItemsd,
    DeleteItemsd,
    CastToTyped,
    Invertd,
)

from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
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


def find_epochmodel_swsize_for_this_experiment_code(ablation_dir):
    ablation_studies = sorted(glob.glob(os.path.join(ablation_dir, '*.csv')))
    swsizes_max_epoch_dict = {}
    
    for studypath in ablation_studies:
        swsize = int(os.path.basename(studypath).split('_')[-1][:-4])
        swsizes_max_epoch_dict[swsize] = {}
        data = pd.read_csv(studypath)
        best_epoch = 2*(1 + np.argmax(data['SZ'+str(swsize)]))
        best_epoch_valid_dsc = np.max(data['SZ'+str(swsize)])
        swsizes_max_epoch_dict[swsize]['Epoch'] = best_epoch
        swsizes_max_epoch_dict[swsize]['Valid DSC'] = best_epoch_valid_dsc
    
    bestbest_epoch = -1
    bestbest_epoch_valid_dsc = -1
    bestbest_swsize = -1

    for item in swsizes_max_epoch_dict:
        validdsc = swsizes_max_epoch_dict[item]['Valid DSC']

        if validdsc > bestbest_epoch_valid_dsc:
            bestbest_epoch_valid_dsc = validdsc 
            bestbest_epoch = swsizes_max_epoch_dict[item]['Epoch']
            bestbest_swsize = item

    return swsizes_max_epoch_dict, bestbest_epoch, bestbest_epoch_valid_dsc, bestbest_swsize
        

#%%
fold = 3
network = 'unet'
disease = 'melanoma'
inputtype = 'ctpt'
inputsize = 'randcrop192'
# extrafeatures = '_nopmbclbccv'
experiment_code =f"{network}_{disease}_fold{str(fold)}_{inputtype}_{inputsize}"

testing_on_disease = 'melanoma'

save_logs_dir = '/data/blobfuse/default/autopet_generalizability_results/saved_logs_folds/segmentation3d'
validdscfname = os.path.join(save_logs_dir, 'fold'+str(fold), network, experiment_code, 'validdice.csv')
data = pd.read_csv(validdscfname)
epoch_max_valid_dsc = 2*(np.argmax(data['Metric']) + 1)
best_model_fname = 'model_ep=' + convert_to_4digits(str(epoch_max_valid_dsc)) +'.pth'
print(f"Using the model at epoch={epoch_max_valid_dsc} with mean valid DSC = {round(data['Metric'].max(), 4)}")
#%%
#%%
save_models_dir = '/data/blobfuse/default/autopet_generalizability_results/saved_models_folds/segmentation3d'
best_model_fname = 'model_ep=' + convert_to_4digits(str(epoch_max_valid_dsc)) +'.pth'
save_models_path = os.path.join(save_models_dir, 'fold'+str(fold), network, experiment_code, best_model_fname)

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


# fold_fpaths_dir = 'data_preprocessing_tools/train_valid_test_splits_paths_default/fold'+str(fold)
# test_fpaths_fname = os.path.join(fold_fpaths_dir, 'testfiles_fold'+str(fold)+'.csv')
# test_fpaths_df = pd.read_csv(test_fpaths_fname)
# ctpaths_test, ptpaths_test, gtpaths_test = list(test_fpaths_df['CTPATHS'].values), list(test_fpaths_df['PTPATHS'].values),  list(test_fpaths_df['GTPATHS'].values)
autopet_diagnosis_csvfpath = f'/home/jhubadmin/Projects/autopet-oncology-generalizability/create_data_split/metadata_{testing_on_disease}.csv'
autopet_images_dir = '/data/blobfuse/default/autopet2022_data/images'
autopet_labels_dir = '/data/blobfuse/default/autopet2022_data/labels'
diagnosis_df = pd.read_csv(autopet_diagnosis_csvfpath)
diagnosis_df_test = diagnosis_df[diagnosis_df['TRAIN/TEST'] == 'TEST']
patientIDs = list(diagnosis_df_test['PatientID'])
ctpaths_test, ptpaths_test, gtpaths_test = create_ctpaths_ptpaths_gtpaths(patientIDs, autopet_images_dir, autopet_labels_dir)
# %%
####################### creating dictionary for train and valid images #######################################
test_data = create_dictionary_ctptgt(ctpaths_test, ptpaths_test, gtpaths_test)

mod_keys = ['CT', 'PT', 'GT']
test_transforms = Compose(
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
save_preds_dir = '/data/blobfuse/default/autopet_generalizability_results/saved_testpreds_folds/segmentation3d'
save_preds_dir = os.path.join(save_preds_dir, 'fold'+str(fold), network, experiment_code)
save_preds_dir = os.path.join(save_preds_dir, f'test_{testing_on_disease}')
os.makedirs(save_preds_dir, exist_ok=True)
#%%
post_transforms = Compose([
    Invertd(
        keys="Pred",
        transform=test_transforms,
        orig_keys="GT",
        meta_keys="pred_meta_dict",
        orig_meta_keys="image_meta_dict",
        meta_key_postfix="meta_dict",
        nearest_interp=False,
        to_tensor=True,
    ),
    AsDiscreted(keys="Pred", argmax=True),
    SaveImaged(keys="Pred", meta_keys="pred_meta_dict", output_dir=save_preds_dir, output_postfix="", separate_folder=False, resample=False),
])
#%%
dataset_test = Dataset(data=test_data, transform=test_transforms)#, cache_rate=1, num_workers=16)
dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=16)

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
model.load_state_dict(torch.load(save_models_path))

#%%
bestbest_swsize=192
model.eval()
with torch.no_grad():
    for data in dataloader_test:
        inputs = data['CTPT'].to(device)
        roi_size = (bestbest_swsize, bestbest_swsize, bestbest_swsize)
        sw_batch_size = 4
        data['Pred'] = sliding_window_inference(inputs, roi_size, sw_batch_size, model)
        data = [post_transforms(i) for i in decollate_batch(data)]
#%%
# Now we will compute all the metrics on the test set with respect to gt masks
def read_image_array(path):
    img =  sitk.ReadImage(path)
    array = np.transpose(sitk.GetArrayFromImage(img), (2,1,0))
    return array

def calculate_dsc(gt, pred):
    dsc = 2.0*np.sum(pred[gt==1])/(np.sum(pred) + np.sum(gt))
    return dsc

def calculate_jaccard(gt, pred):
    intersection = np.sum(pred[gt==1])
    union = np.sum(pred) + np.sum(gt) - intersection
    jsc = intersection/union
    return jsc

def calculate_HD(gt, pred):
    gt_tensor = torch.tensor(gt, dtype=torch.int64)
    pred_tensor = torch.tensor(pred, dtype=torch.int64)

    gt_oh = one_hot(gt_tensor, num_classes=2)
    pred_oh = one_hot(pred_tensor, num_classes=2)

    gt_oh_np = gt_oh.numpy()
    pred_oh_np = pred_oh.numpy()

    gt_oh_np_transp = np.transpose(gt_oh_np, (3, 0, 1, 2))
    pred_oh_np_transp = np.transpose(pred_oh_np, (3, 0, 1, 2))

    gt_new_tensor = torch.tensor(gt_oh_np_transp, dtype=torch.int64)
    pred_new_tensor = torch.tensor(pred_oh_np_transp, dtype=torch.int64)

    gt_final = torch.unsqueeze(gt_new_tensor, dim=0)
    pred_final = torch.unsqueeze(pred_new_tensor, dim=0)
    hd = compute_hausdorff_distance(pred_final, gt_final, include_background=False, percentile=95.0)
    return hd.item()

#%%
predpaths_test = sorted(glob.glob(os.path.join(save_preds_dir, '*.nii.gz')))
gtpaths_test = sorted(gtpaths_test)

# %%
imageids = [os.path.basename(path)[:-7] for path in gtpaths_test]
test_dscs = []
test_jaccards = []
test_hds = []

# %%
for i in range(len(gtpaths_test)):
    gtpath = gtpaths_test[i]
    predpath = predpaths_test[i]

    gtarray = read_image_array(gtpath)
    predarray = read_image_array(predpath)

    dsc = calculate_dsc(gtarray, predarray)
    jsc = calculate_jaccard(gtarray, predarray)
    hd = calculate_HD(gtarray, predarray)

    test_dscs.append(dsc)
    test_jaccards.append(jsc)
    test_hds.append(hd)

    print(imageids[i])
    print("DSC:", dsc)
    print("JSC:", jsc)
    print("HD:", hd)
    print('\n')


# %%
save_testmetrics_dir = '/data/blobfuse/default/autopet_generalizability_results/saved_testmetrics_folds/segmentation3d'
save_testmetrics_dir = os.path.join(save_testmetrics_dir, 'fold'+str(fold), network, experiment_code)
save_testmetrics_dir = os.path.join(save_testmetrics_dir, f'test_{testing_on_disease}')
os.makedirs(save_testmetrics_dir, exist_ok=True)
save_testmetrics_fname = os.path.join(save_testmetrics_dir, 'testmetrics.csv')


data = np.column_stack((imageids, test_dscs, test_jaccards, test_hds))
data_df = pd.DataFrame(data=data, columns=['PatientID', 'DSC', 'Jaccard', '95HD'])
data_df.to_csv(save_testmetrics_fname, index=False)
# %%

mean_dsc = round(np.mean(test_dscs), 4)
std_dsc = round(np.std(test_dscs), 4)
median_dsc  = round(np.median(test_dscs), 4)
mean_jaccard = round(np.mean(test_jaccards), 4)
std_jaccard = round(np.std(test_jaccards), 4)
mean_hd = round(np.nanmean(test_hds), 4)
std_hd = round(np.nanstd(test_hds), 4)
#%%
print(f"Test set performance summary: {inputsize}")
print(f'Train disease: {disease}')
print(f'Test disease: {testing_on_disease}')
print(f"DSC: {mean_dsc} +/- {std_dsc}")
print(f"Median DSC: {median_dsc}")
print(f"Jaccard Index: {mean_jaccard} +/- {std_jaccard}")
print(f"95% HD: {mean_hd} +/- {std_hd}")
# %%

# #%%
# def check_same_geometry(ctpath, ptpath, gtpath):
#     ctimg = sitk.ReadImage(ctpath)
#     ptimg = sitk.ReadImage(ptpath)
#     gtimg = sitk.ReadImage(gtpath)

#     ptid = os.path.basename(gtpath)
#     print(ptid)
#     print(ctimg.GetSize())
#     print(ptimg.GetSize())
#     print(gtimg.GetSize())
#     print('\n')
#     print(ctimg.GetOrigin())
#     print(ptimg.GetOrigin())
#     print(gtimg.GetOrigin())
#     print('\n')
#     print(ctimg.GetSpacing())
#     print(ptimg.GetSpacing())
#     print(gtimg.GetSpacing())
#     print('\n')
#     print(ctimg.GetDirection())
#     print(ptimg.GetDirection())
#     print(gtimg.GetDirection())
#     print('\n')


# for i in range(len(ctpaths_test)):
#     ctpath = ctpaths_test[i]
#     ptpath = ptpaths_test[i]
#     gtpath = gtpaths_test[i]
#     check_same_geometry(ctpath, ptpath, gtpath)

# %%
