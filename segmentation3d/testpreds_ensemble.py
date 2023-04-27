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
# %%
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
    return models, max_valid_dsc

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
#%%
save_logs_dir = '/data/blobfuse/default/autopet_generalizability_results/saved_logs_folds/segmentation3d'
save_models_dir = '/data/blobfuse/default/autopet_generalizability_results/saved_models_folds/segmentation3d'
train_on_disease = 'lungcancer'
test_on_disease = 'lungcancer'
ensemble_type = 'vote' # weighted by Validation score on different folds

network = 'unet'
inputtype = 'ctpt'
inputsize = 'randcrop192'
experiment_code = f"{network}_{train_on_disease}_{ensemble_type}_{inputtype}_{inputsize}"

models, max_dscs = get_all_CVmodels(save_logs_dir, save_models_dir, train_on_disease)
if ensemble_type == 'wtavg':
    model_weights = max_dscs
elif ensemble_type == 'avg':
    model_weights = [1.0, 1.0, 1.0, 1.0, 1.0]
else:
    model_weights = None
#%%
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
    ]
)

dataset_test = Dataset(data=test_data, transform=test_transforms)#, cache_rate=0, num_workers=24)
dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=24)

# %%
def ensemble_evaluate(post_transforms, models):
    evaluator = EnsembleEvaluator(
        device=device,
        val_data_loader=dataloader_test,
        pred_keys=["pred0", "pred1", "pred2", "pred3", "pred4"],
        networks=models,
        inferer=SlidingWindowInferer(roi_size=(192,192, 192), sw_batch_size=4, overlap=0.5),
        postprocessing=post_transforms,
        key_val_metric={
            "test_mean_dice": MeanDice(
                include_background=False,
                output_transform=from_engine(["pred", "label"]),
                save_details=True
            )
        },
    )
    evaluator.run()
    return evaluator
# %%
save_preds_dir = '/data/blobfuse/default/autopet_generalizability_results/saved_testpreds_folds/segmentation3d'
save_preds_dir = os.path.join(save_preds_dir, ensemble_type, network, experiment_code)
save_preds_dir = os.path.join(save_preds_dir, f'test_{test_on_disease}')
os.makedirs(save_preds_dir, exist_ok=True)
#%%

mean_post_transforms = Compose(
    [
        EnsureTyped(keys=["pred0", "pred1", "pred2", "pred3", "pred4"]),
        MeanEnsembled(
            keys=["pred0", "pred1", "pred2", "pred3", "pred4"],
            output_key="pred",
            weights=model_weights,
        ),
        Activationsd(keys="pred", sigmoid=True),
        Invertd(
        keys=["pred","label"],
        transform=test_transforms,
        orig_keys="label",
        meta_keys=["pred_meta_dict", 'label_meta_dict'],
        orig_meta_keys="image_meta_dict",
        meta_key_postfix="meta_dict",
        nearest_interp=False,
        to_tensor=True,
        ),
        AsDiscreted(keys="pred", argmax=True, threshold=0.5),
        SaveImaged(keys="pred", meta_keys="image_meta_dict", output_dir=save_preds_dir, output_postfix="", output_ext=".nii.gz", separate_folder=False, resample=False)
    ]
)

vote_post_transforms = Compose(
    [
        EnsureTyped(keys=["pred0", "pred1", "pred2", "pred3", "pred4"]),
        Activationsd(keys=["pred0", "pred1", "pred2", "pred3", "pred4"], sigmoid=True),
        # transform data into discrete before voting
        AsDiscreted(keys=["pred0", "pred1", "pred2", "pred3", "pred4"], argmax=True, threshold=0.5),
        VoteEnsembled(keys=["pred0", "pred1", "pred2", "pred3", "pred4"], output_key="pred"),
        Invertd(
        keys=["pred","label"],
        transform=test_transforms,
        orig_keys="label",
        meta_keys=["pred_meta_dict", 'label_meta_dict'],
        orig_meta_keys="image_meta_dict",
        meta_key_postfix="meta_dict",
        nearest_interp=False,
        to_tensor=True,
        ),
        AsDiscreted(keys="pred", argmax=True, threshold=0.5),
        SaveImaged(keys="pred", meta_keys="image_meta_dict", output_dir=save_preds_dir, output_postfix="", output_ext=".nii.gz", separate_folder=False, resample=False)
    ]
)

if ensemble_type == 'avg' or ensemble_type == 'wtavg':
    post_transforms = mean_post_transforms
elif ensemble_type == 'vote':
    post_transforms = vote_post_transforms

ensemble_evaluate(post_transforms, models)


# %%
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
# gtpaths_test.pop(13)
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
save_testmetrics_dir = os.path.join(save_testmetrics_dir, ensemble_type, network, experiment_code)
save_testmetrics_dir = os.path.join(save_testmetrics_dir, f'test_{test_on_disease}')
os.makedirs(save_testmetrics_dir, exist_ok=True)
save_testmetrics_fname = os.path.join(save_testmetrics_dir, 'testmetrics.csv')


data = np.column_stack((imageids, test_dscs, test_jaccards, test_hds))
data_df = pd.DataFrame(data=data, columns=['PatientID', 'DSC', 'Jaccard', '95HD'])
data_df.to_csv(save_testmetrics_fname, index=False)

mean_dsc = round(np.mean(test_dscs), 4)
std_dsc = round(np.std(test_dscs), 4)
median_dsc  = round(np.median(test_dscs), 4)
mean_jaccard = round(np.mean(test_jaccards), 4)
std_jaccard = round(np.std(test_jaccards), 4)
mean_hd = round(np.nanmean(test_hds), 4)
std_hd = round(np.nanstd(test_hds), 4)
#%%
print(f"Test set performance summary: {inputsize}")
print(f"DSC: {mean_dsc} +/- {std_dsc}")
print(f"Median DSC: {median_dsc}")
print(f"Jaccard Index: {mean_jaccard} +/- {std_jaccard}")
print(f"95% HD: {mean_hd} +/- {std_hd}")


# %%
