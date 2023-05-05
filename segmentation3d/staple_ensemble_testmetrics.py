#%%
#%%
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd 
import torch
from monai.metrics import compute_hausdorff_distance
from torch.nn.functional import one_hot 
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

def read_image_array(path):
    img =  sitk.ReadImage(path)
    array = np.transpose(sitk.GetArrayFromImage(img), (2,1,0))
    return array

def read_image(path):
    img =  sitk.ReadImage(path)
    return img

#%%
network = 'unet'
train_on_disease = 'melanoma'
inputtype = 'ctpt'
inputsize = 'randcrop192' 
experiment_code = f"{network}_{train_on_disease}_staple_{inputtype}_{inputsize}"

test_on_disease = 'lungcancer'

autopet_diagnosis_csvfpath = f'/home/jhubadmin/Projects/autopet-oncology-generalizability/create_data_split/metadata_{test_on_disease}.csv'
autopet_images_dir = '/data/blobfuse/default/autopet2022_data/images'
autopet_labels_dir = '/data/blobfuse/default/autopet2022_data/labels'
diagnosis_df = pd.read_csv(autopet_diagnosis_csvfpath)
diagnosis_df_test = diagnosis_df[diagnosis_df['TRAIN/TEST'] == 'TEST']
patientIDs = list(diagnosis_df_test['PatientID'])
ctpaths_test, ptpaths_test, gtpaths_test = create_ctpaths_ptpaths_gtpaths(patientIDs, autopet_images_dir, autopet_labels_dir)
ptpaths_test = sorted(ptpaths_test)
gtpaths_test = sorted(gtpaths_test)

save_testpreds_dir = '/data/blobfuse/default/autopet_generalizability_results/saved_testpreds_folds/segmentation3d'
save_testpreds_dir = os.path.join(save_testpreds_dir, 'staple', network, experiment_code, f'test_{test_on_disease}') 
predpaths_test = sorted(glob.glob(os.path.join(save_testpreds_dir, '*.nii.gz')))


# %%
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
save_testmetrics_dir = os.path.join(save_testmetrics_dir, 'staple', network, experiment_code, f'test_{test_on_disease}')
os.makedirs(save_testmetrics_dir, exist_ok=True)
save_testmetrics_fpath = os.path.join(save_testmetrics_dir, 'testmetrics.csv')

data = np.column_stack((imageids, test_dscs, test_jaccards, test_hds))
data_df = pd.DataFrame(data=data, columns=['PatientID', 'DSC', 'Jaccard', '95HD'])
data_df.to_csv(save_testmetrics_fpath, index=False)
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
print(f"DSC: {mean_dsc} +/- {std_dsc}")
print(f"Median DSC: {median_dsc}")
print(f"Jaccard Index: {mean_jaccard} +/- {std_jaccard}")
print(f"95% HD: {mean_hd} +/- {std_hd}")
# %%