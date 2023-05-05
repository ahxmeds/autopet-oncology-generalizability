#%%
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd 

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

def print_sitkimg_props(image):
    print(f"Size: {image.GetSize()}")
    print(f"Spacing: {image.GetSpacing()}")
    print(f"Origin: {image.GetOrigin()}")
    print(f"Direction: {image.GetDirection()}")
#%%
# generate combined segmentation using STAPLE algorithm as implemented in the SITK package
# this function takes in a series of filepaths for 3D Nifti mask images and create a 
# STAPLE segmentation from them
 
def generate_staple(filepaths):
    segmentations_sitk = []
    for path in filepaths:
        seg_sitk = sitk.ReadImage(path, sitk.sitkUInt8)
        segmentations_sitk.append(seg_sitk)

    STAPLE_seg_sitk = sitk.STAPLE(segmentations_sitk, 1.0)
    STAPLE_seg = np.transpose(sitk.GetArrayFromImage(STAPLE_seg_sitk), (2,1,0))
    STAPLE_seg = np.where(STAPLE_seg<0.95, 0, 1) # setting the values to either 0 or 1

    STAPLE_seg_sitk = sitk.GetImageFromArray(np.transpose(STAPLE_seg, (2,1,0)))
    STAPLE_seg_sitk.SetSpacing(segmentations_sitk[0].GetSpacing())
    STAPLE_seg_sitk.SetOrigin(segmentations_sitk[0].GetOrigin())
    STAPLE_seg_sitk.SetDirection(segmentations_sitk[0].GetDirection())
    return STAPLE_seg_sitk

def generate_majorityvoting(filepaths):
    segmentations_sitk = []
    for path in filepaths:
        seg_sitk = sitk.ReadImage(path, sitk.sitkUInt8)
        segmentations_sitk.append(seg_sitk)
    
    labelForUndecidedPixels = 0
    MajorityVote_sitk = sitk.LabelVoting(segmentations_sitk, labelForUndecidedPixels)  
    return MajorityVote_sitk


def plot_images(filepaths):
    patientID = os.path.basename(filepaths[0])[:-7]
    sitk_imgs = [sitk.ReadImage(path) for path in filepaths]
    arrays = [np.transpose(sitk.GetArrayFromImage(img), (2,1,0)) for img in sitk_imgs]

    fig, ax = plt.subplots(1,8, figsize=(20,5))
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(0.7)

    arrays_coronalmip = [np.rot90(np.max(array, axis=1)) for array in arrays] 
    titles = ['PT', 'GT', 'Fold0', 'Fold1', 'Fold2', 'Fold3', 'Fold4', 'STAPLE']
    for i in range(len(ax)):
        ax[i].imshow(arrays_coronalmip[i], cmap='Greys')
        ax[i].set_title(titles[i])
        ax[i].set_axis_off()
    fig.suptitle(patientID, fontsize=15)
    plt.show()
    plt.close('all')

# %%
fold = [0, 1, 2, 3, 4]
network = 'unet'
train_on_disease = 'melanoma'
inputtype = 'ctpt'
inputsize = 'randcrop192' 
experiment_code = [f"{network}_{train_on_disease}_fold{str(f)}_{inputtype}_{inputsize}" for f in fold]

test_on_disease = 'melanoma'

dir = '/data/blobfuse/default/autopet_generalizability_results/saved_testpreds_folds/segmentation3d'
save_testpreds_dir = [os.path.join(dir, f'fold{fold[i]}', network, experiment_code[i], f'test_{test_on_disease}') for i in range(len(fold))]
save_testpreds_fpaths = [sorted(glob.glob(os.path.join(save_testpreds_dir[i], '*.nii.gz'))) for i in range(len(save_testpreds_dir))]

test_experiment_code = f"{network}_{train_on_disease}_staple_{inputtype}_{inputsize}"
savedir = os.path.join(dir, 'staple', network, test_experiment_code, f'test_{test_on_disease}')
os.makedirs(savedir, exist_ok=True)

#%%

autopet_diagnosis_csvfpath = f'/home/jhubadmin/Projects/autopet-oncology-generalizability/create_data_split/metadata_{test_on_disease}.csv'
autopet_images_dir = '/data/blobfuse/default/autopet2022_data/images'
autopet_labels_dir = '/data/blobfuse/default/autopet2022_data/labels'
diagnosis_df = pd.read_csv(autopet_diagnosis_csvfpath)
diagnosis_df_test = diagnosis_df[diagnosis_df['TRAIN/TEST'] == 'TEST']
patientIDs = list(diagnosis_df_test['PatientID'])
ctpaths_test, ptpaths_test, gtpaths_test = create_ctpaths_ptpaths_gtpaths(patientIDs, autopet_images_dir, autopet_labels_dir)
ptpaths_test = sorted(ptpaths_test)
gtpaths_test = sorted(gtpaths_test)
# %%
filedata = np.column_stack(
    (
        ptpaths_test,
        gtpaths_test,
        save_testpreds_fpaths[0],
        save_testpreds_fpaths[1],
        save_testpreds_fpaths[2],
        save_testpreds_fpaths[3],
        save_testpreds_fpaths[4],
    )
)
#%%
index = 0
for fpaths in filedata:
    ptpath, gtpath, fold0path, fold1path, fold2path, fold3path, fold4path = fpaths

    fname = os.path.basename(gtpath)

    filepaths = [fold0path, fold1path, fold2path, fold3path, fold4path]
    
    staple_seg_sitk = generate_staple(filepaths)

    staplefpath = os.path.join(savedir, fname)
    
    sitk.WriteImage(staple_seg_sitk, staplefpath)
    # fpaths_for_plotting = [ptpath, gtpath] + filepaths + [staplefpath]
    # plot_images(fpaths_for_plotting)
    index+=1
    print(f'{index}: Done with {fname}')



# %%
network = 'unet'
train_on_disease = 'melanoma'
inputtype = 'ctpt'
inputsize = 'randcrop192' 
experiment_code = f"{network}_{train_on_disease}_staple_{inputtype}_{inputsize}"

test_on_disease = 'melanoma'

autopet_diagnosis_csvfpath = f'/home/jhubadmin/Projects/autopet-oncology-generalizability/create_data_split/metadata_{test_on_disease}.csv'
autopet_images_dir = '/data/blobfuse/default/autopet2022_data/images'
autopet_labels_dir = '/data/blobfuse/default/autopet2022_data/labels'
diagnosis_df = pd.read_csv(autopet_diagnosis_csvfpath)
diagnosis_df_test = diagnosis_df[diagnosis_df['TRAIN/TEST'] == 'TEST']
patientIDs = list(diagnosis_df_test['PatientID'])
ctpaths_test, ptpaths_test, gtpaths_test = create_ctpaths_ptpaths_gtpaths(patientIDs, autopet_images_dir, autopet_labels_dir)
ptpaths_test = sorted(ptpaths_test)
gtpaths_test = sorted(gtpaths_test)