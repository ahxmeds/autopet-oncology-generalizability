# Towards a Segment Anything Model (SAM) for lesion segmentation in oncological PET/CT images
(This repository is still under construction and the README.md file will be updated)

## Introduction
In this work, we test the generalizability of a convolutional neural network, `UNet` with residual units  trained on PET/CT images of one cancer type to other cancer types. We used three oncological PET/CT datasets (provided by the [autoPET 2022](https://autopet.grand-challenge.org/) challenge) of different cancer types: lymphoma `(n=145)`, lung cancer `(n=168)`, and melanoma `(n=188)`, collected from two institutions. The dataset also contained PET/CT images from healthy control patients `(n=513)`, but those were not used for this work. The dataset is publicly available and can be downloaded via TCIA website from [here](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=93258287).

## Materials and Methods
### Data preprocessing and augmentation
The original CT images and annotations were resampled to the resolution of the original PET images, and CT intensities (in Hounsfield units) were clipped between `(-1024, 1024)`. Both PET (in SUV) and CT intensities were normalized in `(0,1)`. All the images were then resampled to a voxel spacing of `2.0 mm × 2.0 mm × 2.0 mm`. During training, randomly cropped patches of sizes `192 × 192 × 192` were extracted with centers on a foreground or a background voxel with probabilities 5/6 and 1/6, respectively. Spatial augmentations like random affine and 3D elastic deformations were applied to the cropped patches. Input to the network was created by combining the PET and CT patches along the channel dimension. 

### Hardware and network architecture used
All our networks were trained with `nn.DataParallel()` wrapper on a Standard_NC24s_v3 Azure Virtual Machines from Microsoft consisting of 4 NVIDIA GPUs with 16 GiB RAM each and 24 vCPUs with overall 448 GiB RAM. 

A `UNet` with residual units adapted from the `MONAI` [[1]](#1) was used in this work. This network architecture is shown in Figure 1 below and it can be created using the `monai.networks.nets.UNet` class of MONAI as follows:
```
from monai.networks.nets import UNet
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
```
<a href="url"><img src="https://user-images.githubusercontent.com/48228716/236364018-7e7ac66f-3253-4882-b39c-a91e0268de67.png" align="left" height="48" width="48" ></a>

### Loss function, optimizers, scheduler and metrics
The networks were trained using Dice Loss, $L_{Dice}$ (Equation 1): 
![LatexEquation](https://user-images.githubusercontent.com/48228716/236364018-7e7ac66f-3253-4882-b39c-a91e0268de67.png)

Each of the PET/CT data from three different cancer types were randomly split into training (80%) and test (20%) sets. The training of the networks was performed under 5-fold cross-validation of the training set (Figure 1). The network was trained using Dice Loss, (Equation 1): 


## Experiments conducted
![image](https://user-images.githubusercontent.com/48228716/236361695-daf71d3a-86dc-43d6-878a-937a4d951ccb.png)

For each cancer type, the networks were trained to segment the specific single cancer type under 5-fold cross-validation (CV). We evaluated the model on the internal test set of the same cancer type as the training set and then assessed the transferability of the model's lesion segmentation ability on a different cancer type. We further explored different ensembling techniques - Average, Weighted Average, Vote, and STAPLE to combine the five models trained in 5-fold CV as a possible route towards improving model generalizability to new cancer types. The details about the 5-fold split for training/validation and testing for the three cancer types can be found in three .csv files containing the metadata [here](create_data_split/).

Ensemble models on lymphoma        |  Ensemble models on lung cancer | Ensemble models on melanoma
:-------------------------:|:-------------------------:|:-------------------------:
![Lymphoma-unet](https://user-images.githubusercontent.com/48228716/236359920-0c11a08a-007f-44a6-9d7b-b0af59aef487.png) | ![Lung cancer-unet](https://user-images.githubusercontent.com/48228716/236360132-a2ede141-8f52-47bd-b2e3-e9bcb3fc33a0.png)| ![Melanoma-unet](https://user-images.githubusercontent.com/48228716/236359933-7b8f911a-23a5-475e-bbae-cf4c86b11509.png)


A short description of the results is as follows:


For lymphoma-trained ensemble models, we obtained the best Dice similarity coefficient (DSC) (mean, median) of (0.58±0.28, 0.72) on lymphoma test set, and a DSC of (0.44±0.25, 0.45) and (0.43±0.31, 0.43) were achieved on lung cancer and melanoma test sets, respectively. Similarly, for lung cancer-trained ensemble models, the best DSC obtained was (0.71±0.20, 0.77) on lung cancer test set, and a DSC of (0.41±0.28, 0.48) and (0.42±0.27, 0.48) were achieved on lymphoma and melanoma test sets, respectively. Finally, for melanoma-trained ensemble models, the best DSC obtained was (0.52±0.28, 0.61) on melanoma test set, and a DSC of (0.46±0.25, 0.52) and (0.43±0.23, 0.46) were achieved on lymphoma and lung cancer test sets, respectively. We emphasize that ensembling can be a powerful method for generalization, especially in cases when a model is evaluated on a cancer type different from what it was trained on. For internal testing, Weighted Average ensemble generally performed the best, while for testing on a different cancer type, different ensembles performed the best on various training and test set pairs. 

## References
<a id="1">[1]</a> 
MONAI: Medical Open Network for AI,
*AI Toolkit for Healthcare Imaging*
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7459814.svg)](https://doi.org/10.5281/zenodo.7459814)

