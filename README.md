# Towards a Segment Anything Model (SAM) for lesion segmentation in oncological PET/CT images
(This repository is still under construction and the README.md file will be updated)
## Table of contents
- [Introduction](#introduction)
- [Materials and Methods](#materials-and-methods)
    - [Data preprocessing and augmentation](#data-preprocessing-and-augmentation)
    - [Hardware and network architecture](#hardware-and-network-architecture)
    - [Loss function, optimizers, scheduler and metrics](#loss-function-optimizers-scheduler-and-metrics)
- [Experiments](#experiments)
- [Results](#results)
- [Reference](#references)
    
## Introduction
In this work, we test the generalizability of a convolutional neural network, `UNet` with residual units  trained on PET/CT images of one cancer type to other cancer types. We used three oncological PET/CT datasets (provided by the [autoPET 2022](https://autopet.grand-challenge.org/) challenge) of different cancer types: lymphoma `(n=145)`, lung cancer `(n=168)`, and melanoma `(n=188)`, collected from two institutions. The dataset also contained PET/CT images from healthy control patients `(n=513)`, but those were not used for this work. The dataset is publicly available and can be downloaded via TCIA website from [here](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=93258287).

## Materials and Methods
### Data preprocessing and augmentation
The original CT images and annotations were resampled to the resolution of the original PET images, and CT intensities (in Hounsfield units) were clipped between `(-1024, 1024)`. Both PET (in SUV) and CT intensities were normalized in `(0,1)`. All the images were then resampled to a voxel spacing of `2.0 mm × 2.0 mm × 2.0 mm`. During training, randomly cropped patches of sizes `192 × 192 × 192` were extracted with centers on a foreground or a background voxel with probabilities 5/6 and 1/6, respectively. Spatial augmentations like random affine and 3D elastic deformations were applied to the cropped patches. Input to the network was created by combining the PET and CT patches along the channel dimension. The annotation masks contained two labels: `0` for background and `1` for the lesion class.

### Hardware and network architecture
All our networks were trained with `nn.DataParallel(.)` wrapper on a Standard_NC24s_v3 Azure Virtual Machines from Microsoft consisting of 4 NVIDIA GPUs each with a 16 GiB RAM and 24 vCPUs with overall 448 GiB RAM. 

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

<a href="url"><img src="https://user-images.githubusercontent.com/48228716/236365556-459fd8c5-c0c5-491f-8f2b-7bdf11438d60.png" align="center" height=500></a>


### Loss function, optimizers, scheduler and metrics
The networks were trained using Dice Loss, $L_{Dice}$ adapted from `monai.losses.DiceLoss(.)` given by the following equation: 

<a href="url"><img src="https://user-images.githubusercontent.com/48228716/236364018-7e7ac66f-3253-4882-b39c-a91e0268de67.png" align="center" height=90></a>

where, $p_{ij}$ and $g_{ij}$ are the $j^{\text{th}}$ voxels of the $i^{\text{th}}$ cropped patch of the predicted and ground truth segmentation masks, respectively. $n_b$ and $K$ denote the `batch-size` and the `number of voxels in a patch`, respectively. In this work, we set $n_b = 8$ and $K = 192^3$.  The loss for an epoch was calculated by averaging $L_{Dice}$ over all the batches. Adam optimize with an initial learning rate of $10^{-4}$ was used to minimize $L_{Dice}$. The learning rate was reduced to zero at the end of 1200 epochs via the Cosine annealing scheduler.

Dice similarity coefficient (DSC) metric, adapted from `monai.metrics.DiceMetric(.)`, was used for evaluation of the overlap between the ground truth and the predicted mask for the lesion class. Inference was performed using a sliding window method with a window size of `192 × 192 × 192` on the test set images. 


## Experiments
<a href="url"><img src="https://user-images.githubusercontent.com/48228716/236361695-daf71d3a-86dc-43d6-878a-937a4d951ccb.png" align="center" height=300></a>


Each of the PET/CT data from three different cancer types were randomly split into training (80%) and test (20%) sets. For each cancer type, the networks were trained to segment the specific single cancer type under 5-fold cross-validation (CV). We evaluated the model on the internal test set of the same cancer type as the training set and then assessed the transferability of the model's lesion segmentation ability on a different cancer type. We further explored different ensembling techniques - `Average (Avg)`, `Weighted Average (WtAvg)` (with weights equal to the mean DSC on the corresponding validation fold), `Majority Voting (Vote)`, and `STAPLE` [[2]](#)  to combine the five models trained in 5-fold CV as a possible route towards improving model generalizability to new cancer types. The details about the 5-fold split for training/validation and testing for the three cancer types can be found in three .csv files containing the metadata [here](create_data_split/).

## Results
A short description of our network performance with respect to mean and median DSC for different training and test set pairs can be found in the Figure and the Table below,

Ensemble models on lymphoma        |  Ensemble models on lung cancer | Ensemble models on melanoma
:-------------------------:|:-------------------------:|:-------------------------:
![Lymphoma-unet](https://user-images.githubusercontent.com/48228716/236359920-0c11a08a-007f-44a6-9d7b-b0af59aef487.png) | ![Lung cancer-unet](https://user-images.githubusercontent.com/48228716/236360132-a2ede141-8f52-47bd-b2e3-e9bcb3fc33a0.png)| ![Melanoma-unet](https://user-images.githubusercontent.com/48228716/236359933-7b8f911a-23a5-475e-bbae-cf4c86b11509.png)


A short description of the results is as follows:

<table>
<thead>
  <tr>
    <th rowspan="2">Training data</th>
    <th rowspan="2">Ensemble type</th>
    <th colspan="2">DSC Lymphoma (Test)</th>
    <th colspan="2">DSC Lung cancer (Test)</th>
    <th colspan="2">DSC Melanoma (Test)</th>
  </tr>
  <tr>
    <th>mean</th>
    <th>median</th>
    <th>mean</th>
    <th>median</th>
    <th>mean</th>
    <th>median</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="5">Lymphoma</td>
    <td>Average DSC over folds [01234]</td>
    <td>0.5541±0.2774</td>
    <td>0.6791</td>
    <td>0.4021±0.2412</td>
    <td>0.4265</td>
    <td>0.3686±0.286</td>
    <td>0.3194</td>
  </tr>
  <tr>
    <td>Average</td>
    <td>0.5832±0.2772</td>
    <td>0.7196</td>
    <td>0.4161±0.2514</td>
    <td>0..4473</td>
    <td>0.4330±0.3138</td>
    <td>0.4255</td>
  </tr>
  <tr>
    <td>Weighted Average</td>
    <td>0.5838±0.2761</td>
    <td>0.7194</td>
    <td>0.4161±0.2519</td>
    <td>0.4462</td>
    <td>0.4337±0.3139</td>
    <td>0.4249</td>
  </tr>
  <tr>
    <td>Vote</td>
    <td>0.5691±0.2787</td>
    <td>0.707</td>
    <td>0.4031±0.2491</td>
    <td>0.419</td>
    <td>0.4253±0.3133</td>
    <td>0.4282</td>
  </tr>
  <tr>
    <td>STAPLE</td>
    <td>0.5766±0.2839</td>
    <td>0.7057</td>
    <td>0.4374±0.253</td>
    <td>0.4527</td>
    <td>0.4063±0.2933</td>
    <td>0.3914</td>
  </tr>
  <tr>
    <td rowspan="5">Lung cancer</td>
    <td>Average DSC over folds [01234]</td>
    <td>0.3886±0.2497</td>
    <td>0.4234</td>
    <td>0.6909±0.2092</td>
    <td>0.7339</td>
    <td>0.3729±0.2465</td>
    <td>0.3783</td>
  </tr>
  <tr>
    <td>Average</td>
    <td>0.4062±0.2775</td>
    <td>0.4765</td>
    <td>0.7147±0.2023</td>
    <td>0.7626</td>
    <td>0.4206±0.2651</td>
    <td>0.4754</td>
  </tr>
  <tr>
    <td>Weighted Average</td>
    <td>0.4063±0.2775</td>
    <td>0.4753</td>
    <td>0.7148±0.2023</td>
    <td>0.763</td>
    <td>0.4207±0.2650</td>
    <td>0.4768</td>
  </tr>
  <tr>
    <td>Vote</td>
    <td>0.3992±0.2789</td>
    <td>0.4663</td>
    <td>0.7134±0.2026</td>
    <td>0.7704</td>
    <td>0.4248±0.2667</td>
    <td>0.476</td>
  </tr>
  <tr>
    <td>STAPLE</td>
    <td>0.4132±0.2597</td>
    <td>0.4583</td>
    <td>0.708±0.2062</td>
    <td>0.76</td>
    <td>0.381±0.2512</td>
    <td>0.3887</td>
  </tr>
  <tr>
    <td rowspan="5">Melanoma</td>
    <td>Average DSC over folds [01234]</td>
    <td>0.4026±0.2342</td>
    <td>0.4516</td>
    <td>0.4033±0.2283</td>
    <td>0.4237</td>
    <td>0.4737±0.2877</td>
    <td>0.5186</td>
  </tr>
  <tr>
    <td>Average</td>
    <td>0.4136±0.2347</td>
    <td>0.4419</td>
    <td>0.4119±0.2337</td>
    <td>0.4419</td>
    <td>0.5175±0.2831</td>
    <td>0.6038</td>
  </tr>
  <tr>
    <td>Weighted Average</td>
    <td>0.4118±0.2365</td>
    <td>0.4411</td>
    <td>0.4119±0.2336</td>
    <td>0.4395</td>
    <td>0.5191±0.2822</td>
    <td>0.6067</td>
  </tr>
  <tr>
    <td>Vote</td>
    <td>0.3495±0.245</td>
    <td>0.3591</td>
    <td>0.3823±0.235</td>
    <td>0.4173</td>
    <td>0.4736±0.2822</td>
    <td>0.5283</td>
  </tr>
  <tr>
    <td>STAPLE</td>
    <td>0.4575±0.2459</td>
    <td>0.5192</td>
    <td>0.4316±0.2303</td>
    <td>0.4612</td>
    <td>0.5154±0.2938</td>
    <td>0.5887</td>
  </tr>
</tbody>
</table>

## References
<a id="1">[1]</a> 
MONAI: Medical Open Network for AI,
*AI Toolkit for Healthcare Imaging*
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7459814.svg)](https://doi.org/10.5281/zenodo.7459814)

<a id="2">[2]</a> 
Simon K. Warfield, Kelly H. Zou, and William M. Wells, 
*Simultaneous Truth and Performance Level Estimation (STAPLE): An Algorithm for the Validation of Image Segmentation*,
IEEE Trans Med Imaging, 2004.
