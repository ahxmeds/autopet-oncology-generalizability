# Towards a Segment Anything Model (SAM) for lesion segmentation in oncological PET/CT images
(This repository is still under construction and the README.md file will be updated)

In this work, we test the generalizability of a convolutional neural network, `UNet` with residual units (as adapted from MONAI [[1]](#1)) trained on PET/CT images of one cancer type to other cancer types. We used three oncological PET/CT datasets (provided by the [autoPET 2022](https://autopet.grand-challenge.org/) challenge) of different cancer types: lymphoma `(n=145)`, lung cancer `(n=168)`, and melanoma `(n=188)`, collected from two institutions. The dataset also contained PET/CT images from healthy control patients `(n=513)`, but those were not used for this work. The dataset is publicly available and can be downloaded via TCIA website from [here](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=93258287).

For each cancer type, the networks were trained to segment the specific single cancer type under 5-fold cross-validation (CV). We evaluated the model on the internal test set of the same cancer type as the training set and then assessed the transferability of the model's lesion segmentation ability on a different cancer type. We further explored different ensembling techniques - Average, Weighted Average, Vote, and STAPLE to combine the five models trained in 5-fold CV as a possible route towards improving model generalizability to new cancer types. The details about the 5-fold split for training/validation and testing for the three cancer types can be found in three .csv files containing the metadata [here](create_data_split/) 

Ensemble models on lymphoma        |  Ensemble models on lung cancer | Ensemble models on melanoma
:-------------------------:|:-------------------------:|:-------------------------:
![Lymphoma-unet](https://user-images.githubusercontent.com/48228716/236359920-0c11a08a-007f-44a6-9d7b-b0af59aef487.png) | ![Lung cancer-unet](https://user-images.githubusercontent.com/48228716/236360132-a2ede141-8f52-47bd-b2e3-e9bcb3fc33a0.png)| ![Melanoma-unet](https://user-images.githubusercontent.com/48228716/236359933-7b8f911a-23a5-475e-bbae-cf4c86b11509.png)


The detailed results are given below:


For lymphoma-trained ensemble models, we obtained the best Dice similarity coefficient (DSC) (mean, median) of (0.58±0.28, 0.72) on lymphoma test set, and a DSC of (0.44±0.25, 0.45) and (0.43±0.31, 0.43) were achieved on lung cancer and melanoma test sets, respectively. Similarly, for lung cancer-trained ensemble models, the best DSC obtained was (0.71±0.20, 0.77) on lung cancer test set, and a DSC of (0.41±0.28, 0.48) and (0.42±0.27, 0.48) were achieved on lymphoma and melanoma test sets, respectively. Finally, for melanoma-trained ensemble models, the best DSC obtained was (0.52±0.28, 0.61) on melanoma test set, and a DSC of (0.46±0.25, 0.52) and (0.43±0.23, 0.46) were achieved on lymphoma and lung cancer test sets, respectively. We emphasize that ensembling can be a powerful method for generalization, especially in cases when a model is evaluated on a cancer type different from what it was trained on. For internal testing, Weighted Average ensemble generally performed the best, while for testing on a different cancer type, different ensembles performed the best on various training and test set pairs. 

## References
<a id="1">[1]</a> 
MONAI: Medical Open Network for AI,
*AI Toolkit for Healthcare Imaging*
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7459814.svg)](https://doi.org/10.5281/zenodo.7459814)

