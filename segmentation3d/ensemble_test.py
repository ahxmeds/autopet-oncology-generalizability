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
from monai.utils import set_determinism

print_config()
# %%
directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)
# %%
set_determinism(seed=0)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
device = torch.device("cuda:0")
# %%
data_dir = os.path.join(root_dir, "runs")
datalist = []

if not os.path.exists(data_dir):
    os.makedirs(data_dir)
for i in range(60):
    im, seg = create_test_image_3d(128, 128, 128, num_seg_classes=1, channel_dim=-1)

    n = nib.Nifti1Image(im, np.eye(4))
    img_path = os.path.join(data_dir, f"img{i}.nii.gz")
    nib.save(n, img_path)

    n = nib.Nifti1Image(seg, np.eye(4))
    seg_path = os.path.join(data_dir, f"seg{i}.nii.gz")
    nib.save(n, seg_path)

    datalist.append({"image": img_path, "label": seg_path})
# %%
class CVDataset(ABC, CacheDataset):
    """
    Base class to generate cross validation datasets.

    """

    def __init__(
        self,
        data,
        transform,
        cache_num=sys.maxsize,
        cache_rate=1.0,
        num_workers=4,
    ) -> None:
        data = self._split_datalist(datalist=data)
        CacheDataset.__init__(
            self, data, transform, cache_num=cache_num, cache_rate=cache_rate, num_workers=num_workers
        )

    @abstractmethod
    def _split_datalist(self, datalist):
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")
# %%
train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"], channel_dim=-1),
        ScaleIntensityd(keys=["image", "label"]),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=[96, 96, 96],
            pos=1,
            neg=1,
            num_samples=4,
        ),
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[0, 2]),
        EnsureTyped(keys=["image", "label"]),
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"], channel_dim=-1),
        ScaleIntensityd(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"]),
    ]
)
# %%
num = 5
folds = list(range(num))

cvdataset = CrossValidation(
    dataset_cls=CVDataset,
    data=datalist[0:50],
    nfolds=5,
    seed=12345,
    transform=train_transforms,
)

train_dss = [cvdataset.get_dataset(folds=folds[0:i] + folds[(i + 1) :]) for i in folds]
val_dss = [cvdataset.get_dataset(folds=i, transform=val_transforms) for i in range(num)]

train_loaders = [DataLoader(train_dss[i], batch_size=2, shuffle=True, num_workers=4) for i in folds]
val_loaders = [DataLoader(val_dss[i], batch_size=1, num_workers=4) for i in folds]

test_ds = CacheDataset(data=datalist[50:], transform=val_transforms)
test_loader = DataLoader(test_ds, batch_size=1, num_workers=4)

# %%
def train(index):
    net = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    loss = DiceLoss(sigmoid=True)
    opt = torch.optim.Adam(net.parameters(), 1e-3)

    val_post_transforms = Compose(
        [EnsureTyped(keys="pred"), Activationsd(keys="pred", sigmoid=True), AsDiscreted(keys="pred", threshold=0.5)]
    )

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=val_loaders[index],
        network=net,
        inferer=SlidingWindowInferer(roi_size=(96, 96, 96), sw_batch_size=4, overlap=0.5),
        postprocessing=val_post_transforms,
        key_val_metric={
            "val_mean_dice": MeanDice(
                include_background=True,
                output_transform=from_engine(["pred", "label"]),
            )
        },
    )
    train_handlers = [
        ValidationHandler(validator=evaluator, interval=4, epoch_level=True),
        StatsHandler(tag_name="train_loss", output_transform=from_engine(["loss"], first=True)),
    ]

    trainer = SupervisedTrainer(
        device=device,
        max_epochs=4,
        train_data_loader=train_loaders[index],
        network=net,
        optimizer=opt,
        loss_function=loss,
        inferer=SimpleInferer(),
        amp=False,
        train_handlers=train_handlers,
    )
    trainer.run()
    return net
# %%
models = [train(i) for i in range(num)]

# %%
def ensemble_evaluate(post_transforms, models):
    evaluator = EnsembleEvaluator(
        device=device,
        val_data_loader=test_loader,
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
