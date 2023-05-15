from __future__ import annotations

from collections.abc import Sequence
from os import PathLike
from typing import TypedDict

import torch
from monai.data import CacheDataset, DataLoader, Dataset, PatchDataset
from monai.networks import normal_init
from monai.networks.nets import UNet
from monai.transforms import (
    EnsureChannelFirstd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotated,
    ScaleIntensityd,
)
from monai.transforms.compose import Compose
from monai.transforms.io.dictionary import LoadImaged
from torchmetrics import MeanSquaredError

KEYS = ["image", "label"]


class DataDict(TypedDict):
    """Dict describing training data."""

    image: PathLike[str] | str
    label: PathLike[str] | str


def gen_cnn(
    train_files: Sequence[DataDict],
    validate_files: Sequence[DataDict],
    num_patches: int,
) -> UNet:
    """Set up a CNN."""
    load_images = Compose(
        [
            LoadImaged(keys=KEYS),
            EnsureChannelFirstd(keys=KEYS),
            ScaleIntensityd(keys=KEYS),
        ],
    )

    train_imgs_cache = CacheDataset(data=train_files, transform=load_images)
    validate_imgs_cache = CacheDataset(data=validate_files, transform=load_images)

    patcher = RandCropByPosNegLabeld(
        keys=KEYS,
        label_key="image",
        spatial_size=(32, 32, 32),
        pos=1,
        neg=0.0001,
        num_samples=num_patches,
    )
    patch_transforms = Compose(
        [
            RandRotated(
                keys=KEYS,
                range_x=(0.8, 0.8),
                range_y=(0.8, 0.8),
                range_z=(0.8, 0.8),
                prob=0.4,
            ),
            RandFlipd(keys=KEYS, prob=0.2, spatial_axis=1),
        ],
    )
    train_patches_dataset = PatchDataset(
        data=train_imgs_cache,
        patch_func=patcher,
        samples_per_image=num_patches,
        transform=patch_transforms,
    )
    validate_patches_dataset = PatchDataset(
        data=validate_imgs_cache,
        patch_func=patcher,
        samples_per_image=num_patches,
        transform=patch_transforms,
    )

    cnn = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256, 512, 512, 512),
        strides=(2, 2, 2, 2, 1, 1, 1),
        dropout=0.2,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else ("cpu"))
    cnn.apply(normal_init)
    return cnn.to(device)


def train_cnn(
    cnn: UNet,
    train_patches_dataset: Dataset,
    validate_patches_dataset: Dataset,
    max_epochs: int,
    learning_rate: float,
    betas: tuple[float, float],
    batch_size: int,
    mean_squared: MeanSquaredError,
) -> None:
    cnn_opt = torch.optim.Adam(cnn.parameters(), lr=learning_rate, betas=betas)
    train_loader = DataLoader(
        train_patches_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    validate_loader = DataLoader(
        validate_patches_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    epoch_loss_values = [0]
    mse_error = [0]
    for _epoch in range(max_epochs):
        cnn.train()
        epoch_loss = 0
        for train_batch in train_loader:
            cnn_opt.zero_grad()
            gad_images = train_batch["image"].cuda()
            nongad_images = train_batch["label"].cuda()
            degad_images = cnn(gad_images)
            mse_loss = torch.nn.MSELoss()
            loss = torch.sqrt(mse_loss(degad_images, nongad_images))
            loss.backward()
            cnn_opt.step()
            epoch_loss += loss.item()

        epoch_loss_values.append(epoch_loss / training_sample_size)
    cnn.eval()
    with torch.no_grad():
        mean_squared.reset()
        mse_total_epoch = 0
        for validate_batch in validate_loader:
            gad_images = validate_batch["image"].cuda()
            nongad_images = validate_batch["label"].cuda()
            degad_images = cnn(gad_images)
            val_mse = mean_squared(degad_images, nongad_images)
            mse_total_epoch += val_mse
        mse_error.append(mse_total_epoch.item() / validate_sample_size)
