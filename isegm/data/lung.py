from pathlib import Path
import numpy as np
from .base import ISDataset, get_unique_labels
import glob
import os
from monai.data import Dataset, DataLoader
from monai.utils import first
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    EnsureTyped,
    Resized,
)


class LungDataset(ISDataset):
    def __init__(self, sliding_window, dataset_path, split='train', buggy_mask_thresh=0.08, **kwargs):
        super(LungDataset, self).__init__(**kwargs)
        assert split in {'train', 'val'}

        self.sliding_window = sliding_window
        self.dataset_path = Path(dataset_path)
        self.dataset_split = split
        self.train_images = sorted(
            glob.glob(os.path.join(self.dataset_path, "imagesTr", "*.nii.gz")))
        self.train_labels = sorted(
            glob.glob(os.path.join(self.dataset_path, "labelsTr", "*.nii.gz")))
        self.data_dicts = [
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip(self.train_images, self.train_labels)
        ]
        if self.dataset_split == 'train':
            self.dataset_samples = self.data_dicts[:-23]
        else:
            self.dataset_samples = self.data_dicts[-23:]
        self.train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Spacingd(keys=["image", "label"], pixdim=(
                    1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-57, a_max=164,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(96, 96, 96),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=0,
                ),
                EnsureTyped(keys=["image", "label"]),
            ]
        )
        self.val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Spacingd(keys=["image", "label"], pixdim=(
                    1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-57, a_max=164,
                    b_min=0.0, b_max=1.0, clip=True,
                ),

                CropForegroundd(keys=["image", "label"], source_key="image"),
                #Resized(keys=["image", "label"], spatial_size=(256, 256, 96)),
                EnsureTyped(keys=["image", "label"]),
            ]
        )

        self.val_transforms_sliding_window = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Spacingd(keys=["image", "label"], pixdim=(
                    1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-57, a_max=164,
                    b_min=0.0, b_max=1.0, clip=True,
                ),

                CropForegroundd(keys=["image", "label"], source_key="image"),
                EnsureTyped(keys=["image", "label"]),
            ]
        )

    def get_sample(self, index):
        if self.dataset_split == 'train':
            check_ds = Dataset(data=[self.dataset_samples[index]], transform=self.train_transforms)
            check_loader = DataLoader(check_ds, batch_size=1)
            check_data = first(check_loader)
            instances_mask = check_data['label'][0][0].numpy().astype(np.int32)
            image = check_data['image'][0][0].numpy()
        else:
            transform_to_use = self.val_transforms_sliding_window if self.sliding_window else self.val_transforms
            check_ds = Dataset(data=[self.dataset_samples[index]], transform=transform_to_use)
            check_loader = DataLoader(check_ds, batch_size=1)
            check_data = first(check_loader)
            instances_mask = check_data['label'][0][0].numpy().astype(np.int32)
            image = check_data['image'][0][0].numpy()

        image = np.expand_dims(image, axis=0)
        instances_ids = get_unique_labels(instances_mask, exclude_zero=True)

        instances_info = {
            x: {'ignore': False}
            for x in instances_ids
        }
        return {
            'image': image,
            'instances_mask': instances_mask,
            'instances_info': instances_info,
            'image_id': index
        }