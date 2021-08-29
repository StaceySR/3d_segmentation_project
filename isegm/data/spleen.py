from pathlib import Path

import cv2
import numpy as np

from isegm.utils.misc import get_bbox_from_mask
from .base import ISDataset, get_unique_labels
from skimage import io
from PIL import Image
import tifffile as tiff
import nibabel as nib
# import monai
from scipy.ndimage import zoom
import glob
import os
from monai.data import Dataset, DataLoader
from monai.utils import first
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    EnsureTyped,
    EnsureType,
    Invertd,
    Resized,
)

class SpleenDataset(ISDataset):
    def __init__(self, sliding_window, dataset_path, split='train', buggy_mask_thresh=0.08, **kwargs):
        super(SpleenDataset, self).__init__(**kwargs)
        assert split in {'train', 'val'}

        self.sliding_window = sliding_window
        self.dataset_path = Path(dataset_path)
        self.dataset_split = split
        #self._images_path = self.dataset_path / 'data_GT'
        #self._insts_path = self.dataset_path / 'boundary_GT'
        #self._buggy_objects = dict()
        #self._buggy_mask_thresh = buggy_mask_thresh

        # with open(self.dataset_path / f'{split}.txt', 'r') as f:
        #     self.dataset_samples = [x.strip() for x in f.readlines()]
        self.train_images = sorted(
            glob.glob(os.path.join(self.dataset_path, "imagesTr", "*.nii.gz")))
        self.train_labels = sorted(
            glob.glob(os.path.join(self.dataset_path, "labelsTr", "*.nii.gz")))
        self.data_dicts = [
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip(self.train_images, self.train_labels)
        ]
        if self.dataset_split == 'train':
            self.dataset_samples = self.data_dicts[:-9]
        else:
            self.dataset_samples = self.data_dicts[-9:]
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
        # user can also add other random transforms
        # RandAffined(
        #     keys=['image', 'label'],
        #     mode=('bilinear', 'nearest'),
        #     prob=1.0, spatial_size=(96, 96, 96),
        #     rotate_range=(0, 0, np.pi/15),
        #     scale_range=(0.1, 0.1, 0.1)),
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
        Resized(keys=["image", "label"], spatial_size=(256,256,96)), 
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

    def normalize(self, image):
        '''
        This function is to normalize the input grayscale image by
        substracting globle mean and dividing standard diviation for
        visualization.

        Input:  a grayscale image

        Output: normolized grascale image

        '''
        img = image.copy().astype(np.float32)
        img -= np.mean(img)
        img /= np.linalg.norm(img)
        # img = (img - img.min() )
        img = np.clip(img, 0, 255)
        img *= (1. / float(img.max()))
        return (img * 255).astype(np.uint8)

    def get_sample(self, index):  #get the sample
        instances_mask = None
        image = None
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
            #image = zoom(image, (1, float(256/image.shape[1]), float(256/image.shape[2]), float(96/image.shape[3])), order=3)
        #tiff.imwrite('d.tif', image.transpose([2,0,1]))
        #tiff.imwrite('e.tif', instances_mask.transpose([2,0,1]))

        #image_name = self.dataset_samples[index]
        #image_path = str(self._images_path / f'{image_name}')
        #print(image_path)
        #image = tiff.imread(image_path)
        #image = zoom(image.dataobj, (0.5, 0.5, float(74/image.dataobj.shape[2])), order=3)
        #image = image.transpose([2,1,0])

        #image = self.normalize(image)
        #image = image.transpose([1,2,0])
        image = np.expand_dims(image, axis=0) # add the channel dimension

        #mask_name = image_name
        #mask_path = str(self._insts_path / f'{mask_name}')
        #print(mask_path)
        #masks = tiff.imread(mask_path)
        #masks = zoom(masks.dataobj, (0.5, 0.5, float(74/masks.dataobj.shape[2])), order=0)
        #masks = masks.transpose([2,1,0])
        #masks = masks.transpose([1, 2, 0])
        #instances_mask = masks[:, :, :].astype(np.int32)
        instances_ids = get_unique_labels(instances_mask, exclude_zero=True)


        # image_path = str(self._images_path / f'{image_name}.npy')
        # image = np.load(image_path)
        # image = np.expand_dims(image, axis=3) # add the channel dimension
        # # image = cv2.imread('/media/huoy1/48EAE4F7EAE4E264/guihu/22558_2017-04-08 12_34_57-x-ROI_0-x-30474-x-110083-x-464-x-572.png')
        # mask_path = str(self._insts_path / f'{image_name}.npy')
        # masks = np.load(mask_path)
        # instances_mask = masks[:, :, :].astype(np.int32)
        # instances_ids = get_unique_labels(instances_mask, exclude_zero=True)

        instances_info = {
            x: {'ignore': False}
            for x in instances_ids
        }
        # print("FLUOline41")
        return {
            'image': image,
            'instances_mask': instances_mask,
            'instances_info': instances_info,
            'image_id': index
        }