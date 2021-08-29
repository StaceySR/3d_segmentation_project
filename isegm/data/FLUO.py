from pathlib import Path

import cv2
import numpy as np

from isegm.utils.misc import get_bbox_from_mask
from .base import ISDataset, get_unique_labels
from skimage import io
from PIL import Image
import tifffile as tiff
# import monai
from scipy.ndimage import zoom

class FLUODataset(ISDataset):
    def __init__(self, dataset_path, split='train', buggy_mask_thresh=0.08, **kwargs):
        super(FLUODataset, self).__init__(**kwargs)
        assert split in {'train', 'val'}

        self.dataset_path = Path(dataset_path)
        self.dataset_split = split
        self._images_path = self.dataset_path / 'data_GT'
        self._insts_path = self.dataset_path / 'boundary_GT'
        self._buggy_objects = dict()
        self._buggy_mask_thresh = buggy_mask_thresh

        with open(self.dataset_path / f'{split}.txt', 'r') as f:
            self.dataset_samples = [x.strip() for x in f.readlines()]

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
        image_name = self.dataset_samples[index]
        image_path = str(self._images_path / f'{image_name}.tif')
        image = tiff.imread(image_path)
        image = zoom(image, (0.5, 0.5, 0.5), order=3)

        image = self.normalize(image)
        image = image.transpose([1,2,0])
        image = np.expand_dims(image, axis=0) # add the channel dimension


        mask_name = image_name.replace('t','man_seg')
        mask_path = str(self._insts_path / f'{mask_name}.tif')
        masks = tiff.imread(mask_path)
        masks = masks.transpose([1, 2, 0])
        masks = zoom(masks, (0.5, 0.5, 0.5), order=0)
        instances_mask = masks[:, :, :].astype(np.int32)
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
