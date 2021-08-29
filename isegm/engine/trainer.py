import os
import logging
from copy import deepcopy
from collections import defaultdict
import sys

import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize

from isegm.utils.log import logger, TqdmToLogger, SummaryWriterAvg
from isegm.utils.vis import draw_probmap, draw_points
from isegm.utils.misc import save_checkpoint
from isegm.inference import utils
from isegm.inference.clicker import Clicker

import tifffile as tiff
from monai.metrics import DiceMetric
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
)

from typing import Any, Callable, List, Sequence, Tuple, Union

import torch.nn.functional as F

from monai.data.utils import compute_importance_map, dense_patch_slices, get_valid_patch_size
from monai.utils import BlendMode, PytorchPadMode, fall_back_tuple, look_up_option
from monai.inferers.utils import _get_scan_interval

from isegm.data.points_sampler import MultiPointSampler
from isegm.data.base import get_unique_labels


class ISTrainer(object):
    def __init__(self, model, cfg, model_cfg, loss_cfg,
                 trainset, valset,
                 optimizer='adam',
                 optimizer_params=None,
                 image_dump_interval=200,
                 checkpoint_interval=10,
                 tb_dump_period=25,
                 max_interactive_points=0,
                 lr_scheduler=None,
                 metrics=None,
                 additional_val_metrics=None,
                 backbone_lr_mult=0.1,
                 net_inputs=('images', 'points')):
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.max_interactive_points = max_interactive_points
        self.loss_cfg = loss_cfg
        self.val_loss_cfg = deepcopy(loss_cfg)
        self.tb_dump_period = tb_dump_period
        self.net_inputs = net_inputs

        if metrics is None:
            metrics = []
        self.train_metrics = metrics
        self.val_metrics = deepcopy(metrics)
        if additional_val_metrics is not None:
            self.val_metrics.extend(additional_val_metrics)

        self.checkpoint_interval = checkpoint_interval
        self.image_dump_interval = image_dump_interval
        self.task_prefix = ''
        self.sw = None

        self.trainset = trainset
        self.valset = valset

        self.train_data = DataLoader(
            trainset, cfg.batch_size, shuffle=True,
            drop_last=True, pin_memory=True,
            num_workers=cfg.workers
        )

        self.val_data = DataLoader(
            valset, 1, shuffle=False,
            drop_last=True, pin_memory=True,
            num_workers=cfg.workers
        )

        backbone_params, other_params = model.get_trainable_params()
        opt_params = [
            {'params': backbone_params, 'lr': backbone_lr_mult * optimizer_params['lr']},
            {'params': other_params}
        ]
        if optimizer.lower() == 'adam':
            self.optim = torch.optim.Adam(opt_params, **optimizer_params)
        elif optimizer.lower() == 'adamw':
            self.optim = torch.optim.AdamW(opt_params, **optimizer_params)
        elif optimizer.lower() == 'sgd':
            self.optim = torch.optim.SGD(opt_params, **optimizer_params)
        else:
            raise NotImplementedError

        if cfg.multi_gpu:
            model = _CustomDP(model, device_ids=cfg.gpu_ids, output_device=cfg.gpu_ids[0])

        logger.info(model)
        self.device = cfg.device
        self.net = model.to(self.device)
        self.lr = optimizer_params['lr']

        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler(optimizer=self.optim)
            if cfg.start_epoch > 0:
                for _ in range(cfg.start_epoch):
                    self.lr_scheduler.step()

        self.tqdm_out = TqdmToLogger(logger, level=logging.INFO)
        if cfg.input_normalization:
            #print("trainerline98")
            mean = torch.tensor(cfg.input_normalization['mean'], dtype=torch.float32)
            std = torch.tensor(cfg.input_normalization['std'], dtype=torch.float32)

            self.denormalizator = Normalize((-mean / std), (1.0 / std))
        else:
            #print("trainerline104")
            self.denormalizator = lambda x: x

        self._load_weights()

    def training(self, epoch):
        if self.sw is None:
            self.sw = SummaryWriterAvg(log_dir=str(self.cfg.LOGS_PATH),
                                       flush_secs=10, dump_period=self.tb_dump_period)

        log_prefix = 'Train' + self.task_prefix.capitalize()
        tbar = tqdm(self.train_data, file=self.tqdm_out, ncols=100)
        train_loss = 0.0

        for metric in self.train_metrics:
            metric.reset_epoch_stats()
        
        self.net.train()
        for i, batch_data in enumerate(tbar):
            global_step = epoch * len(self.train_data) + i
            #print(i)
            loss, losses_logging, splitted_batch_data, outputs = \
                self.batch_forward(batch_data)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            batch_loss = loss.item()
            train_loss += batch_loss

            for loss_name, loss_values in losses_logging.items():
                self.sw.add_scalar(tag=f'{log_prefix}Losses/{loss_name}',
                                   value=np.array(loss_values).mean(),
                                   global_step=global_step)
            self.sw.add_scalar(tag=f'{log_prefix}Losses/overall',
                               value=batch_loss,
                               global_step=global_step)

            for k, v in self.loss_cfg.items():
                if '_loss' in k and hasattr(v, 'log_states') and self.loss_cfg.get(k + '_weight', 0.0) > 0:
                    v.log_states(self.sw, f'{log_prefix}Losses/{k}', global_step)
            #print("trainerline145")
            if self.image_dump_interval > 0 and global_step % self.image_dump_interval == 1:
                self.save_visualization(splitted_batch_data, outputs, global_step, prefix='train')
            #print("trainerline148")
            self.sw.add_scalar(tag=f'{log_prefix}States/learning_rate',
                               value=self.lr if self.lr_scheduler is None else self.lr_scheduler.get_lr()[-1],
                               global_step=global_step)

            tbar.set_description(f'Epoch {epoch}, training loss {train_loss/(i+1):.6f}')
            for metric in self.train_metrics:
                metric.log_states(self.sw, f'{log_prefix}Metrics/{metric.name}', global_step)
        # print("trainline154")
        for metric in self.train_metrics:
            self.sw.add_scalar(tag=f'{log_prefix}Metrics/{metric.name}',
                               value=metric.get_epoch_value(),
                               global_step=epoch, disable_avg=True)

        save_checkpoint(self.net, self.cfg.CHECKPOINTS_PATH, prefix=self.task_prefix,
                        epoch=None, multi_gpu=self.cfg.multi_gpu)
        if epoch % self.checkpoint_interval == 0:
            save_checkpoint(self.net, self.cfg.CHECKPOINTS_PATH, prefix=self.task_prefix,
                            epoch=epoch, multi_gpu=self.cfg.multi_gpu)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def validation(self, epoch, sliding_window):
        if self.sw is None:
            self.sw = SummaryWriterAvg(log_dir=str(self.cfg.LOGS_PATH),
                                       flush_secs=10, dump_period=self.tb_dump_period)

        log_prefix = 'Val' + self.task_prefix.capitalize()
        tbar = tqdm(self.val_data, file=self.tqdm_out, ncols=100)

        for metric in self.val_metrics:
            metric.reset_epoch_stats()

        num_batches = 0
        val_loss = 0
        losses_logging = defaultdict(list)

        self.net.eval()
        for i, batch_data in enumerate(tbar):
            global_step = epoch * len(self.val_data) + i
            
            if sliding_window:
                n = self.sliding_window_inference2(batch_data, (160, 160, 160), 4)
                output = dict()
                output['instances'] = n
                loss, batch_losses_logging, splitted_batch_data, outputs = \
                    self.eva_loss(batch_data, output)
            else:
                loss, batch_losses_logging, splitted_batch_data, outputs = \
                    self.batch_forward(batch_data, validation=True)

            for loss_name, loss_values in batch_losses_logging.items():
                losses_logging[loss_name].extend(loss_values)

            batch_loss = loss.item()
            val_loss += batch_loss
            num_batches += 1

            tbar.set_description(f'Epoch {epoch}, validation loss: {val_loss/num_batches:.6f}')
            for metric in self.val_metrics:
                metric.log_states(self.sw, f'{log_prefix}Metrics/{metric.name}', global_step)

        for loss_name, loss_values in losses_logging.items():
            self.sw.add_scalar(tag=f'{log_prefix}Losses/{loss_name}', value=np.array(loss_values).mean(),
                               global_step=epoch, disable_avg=True)

        for metric in self.val_metrics:
            self.sw.add_scalar(tag=f'{log_prefix}Metrics/{metric.name}', value=metric.get_epoch_value(),
                               global_step=epoch, disable_avg=True)
        self.sw.add_scalar(tag=f'{log_prefix}Losses/overall', value=val_loss / num_batches,
                           global_step=epoch, disable_avg=True)

    def evaluate(self, sliding_window):
        if self.sw is None:
            self.sw = SummaryWriterAvg(log_dir=str(self.cfg.LOGS_PATH),
                                       flush_secs=10, dump_period=self.tb_dump_period)

        tbar = tqdm(self.val_data, file=self.tqdm_out, ncols=100)

        for metric in self.val_metrics:
            metric.reset_epoch_stats()

        self.net.eval()
        dice_metric = DiceMetric(include_background=False, reduction="mean")
        post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=True, n_classes=2)])
        post_label = Compose([EnsureType(), AsDiscrete(to_onehot=True, n_classes=2)])
        for i, batch_data in enumerate(tbar):
            global_step = len(self.val_data) + i
            instances = batch_data['instances'].squeeze(1)

            if sliding_window:
                n = self.sliding_window_inference2(batch_data, (160, 160, 160), 4)
            else:
                _, _, _, output = \
                    self.batch_forward(batch_data, validation=True)
                n = output['instances']

            print(n.shape)
            predicted_instance_masks = torch.sigmoid(n).detach().cpu().numpy()
                
            predicted_instance_masks = np.where(predicted_instance_masks > 0.49, 1, 0)
            gt_instance_masks = instances.cpu().numpy()
                
            for j in range(predicted_instance_masks.shape[0]):
                dice_metric(y_pred=torch.from_numpy(predicted_instance_masks[j]), y=torch.from_numpy(gt_instance_masks[j]))
                
            metric_org = dice_metric.aggregate().item()
            dice_metric.reset()
            
        print("Metric on original image spacing: ", metric_org)

    def batch_forward(self, batch_data, validation=False):
        if 'instances' in batch_data:
            batch_size, num_points, c, l, h, w = batch_data['instances'].size()
            batch_data['instances'] = batch_data['instances'].view(batch_size * num_points, c, l, h, w)
        metrics = self.val_metrics if validation else self.train_metrics
        losses_logging = defaultdict(list)
        with torch.set_grad_enabled(not validation):
            batch_data = {k: v.to(self.device) for k, v in batch_data.items()}
            image, points = batch_data['images'], batch_data['points']

            output = self.net(image, points)

            loss = 0.0
            loss = self.add_loss('instance_loss', loss, losses_logging, validation,
                                 lambda: (output['instances'], batch_data['instances']))
            loss = self.add_loss('instance_aux_loss', loss, losses_logging, validation,
                                 lambda: (output['instances_aux'], batch_data['instances']))
            with torch.no_grad():
                for m in metrics:
                    m.update(*(output.get(x) for x in m.pred_outputs),
                             *(batch_data[x] for x in m.gt_outputs))
        return loss, losses_logging, batch_data, output

    def eva_loss(self, batch_data, output):
        if 'instances' in batch_data:
            batch_size, num_points, c, l, h, w = batch_data['instances'].size()
            batch_data['instances'] = batch_data['instances'].view(batch_size * num_points, c, l, h, w)
        metrics = self.val_metrics
        losses_logging = defaultdict(list)
        with torch.set_grad_enabled(False):

            loss = 0.0
            loss = self.add_loss('instance_loss', loss, losses_logging, True,
                                 lambda: (output['instances'], batch_data['instances']))
            loss = self.add_loss('instance_aux_loss', loss, losses_logging, True,
                                 lambda: (output['instances_aux'], batch_data['instances']))
            with torch.no_grad():
                for m in metrics:
                    m.update(*(output.get(x) for x in m.pred_outputs),
                             *(batch_data[x] for x in m.gt_outputs))
        return loss, losses_logging, batch_data, output

    def sliding_window_inference2(
        self,
        batch_data,
        roi_size: Union[Sequence[int], int],
        sw_batch_size: int,
        overlap: float = 0.25,
        mode: Union[BlendMode, str] = BlendMode.CONSTANT,
        sigma_scale: Union[Sequence[float], float] = 0.125,
        padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT,
        cval: float = 0.0,
        sw_device: Union[torch.device, str, None] = None,
        device: Union[torch.device, str, None] = None,
        *args: Any,
        **kwargs: Any,
) -> torch.Tensor:
        num_spatial_dims = len(batch_data['images'].shape) - 2
        if overlap < 0 or overlap >= 1:
            raise AssertionError("overlap must be >= 0 and < 1.")

        # determine image spatial size and batch size
        # Note: all input images must have the same image size and batch size
        image_size_ = list(batch_data['images'].shape[2:])
        batch_size = batch_data['images'].shape[0]

        if device is None:
            device = batch_data['images'].device
        if sw_device is None:
            sw_device = batch_data['images'].device

        roi_size = fall_back_tuple(roi_size, image_size_)
        # in case that image size is smaller than roi size
        image_size = tuple(max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims))
        pad_size = []
        for k in range(len(batch_data['images'].shape) - 1, 1, -1):
            diff = max(roi_size[k - 2] - batch_data['images'].shape[k], 0)
            half = diff // 2
            pad_size.extend([half, diff - half])
        batch_data['images'] = F.pad(batch_data['images'], pad=pad_size, mode=look_up_option(padding_mode, PytorchPadMode).value, value=cval)
        instance = torch.squeeze(batch_data['instances'], 1)
        instance = F.pad(instance, pad=pad_size, mode=look_up_option(padding_mode, PytorchPadMode).value, value=cval)

        scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)

        # Store all slices in list
        slices = dense_patch_slices(image_size, roi_size, scan_interval)
        num_win = len(slices)  # number of windows per image
        total_slices = num_win * batch_size  # total number of windows

        # Create window-level importance map
        importance_map = compute_importance_map(
            get_valid_patch_size(image_size, roi_size), mode=mode, sigma_scale=sigma_scale, device=device
        )

        # Perform predictions
        output_image, count_map = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
        _initialized = False
        for slice_g in range(0, total_slices, sw_batch_size):
            slice_range = range(slice_g, min(slice_g + sw_batch_size, total_slices))
            unravel_slice = [
                [slice(int(idx / num_win), int(idx / num_win) + 1), slice(None)] + list(slices[idx % num_win])
                for idx in slice_range
            ]
            image = torch.cat([batch_data['images'][win_slice] for win_slice in unravel_slice]).to(sw_device)
            mask = torch.cat([instance[win_slice] for win_slice in unravel_slice]).to(sw_device)
            points_sampler = MultiPointSampler(12, prob_gamma=0.7,
                                           merge_objects_prob=0.15,
                                           max_num_merged_objects=2)
            numpy_image = image.numpy()[0]
            instances_mask = mask.numpy()[0][0].astype(np.int32)
            instances_ids = get_unique_labels(instances_mask, exclude_zero=True)
            instances_info = {
              x: {'ignore': False}
              for x in instances_ids
            }
            sample = {
              'image': numpy_image,
              'instances_mask': instances_mask,
              'instances_info': instances_info,
            }
            sample['objects_ids'] = [obj_id for obj_id, obj_info in sample['instances_info'].items()
                                 if not obj_info['ignore']]
            points = []
            points_sampler.sample_object(sample)
            points.extend(points_sampler.sample_points())
            points = np.array(points, dtype=np.float32)
            x,y = points.shape
            z = image.shape[0]
            pts = np.empty((z, x, y), dtype=np.float32)
            for i in range(z):
              pts[i] = points
            points = torch.from_numpy(pts)
            mask = mask.unsqueeze(1)
            window_data = {
              'images': image,
              'points': points,
              'instances': mask
            }
            loss, batch_losses_logging, splitted_batch_data, outputs = \
                self.batch_forward(window_data, validation=True)

            if not _initialized:  # init. buffer at the first iteration
                output_classes = outputs['instances'].shape[1]
                output_shape = [batch_size, output_classes] + list(image_size)
                # allocate memory to store the full output and the count for overlapping parts
                output_image = torch.zeros(output_shape, dtype=torch.float32, device=device)
                count_map = torch.zeros(output_shape, dtype=torch.float32, device=device)
                _initialized = True

            # store the result in the proper location of the full output. Apply weights from importance map.
            for idx, original_idx in zip(slice_range, unravel_slice):
                outputs['instances'] = outputs['instances'].to('cpu')
                output_image[original_idx] += importance_map * outputs['instances'][idx - slice_g]
                count_map[original_idx] += importance_map

        # account for any overlapping sections
        output_image = output_image / count_map

        final_slicing: List[slice] = []
        for sp in range(num_spatial_dims):
            slice_dim = slice(pad_size[sp * 2], image_size_[num_spatial_dims - sp - 1] + pad_size[sp * 2])
            final_slicing.insert(0, slice_dim)
        while len(final_slicing) < len(output_image.shape):
            final_slicing.insert(0, slice(None))
        return output_image[final_slicing]

    def add_loss(self, loss_name, total_loss, losses_logging, validation, lambda_loss_inputs):
        loss_cfg = self.loss_cfg if not validation else self.val_loss_cfg
        loss_weight = loss_cfg.get(loss_name + '_weight', 0.0)
        if loss_weight > 0.0:
            loss_criterion = loss_cfg.get(loss_name)
            loss = loss_criterion(*lambda_loss_inputs())
            loss = torch.mean(loss)
            losses_logging[loss_name].append(loss.detach().cpu().numpy())
            loss = loss_weight * loss
            total_loss = total_loss + loss

        return total_loss

    def save_visualization(self, splitted_batch_data, outputs, global_step, prefix):
        output_images_path = self.cfg.VIS_PATH / prefix
        if self.task_prefix:
            output_images_path /= self.task_prefix

        if not output_images_path.exists():
            output_images_path.mkdir(parents=True)
        image_name_prefix = f'{global_step:06d}'
        #print("trainerline260")
        def _save_image(suffix, image):
            tiff.imwrite(str(output_images_path / f'{image_name_prefix}_{suffix}.tif'),
                        image)
        #print("trainerline266")
        images = splitted_batch_data['images']
        print(images.shape)
        points = splitted_batch_data['points']
        instance_masks = splitted_batch_data['instances']
        image_blob, points = images[0], points[0]
        #print("trainerline272")
        #image = self.denormalizator(image_blob).cpu().numpy() * 255
        image = image_blob.cpu().numpy()
        image = image[0]
        gt_instance_masks = instance_masks.cpu().numpy()
        predicted_instance_masks = torch.sigmoid(outputs['instances']).detach().cpu().numpy()
        #print(predicted_instance_masks)
        points = points.detach().cpu().numpy()
        if self.max_interactive_points > 0:
            points = points.reshape((-1, 2 * self.max_interactive_points, 3))
        else:
            points = points.reshape((-1, 1, 2))
        #print("trainerline285")
        num_masks = points.shape[0]
        gt_masks = np.squeeze(gt_instance_masks[:num_masks], axis=1)
        #print(gt_masks.shape)
        predicted_masks = np.squeeze(predicted_instance_masks[:num_masks], axis=1)
        #print(predicted_masks.shape)
        #print("trainerline289")
        viz_image = []
        i = 0
        for gt_mask, point, predicted_mask in zip(gt_masks, points, predicted_masks):
            gt_mask[gt_mask < 0] = 0.25
            gt_mask = gt_mask.transpose([2, 0, 1])
            predicted_mask = predicted_mask.transpose([2, 0, 1])
            image = image.transpose([2, 0, 1])
            #gt_mask = gt_mask  * 255
            #predicted_mask = predicted_mask  * 255
            image = image  * 255
            
            for i in range(image.shape[0]):
              image_i = image[i]
              w, h = image_i.shape
              ret = np.empty((w, h, 3), dtype=np.uint8)
              ret[:, :, 0] = image_i
              ret[:, :, 1] = image_i
              ret[:, :, 2] = image_i
              tiff.imwrite('./c.tif', gt_mask)
              
              timage = draw_points(ret, i, point[:max(1, self.max_interactive_points)], (0, 255, 0))
              if self.max_interactive_points > 0:
                  timage = draw_points(timage, i, point[self.max_interactive_points:], (0, 0, 255))
              
              gt_mask_i = draw_probmap(gt_mask[i])
              predicted_mask_i = draw_probmap(predicted_mask[i])

              if viz_image != []:
                temp = np.hstack((timage, gt_mask_i, predicted_mask_i))
                viz_image=np.vstack((viz_image, temp))
              else:
                viz_image=np.hstack((timage, gt_mask_i, predicted_mask_i))
        result = viz_image.astype(np.uint16)
        _save_image('instance_segmentation', result[:, :, ::-1])

    def _load_weights(self):
        print("self.cfg", self.cfg)
        if self.cfg.weights is not None:
            if os.path.isfile(self.cfg.weights):
                self.net.load_weights(self.cfg.weights)
                self.cfg.weights = None
            else:
                raise RuntimeError(f"=> no checkpoint found at '{self.cfg.weights}'")
        elif self.cfg.resume_exp is not None:
            checkpoints = list(self.cfg.CHECKPOINTS_PATH.glob(f'{self.cfg.resume_prefix}*.pth'))
            assert len(checkpoints) == 1

            checkpoint_path = checkpoints[0]
            logger.info(f'Load checkpoint from path: {checkpoint_path}')
            self.net.load_weights(str(checkpoint_path))
        self.net = self.net.to(self.device)


class _CustomDP(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
