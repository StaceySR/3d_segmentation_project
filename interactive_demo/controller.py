import torch
import numpy as np
from torchvision import transforms

from isegm.inference import clicker
from isegm.inference.predictors import get_predictor
from isegm.utils.vis import draw_with_blend_and_clicks
import tifffile as tiff


class InteractiveController:
    def __init__(self, net, device, predictor_params, update_image_callback, prob_thresh=0.5):
        self.net = net.to(device)
        self.prob_thresh = prob_thresh
        self.clicker = clicker.Clicker()
        self.states = []
        self.probs_history = []
        self.object_count = 0
        self._result_mask = None

        self.image = None
        self.image_nd = None
        self.predictor = None
        self.device = device
        self.update_image_callback = update_image_callback
        self.predictor_params = predictor_params
        self.reset_predictor()

    def set_image(self, image):
        # input_transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize([.485, .456, .406], [.229, .224, .225])
        # ])

        self.image = image
        #print(image.shape)
        im = np.transpose(image, [1, 2, 0])
        im = np.expand_dims(im, axis=0)
        print("im", im.shape)
        self.image_nd_for_flip = im
        self.image_nd_for_brightness = im.astype(np.float64)
        #print(im.shape)
        self.image_nd = torch.from_numpy(im).to(self.device)
        #print(self.image_nd.shape)
        d, h, w = image.shape
        img = np.empty((d, h, w, 3), dtype=np.uint8)
        img[:, :, :, 0] = image
        img[:, :, :, 1] = image
        img[:, :, :, 2] = image
        self.image = img
        self.image_for_brightness = img.astype(np.float64)
        self._result_mask = np.zeros(self.image.shape[:3], dtype=np.uint16)
        self.object_count = 0
        self.reset_last_object(update_image=False)
        self.update_image_callback(reset_canvas=True)

    def add_click(self, x, y, z, is_positive):
        self.states.append({
            'clicker': self.clicker.get_state(),
            'predictor': self.predictor.get_states()
        })

        click = clicker.Click(is_positive=is_positive, coords=(y, x, z))
        self.clicker.add_click(click)
        pred = self.predictor.get_prediction(self.clicker)
        torch.cuda.empty_cache()

        if self.probs_history:
            self.probs_history.append((self.probs_history[-1][0], pred))
        else:
            self.probs_history.append((np.zeros_like(pred), pred))

        self.update_image_callback()

    def undo_click(self):
        if not self.states:
            return

        prev_state = self.states.pop()
        self.clicker.set_state(prev_state['clicker'])
        self.predictor.set_states(prev_state['predictor'])
        self.probs_history.pop()
        self.update_image_callback()

    def partially_finish_object(self):
        object_prob = self.current_object_prob
        if object_prob is None:
            return

        self.probs_history.append((object_prob, np.zeros_like(object_prob)))
        self.states.append(self.states[-1])

        self.clicker.reset_clicks()
        self.reset_predictor()
        self.update_image_callback()

    def finish_object(self):
        object_prob = self.current_object_prob
        if object_prob is None:
            return

        self.object_count += 1
        object_prob1 = object_prob.transpose([2,0,1])
        object_mask = object_prob1 > self.prob_thresh
        print(object_mask)

        self._result_mask[object_mask] = self.object_count
        self.reset_last_object()

    def reset_last_object(self, update_image=True):
        self.states = []
        self.probs_history = []
        self.clicker.reset_clicks()
        self.reset_predictor()
        if update_image:
            self.update_image_callback()

    def reset_predictor(self, predictor_params=None):
        if predictor_params is not None:
            self.predictor_params = predictor_params
        self.predictor = get_predictor(self.net, device=self.device,
                                       **self.predictor_params)
        if self.image_nd is not None:
            self.predictor.set_input_image(self.image_nd)

    @property
    def current_object_prob(self):
        if self.probs_history:
            current_prob_total, current_prob_additive = self.probs_history[-1]
            return np.maximum(current_prob_total, current_prob_additive)
        else:
            return None

    @property
    def is_incomplete_mask(self):
        return len(self.probs_history) > 0

    @property
    def result_mask(self):
        return self._result_mask.copy()

    def get_visualization(self, alpha_blend, click_radius, image_flip, flip_origin, flip_mode, brightness,
                          brightness_origin):
        if self.image is None:
            return None

        if brightness != brightness_origin:
            brightness_ori_to_use = brightness_origin/10+0.5
            brightness_to_use = brightness/10+0.5
            self.image = (self.image / brightness_ori_to_use * brightness_to_use).astype(np.uint8)
            self.image = np.clip(self.image, 0, 255)
            self.image_nd_for_flip = (self.image_nd_for_flip / brightness_ori_to_use * brightness_to_use)\
                .astype(np.uint8)
            self.image_nd_for_flip = np.clip(self.image_nd_for_flip, 0, 255)
            cont_img = np.ascontiguousarray(self.image_nd_for_flip)
            self.image_nd = torch.from_numpy(cont_img).to(self.device)

        if image_flip != flip_origin:
            self.image = np.flip(self.image, axis=1)
            self.image_nd_for_flip = np.flip(self.image_nd_for_flip, axis=1)
            cont_img = np.ascontiguousarray(self.image_nd_for_flip)
            self.image_nd = torch.from_numpy(cont_img).to(self.device)
            self._result_mask = np.zeros(self.image.shape[:3], dtype=np.uint16)
            self.reset_predictor()
            if flip_mode == 'reset pred':
                self.states = []
                self.probs_history = []
                self.clicker.reset_clicks()
            else:
                height = self.image.shape[1]
                clicks_list_flipped = [clicker.Click(is_positive=click.is_positive,
                                                     coords=(
                                                     height - click.coords[0], click.coords[1], click.coords[2]))
                                       for click in self.clicker.get_clicks()]
                self.clicker.reset_clicks()
                for click in clicks_list_flipped:
                    self.states.append({
                        'clicker': self.clicker.get_state(),
                        'predictor': self.predictor.get_states()
                    })

                    click = clicker.Click(is_positive=click.is_positive, coords=(click.coords[0], click.coords[1],
                                                                                 click.coords[2]))
                    self.clicker.add_click(click)
                    pred = self.predictor.get_prediction(self.clicker)
                    torch.cuda.empty_cache()

                    if self.probs_history:
                        self.probs_history.append((self.probs_history[-1][0], pred))
                    else:
                        self.probs_history.append((np.zeros_like(pred), pred))

        results_mask_for_vis = self.result_mask

        if self.probs_history:
            results_mask_for_vis = results_mask_for_vis.transpose([1, 2, 0])
            results_mask_for_vis[self.current_object_prob > self.prob_thresh] = self.object_count + 1
            results_mask_for_vis = results_mask_for_vis.transpose([2, 0, 1])


        vis = draw_with_blend_and_clicks(self.image, mask=results_mask_for_vis, alpha=alpha_blend,
                                         clicks_list=self.clicker.clicks_list, radius=click_radius)


        if self.probs_history:
            total_mask = self.probs_history[-1][0] > self.prob_thresh
            total_mask = total_mask.transpose([2, 0, 1])
            results_mask_for_vis[np.logical_not(total_mask)] = 0
            vis = draw_with_blend_and_clicks(vis, mask=results_mask_for_vis, alpha=alpha_blend)

        return vis
