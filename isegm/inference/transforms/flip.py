import torch

from isegm.inference.clicker import Click
from .base import BaseTransform


class AddHorizontalFlip(BaseTransform):
    def transform(self, image_nd, clicks_lists):
        assert len(image_nd.shape) == 5
        image_nd = torch.cat([image_nd, torch.flip(image_nd, dims=[3])], dim=0)
        #print(image_nd.shape)

        image_width = image_nd.shape[3]
        clicks_lists_flipped = []
        for clicks_list in clicks_lists:
            clicks_list_flipped = [Click(is_positive=click.is_positive,
                                         coords=(click.coords[0], image_width - click.coords[1] - 1, click.coords[2]))
                                   for click in clicks_list]
            clicks_lists_flipped.append(clicks_list_flipped)
        clicks_lists = clicks_lists + clicks_lists_flipped

        return image_nd, clicks_lists

    def inv_transform(self, prob_map):
        #print(prob_map.shape)
        assert len(prob_map.shape) == 5 and prob_map.shape[0] % 2 == 0
        num_maps = prob_map.shape[0] // 2
        #print(num_maps)
        prob_map, prob_map_flipped = prob_map[:num_maps], prob_map[num_maps:]
        #print(prob_map.shape)
        #print(prob_map_flipped.shape)

        return 0.5 * (prob_map + torch.flip(prob_map_flipped, dims=[3]))

    def get_state(self):
        return None

    def set_state(self, state):
        pass

    def reset(self):
        pass
