import random
import torch
import numpy as np
import torchvision.transforms as T
from torchvision.transforms import functional as tfn

class OCIDTransform:
    """Transformer function for OCID_grasp dataset
    """

    def __init__(self,
                 shortest_size,
                 longest_max_size,
                 rgb_mean=None,
                 rgb_std=None,
                 random_flip=False,
                 random_scale=None,
                 rotate_and_scale=False):
        self.shortest_size = shortest_size
        self.longest_max_size = longest_max_size
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std
        self.random_flip = random_flip
        self.random_scale = random_scale
        self.rotate_and_scale = rotate_and_scale

    def _adjusted_scale(self, in_width, in_height, target_size):
        min_size = min(in_width, in_height)
        max_size = max(in_width, in_height)
        scale = target_size / min_size

        if int(max_size * scale) > self.longest_max_size:
            scale = self.longest_max_size / max_size

        return scale
    
    @staticmethod
    # 50% chance of performing a horizontal flip on the image and masks ("->" => "<-")
    def _random_flip(img, msk):
        if random.random() < 0.5:
            img = tfn.hflip(img)
            msk = [tfn.hflip(m) for m in msk] 
            return img, msk
        return img, msk
    
    def _normalize_image(self, img):
        if self.rgb_mean is not None:
            img.sub_(img.new(self.rgb_mean).view(-1, 1, 1))
        if self.rgb_std is not None:
            img.div_(img.new(self.rgb_std).view(-1, 1, 1))
        return img
    
    

