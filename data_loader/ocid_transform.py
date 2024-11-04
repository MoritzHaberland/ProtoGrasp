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
    
    
    
    def _normalize_image(self, img):
    # is there a reason why only one should be None?
    # ?
    # mark    
        if self.rgb_mean is not None:
            img.sub_(img.new(self.rgb_mean).view(-1, 1, 1))
        if self.rgb_std is not None:
            img.div_(img.new(self.rgb_std).view(-1, 1, 1))
        return img
    
    @staticmethod
    def _rotate_2d(pts, cnt, ang_deg):
        ang_rad = torch.tensor(ang_deg * (torch.pi / 180.0))

        # Create the rotation matrix
        rotation_matrix = torch.tensor([[torch.cos(ang_rad), -torch.sin(ang_rad)],
                                         [torch.sin(ang_rad), torch.cos(ang_rad)]])
        
        rotated_pts = torch.mm(pts - cnt, rotation_matrix) + cnt

        return rotated_pts
    
    def __call__(self, img_, msk_, bbox_infos_):
        im_size = [self.shortest_size, self.longest_max_size]

        # Calculate the borders for the center crop
        x_min = int(img_.shape[0] / 2 - int(im_size[0] / 2))
        x_max = int(img_.shape[0] / 2 + int(im_size[0] / 2))
        y_min = int(img_.shape[1] / 2 - int(im_size[1] / 2))
        y_max = int(img_.shape[1] / 2 + int(im_size[1] / 2))

        new_origin = np.array([])

        img = img_[x_min:x_max, y_min:y_max, :]

        msk = msk_[x_min:x_max, y_min:y_max]

        bbox_infos_ = bbox_infos_ - new_origin
        bbox_infos = np.copy(bbox_infos_)

        if self.rotate_and_scale:
            img, msk, bbox_transformed = self._rotateAndScale(img, msk, bbox_infos_)
            bbox_infos = bbox_transformed
        # Random flip
        if self.random_flip:
            img, msk = self._random_flip(img, msk)

        # Adjust scale, possibly at random
        if self.random_scale is not None:
            target_size = self._random_target_size()
        else:
            target_size = self.shortest_size

        ret = self._prepare_frcnn_format(bbox_infos, im_size)
        (x1, y1, theta, x2, y2, cls) = ret
        if len(cls) == 0:
            print('NO valid boxes after augmentation, switch to gt values')
            ret = self._prepare_frcnn_format(bbox_infos_, im_size)
            img = img_[x_min:x_max, y_min:y_max, :]

            msk = msk_[x_min:x_max, y_min:y_max]

        bbox_infos = np.asarray(ret).T
        bbox_infos = bbox_infos.astype(np.float32)

        # Image transformations
        img = tfn.to_tensor(img)
        img = self._normalize_image(img)

        # Label transformations
        msk = np.stack([np.array(m, dtype=np.int32, copy=False) for m in msk], axis=0)

        # Convert labels to torch and extract bounding boxes
        msk = torch.from_numpy(msk.astype(np.long))

        bbx = torch.from_numpy(np.asarray(bbox_infos)).contiguous()
        if bbox_infos.shape[1] != 6:
            assert False

        return dict(img=img, msk=msk, bbx=bbx), im_size

    

    
    
    

