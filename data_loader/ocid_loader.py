import torch
import torchvision.io as torchvision_io
import os
from torch.utils.data import Dataset, DataLoader

# Directory / file names
RGB_DIR = 'rgb'
MASK_DIR = 'seg_mask_labeled_combi'
ANNOTATION_DIR = 'Annotations'
IMAGE_EXT = '.png'
TXT_FILE_EXT = '.txt'

class OcidDataset(Dataset):

    def __init__(self, split_file_path, root_path, split_name, transform):
        super().__init__()
        self.split_file_path = split_file_path
        self.root_path = root_path
        self.split_name = split_name
        self.transform = transform

        self._image_paths = self._load_split()

    def _load_split(self):
        """Load the list of image paths from the split file"""
        split_file = os.path.join(self.split_file_path, f"{self.split_name}{TXT_FILE_EXT}")
        with open(split_file, "r") as fid:
            # remove blanks infront and behind the paths
            image_paths = [x.strip() for x in fid.readlines()]
        return image_paths
    
    def _load_item(self, item):
        """Load the image, mask, and annotations for a given item"""
        seq_path, im_name = item.split(',')

        # Build full paths for the image, mask, and annotation
        sample_path = os.path.join(self.root_path, seq_path)
        img_path = os.path.join(sample_path, RGB_DIR, im_name)
        mask_path = os.path.join(sample_path, MASK_DIR, im_name)
        anno_path = os.path.join(sample_path, ANNOTATION_DIR, im_name.replace(IMAGE_EXT, TXT_FILE_EXT))

        # Load image (already in RGB format)
        img = torchvision_io.read_image(img_path)  # Load image as a tensor (C, H, W)

        # Load annotations (grasp detection boxes)
        points_list = []
        boxes_list = []

        with open(anno_path, "r") as f:
            for count, line in enumerate(f):
                line = line.rstrip()  # Remove any trailing whitespace, including newlines
                [x, y] = line.split(' ') 

                x = float(x)
                y = float(y)

                pt = (x, y)
                points_list.append(pt)  # Append the point to points_list

                if len(points_list) == 4:  # collect 4 point and store them in the box list
                    boxes_list.append(points_list)
                    points_list = []  

        # Convert points to a tensor
        box_arry = torch.tensor(boxes_list, dtype=torch.float32)

        # Load mask image
        msk = torchvision_io.read_image(mask_path, mode=torchvision_io.ImageReadMode.UNCHANGED)

        return img, msk, box_arry

    def __getitem__(self, index):
        pass
    
    def __len__(self):
        pass