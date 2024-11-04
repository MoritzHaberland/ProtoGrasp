import torch
import torchvision.io as torchvision_io
import numpy as np
import os
import math
from torch.utils.data import Dataset, DataLoader

# Directory / file names
RGB_DIR = 'rgb'
MASK_DIR = 'seg_mask_labeled_combi'
ANNOTATION_DIR = 'Annotations'
IMAGE_EXT = '.png'
TXT_FILE_EXT = '.txt'

class OcidDataset(Dataset):

    def __init__(self, split_file_path, root_path, split_name, transform, number_of_angle_classes):
        super().__init__()
        self.split_file_path = split_file_path
        self.root_path = root_path
        self.split_name = split_name
        self.transform = transform
        
        angle_intervals = 360 / number_of_angle_classes  
        self.angles = np.array([i * angle_intervals for i in range(number_of_angle_classes)])

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
        points = []
        boxes_corners = []

        with open(anno_path, "r") as f:
            for count, line in enumerate(f):
                line = line.rstrip()  # Remove any trailing whitespace, including newlines
                [x, y] = line.split(' ') 

                x = float(x)
                y = float(y)

                pt = (x, y)
                points.append(pt)  # Append the point to points_list

                if len(points) == 4:  # collect 4 point and store them in the box list
                    boxes_corners.append(points)
                    points = []  

        # Convert points to a tensor
        boxes_corners = torch.tensor(boxes_corners, dtype=torch.float32)

        # Load mask image
        msk = torchvision_io.read_image(mask_path, mode=torchvision_io.ImageReadMode.UNCHANGED)

        return img, msk, boxes_corners
    
    def _get_unrotated_box_with_angle(self, corners):
        # Define corners for readability
        x1, y1 = corners[0]
        x2, y2 = corners[1]
        x3, y3 = corners[2]
        x4, y4 = corners[3]

        # Calculate the center as the midpoint between two opposite corners
        x_center = (x1 + x3) / 2
        y_center = (y1 + y3) / 2

        # Calculate width and height
        width = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)  # Distance between first and second corner
        height = math.sqrt((x4 - x1) ** 2 + (y4 - y1) ** 2)  # Distance between first and fourth corner

        # calculate the unrotated upper left and lower right corner
        xmin = x_center - width/2
        ymin = y_center - height/2
        xmax = x_center + width/2
        ymax = y_center + height/2

        # Calculate the rotation angle (in degrees) relative to the horizontal axis
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

        return [xmin, ymin, xmax, ymax], angle
    
    def _find_closest_index_class(self, angle):
        # Normalize the target angle to be within [0, 360)

        target_angle = angle % 360
        
        # Calculate the differences from the target angle to all class angles
        differences = np.abs(self.angles - target_angle)

        # Find the index of the minimum difference
        closest_index = np.argmin(differences)
        
        # Return the closest class index and its mean angle
        return closest_index, self.angles[closest_index]

    def __getitem__(self, index):
        item_path = self._image_paths[index]
        img, msk, boxes_cornes = self._load_item(item_path)

        # Apply transformations if specified
        if self.transform:
            img = self.transform(img)

        # Convert each bounding box in `box_corners` to (x, y, w, h, theta) after any transformations
        boxes_xyxy = []
        angles = []
        for corners in boxes_cornes:
            xyxy,a = self._get_unrotated_box_with_angle(corners)
            boxes_xyxy.append(xyxy)
            angles.append(self._find_closest_index_class(a))

        boxes_xyxy = torch.tensor(boxes_xyxy, dtype=torch.float32)
        angles = torch.tensor(angles, dtype=torch.int)

        return {
            "image": img,
            "mask": msk,
            "boxes_rotated_corners": boxes_cornes,
            "boxes_unrotated": boxes_xyxy,
            "angles": angles
        } 

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self._image_paths)