from ml_collections import ConfigDict

def get_segmentation_config():
    segmentation = ConfigDict()
    segmentation.num_classes = 2  # Classes: graspable, non-graspable
    return segmentation
