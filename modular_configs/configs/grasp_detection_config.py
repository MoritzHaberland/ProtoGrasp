from ml_collections import ConfigDict

def get_grasp_detection_config():
    grasp_detection = ConfigDict()
    grasp_detection.num_classes = 18  # Number of orientation classes
    grasp_detection.fc_units = 1024  # Number of units in the fully connected layers
    return grasp_detection
