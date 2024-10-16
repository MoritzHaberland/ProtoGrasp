from ml_collections import config_dict
from modular_configs.configs.resnet101_backbone_config import get_backbone_config
from modular_configs.configs.grasp_detection_config import get_grasp_detection_config
from modular_configs.configs.segmentation_config import get_segmentation_config
from modular_configs.configs.grasp_refinement_config import get_grasp_refinement_config
from modular_configs.configs.training_config import get_training_config
from modular_configs.configs.loss_weights_config import get_loss_weights_config

def get_config():
    config = config_dict.ConfigDict()

    # Basic setup
    config.project_name = "ProtoGrasp"

    # Hierarchical structure
    config.model = config_dict.ConfigDict()
    config.model.backbone = get_backbone_config()
    config.model.grasp_detection = get_grasp_detection_config()
    config.model.segmentation = get_segmentation_config()
    config.model.grasp_refinement = get_grasp_refinement_config()

    config.training = get_training_config()
    config.loss_weights = get_loss_weights_config()

    return config
