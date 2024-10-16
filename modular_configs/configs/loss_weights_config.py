from ml_collections import ConfigDict

def get_loss_weights_config():
    loss_weights = ConfigDict()
    loss_weights.grasp_loss = 1.0
    loss_weights.segmentation_loss = 0.8
    loss_weights.refinement_loss = 0.8
    return loss_weights
