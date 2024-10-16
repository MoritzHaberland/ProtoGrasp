from ml_collections import ConfigDict

def get_grasp_refinement_config():
    refinement = ConfigDict()
    refinement.fc_units = 1024  # Fully connected layer units for grasp refinement
    return refinement
