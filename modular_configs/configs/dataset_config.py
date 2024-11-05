import os
from ml_collections import ConfigDict

def get_dataset_config():
    dataset = ConfigDict()
    dataset.root_path = "/home/mhaberland/ProtoGrasp/datasets/OCID_grasp"
    dataset.split_file_path = "/home/mhaberland/ProtoGrasp/datasets/OCID_grasp/data_split/"

    return dataset
