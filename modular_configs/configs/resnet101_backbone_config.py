from ml_collections import ConfigDict

def get_backbone_config():
    backbone = ConfigDict()
    backbone.type = 'resnet101' 
    backbone.pretrained = True
    return backbone