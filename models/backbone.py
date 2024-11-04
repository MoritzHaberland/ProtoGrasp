import torch.nn as nn
import torchvision.models as models

class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        # Remove last fully connected layer and pooling
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])  

    def forward(self, x):
        return self.resnet(x)
