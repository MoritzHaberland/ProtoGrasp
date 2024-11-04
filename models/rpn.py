import torch.nn as nn
import torch.nn.functional as F

class RPN(nn.Module):
    def __init__(self, in_channels):
        super(RPN, self).__init__()
        self.conv = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)
        self.cls_layer = nn.Conv2d(512, 18 * 2, kernel_size=1)  # 2 for foreground and background
        self.reg_layer = nn.Conv2d(512, 18 * 4, kernel_size=1)  # 4 for bounding box deltas

    def forward(self, x):
        x = F.relu(self.conv(x))
        rpn_cls_logits = self.cls_layer(x)  # Classification logits
        rpn_reg_deltas = self.reg_layer(x)  # Regression deltas
        return rpn_cls_logits, rpn_reg_deltas
