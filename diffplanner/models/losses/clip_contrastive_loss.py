import torch
import torch.nn as nn
import numpy as np
import time
import math
import torch.nn.functional as F
from ..builder import LOSSES

@LOSSES.register_module()
class Clip_ContrastiveLoss(torch.nn.Module):
    def __init__(self):
        super(Clip_ContrastiveLoss, self).__init__()

    def forward(self, logits_per_motion, logits_per_text, labels):
        motion_loss = F.cross_entropy(logits_per_motion, labels)
        text_loss  = F.cross_entropy(logits_per_text, labels)
        loss = (motion_loss + text_loss) / 2
        return loss
