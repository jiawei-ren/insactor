import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import clip

from .base_architecture import BaseArchitecture
from ..builder import (
    ARCHITECTURES,
    build_architecture,
    build_submodule,
    build_loss
)


@ARCHITECTURES.register_module()
class Clip_ContrastModel(BaseArchitecture):

    def __init__(self,
                 motion_encoder=None,
                 text_encoder=None,
                 contrastive_loss=None,
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg, **kwargs)
        self.motion_encoder = build_submodule(motion_encoder)
        self.text_encoder = build_submodule(text_encoder)
        self.contrastive_loss = build_loss(contrastive_loss)

    def forward(self, **kwargs):
        motion, motion_mask = kwargs['motion'].float(), kwargs['motion_mask'].float()
        B, T = motion.shape[:2]
        text = []
        for i in range(B):
            text.append(kwargs['motion_metas'][i]['text'])
        
        mu, logvar = self.motion_encoder(motion, motion_mask)
        text_features = self.text_encoder(text, motion.device)
        motion_features = mu
        
        pre_logit_scale = nn.Parameter(torch.zeros(1))
        logit_scale = pre_logit_scale.exp().cuda()
        
        logits_per_motion = logit_scale * motion_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ motion_features.t()
        
        labels = torch.arange(len(logits_per_motion)).to(logits_per_motion.device)

        loss = dict()
        loss['loss'] = self.contrastive_loss(logits_per_motion, logits_per_text, labels)
            
        return loss
        