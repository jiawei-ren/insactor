import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from .base_architecture import BaseArchitecture
from ..builder import (
    ARCHITECTURES,
    build_architecture,
    build_submodule,
    build_loss
)

@ARCHITECTURES.register_module()
class ContrastModel(BaseArchitecture):

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

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar / 2)

        eps = std.data.new(std.size()).normal_()
        latent_code = eps.mul(std).add_(mu)
        return latent_code

    def forward(self, **kwargs):
        motion, motion_mask = kwargs['motion'].float(), kwargs['motion_mask'].float()
        B, T = motion.shape[:2]
        text = []
        for i in range(B):
            text.append(kwargs['motion_metas'][i]['text'])
        
        mu, logvar = self.motion_encoder(motion, motion_mask)
        text_features = self.text_encoder(text, motion.device)
        motion_features = mu
        
        '''Positive pairs'''
        pos_labels = torch.zeros(B).to(motion.device)
        self.loss_pos = self.contrastive_loss(text_features, motion_features, pos_labels)

        '''Negative Pairs, shifting index'''
        neg_labels = torch.ones(B).to(motion.device)
        shift = np.random.randint(0, B-1)
        new_idx = np.arange(shift, B + shift) % B
        mis_motion_embedding = motion_features.clone()[new_idx]
        self.loss_neg = self.contrastive_loss(text_features, mis_motion_embedding, neg_labels)
        self.loss = self.loss_pos + self.loss_neg

        loss_logs = dict()
        loss_logs['loss'] = self.loss
        loss_logs['loss_pos'] = self.loss_pos
        loss_logs['loss_neg'] = self.loss_neg
        return loss_logs
        