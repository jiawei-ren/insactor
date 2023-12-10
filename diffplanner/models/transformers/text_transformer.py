import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import clip

from collections import OrderedDict

from ..builder import SUBMODULES

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad for all the networks.

    Args:
        nets (nn.Module | list[nn.Module]): A list of networks or a single
            network.
        requires_grad (bool): Whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


@SUBMODULES.register_module()
class TextEncoder(nn.Module):
    def __init__(self,
                 pretrained_model='clip',
                 text_latent_dim=512,
                 time_embed_dim=2048,
                 dropout=0,
                 num_text_layers=4,
                 text_num_heads=4,
                 text_ff_size=2048,  
                 use_text_proj=True):
        super().__init__()
        activation = 'gelu'
        self.time_embed_dim = time_embed_dim
        self.use_text_proj = use_text_proj

        if pretrained_model == 'clip':
            self.clip, _ = clip.load('ViT-B/32', "cpu")
            set_requires_grad(self.clip, False)
            if text_latent_dim != 512:
                self.text_pre_proj = nn.Linear(512, text_latent_dim)
            else:
                self.text_pre_proj = nn.Identity()
        else:
            raise NotImplementedError()
        
        if num_text_layers > 0:
            self.use_text_finetune = True
            textTransEncoderLayer = nn.TransformerEncoderLayer(
                d_model=text_latent_dim,
                nhead=text_num_heads,
                dim_feedforward=text_ff_size,
                dropout=dropout,
                activation=activation)
            self.textTransEncoder = nn.TransformerEncoder(
                textTransEncoderLayer,
                num_layers=num_text_layers)
        else:
            self.use_text_finetune = False
        self.text_ln = nn.LayerNorm(text_latent_dim)
        if self.use_text_proj:
            self.text_proj = nn.Sequential(
                nn.Linear(text_latent_dim, self.time_embed_dim)
            )
        
    def forward(self, text, token=None, device=None):
        with torch.no_grad():
            text = clip.tokenize(text, truncate=True).to(device)
            x = self.clip.token_embedding(text).type(self.clip.dtype)

            x = x + self.clip.positional_embedding.type(self.clip.dtype)
            x = x.permute(1, 0, 2)
            x = self.clip.transformer(x)
            x = self.clip.ln_final(x).type(self.clip.dtype)

        x = self.text_pre_proj(x)
        xf_out = self.textTransEncoder(x)
        xf_out = self.text_ln(xf_out)
        if self.use_text_proj:
            xf_proj = self.text_proj(xf_out[text.argmax(dim=-1), torch.arange(xf_out.shape[1])])
            return xf_proj
        else:
            xf_out = xf_out.permute(1, 0, 2)
            return xf_out
        
    def load_pretrained(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        text_ckpt = OrderedDict()
        for key in checkpoint['state_dict'].keys():
            if key.startswith('text_encoder'):
                new_key = key[13:]
                text_ckpt[new_key] = checkpoint['state_dict'][key]
        self.load_state_dict(text_ckpt, strict=False)