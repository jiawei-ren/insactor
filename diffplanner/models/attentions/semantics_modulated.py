import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.stylization_block import StylizationBlock
from ..builder import ATTENTIONS


@ATTENTIONS.register_module()
class SemanticsModulatedAttention(nn.Module):

    def __init__(self, latent_dim,
                       text_latent_dim,
                       num_heads,
                       dropout,
                       time_embed_dim,
                       use_retrieval=False):
        super().__init__()
        self.num_heads = num_heads

        self.norm = nn.LayerNorm(latent_dim)
        self.text_norm = nn.LayerNorm(text_latent_dim)
        
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key_text = nn.Linear(text_latent_dim, latent_dim)
        self.value_text = nn.Linear(text_latent_dim, latent_dim)
        self.key_motion = nn.Linear(latent_dim, latent_dim)
        self.value_motion = nn.Linear(latent_dim, latent_dim)
        
        self.use_retrieval = use_retrieval
        if self.use_retrieval:
            self.retr_norm = nn.LayerNorm(latent_dim)
            self.key_retr = nn.Linear(latent_dim, latent_dim)
            self.value_retr = nn.Linear(latent_dim, latent_dim)

        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)
    
    def forward(self, x, xf, emb, src_mask, cond_type, re_feat=None):
        """
        x: B, T, D
        xf: B, N, L
        """
        B, T, D = x.shape
        # re_motion = re_dict['re_motion']
        # re_text = re_dict['re_text']
        # retr = torch.cat((re_text, re_motion), dim=-1)
        if self.use_retrieval:
            N = xf.shape[1] + x.shape[1] + re_feat.shape[1]
        else:
            N = xf.shape[1] + x.shape[1]
        H = self.num_heads
        # B, T, D
        query = self.query(self.norm(x))
        # B, N, D
        text_cond_type = 0 if cond_type % 10 == 0 else 1
        retr_cond_type = 0 if cond_type // 10 == 0 else 1
        if self.use_retrieval:
            key = torch.cat((
                self.key_text(self.text_norm(xf)) + (1 - text_cond_type) * -1000000,
                self.key_retr(self.retr_norm(re_feat)) + (1 - retr_cond_type) * -1000000,
                self.key_motion(self.norm(x)) + (1 - src_mask) * -1000000
            ), dim=1)
        else:
            key = torch.cat((
                self.key_text(self.text_norm(xf)) + (1 - text_cond_type) * -1000000,
                self.key_motion(self.norm(x)) + (1 - src_mask) * -1000000
            ), dim=1)
        query = F.softmax(query.view(B, T, H, -1), dim=-1)
        key = F.softmax(key.view(B, N, H, -1), dim=1)
        # B, N, H, HD
        if self.use_retrieval:
            value = torch.cat((
                self.value_text(self.text_norm(xf)) * text_cond_type,
                self.value_retr(self.retr_norm(re_feat)) * retr_cond_type,
                self.value_motion(self.norm(x)) * src_mask,
            ), dim=1).view(B, N, H, -1)
        else:
            value = torch.cat((
                self.value_text(self.text_norm(xf)) * text_cond_type,
                self.value_motion(self.norm(x)) * src_mask,
            ), dim=1).view(B, N, H, -1)
        # B, H, HD, HD
        attention = torch.einsum('bnhd,bnhl->bhdl', key, value)
        y = torch.einsum('bnhd,bhdl->bnhl', query, attention).reshape(B, T, D)
        y = x + self.proj_out(y, emb)
        return y