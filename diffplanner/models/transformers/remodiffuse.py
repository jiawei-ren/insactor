from cv2 import norm
import torch
import torch.nn.functional as F
from torch import layer_norm, nn
import numpy as np
import clip
import random
import math

from ..builder import SUBMODULES
from .diffusion_transformer import DiffusionTransformer


class RetrievalDatabase(nn.Module):

    def __init__(self,
                 num_retrieval=None,
                 use_motion=False,
                 use_text=False,
                 retrieval_file=None,
                 latent_dim=512,
                 output_dim=512,
                 num_layers=2,
                 max_seq_len=196,
                 num_heads=8,
                 ff_size=1024,
                 dropout=0):
        super().__init__()
        self.num_retrieval = num_retrieval
        self.use_motion = use_motion
        self.use_text = use_text
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        data = np.load(retrieval_file)
        self.text_features = torch.Tensor(data['text_features'])
        self.captions = data['captions']
        self.motions = data['motions']
        self.m_lengths = data['m_lengths']
        if 'text_seq_features' in data.keys():
            self.text_seq_features = data['text_seq_features']

        self.latent_dim = latent_dim
        self.output_dim = output_dim
        if self.use_motion:
            self.motion_proj = nn.Linear(self.motions.shape[-1], self.latent_dim)
            self.motion_pos_embedding = nn.Parameter(torch.randn(max_seq_len, self.latent_dim))
            TransEncoderLayer = nn.TransformerEncoderLayer(
                d_model=self.latent_dim,
                nhead=num_heads,
                dim_feedforward=ff_size,
                dropout=dropout,
                activation="gelu",
                batch_first=True)
            self.motion_encoder = nn.TransformerEncoder(
                TransEncoderLayer,
                num_layers=num_layers)
        if self.use_text:
            TransEncoderLayer = nn.TransformerEncoderLayer(
                d_model=self.latent_dim,
                nhead=num_heads,
                dim_feedforward=ff_size,
                dropout=dropout,
                activation="gelu",
                batch_first=True)
            self.text_encoder = nn.TransformerEncoder(
                TransEncoderLayer,
                num_layers=num_layers)
        self.results = {}

    def extract_text_feature(self, text, clip_model, device):
        text = clip.tokenize([text]).to(device)
        with torch.no_grad():
            text_features = clip_model.encode_text(text)
        return text_features
    
    def encode_text(self, text, device):
        with torch.no_grad():
            text = clip.tokenize(text, truncate=True).to(device)
            x = self.clip.token_embedding(text).type(self.clip.dtype)  # [batch_size, n_ctx, d_model]

            x = x + self.clip.positional_embedding.type(self.clip.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.clip.transformer(x)
            x = self.clip.ln_final(x).type(self.clip.dtype)

        # B, T, D
        xf_out = x.permute(1, 0, 2)
        return xf_out

    def encoder_text_batch(self, text, device):
        output = []
        batch_size = 128
        cur_idx = 0
        N = len(text)
        while cur_idx < N:
            cur_text = text[cur_idx: cur_idx + batch_size]
            cur_idx = cur_idx + batch_size
            features = self.encode_text(cur_text, device)
            output.append(features)
            print(cur_idx)
        output = torch.cat(output, dim=0)
        return output

    def retrieval(self, caption, length, clip_model, device):
        value = hash(caption)
        if value in self.results:
            return self.results[value]
        text_feature = self.extract_text_feature(caption, clip_model, device)
        
        rel_length = torch.LongTensor([length]).to(device)
        rel_length = torch.abs(rel_length - length) / torch.clamp(rel_length, min=length)
        score = F.cosine_similarity(self.text_features.to(device), text_feature) * torch.exp(-rel_length)
        indexes = torch.argsort(score, descending=True)
        data = []
        cnt = 0
        for idx in indexes:
            caption, motion, m_length = self.captions[idx], self.motions[idx], self.m_lengths[idx]
            if m_length != length:
                cnt += 1
                data.append(idx.item())
                if cnt == self.num_retrieval:
                    self.results[value] = data
                    return data
        assert False

    def generate_src_mask(self, T, length):
        B = len(length)
        src_mask = torch.ones(B, T)
        for i in range(B):
            for j in range(length[i], T):
                src_mask[i, j] = 0
        return src_mask

    def forward(self, captions, lengths, clip_model, device):
        B = len(captions)
        all_indexes = []
        for b_ix in range(B):
            batch_indexes = self.retrieval(captions[b_ix], lengths[b_ix], clip_model, device)
            all_indexes.extend(batch_indexes)
        all_indexes = np.array(all_indexes)
        N = all_indexes.shape[0]
        all_motions = torch.Tensor(self.motions[all_indexes]).to(device)
        all_m_lengths = torch.Tensor(self.m_lengths[all_indexes]).long()
        all_captions = self.captions[all_indexes].tolist()
            
        T = all_motions.shape[1]
        src_mask = self.generate_src_mask(T, all_m_lengths).to(device)
        re_motion = self.motion_proj(all_motions) + self.motion_pos_embedding.unsqueeze(0)
        re_motion = self.motion_encoder(re_motion, src_key_padding_mask=1 - src_mask)
        re_motion = re_motion.view(B, self.num_retrieval, T, -1).contiguous()
        # stride 4
        re_motion = re_motion[:, :, ::4, :].contiguous()
        
        src_mask = src_mask[:, ::4].contiguous()
        src_mask = src_mask.view(B, -1).contiguous()

        T = 77
        all_text_seq_features = torch.Tensor(self.text_seq_features[all_indexes]).to(device)
        re_text = self.text_encoder(all_text_seq_features).view(B, self.num_retrieval, T, -1).contiguous()
        re_text = re_text[:, :, -1, :].contiguous().unsqueeze(2)
        re_feat = re_motion + re_motion
        
        T = re_motion.shape[2]
        re_feat = re_feat.view(B, self.num_retrieval * T, -1).contiguous()
        return re_feat


@SUBMODULES.register_module()
class ReMoDiffuseTransformer(DiffusionTransformer):
    def __init__(self, guide_scale=None, retr_guide_scale=None, retrieval_cfg=None, waypoint=False, **kwargs):
        super().__init__(**kwargs)
        self.guide_scale = guide_scale
        self.retr_guide_scale = retr_guide_scale
        self.waypoint = waypoint
        if retrieval_cfg is None:
            self.use_retrieval = False
        else:
            self.database = RetrievalDatabase(**retrieval_cfg)
            self.use_retrieval = True
        
    def get_precompute_condition(self, text=None, motion_length=None, xf_out=None, re_feat=None, device=None):
        if xf_out is None:
            xf_out = self.encode_text(text, device)
        output = {'xf_out': xf_out}
        if self.use_retrieval:
            if re_feat is None:
                re_feat = self.database(text, motion_length, self.clip, device)
            output['re_feat'] = re_feat
        return output

    def forward_train(self, h=None, src_mask=None, emb=None, xf_out=None, re_feat=None, **kwargs):
        B, T = h.shape[0], h.shape[1]
        cond_type = random.randint(0, 99)
        for module in self.temporal_decoder_blocks:
            h = module(x=h, xf=xf_out, emb=emb, src_mask=src_mask, cond_type=cond_type, re_feat=re_feat)

        output = self.out(h).view(B, T, -1).contiguous()
        return output
    
    def forward_test(self, h=None, src_mask=None, emb=None, xf_out=None, re_feat=None, **kwargs):
        B, T = h.shape[0], h.shape[1]
        scale = self.guide_scale
        
        # for waypoint heading
        if self.waypoint:
            src_mask = src_mask * 0 + 1

        # # for unconditional
        # cond_type = 0
        # for module in self.temporal_decoder_blocks: 
        #     h = module(x=h, xf=xf_out, emb=emb, src_mask=src_mask, cond_type=cond_type, re_feat=re_feat) 
        #     output = self.out(h).view(B, T, -1).contiguous() 
        # return output
        
        if abs(scale - 1.0) < 0.000001:
            cond_type = 99
            for module in self.temporal_decoder_blocks:
                h = module(x=h, xf=xf_out, emb=emb, src_mask=src_mask, cond_type=cond_type, re_feat=re_feat)
            output = self.out(h).view(B, T, -1).contiguous()
        elif not self.use_retrieval:
            h0 = h
            cond_type = 0
            for module in self.temporal_decoder_blocks:
                h0 = module(x=h0, xf=xf_out, emb=emb, src_mask=src_mask, cond_type=cond_type, re_feat=re_feat)
            out0 = self.out(h0).view(B, T, -1).contiguous()
            h1 = h
            cond_type = 1
            for module in self.temporal_decoder_blocks:
                h1 = module(x=h1, xf=xf_out, emb=emb, src_mask=src_mask, cond_type=cond_type, re_feat=re_feat)
            out1 = self.out(h1).view(B, T, -1).contiguous()
            output = out0 + scale * (out1 - out0)
        else:
            if random.randint(0, 1) == 0:
                first_cond_type = 10
                second_cond_type = 99
                scale = self.guide_scale
            else:
                first_cond_type = 1
                second_cond_type = 99
                scale = self.retr_guide_scale
            h0 = h
            for module in self.temporal_decoder_blocks:
                h0 = module(x=h0, xf=xf_out, emb=emb, src_mask=src_mask, cond_type=first_cond_type, re_feat=re_feat)
            out0 = self.out(h0).view(B, T, -1).contiguous()
            h1 = h
            
            for module in self.temporal_decoder_blocks:
                h1 = module(x=h1, xf=xf_out, emb=emb, src_mask=src_mask, cond_type=second_cond_type, re_feat=re_feat)
            out1 = self.out(h1).view(B, T, -1).contiguous()
            output = out0 + scale * (out1 - out0)

        return output
