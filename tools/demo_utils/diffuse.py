import mmcv
import numpy as np
import torch
from diffplanner.models import build_architecture
from mmcv.runner import load_checkpoint


config_path = "configs/planner/human.py"
ckpt_path = "pretrained_models/planner_humanml.pth"
mean_path = "asset/normalization_params/human_pml3d/mean.npy"
std_path = "asset/normalization_params/human_pml3d/std.npy"
x_dim=247


mean = np.load(mean_path)
std = np.load(std_path)

cfg = mmcv.Config.fromfile(config_path)
cfg.model.model.guide_scale = 1.
model = build_architecture(cfg.model)
model.eval()

load_checkpoint(model, ckpt_path, map_location='cpu')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
B = 4


def get_transl(transl):
        x_beg, y_beg = 0, 0
        x_ed, y_ed = transl

        x_beg = -0.5 * x_ed
        y_beg = -0.5 * y_ed

        x_ed = 0.5 * x_ed
        y_ed = 0.5 * y_ed

        x_mean, x_std, y_mean, y_std = mean[0], std[0], mean[1], std[1]

        x_beg = (x_beg - x_mean) / x_std
        y_beg = (y_beg - y_mean) / y_std

        x_ed = (x_ed - x_mean) / x_std if x_ed is not None else None
        y_ed = (y_ed - y_mean) / y_std if y_ed is not None else None

        trans_req = np.stack([x_beg, y_beg, x_ed, y_ed], -1)
        trans_req = torch.from_numpy(trans_req).cuda().float()

        return trans_req

def infer_motion_diffusion(text, pre_seq, transl, motion_length):
    print('start diffusion!')

    motion = torch.zeros(B, motion_length, x_dim).to(device)
    motion_mask = torch.ones(B, motion_length).to(device)
    motion_length = torch.Tensor([motion_length] * B).long().to(device)

    input = {
        'motion': motion,
        'motion_mask': motion_mask,
        'motion_length': motion_length,
        'motion_metas': [{'text': text}] * B
    }

    def preprocess_pre_seq(pre_seq):
        pre_seq = (pre_seq - mean) / (std + 1e-6)
        return  torch.tensor(pre_seq).to(device)
    
    if pre_seq is not None:
        pre_seq = preprocess_pre_seq(pre_seq)

    if transl is not None:
        transl = get_transl(transl)

    with torch.no_grad():
        input['inference_kwargs'] = {}

        input['inference_kwargs']['pre_seq'] = pre_seq
        input['inference_kwargs']['trans_req'] = transl
        
        output_new_list = []
        all_output = model(**input)
        for i in range(B):
            output_new = all_output[i]['pred_motion']
            output_new_list.append(output_new)
        output_new = torch.stack(output_new_list, dim=0)
        
        pred_motion = output_new.cpu().detach().numpy()
        pred_motion = pred_motion * std + mean

    return pred_motion