import numpy as np
import torch

from ..get_model import get_motion_model
from .base_evaluator import BaseEvaluator
from ..utils import calculate_diversity


class DiversityEvaluator(BaseEvaluator):
    
    def __init__(self,
                 data_len=0,
                 motion_encoder_name=None,
                 motion_encoder_path=None,
                 num_samples=300,
                 batch_size=None,
                 drop_last=False,
                 replication_times=1,
                 replication_reduction='statistics',
                 **kwargs):
        super().__init__(
            replication_times=replication_times,
            replication_reduction=replication_reduction,
            batch_size=batch_size,
            drop_last=drop_last,
            eval_begin_idx=0,
            eval_end_idx=data_len
        )
        self.num_samples = num_samples
        self.append_indexes = None
        self.motion_encoder_name = motion_encoder_name
        self.motion_encoder = get_motion_model(motion_encoder_name, motion_encoder_path)
        self.model_list = [self.motion_encoder]
        
    def single_evaluate(self, results, gt_flag):
        results = self.prepare_results(results)
        motion = results['motion']
        pred_motion = results['pred_motion']
        motion_length = results['motion_length']
        motion_mask = results['motion_mask']
        self.motion_encoder.to(motion.device)
        with torch.no_grad():
            if self.motion_encoder_name == 'kit_ttc':
                pred_motion_emb = (self.motion_encoder(pred_motion, motion_mask))[0]
                pred_motion_emb = pred_motion_emb.cpu().detach().numpy()
                gt_motion_emb = (self.motion_encoder(motion.to(torch.float32), motion_mask))[0]
                gt_motion_emb = gt_motion_emb.cpu().detach().numpy()
            else:
                pred_motion_emb = self.motion_encoder(pred_motion, motion_length, motion_mask).cpu().detach().numpy()
                gt_motion_emb = self.motion_encoder(motion, motion_length, motion_mask).cpu().detach().numpy()
            diversity = calculate_diversity(pred_motion_emb, self.num_samples)
            gt_diversity = calculate_diversity(gt_motion_emb, self.num_samples)
        if gt_flag == 0:
            return diversity
        else:
            return gt_diversity
        
    def parse_values(self, values, gt_values):
        metrics = {}
        metrics['Diversity (mean)'] = values[0]
        metrics['Ground Truth Diversity (mean)'] = gt_values[0]
        metrics['Diversity (conf)'] = values[1]
        metrics['Ground Truth Diversity (conf)'] = gt_values[1]
        return metrics
