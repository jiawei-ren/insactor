import argparse
import os
import os.path as osp

import mmcv
import torch
from mmcv import DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (
    get_dist_info,
    init_dist,
    load_checkpoint,
    wrap_fp16_model,
)
import sys

from diffplanner.apis import multi_gpu_test, single_gpu_test
from diffplanner.datasets import build_dataloader, build_dataset
from diffplanner.models import build_architecture

from tqdm import tqdm

torch.manual_seed(0)
import random
random.seed(0)
import numpy as np
np.random.seed(0)

def parse_args():
    parser = argparse.ArgumentParser(description='mmhuman3d test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        '--work-dir', help='the dir to save evaluation results')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--gpu_collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default='cuda',
        help='device used for testing')
    parser.add_argument(
        '--physmode',
        choices=['none', 'normal'],
        default='none',
        help='physmode')
    parser.add_argument(
        '--perturb',
        choices=['false', 'true'],
        default='false')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    # the extra round_up data will be removed during gpu/cpu collect
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        round_up=False)

    # build the model and load checkpoint
    model = build_architecture(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    if not distributed:
        if args.device == 'cpu':
            model = model.cpu()
        else:
            model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)
        
    # for i in tqdm(range(0, len(outputs))):
    #     outputs[i]['pred_motion'] = outputs[i]['motion']


    phys_mode = args.physmode
    phys_setting = 'kit' if 'kit' in args.config else 'human'

    if phys_mode == 'normal':
        ##########################
        # This part is for physics simulation
        import numpy as np
        from simulate.rollout import execute_actions
        from simulate.scene import add_scene_to_traj, remove_scene_from_traj

        if phys_setting == 'kit':
            mean_path = 'data/datasets/kit_pml/mean.npy'
            std_path = 'data/datasets/kit_pml/std.npy'
        else:
            mean_path = 'data/datasets/human_pml3d/mean.npy'
            std_path = 'data/datasets/human_pml3d/std.npy'

        mean = np.load(mean_path)
        std = np.load(std_path)

        if phys_setting == 'kit':
            num_env = 316
        else:
            num_env = 2236


        print(len(outputs))
        for i in tqdm(range(0, len(outputs), num_env)):

            pred_motion = torch.stack([outputs[j]['pred_motion'] for j in range(i, i+num_env)], 0)
            pred_motion = pred_motion.detach().cpu().numpy()
            pred_motion = pred_motion * std + mean 
            # np.save('./planned_traj.npy', pred_motion.transpose(1,0,2))
            pred_motion = add_scene_to_traj(pred_motion, None, scene='vis')

            pred_motion_phys = execute_actions(pred_motion, num_env, perturb=args.perturb == 'true')

            assert pred_motion_phys.shape == pred_motion.shape, (pred_motion_phys.shape, pred_motion.shape)
            # np.save('./simulated_traj.npy', pred_motion_phys.transpose(1,0,2))
            pred_motion_phys = remove_scene_from_traj(pred_motion_phys)[0]
            pred_motion_phys = np.array(pred_motion_phys)
            pred_motion_phys = (pred_motion_phys-mean) / (std+1e-9)
            pred_motion_phys = torch.from_numpy(pred_motion_phys).cuda().float()
            for j in range(i, i+num_env):
                outputs[j]['pred_motion'] = pred_motion_phys[j-i]

        ################


    # for i in tqdm(range(0, len(outputs))):
    #     outputs[i]['pred_motion'][outputs[i]['motion_length']:] = outputs[i]['motion'][outputs[i]['motion_length']:]

    rank, _ = get_dist_info()
    if rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        results = dataset.evaluate(outputs, args.work_dir)
        for k, v in results.items():
            print(f'\n{k} : {v:.4f}')

    if args.out and rank == 0:
        print(f'\nwriting results to {args.out}')
        mmcv.dump(results, args.out)


if __name__ == '__main__':
    main()