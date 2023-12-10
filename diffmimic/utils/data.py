import random

import numpy as np
from torch.utils import data
from torch.utils.data.sampler import RandomSampler
import glob
import torch
import os


def numpy_collate(batch):
  if isinstance(batch[0], np.ndarray):
    return np.stack(batch)
  elif isinstance(batch[0], (tuple,list)):
    transposed = zip(*batch)
    return [numpy_collate(samples) for samples in transposed]
  else:
    return np.array(batch)


class NumpyLoader(data.DataLoader):
  def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):
    super(self.__class__, self).__init__(dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=numpy_collate,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn)


class MotionDataset(data.Dataset):
    def __init__(self, motion_dir, seq_len=60, subset=None, all=False, resample=False):
        # motion_path_list = glob.glob(os.path.join(motion_dir, "*.npy"))
        if 'kit' in motion_dir:
            list_path = 'data/kitml_datalist.txt'
            text_dir = 'data/kit_texts'
        elif 'human' in motion_dir:
            # list_path = '/scratch/users/ntu/cheeguan/jwren/data/humanml3d.txt'
            list_path = 'data/humanml3d_datalist.txt'
            text_dir = 'data/humanml3d_text'
        else:
            raise NotImplementedError
        with open(list_path) as f:
            motion_list = f.read().splitlines()
            motion_path_list = [os.path.join(motion_dir, fname) for fname in motion_list]
            motion_path_list = [x for x in motion_path_list if os.path.isfile(x)]
        if subset is not None:
            motion_path_list = motion_path_list[:subset]
        self.frame_list = []
        if all:
            for fpath in motion_path_list:
                valid_start_frames = [(fpath, 0)]
                self.frame_list += valid_start_frames
        else:
            if resample:
                for fpath in motion_path_list:
                    motion = np.array(np.load(fpath))
                    text_path = os.path.join(text_dir, os.path.basename(fpath).replace('.npy', '.txt'))
                    valid_start_frames = [(fpath, x) for x in range(max(0, motion.shape[0]-seq_len)+1)]
                    with open(text_path) as f:
                        text_description = f.read().splitlines()
                        text_description = [x.split('#')[0] for x in text_description]
                        text_description = ' '.join(text_description)
                    if 'walk' not in text_description:
                        self.frame_list += valid_start_frames * 10
                    else:
                        self.frame_list += valid_start_frames
            else:
                for fpath in motion_path_list:
                    motion = np.array(np.load(fpath))
                    valid_start_frames = [(fpath, x) for x in range(max(0, motion.shape[0]-seq_len)+1)]
                    self.frame_list += valid_start_frames


        # self.motion_path_list = glob.glob(os.path.join(motion_dir, "*.npy"))
        # self.motion_path_list = [fn for fn in self.motion_path_list if np.array(np.load(fn)).shape[0] >= seq_len]
        self.seq_len = seq_len
        print(len(self.frame_list))

    def __len__(self):
        return len(self.frame_list)

    def __getitem__(self, idx):
        fpath, start_idx = self.frame_list[idx]
        demo_traj = np.array(np.load(fpath))
        demo_traj = demo_traj[start_idx:start_idx+self.seq_len]
        mask = np.ones(self.seq_len)
        demo_size = demo_traj.shape[0]
        padding_size = self.seq_len - demo_size
        # padded_demo_traj = demo_traj
        # # padding
        # while padded_demo_traj.shape[0] < self.seq_len:
        #     mask[padded_demo_traj.shape[0]] *= 0.
        #     padded_demo_traj = np.concatenate([padded_demo_traj, demo_traj])[:self.seq_len]
        # return padded_demo_traj, mask
        if padding_size > 0:
            demo_traj = np.concatenate([demo_traj, np.zeros((padding_size,)+demo_traj.shape[1:])])
            mask[demo_size-1:] *= 0.
        return demo_traj, mask
    


class MotionDatasetRollout(data.Dataset):
    def __init__(self, motion_dir, seg_info=None):
        if 'kit' in motion_dir:
            list_path = 'data/kitml_datalist.txt'
        elif 'human' in motion_dir:
            list_path = 'data/humanml3d_datalist.txt'
        else:
            raise NotImplementedError
        with open(list_path) as f:
            self.motion_list = f.read().splitlines()
            if 'kit' in motion_dir:
                # self.motion_list = ['00586.npy', '00603.npy', '00606.npy', '02916.npy','02351.npy', '02593.npy']
                # self.motion_list = ['02916.npy','02351.npy', '02593.npy'] # 360 jump
                # self.motion_list = ['00607.npy', '00608.npy', '00609.npy', '00610.npy','00611.npy', '00612.npy', '01236.npy'] # kick
                self.motion_list = ['03329.npy', '02797.npy', '00653.npy', '01317.npy', '02320.npy', '02772.npy', '02355.npy', '02625.npy', '02916.npy', '03299.npy'] # acrobatic
            #    self.motion_list = ['02285.npy', '02291.npy', '02292.npy', '02626.npy', '03449.npy', '03311.npy', '03304.npy', '02823.npy', '02628.npy', '02283.npy', '01234.npy'] # dance
            else:
                self.motion_list = ['000000.npy', '000002.npy', '000264.npy', '000499.npy']
            # self.motion_list = [fname for fname in self.motion_list if not os.path.isfile(os.path.join('/mnt/lustre/share/jwren/cvae_kitml_traj_valid_v3/', fname))]
            # self.motion_list = [fname for fname in self.motion_list if not os.path.isfile(os.path.join('/mnt/lustre/share/jwren/cvae_kitml_traj_invalid_v3/', fname))]


            if seg_info is not None:
                num, idx = seg_info
                total_len = len(self.motion_list)
                seg_len = (total_len // num) + 1
                idx_start = seg_len * idx
                idx_end = seg_len * (idx + 1)
                self.motion_list = self.motion_list[idx_start:idx_end]
            motion_path_list = [os.path.join(motion_dir, fname) for fname in self.motion_list]
            self.motion_path_list = [x for x in motion_path_list if os.path.isfile(x)]
        print(len(self.motion_path_list))

    def __len__(self):
        return len(self.motion_path_list)

    def __getitem__(self, idx):
        fpath, fname = self.motion_path_list[idx], self.motion_list[idx]
        demo_traj = np.array(np.load(fpath))

        return demo_traj, fname


class InfiniteSampler(data.sampler.Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        while True:
            yield from iter(torch.randperm(len(self.data_source)).tolist())

    def __len__(self):
        return 2 ** 31