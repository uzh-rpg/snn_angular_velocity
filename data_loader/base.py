import numpy as np
import  h5py
import os
from pathlib import Path
import torch
from torch.utils.data.dataset import Dataset

from .utils import SpikeRepresentationGenerator


class DatasetBase(Dataset):
    def __init__(self, mode: str, data_augm_probability: float=None):
        assert mode in {'train', 'val', 'test'}
        if mode != 'test':
            raise NotImplementedError("Only testing supported for now")
        self.mode = mode
        self.height = 180
        self.width = 240
        self.subsequences = list()
        self.repr_generator = SpikeRepresentationGenerator(self.height, self.width)
        self.data_augm_prob = data_augm_probability
        if self.data_augm_prob:
            assert 0. <= self.data_augm_prob < 1.
        self.data_augmentation = DataAugmentation(data_augm_probability)
        self.nTimeBins = None

    def getHeightAndWidth(self):
        assert self.height
        assert self.width
        return self.height, self.width

    @staticmethod
    def isDataFile(filepath: str):
        suffix = Path(filepath).suffix
        return suffix == '.h5' or suffix == '.npz'

    def __len__(self):
        assert self.subsequences
        return len(self.subsequences)

    def __getitem__(self, index: int):
        assert self.subsequences
        subseq_file = self.subsequences[index]

        data = dict()
        if Path(subseq_file).suffix == '.npz':
            np_data = np.load(subseq_file)

            data['ev_xy'] = torch.from_numpy(np_data['ev_xy'])
            data['ev_pol'] = torch.from_numpy(np_data['ev_pol'])
            data['ev_ts_us'] = torch.from_numpy(np_data['ev_ts'])
            data['ang_xyz'] = torch.from_numpy(np_data['ang_xyz'])
            data['ang_ts_us'] = torch.from_numpy(np_data['ang_ts'])
        else:
            assert Path(subseq_file).suffix == '.h5'
            with h5py.File(subseq_file, "r") as hf:
                data['ev_xy'] = torch.from_numpy(hf['ev_xy'][()])
                data['ev_ts_us'] = torch.from_numpy(hf['ev_ts'][()])
                data['ev_pol'] = torch.from_numpy(hf['ev_pol'][()])
                data['ang_xyz'] = torch.from_numpy(hf['ang_xyz'][()])
                data['ang_ts_us'] = torch.from_numpy(hf['ang_ts'][()])

        if self.nTimeBins is None:
            self.nTimeBins = data['ang_ts_us'].numel()
        else:
            assert self.nTimeBins == data['ang_ts_us'].numel()

        if self.mode == 'train' and self.data_augm_prob is not None:
            data = self.data_augmentation.apply(data, self.height, self.width)

        assert self.repr_generator
        spike_tensor = self.repr_generator.getSlayerSpikeTensor(data['ev_pol'],
                                                                data['ev_xy'],
                                                                data['ev_ts_us'],
                                                                data['ang_ts_us'])
        data['ang_xyz'] = data['ang_xyz'].t()
        assert data['ang_xyz'].size(0) == 3
        out = {
            'file_number': int(''.join(filter(str.isdigit, Path(subseq_file).stem))),
            'spike_tensor': spike_tensor,
            'angular_velocity': data['ang_xyz']
        }
        assert len(data) == 5
        return out


class DataAugmentation:
    def __init__(self, prob: float):
        self._prob = prob
        if prob is not None:
            assert 0 <= prob < 1

    def apply(self, data, height, width):
        raise NotImplementedError("Data augmention is not supported for now")
