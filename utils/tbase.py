import os
import torch
from model import getNetwork
#from utils import moveToGPUDevice
from .gpu import moveToGPUDevice


class TBase:
    def __init__(self, data_dir, log_config, general_config):
        self.data_dir = data_dir
        assert os.path.isdir(self.data_dir)

        self.log_config = log_config
        self.general_config = general_config

        self.device = self.general_config['hardware']['gpuDevice']
        self.dtype = self.general_config['model']['dtype']

    def _loadNetFromCheckpoint(self):
        ckpt_file = self.general_config['model']['CkptFile']
        print('Loading checkpoint from {}'.format(ckpt_file))
        assert ckpt_file
        checkpoint = torch.load(ckpt_file,
                map_location=self.general_config['hardware']['gpuDevice'])

        self.net = getNetwork(self.general_config['model']['type'],
                self.general_config['simulation'])
        self.net.load_state_dict(checkpoint['model_state_dict'])
        moveToGPUDevice(self.net, self.device, self.dtype)
        self.log_config.copyModelFile(self.net)
